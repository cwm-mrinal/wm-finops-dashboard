import streamlit as st
import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from fpdf import FPDF
from io import BytesIO
import plotly.io as pio
from sklearn.linear_model import LinearRegression
import numpy as np
import logging
import time
import base64
from PIL import Image
from aws_accounts import sts

# Constants for Bedrock
REGION = 'ap-south-1'
MODEL_ID = 'arn:aws:bedrock:ap-south-1:036160411876:inference-profile/apac.anthropic.claude-sonnet-4-20250514-v1:0'
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1
MAX_TOKENS = 8000

# Setup logging
logging.basicConfig(filename='finops_dashboard.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# ---------- Role Assumption ----------
def assume_role(account_id: str):
    try:
        sts_response = sts(account_id)
        if sts_response["status"] != 200:
            raise Exception(f"Failed to create session for account {account_id}: {sts_response['message']}")
        return boto3.Session(
            aws_access_key_id=sts_response["data"]["AccessKeyId"],
            aws_secret_access_key=sts_response["data"]["SecretAccessKey"],
            aws_session_token=sts_response["data"]["SessionToken"],
        )
    except Exception as e:
        logger.error(f"Role assumption failed: {str(e)}")
        raise

# Initialize session state for AWS credentials and configuration
if 'aws_config' not in st.session_state:
    st.session_state.aws_config = {
        'profiles': ['default'],
        'account_id': '',
        'regions': ['us-east-1'],
        'start_date': (datetime.now().date() - timedelta(days=30)),
        'end_date': datetime.now().date(),
        'tags': [],
        'combine': False,
        'report_name': 'aws_finops_report',
        'report_types': ['csv'],
        'output_dir': './reports',
        'audit': False,
        'trend': False,
        'cost_threshold': 100.0,
        'forecast_days': 30
    }

# Function to create a session (either profile-based or role-based)
def create_session(profile, account_id):
    if account_id:
        try:
            return assume_role(account_id)
        except Exception as e:
            st.error(f"Role assumption failed for account {account_id}: {str(e)}")
            return None
    else:
        return boto3.Session(profile_name=profile)

# Function to get AWS Cost Explorer data
@st.cache_data(ttl=3600)
def get_cost_data(start_date, end_date, profiles, account_id, tags, granularity='MONTHLY'):
    cost_data = []
    account_ids = {}
    for profile in profiles:
        try:
            session = create_session(profile, account_id)
            if session is None:
                continue
            sts_client = session.client('sts')
            account_id_fetch = sts_client.get_caller_identity()['Account']
            account_ids[profile] = account_id_fetch
            ce_client = session.client('ce')
            query = {
                'TimePeriod': {'Start': start_date.strftime('%Y-%m-%d'), 'End': end_date.strftime('%Y-%m-%d')},
                'Granularity': granularity,
                'Metrics': ['UnblendedCost', 'UsageQuantity'],
                'GroupBy': [{'Type': 'DIMENSION', 'Key': 'SERVICE'}, {'Type': 'TAG', 'Key': tags[0].split('=')[0]}] if tags else [{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            }
            if tags:
                query['Filter'] = {'Tags': {'Key': tags[0].split('=')[0], 'Values': [tags[0].split('=')[1]]}}
            response = ce_client.get_cost_and_usage(**query)
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    profile_or_account = account_id_fetch if st.session_state.aws_config['combine'] else profile
                    cost_data.append({
                        'Profile': profile_or_account,
                        'Service': group['Keys'][0],
                        'Tag': group['Keys'][1] if tags else 'N/A',
                        'Cost': float(group['Metrics']['UnblendedCost']['Amount']),
                        'Usage': float(group['Metrics']['UsageQuantity']['Amount']),
                        'Currency': group['Metrics']['UnblendedCost']['Unit'],
                        'Date': result['TimePeriod']['Start']
                    })
        except Exception as e:
            st.error(f"Error fetching cost data for profile {profile}: {str(e)}")
            logger.error(f"Cost data error for {profile}: {str(e)}")
    return pd.DataFrame(cost_data)

# Function to get EC2 instance status
@st.cache_data(ttl=3600)
def get_ec2_status(profiles, account_id, regions):
    ec2_data = []
    for profile in profiles:
        session = create_session(profile, account_id)
        if session is None:
            continue
        for region in regions:
            try:
                ec2_client = session.client('ec2', region_name=region)
                response = ec2_client.describe_instances()
                for reservation in response['Reservations']:
                    for instance in reservation['Instances']:
                        ec2_data.append({
                            'Profile': profile,
                            'Region': region,
                            'InstanceId': instance['InstanceId'],
                            'InstanceType': instance.get('InstanceType', 'N/A'),
                            'State': instance['State']['Name'],
                            'LaunchTime': instance.get('LaunchTime', 'N/A'),
                            'Tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                        })
            except Exception as e:
                st.error(f"Error fetching EC2 data for profile {profile} in region {region}: {str(e)}")
                logger.error(f"EC2 data error for {profile} in {region}: {str(e)}")
    return pd.DataFrame(ec2_data)

# Function to get budget data
@st.cache_data(ttl=3600)
def get_budget_data(profiles, account_id):
    budget_data = []
    for profile in profiles:
        session = create_session(profile, account_id)
        if session is None:
            continue
        try:
            budgets_client = session.client('budgets')
            response = budgets_client.describe_budgets(AccountId=session.client('sts').get_caller_identity()['Account'])
            for budget in response['Budgets']:
                actual_spend = float(budget.get('CalculatedSpend', {}).get('ActualSpend', {}).get('Amount', 0))
                limit = float(budget['BudgetLimit']['Amount'])
                budget_data.append({
                    'Profile': profile,
                    'BudgetName': budget['BudgetName'],
                    'Limit': limit,
                    'ActualSpend': actual_spend,
                    'Utilization': (actual_spend / limit * 100) if limit > 0 else 0,
                    'Currency': budget['BudgetLimit']['Unit']
                })
        except Exception as e:
            st.error(f"Error fetching budget data for profile {profile}: {str(e)}")
            logger.error(f"Budget data error for {profile}: {str(e)}")
    return pd.DataFrame(budget_data)

# Function to get audit report
@st.cache_data(ttl=3600)
def get_audit_report(profiles, account_id, regions):
    audit_data = []
    for profile in profiles:
        session = create_session(profile, account_id)
        if session is None:
            continue
        for region in regions:
            try:
                ec2_client = session.client('ec2', region_name=region)
                response = ec2_client.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['stopped']}])
                stopped_instances = [instance['InstanceId'] for reservation in response['Reservations'] for instance in reservation['Instances']]
                response = ec2_client.describe_volumes(Filters=[{'Name': 'status', 'Values': ['available']}])
                unused_volumes = [volume['VolumeId'] for volume in response['Volumes']]
                response = ec2_client.describe_addresses()
                unused_eips = [address['PublicIp'] for address in response['Addresses'] if 'InstanceId' not in address]
                audit_data.append({
                    'Profile': profile,
                    'Region': region,
                    'StoppedInstances': ', '.join(stopped_instances) if stopped_instances else 'None',
                    'UnusedVolumes': ', '.join(unused_volumes) if unused_volumes else 'None',
                    'UnusedEIPs': ', '.join(unused_eips) if unused_eips else 'None'
                })
            except Exception as e:
                st.error(f"Error fetching audit data for profile {profile} in region {region}: {str(e)}")
                logger.error(f"Audit data error for {profile} in {region}: {str(e)}")
    return pd.DataFrame(audit_data)

# Function to detect cost anomalies
def detect_anomalies(cost_df, threshold):
    anomalies = []
    if not cost_df.empty:
        grouped = cost_df.groupby(['Profile', 'Service'])['Cost'].mean().reset_index()
        for _, row in grouped.iterrows():
            if row['Cost'] > threshold:
                anomalies.append({
                    'Profile': row['Profile'],
                    'Service': row['Service'],
                    'AverageCost': row['Cost'],
                    'Threshold': threshold
                })
    return pd.DataFrame(anomalies)

# Function for cost forecasting
def forecast_costs(trend_df, forecast_days):
    if trend_df.empty:
        return pd.DataFrame()
    trend_df['Date'] = pd.to_datetime(trend_df['Date'])
    trend_df['Days'] = (trend_df['Date'] - trend_df['Date'].min()).dt.days
    X = trend_df[['Days']].values
    y = trend_df['Cost'].values
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.array([[i] for i in range(trend_df['Days'].max() + 1, trend_df['Days'].max() + forecast_days + 1)])
    future_dates = [trend_df['Date'].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
    predictions = model.predict(future_days)
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Cost': predictions,
        'Profile': 'Forecast',
        'Service': 'All'
    })
    return forecast_df

# Function to get recommendations from AWS Bedrock
def get_recommendations(cost_df, audit_df):
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {str(e)}")
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return "Unable to generate recommendations at this time."

    try:
        cost_summary = {}
        if not cost_df.empty:
            cost_summary = {
                "total_cost": float(cost_df['Cost'].sum()),
                "services": cost_df.groupby('Service')['Cost'].sum().to_dict(),
                "profiles": cost_df['Profile'].unique().tolist()
            }
        else:
            cost_summary = {"summary": "No cost data available"}

        audit_summary = {}
        if not audit_df.empty:
            audit_summary = {
                "stopped_instances": audit_df['StoppedInstances'].tolist(),
                "unused_volumes": audit_df['UnusedVolumes'].tolist(),
                "unused_eips": audit_df['UnusedEIPs'].tolist()
            }
        else:
            audit_summary = {"summary": "No audit data available"}

        infrastructure_data = {
            "cost_data": cost_summary,
            "audit_data": audit_summary
        }

        query = "Provide cost optimization recommendations based on the provided AWS cost and audit data."

        system_prompt = """
You are a support assistant specializing in AWS billing, usage analysis, and cost optimization. You help customers address issues related to unexpected charges, cost spikes, and budget control—without requesting more information.

Your Objectives:

Acknowledge the concern with empathy and professionalism.

Make intelligent assumptions when the issue is vague.

Provide 7–8 actionable suggestions to help the customer reduce or analyze costs.

Maintain a confident, solution-focused, and supportive tone.

Structured Response Format and Style:

Hello,

Thank you for reaching out. We understand how concerning unexpected billing charges or cost spikes can be, and we're here to help you take control of the situation quickly and effectively.

[Optional – Restate or summarize the issue]
It appears you're noticing [a sudden cost increase / unexpected usage / higher than expected billing on a service or project].

Here are a few actions you can take right away to review and reduce your AWS costs:

Use AWS Cost Explorer

Review cost trends and group by service, account, or tag to pinpoint changes.

Use the “Daily Granularity” view to isolate when the spike occurred.

Check for Idle or Underutilized Resources

Look for unused EC2 instances, idle RDS databases, or EBS volumes with minimal I/O.

Use Trusted Advisor or Compute Optimizer to identify and right-size inefficient resources.

Review Budgets and Enable Alerts

Set up AWS Budgets with alert thresholds to proactively monitor spend.

Enable Cost Anomaly Detection for early warnings about unusual usage.

We recommend implementing these steps as soon as possible to mitigate further costs. We're confident this will help you regain control over your AWS spend.

Please don’t hesitate to reach out if the situation escalates—we’re here to support you every step of the way.

Best regards,
Workmates Support
"""

        prompt = f"{system_prompt}\n\nInfrastructure Summary:\n{json.dumps(infrastructure_data, indent=2)}\n\nUser Query:\n{query}"

        if len(prompt) > 100000:
            logger.warning("Prompt size exceeds safe limit, truncating data")
            prompt = f"{system_prompt}\n\nInfrastructure Summary: (Truncated due to size)\n{json.dumps({'summary': 'Large dataset, please specify service'}, indent=2)}\n\nUser Query:\n{query}"

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(json.dumps({"event": "invoke_model_attempt", "attempt": attempt, "query": query[:100]}))
                response = bedrock_runtime.invoke_model(
                    modelId=MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload).encode("utf-8")
                )
                response_body = json.loads(response["body"].read())
                model_text = response_body["content"][0]["text"]
                logger.info(json.dumps({"event": "model_raw_output", "text": model_text[:200]}))
                return model_text
            except Exception as e:
                logger.error(f"Bedrock attempt {attempt} failed: {str(e)}")
                if attempt == MAX_RETRIES:
                    raise Exception(f"Bedrock query failed after {MAX_RETRIES} attempts: {str(e)}")
                time.sleep(RETRY_DELAY_SECONDS)
        return "Error: Unable to process query with Bedrock."
    except Exception as e:
        logger.error(f"Bedrock query error: {str(e)}")
        st.error(f"Error generating recommendations: {str(e)}")
        return f"Error processing query: {str(e)}"

# Function to export data
def export_data(df, report_name, report_types, output_dir, fig=None):
    os.makedirs(output_dir, exist_ok=True)
    files = []
    for report_type in report_types:
        try:
            if report_type == 'csv':
                path = os.path.join(output_dir, f"{report_name}.csv")
                df.to_csv(path, index=False)
                files.append(path)
            elif report_type == 'json':
                path = os.path.join(output_dir, f"{report_name}.json")
                df.to_json(path, orient='records', lines=True)
                files.append(path)
            elif report_type == 'pdf':
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"Report: {report_name}", ln=True, align='C')
                pdf.ln(10)
                for i, row in df.iterrows():
                    for col in df.columns:
                        pdf.cell(0, 10, f"{col}: {row[col]}", ln=True)
                if fig:
                    img_buffer = BytesIO()
                    pio.write_image(fig, file=img_buffer, format='png')
                    img_buffer.seek(0)
                    pdf.image(img_buffer, x=10, w=190)
                path = os.path.join(output_dir, f"{report_name}.pdf")
                pdf.output(path)
                files.append(path)
        except Exception as e:
            st.error(f"Error exporting {report_type}: {str(e)}")
            logger.error(f"Export error for {report_type}: {str(e)}")
    return files

# Function to get base64 image (for logo)
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Streamlit UI Configuration
st.set_page_config(page_title="AWS FinOps Dashboard", layout="wide", initial_sidebar_state="expanded")

# Display logo and title
logo_path = "CWM-New-Logo.png"
logo_base64 = get_base64_image(logo_path)
if logo_base64:
    st.markdown(f"""
        <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
            <img src='data:image/png;base64,{logo_base64}' width='100' style='margin-right: 1rem;'>
            <h1 style='margin: 0;'>AWS FinOps Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    st.title("AWS FinOps Dashboard")

st.markdown("Monitor, optimize, and forecast your AWS costs with advanced insights and AI-powered recommendations.")

# Enhanced CSS Styling with more animations and responsiveness
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    .stSidebar .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: background-color 0.3s, transform 0.2s;
    }
    .stSidebar .stButton > button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .main .block-container {
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 1rem;
        transition: box-shadow 0.3s;
    }
    .main .block-container:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-container .stMetric {
        font-size: 1.2rem;
        color: #333;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
        background-color: #f8f9fa;
        color: #333;
        transition: background-color 0.3s, color 0.3s;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    .st-expander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #fafafa;
        transition: background-color 0.3s;
    }
    .st-expander:hover {
        background-color: #f0f0f0;
    }
    .stButton > button {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: background-color 0.3s, transform 0.2s;
    }
    .stButton > button:hover {
        background-color: #218838;
        transform: scale(1.05);
    }
    .stDownloadButton > button {
        background-color: #6c757d;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        transition: background-color 0.3s, transform 0.2s;
    }
    .stDownloadButton > button:hover {
        background-color: #5a6268;
        transform: scale(1.05);
    }
    .stTextInput > div > input,
    .stDateInput > div > input,
    .stNumberInput > div > input {
        border-radius: 6px;
        border: 1px solid #ced4da;
        padding: 8px;
        transition: border-color 0.3s;
    }
    .stTextInput > div > input:focus,
    .stDateInput > div > input:focus,
    .stNumberInput > div > input:focus {
        border-color: #007bff;
    }
    h1, h2, h3 {
        color: #1a3c34;
        font-weight: 600;
    }
    .stProgress > div > div {
        background-color: #007bff;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        .metric-container {
            margin-bottom: 1rem;
        }
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration with icons
with st.sidebar:
    st.header("⚙️ Configuration")
    if st.button("🔄 Refresh Data", key="refresh_data"):
        st.cache_data.clear()
        st.session_state.pop('cost_df', None)
        st.success("Data refreshed successfully!")
    with st.expander("🔑 AWS Credentials & Filters", expanded=True):
        st.subheader("AWS Settings")
        account_id = st.text_input("AWS Account ID", value="", help="Enter AWS Account ID for role assumption (required).")
        profiles = st.text_input("AWS Profiles", value="default", help="Comma-separated AWS profiles (e.g., profile1,profile2).")
        regions = st.text_input("Regions", value="us-east-1", help="Comma-separated regions (e.g., us-east-1,us-west-2).")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=st.session_state.aws_config['start_date'], help="Start date for cost analysis.")
        with col2:
            end_date = st.date_input("End Date", value=st.session_state.aws_config['end_date'], help="End date for cost analysis.")
        tags = st.text_input("Tags", value="", placeholder="e.g., Team=DevOps,Env=Prod", help="Filter by tags in key=value format.")
        combine = st.checkbox("Combine Profiles", value=st.session_state.aws_config['combine'], help="Aggregate data from profiles in the same account.")

    with st.expander("📊 Report Settings", expanded=False):
        report_name = st.text_input("Report Name", value="aws_finops_report", help="Name for exported report files.")
        report_types = st.multiselect("Report Types", options=["csv", "json", "pdf"], default=["csv"], help="Select export formats.")
        output_dir = st.text_input("Output Directory", value="./reports", help="Directory to save reports.")

    with st.expander("📈 Analysis Options", expanded=False):
        audit = st.checkbox("Run Audit Report", value=st.session_state.aws_config['audit'], help="Include stopped instances, unused volumes, and EIPs.")
        trend = st.checkbox("Show Cost Trend & Forecast", value=st.session_state.aws_config['trend'], help="Include cost trend and forecast analysis.")
        cost_threshold = st.number_input("Cost Anomaly Threshold ($)", min_value=0.0, value=st.session_state.aws_config['cost_threshold'], step=10.0, help="Threshold for anomaly detection.")
        forecast_days = st.number_input("Forecast Days", min_value=1, max_value=90, value=st.session_state.aws_config['forecast_days'], help="Days for cost forecasting.")

    if st.button("✅ Apply Configuration", key="apply_config"):
        st.session_state.aws_config.update({
            'profiles': [p.strip() for p in profiles.split(',')],
            'account_id': account_id.strip(),
            'regions': [r.strip() for r in regions.split(',')],
            'start_date': start_date,
            'end_date': end_date,
            'tags': [t.strip() for t in tags.split(',') if t.strip()],
            'combine': combine,
            'report_name': report_name,
            'report_types': report_types,
            'output_dir': output_dir,
            'audit': audit,
            'trend': trend,
            'cost_threshold': cost_threshold,
            'forecast_days': forecast_days
        })
        st.success("Configuration updated successfully!")

# Main Content
config = st.session_state.aws_config

# Validation for Account ID and Region
if not config['account_id'].strip() or not config['regions'][0].strip():
    st.info("Please enter a valid AWS Account ID and at least one Region in the sidebar to access the dashboard.")
else:
    # Progress bar for data loading
    progress_bar = st.progress(0)
    progress_step = 0.2

    # Dashboard Summary Metrics with additional metrics
    st.header("Dashboard Summary")
    if 'cost_df' not in st.session_state:
        st.session_state.cost_df = get_cost_data(config['start_date'], config['end_date'], tuple(config['profiles']), config['account_id'], tuple(config['tags']))
    cost_df = st.session_state.cost_df
    total_cost = cost_df['Cost'].sum() if not cost_df.empty else 0
    avg_cost = cost_df['Cost'].mean() if not cost_df.empty else 0
    num_services = cost_df['Service'].nunique() if not cost_df.empty else 0
    total_usage = cost_df['Usage'].sum() if not cost_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Cost", f"${total_cost:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Average Daily Cost", f"${avg_cost:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Number of Services", num_services)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Usage", f"{total_usage:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Cost Analysis", "🖥️ EC2 Status", "💰 Budgets", "⚠️ Cost Anomalies", "💡 Recommendations"])

    # Cost Analysis
    with tab1:
        st.header("Cost Analysis")
        progress_bar.progress(progress_step)
        if not cost_df.empty:
            with st.expander("View Detailed Data", expanded=False):
                st.subheader("Filter Data")
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    selected_services = st.multiselect("Select Services", options=cost_df['Service'].unique(), default=cost_df['Service'].unique(), key="cost_services")
                with col_filter2:
                    selected_tags = st.multiselect("Select Tags", options=cost_df['Tag'].unique(), default=cost_df['Tag'].unique(), key="cost_tags")
                filtered_cost_df = cost_df[cost_df['Service'].isin(selected_services) & cost_df['Tag'].isin(selected_tags)]
                st.dataframe(filtered_cost_df, use_container_width=True, height=300)
            fig_bar = px.bar(filtered_cost_df, x='Service', y='Cost', color='Profile', title="Cost by Service", barmode='stack')
            st.plotly_chart(fig_bar, use_container_width=True)
            fig_pie = px.pie(filtered_cost_df, values='Cost', names='Service', title="Cost Breakdown by Service", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)
            if config['report_types']:
                files = export_data(filtered_cost_df, config['report_name'], config['report_types'], config['output_dir'], fig_bar)
                st.subheader("Download Reports")
                cols = st.columns(min(3, len(files)))
                for i, file in enumerate(files):
                    with open(file, 'rb') as f:
                        with cols[i]:
                            st.download_button(f"Download {os.path.basename(file).split('.')[-1].upper()}", f, file_name=os.path.basename(file))
        else:
            st.info("No cost data available for the selected time range and profiles.")
        progress_bar.progress(progress_step + 0.2)

    # EC2 Instance Status
    with tab2:
        st.header("EC2 Instance Status")
        ec2_df = get_ec2_status(tuple(config['profiles']), config['account_id'], tuple(config['regions']))
        if not ec2_df.empty:
            with st.expander("View Detailed Data", expanded=False):
                st.subheader("Filter Data")
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    selected_states = st.multiselect("Select Instance States", options=ec2_df['State'].unique(), default=ec2_df['State'].unique(), key="ec2_states")
                with col_filter2:
                    selected_instance_types = st.multiselect("Select Instance Types", options=ec2_df['InstanceType'].unique(), default=ec2_df['InstanceType'].unique(), key="ec2_types")
                filtered_ec2_df = ec2_df[ec2_df['State'].isin(selected_states) & ec2_df['InstanceType'].isin(selected_instance_types)]
                st.dataframe(filtered_ec2_df, use_container_width=True, height=300)
            fig_hist = px.histogram(filtered_ec2_df, x='State', color='Profile', title="EC2 Instance Status", barmode='group')
            st.plotly_chart(fig_hist, use_container_width=True)
            fig_pie_ec2 = px.pie(filtered_ec2_df, names='State', title="EC2 Status Distribution", hole=0.3)
            st.plotly_chart(fig_pie_ec2, use_container_width=True)
            if config['report_types']:
                files = export_data(filtered_ec2_df, f"{config['report_name']}_ec2", config['report_types'], config['output_dir'], fig_hist)
                st.subheader("Download Reports")
                cols = st.columns(min(3, len(files)))
                for i, file in enumerate(files):
                    with open(file, 'rb') as f:
                        with cols[i]:
                            st.download_button(f"Download {os.path.basename(file).split('.')[-1].upper()}", f, file_name=os.path.basename(file))
        else:
            st.info("No EC2 instance data available for the selected profiles and regions.")
        progress_bar.progress(progress_step + 0.4)

    # Budget Information
    with tab3:
        st.header("Budget Information")
        budget_df = get_budget_data(tuple(config['profiles']), config['account_id'])
        if not budget_df.empty:
            with st.expander("View Detailed Data", expanded=False):
                st.dataframe(budget_df, use_container_width=True, height=300)
            fig_budget = go.Figure(data=[
                go.Bar(name='Limit', x=budget_df['BudgetName'], y=budget_df['Limit']),
                go.Bar(name='Actual Spend', x=budget_df['BudgetName'], y=budget_df['ActualSpend']),
                go.Bar(name='Utilization (%)', x=budget_df['BudgetName'], y=budget_df['Utilization'])
            ])
            fig_budget.update_layout(barmode='group', title="Budget Limits vs Actual Spend vs Utilization")
            st.plotly_chart(fig_budget, use_container_width=True)
            if config['report_types']:
                files = export_data(budget_df, f"{config['report_name']}_budget", config['report_types'], config['output_dir'], fig_budget)
                st.subheader("Download Reports")
                cols = st.columns(min(3, len(files)))
                for i, file in enumerate(files):
                    with open(file, 'rb') as f:
                        with cols[i]:
                            st.download_button(f"Download {os.path.basename(file).split('.')[-1].upper()}", f, file_name=os.path.basename(file))
        else:
            st.info("No budget data available for the selected profiles.")
        progress_bar.progress(progress_step + 0.6)

    # Cost Anomalies
    with tab4:
        st.header("Cost Anomalies")
        anomaly_df = detect_anomalies(cost_df, config['cost_threshold'])
        if not anomaly_df.empty:
            with st.expander("View Detailed Data", expanded=False):
                st.dataframe(anomaly_df, use_container_width=True, height=300)
            fig_anomaly = px.bar(anomaly_df, x='Service', y='AverageCost', color='Profile', title="Cost Anomalies")
            st.plotly_chart(fig_anomaly, use_container_width=True)
            if config['report_types']:
                files = export_data(anomaly_df, f"{config['report_name']}_anomalies", config['report_types'], config['output_dir'], fig_anomaly)
                st.subheader("Download Reports")
                cols = st.columns(min(3, len(files)))
                for i, file in enumerate(files):
                    with open(file, 'rb') as f:
                        with cols[i]:
                            st.download_button(f"Download {os.path.basename(file).split('.')[-1].upper()}", f, file_name=os.path.basename(file))
        else:
            st.info("No anomalies detected based on the current threshold.")
        progress_bar.progress(progress_step + 0.8)

    # Recommendations
    with tab5:
        st.header("Cost Optimization Recommendations")
        if st.button("Generate Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations = get_recommendations(cost_df, get_audit_report(tuple(config['profiles']), config['account_id'], tuple(config['regions'])))
                st.markdown(recommendations)
                if 'pdf' in config['report_types']:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, recommendations)
                    pdf_file = os.path.join(config['output_dir'], f"{config['report_name']}_recommendations.pdf")
                    pdf.output(pdf_file)
                    with open(pdf_file, 'rb') as f:
                        st.download_button("Download Recommendations PDF", f, file_name=os.path.basename(pdf_file))
        progress_bar.progress(1.0)

    # Cost Trend and Forecast Analysis
    if config['trend']:
        with tab1:
            st.subheader("Cost Trend & Forecast Analysis")
            trend_start = config['end_date'] - timedelta(days=180)
            trend_df = get_cost_data(trend_start, config['end_date'], tuple(config['profiles']), config['account_id'], tuple(config['tags']), 'MONTHLY')
            if not trend_df.empty:
                with st.expander("View Detailed Trend Data", expanded=False):
                    col_filter1, col_filter2 = st.columns(2)
                    with col_filter1:
                        selected_trend_services = st.multiselect("Select Services for Trend", options=trend_df['Service'].unique(), default=trend_df['Service'].unique(), key="trend_services")
                    with col_filter2:
                        selected_trend_tags = st.multiselect("Select Tags for Trend", options=trend_df['Tag'].unique(), default=trend_df['Tag'].unique(), key="trend_tags")
                    filtered_trend_df = trend_df[trend_df['Service'].isin(selected_trend_services) & trend_df['Tag'].isin(selected_trend_tags)]
                    st.dataframe(filtered_trend_df, use_container_width=True, height=300)
                forecast_df = forecast_costs(filtered_trend_df, config['forecast_days'])
                combined_df = pd.concat([filtered_trend_df, forecast_df])
                fig_trend = px.line(combined_df, x='Date', y='Cost', color='Profile', title=f"Cost Trend & Forecast (Next {config['forecast_days']} Days)")
                st.plotly_chart(fig_trend, use_container_width=True)
                if config['report_types']:
                    files = export_data(combined_df, f"{config['report_name']}_trend_forecast", config['report_types'], config['output_dir'], fig_trend)
                    st.subheader("Download Trend Reports")
                    cols = st.columns(min(3, len(files)))
                    for i, file in enumerate(files):
                        with open(file, 'rb') as f:
                            with cols[i]:
                                st.download_button(f"Download Trend {os.path.basename(file).split('.')[-1].upper()}", f, file_name=os.path.basename(file))
            else:
                st.info("No trend data available for the selected time range and profiles.")

    # Audit Report
    if config['audit']:
        with tab2:
            st.subheader("Audit Report")
            audit_df = get_audit_report(tuple(config['profiles']), config['account_id'], tuple(config['regions']))
            if not audit_df.empty:
                with st.expander("View Detailed Audit Data", expanded=False):
                    st.dataframe(audit_df, use_container_width=True, height=300)
                if config['report_types']:
                    files = export_data(audit_df, f"{config['report_name']}_audit", config['report_types'], config['output_dir'])
                    st.subheader("Download Audit Reports")
                    cols = st.columns(min(3, len(files)))
                    for i, file in enumerate(files):
                        with open(file, 'rb') as f:
                            with cols[i]:
                                st.download_button(f"Download Audit {os.path.basename(file).split('.')[-1].upper()}", f, file_name=os.path.basename(file))
            else:
                st.info("No audit data available for the selected profiles and regions.")

# Footer
st.markdown('<div class="footer">© 2025 AWS FinOps Dashboard | Powered by Streamlit & AWS</div>', unsafe_allow_html=True)