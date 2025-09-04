import boto3, json

def sts(accountID, secret_name="CWM-Secret-Manager", region_name="ap-south-1"):
    client = boto3.client("secretsmanager", region_name=region_name)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = response.get("SecretString", None)

        if not secret:
            return {
                "status": 400,
                "message": "Secret not found in Secrets Manager",
                "data": None
            }

        secret = json.loads(secret)
        access_key = secret.get("ACCESS_KEY", None)
        secret_key = secret.get("SECRET_KEY", None)

        if not access_key or not secret_key:
            return {
                "status": 400,
                "message": "Missing access key or secret key",
                "data": None
            }

        sts_client = boto3.client("sts", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        role_arn = f"arn:aws:iam::{accountID}:role/CWMSessionRole"
        assumed_role = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName="CWM-MSR-Session"
        )
        credentials = assumed_role['Credentials']

        return {
            "status": 200,
            "message": "Successfully assumed role",
            "data": credentials
        }
    except Exception as e:
        return {
            "status": 500,
            "message": f"Error assuming role: {str(e)}",
            "data": None
        }
