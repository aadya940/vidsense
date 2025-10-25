import os
import boto3
from botocore.exceptions import ClientError


def _s3_file_exists(s3_client, bucket, key):
    """Check if a file exists in S3 without downloading it."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise  # Re-raise for other errors


def _push_video_to_s3(video_path, bucket_name, object_path, overwrite=False):
    try:
        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

        # Check if file exists and we shouldn't overwrite
        if not overwrite and _s3_file_exists(s3, bucket_name, object_path):
            print(
                f"File s3://{bucket_name}/{object_path} already exists. Skipping upload."
            )
            return False

        # Verify local file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Local file not found: {video_path}")

        print(f"Uploading {video_path} to s3://{bucket_name}/{object_path}...")
        s3.upload_file(video_path, bucket_name, object_path)
        print("Upload successful!")
        return True

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "NoSuchBucket":
            print(f"Error: The bucket '{bucket_name}' does not exist.")
        elif error_code == "AccessDenied":
            print(
                f"Error: Access denied to bucket '{bucket_name}'. Check your IAM permissions."
            )
        else:
            print(f"AWS Error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise
