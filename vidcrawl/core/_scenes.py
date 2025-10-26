import boto3
import os
import time


from .datamodel import VideoCut

# Initialize Rekognition client
rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)


def start_scene_detection():
    """
    Args
    ----
    video_path: str
        Path to the video file in the S3 bucket.

    Returns
    -------
    job_id: str
        ID of the scene detection job.
    """
    response = rekognition.start_segment_detection(
        Video={"S3Object": {"Bucket": "aws-hack-bucket", "Name": "videos/demo.mp4"}},
        SegmentTypes=["SHOT"],
    )

    job_id = response["JobId"]
    return job_id


def poll_results(job_id):
    while True:
        result = rekognition.get_segment_detection(JobId=job_id)
        print("Current Result: ", result)
        if result["JobStatus"] == "SUCCEEDED":
            break
        elif result["JobStatus"] == "FAILED":
            raise Exception(result["FailureReason"])
        time.sleep(2)
    return result


def get_scene_detection_results(job_id):
    """
    Args
    ----
    job_id: str
        ID of the scene detection job.

    Returns
    -------
    job_status: str
        Status of the scene detection job.
    """
    res = poll_results(job_id)
    if res["JobStatus"] == "SUCCEEDED":
        _output = res["Segments"]

    video_cuts = [
        VideoCut(
            cut_id=f"cut_{seg['ShotSegment']['Index']}",
            start=seg["StartTimestampMillis"] / 1000.0,
            end=seg["EndTimestampMillis"] / 1000.0,
            start_frame=seg["StartFrameNumber"],
            end_frame=seg["EndFrameNumber"],
            source_video="videos/demo.mp4",
            local_source_video="demo.mp4",
        )
        for seg in _output
    ]
    return video_cuts
