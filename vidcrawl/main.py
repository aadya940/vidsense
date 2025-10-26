from dotenv import load_dotenv
import os

from .core import _push_video_to_s3


def push_video_to_s3(video_path):
    load_dotenv()
    _push_video_to_s3(video_path, "aws-hack-bucket", "videos/demo.mp4")
    