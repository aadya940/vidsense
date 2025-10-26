from typing import List, Tuple
import boto3
import os
from io import BytesIO
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import ffmpeg

from .datamodel import AudioClip, VideoClip
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)


def calculate_frame_difference(frame1, frame2):
    """Calculate difference between two frames."""
    if frame1 is None or frame2 is None:
        return 0

    # Convert to grayscale and calculate absolute difference
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    return np.mean(diff)


def extract_best_frames_fast(
    video_path: str,
    output_dir: str,
    num_frames: int = 50,
    sample_rate: int = 10,
) -> List[Tuple[str, float]]:
    """Fast extraction of visually distinct frames by sampling.

    Args:
        video_path: Path to source video
        output_dir: Directory to save keyframes
        num_frames: Total number of frames to extract
        sample_rate: Analyze every Nth frame (higher = faster but less accurate)

    Returns:
        List of tuples (frame_path, timestamp_in_seconds)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {duration:.1f}s, {total_frames} frames @ {fps:.1f}fps")
    print(f"Sampling every {sample_rate} frames for speed...")

    frame_scores = []
    prev_frame = None
    frame_num = 0

    # Sample frames and calculate content scores
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame
        if frame_num % sample_rate == 0:
            # Score based on difference from previous frame
            score = calculate_frame_difference(prev_frame, frame)
            timestamp = frame_num / fps

            frame_scores.append(
                {
                    "frame_num": frame_num,
                    "timestamp": timestamp,
                    "score": score,
                    "frame": frame.copy(),  # Store the frame
                }
            )

            prev_frame = frame

            if len(frame_scores) % 100 == 0:
                print(f"  Sampled {len(frame_scores)} frames...")

        frame_num += 1

    cap.release()

    print(f"Analyzed {len(frame_scores)} sampled frames, selecting top {num_frames}...")

    # Sort by score and select top N
    frame_scores.sort(key=lambda x: x["score"], reverse=True)
    selected = frame_scores[:num_frames]

    # Sort selected frames chronologically
    selected.sort(key=lambda x: x["frame_num"])

    # Save selected frames
    keyframe_data = []
    for idx, frame_info in enumerate(selected):
        output_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
        cv2.imwrite(output_path, frame_info["frame"])
        keyframe_data.append((output_path, frame_info["timestamp"]))

        if (idx + 1) % 10 == 0:
            print(f"  Saved {idx + 1}/{num_frames} keyframes...")

    print(f"✓ Extracted {len(keyframe_data)} best frames")

    return keyframe_data


def separator(
    video_path: str, num_frames: int = 40, sample_rate: int = 10
) -> Tuple[AudioClip, VideoClip]:
    """Extract best frames and audio from entire video (fast version).

    Args:
        video_path:
            Path to video file
        num_frames:
            Number of keyframes to extract
        sample_rate:
            Analyze every Nth frame (10 = 10x faster)

    Returns:
        Tuple of (AudioClip, VideoClip) for the entire video
    """

    print(f"Processing video: {video_path}")
    print(f"Extracting {num_frames} best frames (sample rate: 1/{sample_rate})...")

    temp_dir = tempfile.mkdtemp()
    keyframes_dir = os.path.join(temp_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    try:
        # 1. Extract best frames (fast!)
        keyframe_data = extract_best_frames_fast(
            video_path=video_path,
            output_dir=keyframes_dir,
            num_frames=num_frames,
            sample_rate=sample_rate,
        )

        # 2. Get video duration
        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])

        # 3. Extract audio from entire video
        print("Extracting audio...")
        audio_out, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="mp3", acodec="libmp3lame", vn=None)
            .run(capture_stdout=True, capture_stderr=True)
        )

        # 4. Upload to S3 in parallel
        print("Uploading to S3...")
        with ThreadPoolExecutor(max_workers=16) as upload_executor:
            upload_futures = []

            # Upload keyframes
            for idx, (kf_path, timestamp) in enumerate(keyframe_data):
                future = upload_executor.submit(
                    s3.upload_file,
                    kf_path,
                    "aws-hack-bucket",
                    f"keyframes/video_{idx}.jpg",
                )
                upload_futures.append(future)

            # Upload audio
            audio_future = upload_executor.submit(
                s3.upload_fileobj,
                BytesIO(audio_out),
                "aws-hack-bucket",
                f"audio/video_full.mp3",
            )
            upload_futures.append(audio_future)

            # Wait for all uploads
            for future in upload_futures:
                future.result()

        audio_clip = AudioClip(
            audio_data=[f"audio/video_full.mp3"],
            start=0,
            end=duration,
            source_cut=None,
        )

        # Store both keyframe paths and their timestamps
        video_clip = VideoClip(
            start=0,
            end=duration,
            keyframes=[
                f"keyframes/video_{idx}.jpg" for idx in range(len(keyframe_data))
            ],
            timestamps=[timestamp for _, timestamp in keyframe_data],  # Add timestamps
            source_cut=None,
        )

        print("✓ Processing complete!")
        return audio_clip, video_clip

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
