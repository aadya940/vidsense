from typing import List, Tuple
import boto3
import os
from io import BytesIO
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    sample_rate: int = 20  # Increase for speed
) -> List[Tuple[str, float]]:
    """
    Optimized frame extraction using direct seeking.
    
    IMPROVEMENTS:
    - Seek directly to frame positions (don't read all frames)
    - Higher sample rate (20 instead of 10)
    - Parallel frame saving
    - Lower JPEG quality for faster I/O    
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Optimized extraction: sampling every {sample_rate} frames")
    
    frame_scores = []
    prev_frame = None
    
    # Phase 1: Score frames (with seeking)
    for frame_num in range(0, total_frames, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simple scoring (can be even faster with lower resolution)
        if prev_frame is not None:
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = np.mean(cv2.absdiff(gray1, gray2))
        else:
            score = 0
        
        frame_scores.append({
            'frame_num': frame_num,
            'timestamp': frame_num / fps,
            'score': score,
        })
        prev_frame = frame
    
    cap.release()
    
    # Phase 2: Select and save best frames
    frame_scores.sort(key=lambda x: x['score'], reverse=True)
    selected = frame_scores[:num_frames]
    selected.sort(key=lambda x: x['frame_num'])
    
    # Phase 3: Parallel frame extraction and saving
    def extract_and_save(frame_info, idx):
        cap_local = cv2.VideoCapture(video_path)
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, frame_info['frame_num'])
        ret, frame = cap_local.read()
        cap_local.release()
        
        if ret:
            output_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            # Lower quality for faster I/O (80 instead of 95)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return (output_path, frame_info['timestamp'])
        return None
    
    keyframe_data = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(extract_and_save, info, idx)
            for idx, info in enumerate(selected)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                keyframe_data.append(result)
    
    # Sort by timestamp
    keyframe_data.sort(key=lambda x: x[1])
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
