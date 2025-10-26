from typing import List, Union
import boto3
import os
from dotenv import load_dotenv
from .datamodel import VideoClip

load_dotenv()

bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)


def create_images_prompt(video_clips: List[VideoClip], max_frames: int = 100) -> list:
    """Create content with frames from video clips.

    Args:
        video_clips:
            List of VideoClip objects
        max_frames:
            Maximum total frames to send (to avoid token limits)
    """
    content = [
        {
            "text": (
                "You are a professional sports video analyst. Analyze the provided video frames and create a detailed timeline of events.\n\n"
                "For each significant moment you observe, provide:\n"
                "1. **Timestamp** - The approximate time when the event occurs (format: MM:SS or seconds)\n"
                "2. **Event Type** - What's happening (e.g., Goal, Save, Pass, Tackle, Corner Kick, Throw-in, etc.)\n"
                "3. **Description** - Brief but specific details about the action, players involved, and outcome\n"
                "4. **Context** - Field position, team dynamics, or tactical significance if relevant\n\n"
                "Format your response as a chronological event list. Be precise with timestamps based on the frame timing provided below.\n"
                "Focus on key moments: goals, near-misses, defensive plays, set pieces, and significant game flow changes.\n\n"
                "Here are the frames with their timestamps:"
            ),
        }
    ]

    total_frames = 0
    frames_per_clip = (
        max_frames // len(video_clips) if len(video_clips) > 0 else max_frames
    )

    # Track frame timestamps for better context
    frame_timestamps = []

    for clip in video_clips:
        # Limit frames to stay under max_frames total
        selected_frames = clip.keyframes[:frames_per_clip]

        # Use stored timestamps if available, otherwise calculate
        if clip.timestamps and len(clip.timestamps) == len(clip.keyframes):
            frame_times = clip.timestamps[:frames_per_clip]
        else:
            # Fallback: calculate approximate timestamp for each frame within the clip
            clip_duration = clip.end - clip.start
            time_per_frame = (
                clip_duration / len(selected_frames) if len(selected_frames) > 1 else 0
            )
            frame_times = [
                clip.start + (idx * time_per_frame)
                for idx in range(len(selected_frames))
            ]

        if len(selected_frames) > 0:
            for idx, (frame_key, frame_time) in enumerate(
                zip(selected_frames, frame_times)
            ):
                frame_timestamps.append(frame_time)

                # Format timestamp nicely
                minutes = int(frame_time // 60)
                seconds = int(frame_time % 60)

                content.append(
                    {
                        "text": f"\n[Frame at {minutes:02d}:{seconds:02d} ({frame_time:.1f}s)]"
                    }
                )

                s3_path = f"s3://aws-hack-bucket/{frame_key}"
                content.append(
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"s3Location": {"uri": s3_path}},
                        }
                    }
                )
                total_frames += 1

                # Hard stop at max_frames
                if total_frames >= max_frames:
                    break

        if total_frames >= max_frames:
            break

    # Add closing instruction
    content.append(
        {
            "text": (
                "\n\nNow provide a chronological analysis of the video with timestamped events. "
                "Use the frame timestamps above to accurately reference when events occur."
            )
        }
    )

    print(f"Sending {total_frames} frames across {len(video_clips)} clips to model...")
    print(f"Time range: {frame_timestamps[0]:.1f}s to {frame_timestamps[-1]:.1f}s")
    return content


def analyse_images(
    video_clips: Union[VideoClip, List[VideoClip]],
    model_id: str = "us.amazon.nova-lite-v1:0",
    max_frames: int = 100,
) -> str:
    """
    Analyze video clips with optimized settings.

    Args:
        video_clips: Single VideoClip or list of VideoClip objects
        model_id: Bedrock model ID
        max_frames: Maximum frames to send (affects cost/speed)

    Model options (ordered by speed):
    - "us.amazon.nova-lite-v1:0" (fastest, cheapest)
    - "us.amazon.nova-pro-v1:0" (balanced)
    - "us.amazon.nova-premier-v1:0" (slowest, most capable)
    """

    # Handle both single VideoClip and List[VideoClip]
    if isinstance(video_clips, VideoClip):
        video_clips = [video_clips]
    elif not isinstance(video_clips, list):
        raise TypeError(
            f"Expected VideoClip or List[VideoClip], got {type(video_clips)}"
        )

    content = create_images_prompt(video_clips, max_frames)

    # Use converse() - it's the correct API
    response = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": content}],
        inferenceConfig={
            "maxTokens": 4000,  # Reduced for faster response
            "temperature": 0.3,  # Lower for more focused output
        },
    )

    return response["output"]["message"]["content"][0]["text"]
