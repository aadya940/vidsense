from typing import List, Optional
import boto3
import os
from dotenv import load_dotenv
from .core.datamodel import AudioClip, VideoClip, Transcript

load_dotenv()


def get_aws_client(service_name: str):
    """Create AWS client with consistent credentials."""
    try:
        return boto3.client(
            service_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
    except Exception as e:
        print(f"Failed to create {service_name} client: {e}")
        raise


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def merge_timeline(
    visual_analysis: str, transcripts: List[Transcript], video_duration: float
) -> str:
    """
    Merge visual analysis and transcripts into a chronological markdown timeline.

    Args:
        visual_analysis: Text from analyse_images() containing visual events
        transcripts: List of Transcript objects from transcribe_audio_s3()
        video_duration: Total video duration in seconds

    Returns:
        Formatted markdown string with merged timeline
    """

    # Build markdown document
    markdown = []

    # Header
    markdown.append("# 🎬 Video Analysis Report\n")
    markdown.append(f"**Duration:** {format_timestamp(video_duration)}\n")
    markdown.append(f"**Total Transcript Segments:** {len(transcripts)}\n")
    markdown.append("\n---\n")

    # Visual Analysis Section
    markdown.append("## 📹 Visual Analysis\n")
    markdown.append(visual_analysis)
    markdown.append("\n\n---\n")

    # Audio Transcription Section
    markdown.append("## 🎤 Audio Transcription\n")

    if not transcripts:
        markdown.append("*No transcription available*\n")
    else:
        for i, t in enumerate(transcripts, 1):
            time_range = f"{format_timestamp(t.start)} - {format_timestamp(t.end)}"
            confidence_stars = "⭐" * min(5, int(t.confidence * 5))

            markdown.append(f"### [{time_range}] Segment {i}\n")
            markdown.append(
                f"**Confidence:** {confidence_stars} ({t.confidence:.2f})\n\n"
            )
            markdown.append(f"{t.text}\n\n")

    return "".join(markdown)


def create_unified_report(
    visual_analysis: str,
    transcripts: List[Transcript],
    video_clip: VideoClip,
    audio_clip: AudioClip,
    model_id: str = "us.amazon.nova-lite-v1:0",
) -> str:
    """
    Use AI to create a unified, intelligent video report by combining all data.

    Args:
        visual_analysis: Visual analysis from analyse_images()
        transcripts: Audio transcripts from transcribe_audio_s3()
        video_clip: VideoClip object with metadata
        audio_clip: AudioClip object with metadata
        model_id: Bedrock model for synthesis

    Returns:
        AI-generated unified markdown report
    """

    print("Creating unified report with AI synthesis...")

    # Create bedrock client
    bedrock = get_aws_client("bedrock-runtime")

    # Prepare transcript text
    transcript_text = "\n".join(
        [f"[{format_timestamp(t.start)}] {t.text}" for t in transcripts]
    )

    # Create prompt for synthesis
    prompt = f"""You are a professional sports video analyst. 
    You have analyzed a video using both visual and audio analysis.

    ## VISUAL ANALYSIS:
    {visual_analysis}

    ## AUDIO TRANSCRIPTION:
    {transcript_text if transcript_text else "No audio transcription available"}

    ## YOUR TASK:
    Create a comprehensive, engaging markdown report that synthesizes BOTH the visual and audio information into a unified timeline. Your report should:

    1. **Create a Unified Timeline**: Merge visual events with audio commentary chronologically
    2. **Cross-Reference**: Connect what's seen with what's heard (e.g., "Goal scored at 12:34 - commentator excitement confirms...")
    3. **Fill Gaps**: Use audio to clarify visual events and vice versa
    4. **Key Moments**: Highlight the most important events with context from both sources
    5. **Narrative Flow**: Tell the story of the video in an engaging, readable format

    ## FORMAT:
    Use markdown with:
    - Clear section headers (##, ###)
    - Timestamps in [MM:SS] format
    - Bold for emphasis
    - Bullet points for lists
    - Blockquotes for notable commentary

    Make it engaging, informative, and easy to follow!
    """

    try:
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": 4096,
                "temperature": 0.7,  # Slightly creative
            },
        )

        unified_report = response["output"]["message"]["content"][0]["text"]
        print("✓ Unified report generated!")
        return unified_report

    except Exception as e:
        print(f"Error creating unified report: {e}")
        # Fallback to simple merge
        return merge_timeline(visual_analysis, transcripts, video_clip.end)


def save_report(report: str, output_path: str = "video_report.md") -> str:
    """
    Save markdown report to file.

    Args:
        report: Markdown content
        output_path: Output file path

    Returns:
        Path to saved file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✓ Report saved to: {output_path}")
    return output_path
