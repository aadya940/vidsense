from typing import List
import boto3
import os
from dotenv import load_dotenv

from .datamodel import AudioClip, Transcript

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)
transcribe = boto3.client(
    "transcribe",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)


def transcribe_audio_s3(
    audio_clip: AudioClip,
    bucket_name: str = "aws-hack-bucket",
    language_code: str = "en-US",
) -> List[Transcript]:
    """
    Transcribe audio from S3 using AWS Transcribe.

    Args:
        audio_clip: AudioClip with S3 paths
        bucket_name: S3 bucket name
        language_code: Language code (en-US, es-ES, etc.)

    Returns:
        List of Transcript objects with timestamps
    """
    import time
    import json

    if not audio_clip.audio_data:
        print("No audio data to transcribe")
        return []

    # Use the first audio file (assuming single audio per clip)
    audio_s3_key = audio_clip.audio_data[0]
    job_name = f"transcribe_{int(time.time())}_{audio_s3_key.replace('/', '_')}"

    print(f"Starting transcription job: {job_name}")
    print(f"Audio: s3://{bucket_name}/{audio_s3_key}")

    # Start transcription job
    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": f"s3://{bucket_name}/{audio_s3_key}"},
            MediaFormat="mp3",
            LanguageCode=language_code,
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": 10,  # Adjust based on expected speakers
            },
        )

        # Wait for completion
        print("Waiting for transcription to complete...")
        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]

            if job_status == "COMPLETED":
                print("✓ Transcription complete!")
                break
            elif job_status == "FAILED":
                error = status["TranscriptionJob"].get("FailureReason", "Unknown error")
                raise Exception(f"Transcription failed: {error}")

            time.sleep(5)  # Check every 5 seconds

        # Get transcript URL
        transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]

        # Download and parse transcript
        import urllib.request

        with urllib.request.urlopen(transcript_uri) as response:
            transcript_json = json.loads(response.read())

        # Parse into Transcript objects
        transcripts = []

        # Get items with timestamps
        items = transcript_json["results"].get("items", [])

        # Group by segments (sentences/phrases)
        current_text = []
        current_start = None
        current_end = None

        for item in items:
            if item["type"] == "pronunciation":
                word = item["alternatives"][0]["content"]
                start_time = float(item["start_time"])
                end_time = float(item["end_time"])
                confidence = float(item["alternatives"][0]["confidence"])

                if current_start is None:
                    current_start = start_time

                current_text.append(word)
                current_end = end_time

                # Create segment every ~10 words or at punctuation
                if len(current_text) >= 10:
                    transcripts.append(
                        Transcript(
                            text=" ".join(current_text),
                            start=audio_clip.start + current_start,
                            end=audio_clip.start + current_end,
                            confidence=confidence,
                        )
                    )
                    current_text = []
                    current_start = None

            elif item["type"] == "punctuation" and current_text:
                # End segment at punctuation
                current_text[-1] += item["alternatives"][0]["content"]

                if item["alternatives"][0]["content"] in [".", "!", "?"]:
                    transcripts.append(
                        Transcript(
                            text=" ".join(current_text),
                            start=audio_clip.start + current_start,
                            end=audio_clip.start + current_end,
                            confidence=1.0,
                        )
                    )
                    current_text = []
                    current_start = None

        # Add remaining text
        if current_text:
            transcripts.append(
                Transcript(
                    text=" ".join(current_text),
                    start=audio_clip.start + current_start,
                    end=audio_clip.start + current_end,
                    confidence=1.0,
                )
            )

        print(f"✓ Parsed {len(transcripts)} transcript segments")

        # Cleanup: Delete transcription job
        try:
            transcribe.delete_transcription_job(TranscriptionJobName=job_name)
        except:
            pass

        return transcripts

    except Exception as e:
        print(f"Error during transcription: {e}")
        # Cleanup on error
        try:
            transcribe.delete_transcription_job(TranscriptionJobName=job_name)
        except:
            pass
        return []
