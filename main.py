from vidcrawl import (
    separator,
    analyse_images,
    transcribe_audio_s3,
)
from vidcrawl._merger import create_unified_report, save_report

if __name__ == "__main__":
    audio, video = separator("output.mp4")
    visual_analysis = analyse_images(video)
    transcripts = transcribe_audio_s3(audio)
    report_str = create_unified_report(visual_analysis, transcripts, video, audio)
    print(report_str)

    save_report(report_str)


# import os
# import dotenv
# dotenv.load_dotenv()

# os.system("aws s3 rm s3://aws-hack-bucket --recursive")
