# from vidcrawl import (
#     separator,
#     analyse_images,
#     transcribe_audio_s3,
# )
# from vidcrawl._merger import create_unified_report, save_report
# import os

# from typing import Tuple, List
# from concurrent.futures import ThreadPoolExecutor
# from vidcrawl.core.datamodel import Transcript, AudioClip, VideoClip

# import time

# def analyze_video_parallel(
#     video_path: str,
#     num_frames: int = 40,
#     sample_rate: int = 10
# ) -> Tuple[str, List[Transcript], AudioClip, VideoClip]:
#     """
#     Optimized pipeline: Run transcription and visual analysis in parallel.
    
#     Returns:
#         (visual_analysis, transcripts, audio_clip, video_clip)
#     """    
#     print("🚀 Starting PARALLEL video analysis...")
#     start = time.time()
    
#     # Step 1: Extract (still sequential, but see optimization below)
#     print("\n[1/3] Extracting audio & video...")
#     extract_start = time.time()
#     audio_clip, video_clip = separator(video_path, num_frames, sample_rate)
#     print(f"✓ Extraction done in {time.time() - extract_start:.1f}s")
    
#     # Step 2 & 3: Analyze in PARALLEL
#     print("\n[2/3] Running visual + audio analysis in parallel...")
#     parallel_start = time.time()
    
#     with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#         # Submit both tasks simultaneously
#         visual_future = executor.submit(analyse_images, video_clip)
#         audio_future = executor.submit(transcribe_audio_s3, audio_clip)
        
#         # Wait for both to complete
#         visual_analysis = visual_future.result()
#         transcripts = audio_future.result()
    
#     print(f"✓ Parallel analysis done in {time.time() - parallel_start:.1f}s")
#     print(f"\n🎉 Total time: {time.time() - start:.1f}s")
    
#     return visual_analysis, transcripts, audio_clip, video_clip


# if __name__ == "__main__":
#     a = time.time()
#     visual_analysis, transcripts, audio_clip, video_clip = analyze_video_parallel("output.mp4")
#     report_str = create_unified_report(visual_analysis, transcripts, video_clip, audio_clip)
#     print(report_str)
#     print("Time taken: ", time.time() - a)
    
#     save_report(report_str)


# import os
# import dotenv
# dotenv.load_dotenv()

# os.system("aws s3 rm s3://aws-hack-bucket --recursive")

