from typing import List
import ffmpeg
import boto3
import os
from io import BytesIO
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

from .datamodel import VideoCut

from dotenv import load_dotenv
load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)


def extract_content_aware_frames(
    video_path: str, 
    output_dir: str, 
    start_time: float, 
    duration: float, 
    threshold: float = 5
) -> List[str]:
    """Finds content-based scene changes *only within a specific time segment*
    and saves the first frame of each new scene.
    """
    video = open_video(video_path)
    
    fps = video.frame_rate
    
    start_frame = FrameTimecode(timecode=start_time, fps=fps)
    end_frame = FrameTimecode(timecode=start_time + duration, fps=fps)

    video.seek(start_frame)
    
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    print(f"Detecting scenes from {start_frame.get_timecode()} to {end_frame.get_timecode()}...")
    
    scene_manager.detect_scenes(
        video, 
        end_time=end_frame,
        show_progress=False 
    )

    scene_list = scene_manager.get_scene_list()

    # 4. Save the images
    if scene_list:
        print(f"Saving {len(scene_list)} keyframe(s) to '{output_dir}'...")
        
        image_paths_dict = save_images(
            scene_list=scene_list,      # 1st arg: The list of scenes
            video=video,                # 2nd arg: The video object
            num_images=1,               # Get 1 image per scene
            output_dir=output_dir,
            image_name_template='$SCENE_NUMBER-$FRAME_NUMBER'
        )
        
        # The doc says the return is a dict: {scene_num: [list_of_paths]}
        # We need to flatten this list of lists.
        all_paths = []
        for path_list in image_paths_dict.values():
            for relative_path in path_list:
                full_path = os.path.join(output_dir, relative_path)
                all_paths.append(full_path)
        
        return sorted(all_paths)
        
    else:
        print("No new scenes detected in this segment.")
        return []


def process_single_cut(cut: VideoCut) -> None:
    """Process one cut with parallel uploads"""
    start_time = cut.start
    duration = cut.end - cut.start

    temp_dir = tempfile.mkdtemp()
    # No more 'segment_path'
    keyframes_dir = os.path.join(temp_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    try:
        # 1. Extract keyframes directly from the source video
        keyframe_files = extract_content_aware_frames(
            video_path=cut.local_source_video,
            output_dir=keyframes_dir,
            start_time=start_time,
            duration=duration,
        )

        # 2. Extract audio directly from the source video
        audio_out, _ = (
            ffmpeg.input(
                cut.local_source_video, ss=start_time, t=duration
            )  # Pass ss/t here
            .output("pipe:", format="mp3", acodec="libmp3lame", vn=None)
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Parallel S3 uploads (this part doesn't change)
        with ThreadPoolExecutor(max_workers=16) as upload_executor:
            upload_futures = []

            # Upload keyframes
            for idx, kf_path in enumerate(keyframe_files):
                future = upload_executor.submit(
                    s3.upload_file,
                    kf_path,
                    "aws-hack-bucket",
                    f"keyframes/{cut.cut_id}_{idx}.jpg",
                )
                upload_futures.append(future)

            # Upload audio
            audio_future = upload_executor.submit(
                s3.upload_fileobj,
                BytesIO(audio_out),
                "aws-hack-bucket",
                f"audio/{cut.cut_id}.mp3",
            )
            upload_futures.append(audio_future)

            # Wait for all uploads
            for future in upload_futures:
                future.result()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def separator(video_cuts: List[VideoCut], max_workers: int = 4) -> None:
    """Process cuts in parallel but report in chronological order"""

    print(f"Processing {len(video_cuts)} cuts with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = [executor.submit(process_single_cut, cut) for cut in video_cuts]

        # Wait for results IN ORDER
        for i, future in enumerate(futures):
            try:
                future.result()  # Blocks until THIS cut completes
                print(
                    f"✓ Cut {i+1}/{len(video_cuts)} ({video_cuts[i].cut_id}) complete"
                )
            except Exception as e:
                print(f"✗ Cut {i+1} ({video_cuts[i].cut_id}) failed: {e}")

    print("All cuts processed!")
