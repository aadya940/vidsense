from vidcrawl import push_video_to_s3, start_scene_detection, get_scene_detection_results, separator

if __name__ == '__main__':
    push_video_to_s3('demo.mp4')        # Works.
    job_id = start_scene_detection()
    res = get_scene_detection_results(job_id)
    separator(res)
