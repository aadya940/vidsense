from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VideoCut:
    """One continuous shot (from PySceneDetect)"""

    cut_id: str
    start: float  # seconds
    end: float
    start_frame: int
    end_frame: int
    source_video: str
    local_source_video: str


@dataclass
class AudioClip:
    """Audio segment for a VideoCut."""

    audio_data: List[str]  # List of Paths in the S3 Bucket.
    start: float
    end: float
    source_cut: VideoCut


@dataclass
class VideoClip:
    """Represents extracted video frames with metadata."""

    start: float
    end: float
    keyframes: List[str]  # S3 paths to keyframe images
    timestamps: Optional[List[float]] = None  # Timestamp for each keyframe
    source_cut: Optional[VideoCut] = None


# OUTPUT LAYER: Analysis Results
@dataclass
class Transcript:
    """Spoken words from audio"""

    text: str
    start: float
    end: float
    confidence: float


@dataclass
class VisualAnalysis:
    """Vision model analysis of a keyframe"""

    description: str
    objects: List[str]
    scene_type: str
    timestamp: float


@dataclass
class Scene:
    """Analysis results for one VideoCut"""

    scene_id: str
    source_cut: VideoCut
    audio_clip: AudioClip
    video_clip: VideoClip

    # Analysis results (populated by analyzers)
    transcripts: List[Transcript]
    visual_analyses: List[VisualAnalysis]


@dataclass
class VideoTimeline:
    """Complete understanding of the video"""

    source_video: str
    scenes: List[Scene]


## Clean Flow
## FullVideo → VideoCuts → (AudioClips + VideoClips) → Scenes → VideoTimeline
