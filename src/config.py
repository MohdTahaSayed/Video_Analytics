from dataclasses import dataclass


@dataclass
class VideoConfig:
    target_fps: float = 1.0


@dataclass
class LaneConfig:
    roi_top_ratio: float = 0.57
    roi_bottom_ratio: float = 0.99

    canny_low: int = 50
    canny_high: int = 150

    hough_threshold: int = 20
    hough_min_line_length: int = 20
    hough_max_line_gap: int = 30

    min_lane_width_px: float = 10.0


@dataclass
class PipelineConfig:
    video = VideoConfig()
    lane = LaneConfig()