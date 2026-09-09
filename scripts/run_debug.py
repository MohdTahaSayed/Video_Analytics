import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import os

from src.lane.detector import LaneDetector


def draw_polynomial(frame, fit, thickness=3):

    if fit is None:
        return

    h, w = frame.shape[:2]

    ys = np.linspace(
        int(0.48 * h),
        h - 1,
        100
    )

    xs = np.polyval(
        fit,
        ys
    )

    points = []

    for x, y in zip(xs, ys):

        if 0 <= x < w:

            points.append([
                int(x),
                int(y)
            ])

    if len(points) >= 2:

        points = np.array(
            points,
            dtype=np.int32
        )

        cv2.polylines(
            frame,
            [points],
            False,
            (0, 255, 0),
            thickness
        )


def draw_debug(frame, result):

    output = frame.copy()

    h, w = output.shape[:2]

    # =========================================================
    # ROI
    # =========================================================

    roi_polygon = result.get(
        "roi_polygon"
    )

    if roi_polygon is not None:

        cv2.polylines(
            output,
            roi_polygon,
            True,
            (255, 255, 0),
            2
        )

    # =========================================================
    # Lane boundaries
    # =========================================================

    left_fit = result.get(
        "left_fit"
    )

    right_fit = result.get(
        "right_fit"
    )

    draw_polynomial(
        output,
        left_fit,
        4
    )

    draw_polynomial(
        output,
        right_fit,
        4
    )

    # =========================================================
    # Lane center
    # =========================================================

    lane_center = result.get(
        "lane_center"
    )

    camera_center = result.get(
        "camera_center"
    )

    if lane_center is not None:

        y = int(
            0.90 * h
        )

        cv2.circle(
            output,
            (
                int(lane_center),
                y
            ),
            7,
            (0, 0, 255),
            -1
        )

        cv2.line(
            output,
            (
                int(lane_center),
                y
            ),
            (
                int(lane_center),
                h
            ),
            (0, 0, 255),
            2
        )

    # Camera center

    if camera_center is not None:

        cv2.line(
            output,
            (
                int(camera_center),
                int(0.70 * h)
            ),
            (
                int(camera_center),
                h
            ),
            (255, 0, 0),
            2
        )

    # =========================================================
    # Information panel
    # =========================================================

    confidence = result.get(
        "confidence",
        0.0
    )

    valid = result.get(
        "valid",
        False
    )

    left_valid = result.get(
        "left_valid",
        False
    )

    right_valid = result.get(
        "right_valid",
        False
    )

    offset = result.get(
        "lateral_offset_pixels"
    )

    panel = output.copy()

    cv2.rectangle(
        panel,
        (10, 10),
        (330, 155),
        (0, 0, 0),
        -1
    )

    output = cv2.addWeighted(
        panel,
        0.65,
        output,
        0.35,
        0
    )

    cv2.putText(
        output,
        f"Confidence: {confidence:.2f}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Left: {'OK' if left_valid else 'MISS'}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Right: {'OK' if right_valid else 'MISS'}",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Lane valid: {valid}",
        (20, 119),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    if offset is not None:

        cv2.putText(
            output,
            f"Offset(px): {offset:.1f}",
            (20, 146),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return output


def main():

    # Fixed input path - using the correct location
    input_video = (
        "data/input/VBOX0011 - Trim.mp4"
    )

    # Fixed output path - using data/output directory
    output_video = (
        "data/output/debug_m2c.mp4"
    )

    # Create output directory if it doesn't exist
    os.makedirs(
        "data/output",
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        input_video
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {input_video}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Resolution: {width}x{height}"
    )

    detector = LaneDetector(
        width,
        height
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detector.detect(
            frame
        )

        output = draw_debug(
            frame,
            result
        )

        writer.write(
            output
        )

        frame_count += 1

        # Print progress every 250 frames.
        if frame_count % 250 == 0:

            seconds = (
                frame_count / fps
            )

            print(
                f"Processed: "
                f"{seconds:.1f}s"
            )

    cap.release()
    writer.release()

    print(
        f"\nDone."
    )

    print(
        f"Output: {output_video}"
    )


if __name__ == "__main__":
    main()