import sys
from pathlib import Path

import cv2
import numpy as np

# Allow imports from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lane.detector import LaneDetector
from src.lane.geometry import evaluate_polynomial


def draw_debug(frame, result):

    output = frame.copy()

    h, w = frame.shape[:2]

    # ---------------------------------------------------------
    # ROI
    # ---------------------------------------------------------

    cv2.polylines(
        output,
        [result["roi_polygon"]],
        True,
        (255, 200, 0),
        2
    )

    # ---------------------------------------------------------
    # Raw Hough lines
    # ---------------------------------------------------------

    lines = result["hough_lines"]

    if lines is not None:

        # Fix: Ensure consistent shape (N, 4) regardless of OpenCV version
        lines = np.asarray(lines).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                (160, 160, 160),
                1
            )

    # ---------------------------------------------------------
    # Fitted curves
    # ---------------------------------------------------------

    y_values = np.linspace(
        int(0.57 * h),
        int(0.99 * h),
        100
    )

    left = result["left"]
    right = result["right"]

    left_x = evaluate_polynomial(
        left,
        y_values
    )

    right_x = evaluate_polynomial(
        right,
        y_values
    )

    if left_x is not None:

        points = np.column_stack([
            left_x,
            y_values
        ]).astype(np.int32)

        cv2.polylines(
            output,
            [points],
            False,
            (0, 255, 0),
            4
        )

    if right_x is not None:

        points = np.column_stack([
            right_x,
            y_values
        ]).astype(np.int32)

        cv2.polylines(
            output,
            [points],
            False,
            (0, 255, 0),
            4
        )

    # ---------------------------------------------------------
    # Camera / ego center
    # ---------------------------------------------------------

    ego_x = w / 2

    cv2.line(
        output,
        (int(ego_x), int(0.57 * h)),
        (int(ego_x), h),
        (255, 0, 0),
        2
    )

    # ---------------------------------------------------------
    # Lane center
    # ---------------------------------------------------------

    y_ref = int(0.90 * h)

    if left is not None and right is not None:

        x_left = float(
            evaluate_polynomial(
                left,
                y_ref
            )
        )

        x_right = float(
            evaluate_polynomial(
                right,
                y_ref
            )
        )

        if x_left < x_right:

            lane_width = x_right - x_left

            if lane_width > 10:

                lane_center = (
                    x_left + x_right
                ) / 2

                ego_position = (
                    ego_x - x_left
                ) / lane_width

                cv2.circle(
                    output,
                    (
                        int(lane_center),
                        y_ref
                    ),
                    7,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    output,
                    f"Ego position: {ego_position:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    output,
                    f"Lane width: {lane_width:.1f}px",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

    return output


def main():

    input_path = (
        PROJECT_ROOT
        / "data"
        / "input"
        / "VBOX0011 - Trim.mp4"
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "output"
        / "lane_debug_30s.mp4"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open: {input_path}"
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

    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    detector = LaneDetector()

    frame_number = 0

    max_frames = int(
        fps * 30
    )

    while frame_number < max_frames:

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

        frame_number += 1

        if frame_number % int(fps * 5) == 0:

            print(
                f"Processed "
                f"{frame_number / fps:.1f}s"
            )

    cap.release()
    writer.release()

    print()
    print("Finished.")
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()