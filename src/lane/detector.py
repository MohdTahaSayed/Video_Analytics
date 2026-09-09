import cv2
import numpy as np

from src.preprocessing import (
    gaussian_blur,
    grayscale,
    canny,
    white_yellow_mask
)

from src.lane.geometry import (
    fit_polynomial
)


class LaneDetector:

    def __init__(
        self,
        canny_low=50,
        canny_high=150,
        hough_threshold=20,
        min_line_length=20,
        max_line_gap=30
    ):

        self.canny_low = canny_low
        self.canny_high = canny_high

        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    # ---------------------------------------------------------
    # ROI
    # ---------------------------------------------------------

    def create_roi(self, frame):

        h, w = frame.shape[:2]

        polygon = np.array([
            [int(0.05 * w), int(0.99 * h)],
            [int(0.38 * w), int(0.57 * h)],
            [int(0.62 * w), int(0.57 * h)],
            [int(0.99 * w), int(0.99 * h)]
        ], dtype=np.int32)

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

        return mask, polygon

    # ---------------------------------------------------------
    # Visualization helpers
    # ---------------------------------------------------------

    def draw_candidates(self, frame, result):
        """
        Draw raw Hough candidates used by the lane detector.
        """

        output = frame.copy()

        lines = result["hough_lines"]

        if lines is None:
            return output

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

        return output

    # ---------------------------------------------------------
    # Main detection
    # ---------------------------------------------------------

    def detect(self, frame):

        h, w = frame.shape[:2]

        roi_mask, roi_polygon = self.create_roi(frame)

        # -----------------------------------------------------
        # 1. Color evidence
        # -----------------------------------------------------

        color_mask = white_yellow_mask(frame)

        color_mask = cv2.bitwise_and(
            color_mask,
            roi_mask
        )

        # -----------------------------------------------------
        # 2. Edge evidence
        # -----------------------------------------------------

        gray = grayscale(frame)

        blurred = gaussian_blur(
            gray,
            (5, 5)
        )

        edges = cv2.Canny(
            blurred,
            self.canny_low,
            self.canny_high
        )

        edges = cv2.bitwise_and(
            edges,
            roi_mask
        )

        # -----------------------------------------------------
        # 3. Combine
        # -----------------------------------------------------

        lane_edges = cv2.bitwise_and(
            edges,
            color_mask
        )

        # -----------------------------------------------------
        # 4. Hough candidate lines
        # -----------------------------------------------------

        lines = cv2.HoughLinesP(
            lane_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        left_points = []
        right_points = []

        if lines is not None:

            # Fix: Ensure consistent shape (N, 4) regardless of OpenCV version
            lines = np.asarray(lines).reshape(-1, 4)

            for x1, y1, x2, y2 in lines:

                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)

                dx = x2 - x1
                dy = y2 - y1

                # Ignore nearly horizontal structures.
                # This helps reject zebra crossings and
                # other perpendicular road markings.

                if abs(dy) < 8:
                    continue

                length = np.hypot(dx, dy)

                if length < 20:
                    continue

                # x as a function of y.
                #
                # This represents the horizontal movement of
                # a candidate line as we move vertically
                # through the image.

                slope = dx / dy

                # Left lane boundary
                if slope < -0.12:

                    left_points.extend([
                        (x1, y1),
                        (x2, y2)
                    ])

                # Right lane boundary
                elif slope > 0.12:

                    right_points.extend([
                        (x1, y1),
                        (x2, y2)
                    ])

        # -----------------------------------------------------
        # 5. Polynomial fitting
        # -----------------------------------------------------

        left_model = fit_polynomial(
            left_points
        )

        right_model = fit_polynomial(
            right_points
        )

        return {
            "left": left_model,
            "right": right_model,
            "roi_polygon": roi_polygon,
            "roi_mask": roi_mask,
            "color_mask": color_mask,
            "edges": edges,
            "lane_edges": lane_edges,
            "hough_lines": lines
        }