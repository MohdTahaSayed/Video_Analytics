import cv2
import numpy as np


class LaneDetector:

    def __init__(self, width=720, height=576):

        self.width = width
        self.height = height

        # Previous independently valid models
        self.prev_left = None
        self.prev_right = None

        # Smoothed models
        self.smooth_left = None
        self.smooth_right = None

        # Independent miss counters
        self.left_missed = 0
        self.right_missed = 0

        # Maximum number of frames for which an old
        # model can be temporarily retained.
        self.max_missed = 15

        # Temporal smoothing.
        # Higher = react faster.
        self.alpha = 0.20

    # =========================================================
    # ROI
    # =========================================================

    def get_roi_polygon(self, w, h):

        return np.array([
            [
                (int(0.05 * w), h),
                (int(0.30 * w), int(0.48 * h)),
                (int(0.70 * w), int(0.48 * h)),
                (int(0.95 * w), h)
            ]
        ], dtype=np.int32)

    def apply_roi(self, mask):

        h, w = mask.shape[:2]

        roi_polygon = self.get_roi_polygon(w, h)

        roi_mask = np.zeros_like(mask)

        cv2.fillPoly(
            roi_mask,
            roi_polygon,
            255
        )

        return cv2.bitwise_and(
            mask,
            roi_mask
        )

    # =========================================================
    # Lane evidence
    # =========================================================

    def create_lane_mask(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # -----------------------------------------------------
        # White / bright lane markings
        # -----------------------------------------------------

        white_lower = np.array(
            [0, 0, 145],
            dtype=np.uint8
        )

        white_upper = np.array(
            [180, 105, 255],
            dtype=np.uint8
        )

        white_mask = cv2.inRange(
            hsv,
            white_lower,
            white_upper
        )

        # -----------------------------------------------------
        # Yellow lane markings
        # -----------------------------------------------------

        yellow_lower = np.array(
            [15, 45, 70],
            dtype=np.uint8
        )

        yellow_upper = np.array(
            [45, 255, 255],
            dtype=np.uint8
        )

        yellow_mask = cv2.inRange(
            hsv,
            yellow_lower,
            yellow_upper
        )

        mask = cv2.bitwise_or(
            white_mask,
            yellow_mask
        )

        # Remove tiny isolated noise.
        kernel_open = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel_open
        )

        # Connect small gaps in lane markings.
        kernel_close = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel_close
        )

        return mask

    # =========================================================
    # Connected-component filtering
    # =========================================================

    def filter_components(self, mask):

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        filtered = np.zeros_like(mask)

        h, w = mask.shape

        for label in range(1, num_labels):

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]

            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]

            area = stats[label, cv2.CC_STAT_AREA]

            if area < 15:
                continue

            # Avoid giant filled objects such as truck bodies.
            if area > 0.10 * w * h:
                continue

            # Lane markings should generally be elongated.
            aspect = max(width, height) / max(
                min(width, height),
                1
            )

            if aspect < 1.5:
                continue

            # Very large horizontal structures are unlikely
            # to be lane boundaries.
            if width > 0.65 * w and height < 0.15 * h:
                continue

            component = (
                labels == label
            ).astype(np.uint8) * 255

            filtered = cv2.bitwise_or(
                filtered,
                component
            )

        return filtered

    # =========================================================
    # Expected lane position
    # =========================================================

    def evaluate_fit(self, fit, y):

        if fit is None:
            return None

        return np.polyval(
            fit,
            y
        )

    def estimate_lane_center(self, y):

        if (
            self.smooth_left is not None and
            self.smooth_right is not None
        ):

            left = self.evaluate_fit(
                self.smooth_left,
                y
            )

            right = self.evaluate_fit(
                self.smooth_right,
                y
            )

            return (left + right) / 2.0

        # If we don't have both lanes yet,
        # camera center is only a temporary prior.
        return self.width / 2.0

    # =========================================================
    # Independent side fitting
    # =========================================================

    def fit_side(self, xs, ys):

        if len(xs) < 40:
            return None

        if len(np.unique(ys)) < 10:
            return None

        try:

            fit = np.polyfit(
                ys,
                xs,
                2
            )

        except np.linalg.LinAlgError:

            return None

        return fit

    # =========================================================
    # Curvature validation
    # =========================================================

    def validate_polynomial(self, fit):

        if fit is None:
            return False

        a, b, c = fit

        # Reject extreme curvature.
        #
        # The exact threshold is intentionally conservative
        # for this 720x576 camera geometry.
        if abs(a) > 0.01:
            return False

        if not np.all(
            np.isfinite(fit)
        ):
            return False

        return True

    # =========================================================
    # Individual lane validation
    # =========================================================

    def validate_side(self, fit, side):

        if not self.validate_polynomial(fit):
            return False

        h = self.height
        w = self.width

        y_values = np.array([
            int(0.55 * h),
            int(0.70 * h),
            int(0.85 * h),
            int(0.98 * h)
        ])

        x_values = np.polyval(
            fit,
            y_values
        )

        # Must remain inside a reasonable image region.
        if np.any(
            x_values < -0.15 * w
        ):
            return False

        if np.any(
            x_values > 1.15 * w
        ):
            return False

        # Expected side.
        if side == "left":

            if np.median(x_values) > 0.65 * w:
                return False

        else:

            if np.median(x_values) < 0.35 * w:
                return False

        return True

    # =========================================================
    # Pair validation
    # =========================================================

    def validate_pair(self, left, right):

        if left is None or right is None:
            return False

        h = self.height
        w = self.width

        y_values = np.array([
            int(0.55 * h),
            int(0.70 * h),
            int(0.85 * h),
            int(0.98 * h)
        ])

        left_x = np.polyval(
            left,
            y_values
        )

        right_x = np.polyval(
            right,
            y_values
        )

        widths = right_x - left_x

        # Right must remain right of left.
        if np.any(widths <= 40):
            return False

        # Don't accept absurdly wide lanes.
        if np.any(widths > 0.95 * w):
            return False

        # Lane width should not explode with y.
        if np.max(widths) / max(
            np.min(widths),
            1
        ) > 4.0:
            return False

        return True

    # =========================================================
    # Temporal smoothing
    # =========================================================

    def smooth_fit(self, previous, current):

        if current is None:
            return previous

        if previous is None:
            return current

        return (
            self.alpha * current +
            (1.0 - self.alpha) * previous
        )

    # =========================================================
    # Candidate assignment
    # =========================================================

    def assign_candidates(self, mask):

        ys, xs = np.nonzero(mask)

        if len(xs) == 0:

            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([])
            )

        h, w = mask.shape

        # Use the bottom of the road because this
        # gives the strongest lane separation.

        y_reference = int(
            0.90 * h
        )

        center = self.estimate_lane_center(
            y_reference
        )

        # If both previous lanes exist, calculate
        # their expected positions.
        expected_left = None
        expected_right = None

        if (
            self.smooth_left is not None and
            self.smooth_right is not None
        ):

            expected_left = np.polyval(
                self.smooth_left,
                ys
            )

            expected_right = np.polyval(
                self.smooth_right,
                ys
            )

        if (
            expected_left is not None and
            expected_right is not None
        ):

            left_distance = np.abs(
                xs - expected_left
            )

            right_distance = np.abs(
                xs - expected_right
            )

            left_selection = (
                (left_distance < 100) &
                (xs < center)
            )

            right_selection = (
                (right_distance < 100) &
                (xs > center)
            )

        else:

            # Initialization stage.
            #
            # This is only used until we have a
            # reliable lane model.

            left_selection = (
                xs < 0.50 * w
            )

            right_selection = (
                xs >= 0.50 * w
            )

        return (
            xs[left_selection],
            ys[left_selection],
            xs[right_selection],
            ys[right_selection]
        )

    # =========================================================
    # Main detection
    # =========================================================

    def detect(self, frame):

        h, w = frame.shape[:2]

        self.height = h
        self.width = w

        # -----------------------------------------------------
        # 1. Lane evidence
        # -----------------------------------------------------

        mask = self.create_lane_mask(
            frame
        )

        # -----------------------------------------------------
        # 2. ROI
        # -----------------------------------------------------

        mask = self.apply_roi(
            mask
        )

        # -----------------------------------------------------
        # 3. Component filtering
        # -----------------------------------------------------

        mask = self.filter_components(
            mask
        )

        # -----------------------------------------------------
        # 4. Candidate assignment
        # -----------------------------------------------------

        (
            left_x,
            left_y,
            right_x,
            right_y
        ) = self.assign_candidates(
            mask
        )

        # -----------------------------------------------------
        # 5. Fit independently
        # -----------------------------------------------------

        left_fit = self.fit_side(
            left_x,
            left_y
        )

        right_fit = self.fit_side(
            right_x,
            right_y
        )

        # -----------------------------------------------------
        # 6. Independently validate
        # -----------------------------------------------------

        left_valid = self.validate_side(
            left_fit,
            "left"
        )

        right_valid = self.validate_side(
            right_fit,
            "right"
        )

        # -----------------------------------------------------
        # 7. Update LEFT independently
        # -----------------------------------------------------

        if left_valid:

            self.left_missed = 0

            self.smooth_left = self.smooth_fit(
                self.smooth_left,
                left_fit
            )

            self.prev_left = (
                self.smooth_left.copy()
            )

        else:

            self.left_missed += 1

            if self.left_missed > self.max_missed:

                self.smooth_left = None

        # -----------------------------------------------------
        # 8. Update RIGHT independently
        # -----------------------------------------------------

        if right_valid:

            self.right_missed = 0

            self.smooth_right = self.smooth_fit(
                self.smooth_right,
                right_fit
            )

            self.prev_right = (
                self.smooth_right.copy()
            )

        else:

            self.right_missed += 1

            if self.right_missed > self.max_missed:

                self.smooth_right = None

        # -----------------------------------------------------
        # 9. Final pair validation
        # -----------------------------------------------------

        pair_valid = self.validate_pair(
            self.smooth_left,
            self.smooth_right
        )

        # -----------------------------------------------------
        # 10. Lane center
        # -----------------------------------------------------

        lane_center = None

        if (
            self.smooth_left is not None and
            self.smooth_right is not None and
            pair_valid
        ):

            y_eval = int(
                0.90 * h
            )

            left_bottom = np.polyval(
                self.smooth_left,
                y_eval
            )

            right_bottom = np.polyval(
                self.smooth_right,
                y_eval
            )

            lane_center = (
                left_bottom +
                right_bottom
            ) / 2.0

        # -----------------------------------------------------
        # 11. Camera center
        # -----------------------------------------------------

        camera_center = w / 2.0

        lateral_offset = None

        if lane_center is not None:

            lateral_offset = (
                camera_center -
                lane_center
            )

        # -----------------------------------------------------
        # 12. Confidence
        # -----------------------------------------------------

        confidence = 0.0

        if left_valid:
            confidence += 0.35

        if right_valid:
            confidence += 0.35

        if pair_valid:
            confidence += 0.30

        # Confidence must collapse when there is no
        # usable pair for an extended period.

        if not pair_valid:

            confidence *= 0.5

        return {

            "left_fit": self.smooth_left,

            "right_fit": self.smooth_right,

            "lane_center": lane_center,

            "camera_center": camera_center,

            "lateral_offset_pixels":
                lateral_offset,

            "confidence":
                confidence,

            "valid":
                pair_valid,

            "left_valid":
                left_valid,

            "right_valid":
                right_valid,

            "mask":
                mask,

            "roi_polygon":
                self.get_roi_polygon(
                    w,
                    h
                ),

            # Kept only for compatibility with
            # the current debug script.
            "hough_lines":
                None
        }