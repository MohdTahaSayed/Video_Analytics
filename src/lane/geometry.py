import numpy as np


def fit_polynomial(points):
    """
    Fit a second-order polynomial:

        x(y) = ay^2 + by + c

    We use x as a function of y because lane boundaries are
    approximately one x-position for each image row.
    """

    if points is None or len(points) < 6:
        return None

    points = np.asarray(points, dtype=np.float64)

    x = points[:, 0]
    y = points[:, 1]

    # Center y values to improve numerical conditioning.
    y_mean = np.mean(y)
    y_normalized = y - y_mean

    A = np.column_stack([
        y_normalized ** 2,
        y_normalized,
        np.ones_like(y_normalized)
    ])

    try:
        coefficients, _, _, _ = np.linalg.lstsq(
            A,
            x,
            rcond=None
        )
    except np.linalg.LinAlgError:
        return None

    return {
        "coefficients": coefficients,
        "y_mean": y_mean
    }


def evaluate_polynomial(model, y):
    """
    Evaluate the fitted x(y) polynomial.
    """

    if model is None:
        return None

    coefficients = model["coefficients"]
    y_mean = model["y_mean"]

    y_normalized = np.asarray(y) - y_mean

    a, b, c = coefficients

    return (
        a * y_normalized ** 2
        + b * y_normalized
        + c
    )


def lane_center(left_x, right_x):
    """
    Center of the detected lane at a particular y.
    """

    return (left_x + right_x) / 2.0


def normalized_lane_position(
    ego_x,
    left_x,
    right_x
):
    """
    Normalized ego position:

        0 -> left lane boundary
        0.5 -> lane center
        1 -> right lane boundary
    """

    lane_width = right_x - left_x

    if lane_width <= 0:
        return None

    return (
        ego_x - left_x
    ) / lane_width