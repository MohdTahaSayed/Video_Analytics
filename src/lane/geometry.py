import numpy as np


def fit_polynomial(points):
    """
    Fit a 2nd degree polynomial to a set of (x, y) points.
    
    Args:
        points: List of (x, y) points
        
    Returns:
        numpy array of polynomial coefficients [a, b, c] where x = a*y^2 + b*y + c
        or None if insufficient points
    """
    if not points or len(points) < 50:
        return None
    
    points = np.array(points)
    
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    try:
        # Fit polynomial: x = a*y^2 + b*y + c
        coefficients = np.polyfit(y, x, 2)
        return coefficients
    except np.linalg.LinAlgError:
        return None


def evaluate_polynomial(model, y_values):
    """
    Evaluate a polynomial model at given y values.
    
    Args:
        model: Either a numpy array of coefficients [a, b, c] or a dictionary
               with a 'coefficients' key containing the array
        y_values: Array of y values to evaluate at
        
    Returns:
        Array of x values, or None if model is None
    """
    if model is None:
        return None
    
    # Handle both dictionary and array formats
    if isinstance(model, dict):
        coefficients = model.get("coefficients")
        if coefficients is None:
            return None
    else:
        # Assume it's already a numpy array or list of coefficients
        coefficients = model
    
    # Convert to numpy array if needed
    coefficients = np.asarray(coefficients)
    
    # Evaluate polynomial at y_values
    return np.polyval(coefficients, y_values)


def get_lane_width(left_model, right_model, y_eval):
    """
    Calculate lane width at a specific y position.
    
    Args:
        left_model: Left lane polynomial model
        right_model: Right lane polynomial model
        y_eval: y position to evaluate at
        
    Returns:
        Lane width in pixels, or None if either model is None
    """
    if left_model is None or right_model is None:
        return None
    
    left_x = evaluate_polynomial(left_model, y_eval)
    right_x = evaluate_polynomial(right_model, y_eval)
    
    if left_x is None or right_x is None:
        return None
    
    return float(right_x - left_x)


def validate_polynomial(model):
    """
    Validate a polynomial model.
    
    Args:
        model: Polynomial coefficients [a, b, c]
        
    Returns:
        True if valid, False otherwise
    """
    if model is None:
        return False
    
    # Check if coefficients are reasonable
    if isinstance(model, dict):
        coeffs = model.get("coefficients")
        if coeffs is None:
            return False
    else:
        coeffs = model
    
    coeffs = np.asarray(coeffs)
    
    # Check for NaN or inf
    if not np.isfinite(coeffs).all():
        return False
    
    # Check curvature isn't too extreme
    if len(coeffs) >= 3:
        if abs(coeffs[0]) > 0.01:  # Quadratic coefficient
            return False
    
    return True