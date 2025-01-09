from typing import List, Tuple, Dict, Any
import math
import numpy as np
from scipy.spatial.distance import cdist

def get_polygon_center_and_radius(polygon_points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], float]:
    """Calculate the center point and radius of a polygon."""
    x_coords, y_coords = zip(*polygon_points)
    center_x = sum(x_coords) / len(polygon_points)
    center_y = sum(y_coords) / len(polygon_points)

    radius = max(
        math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in polygon_points
    )
    return (center_x, center_y), radius

def check_collision(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]], min_distance: float) -> bool:
    """Check if two polygons are too close."""
    center1, radius1 = get_polygon_center_and_radius(points1)
    center2, radius2 = get_polygon_center_and_radius(points2)

    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    distance = math.sqrt(dx**2 + dy**2)

    return distance < (radius1 + radius2 + min_distance)

def check_polygon_collision(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]], min_distance: float) -> bool:
    """Check if two polygons are too close using actual geometry."""
    p1 = np.array(poly1)
    p2 = np.array(poly2)
    distances = cdist(p1, p2)
    min_dist = np.min(distances)
    return min_dist < min_distance

def get_polygon_bounds(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Get the bounding box of a polygon."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def rotate_polygon(points: List[Tuple[float, float]], angle_degrees: float) -> List[Tuple[float, float]]:
    """Rotate polygon around its centroid by given angle in degrees."""
    points_array = np.array(points)
    centroid = np.mean(points_array, axis=0)
    angle_rad = np.radians(angle_degrees)
    
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)],
    ])

    centered = points_array - centroid
    rotated = np.dot(centered, rot_matrix.T)
    result = rotated + centroid

    return [(float(x), float(y)) for x, y in result] 