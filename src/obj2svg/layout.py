from typing import List, Tuple, Dict, Any
import math
from .geometry import get_polygon_bounds

def create_grid(width: float, height: float, cell_size: float) -> Dict[Tuple[int, int], List[Any]]:
    """Create a spatial grid for quick collision checks."""
    cols = int(width / cell_size) + 1
    rows = int(height / cell_size) + 1
    return {(i, j): [] for i in range(rows) for j in range(cols)}

def get_grid_cells(bounds: Tuple[float, float, float, float], cell_size: float) -> List[Tuple[int, int]]:
    """Get grid cells that a shape occupies based on its bounds."""
    min_x, min_y, max_x, max_y = bounds
    start_col = max(0, int(min_x / cell_size))
    end_col = int(max_x / cell_size) + 1
    start_row = max(0, int(min_y / cell_size))
    end_row = int(max_y / cell_size) + 1
    return [(row, col) for row in range(start_row, end_row) for col in range(start_col, end_col)]

def quick_collision_check(bounds1: Tuple[float, float, float, float], bounds2: Tuple[float, float, float, float], min_distance: float) -> bool:
    """Fast AABB collision check before detailed polygon check."""
    min_x1, min_y1, max_x1, max_y1 = bounds1
    min_x2, min_y2, max_x2, max_y2 = bounds2
    return not (
        max_x1 + min_distance < min_x2
        or min_x1 > max_x2 + min_distance
        or max_y1 + min_distance < min_y2
        or min_y1 > max_y2 + min_distance
    ) 