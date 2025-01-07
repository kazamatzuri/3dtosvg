import sys
import os
import math
from typing import List, Tuple, Dict, Any
import numpy as np
import svgwrite
from scipy.spatial.distance import cdist

from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)

# Conversion factor
INCH_TO_MM = 25.4


def debug_shapes(polygons: List[Dict[str, Any]], svg_width: float, svg_height: float) -> None:
    """Log details of polygons for debugging."""
    for i, poly in enumerate(polygons):
        center = poly["center"]
        radius = poly["radius"]
        print(f"Shape {i}: Center={center}, Radius={radius}")

        # Ensure shape is within canvas bounds
        if center[0] - radius < 0 or center[0] + radius > svg_width:
            print(f"Shape {i} is out of bounds horizontally!")
        if center[1] - radius < 0 or center[1] + radius > svg_height:
            print(f"Shape {i} is out of bounds vertically!")


def get_polygon_center_and_radius(polygon_points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], float]:
    """Calculate the center point and radius of a polygon."""
    x_coords, y_coords = zip(*polygon_points)
    center_x = sum(x_coords) / len(polygon_points)
    center_y = sum(y_coords) / len(polygon_points)

    # Calculate radius as maximum distance from center to any point
    radius = max(
        math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in polygon_points
    )
    return (center_x, center_y), radius


def check_collision(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]], min_distance: float) -> bool:
    """Check if two polygons are too close."""
    center1, radius1 = get_polygon_center_and_radius(points1)
    center2, radius2 = get_polygon_center_and_radius(points2)

    # Calculate center-to-center distance
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    distance = math.sqrt(dx**2 + dy**2)

    # Check if polygons are too close
    return distance < (radius1 + radius2 + min_distance)


def check_polygon_collision(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]], min_distance: float) -> bool:
    """Check if two polygons are too close using actual geometry."""
    # Convert to numpy arrays for easier computation
    p1 = np.array(poly1)
    p2 = np.array(poly2)

    # Use scipy's distance calculation between point sets
    distances = cdist(p1, p2)
    min_dist = np.min(distances)

    # Changed back to < but now SUBTRACT min_distance from comparison
    return min_dist < min_distance


def rotate_polygon(points: List[Tuple[float, float]], angle_degrees: float) -> List[Tuple[float, float]]:
    """Rotate polygon around its centroid by given angle in degrees."""
    # Convert to numpy array
    points_array = np.array(points)

    # Calculate centroid
    centroid = np.mean(points_array, axis=0)

    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)

    # Create rotation matrix
    rot_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Translate to origin, rotate, and translate back
    centered = points_array - centroid
    rotated = np.dot(centered, rot_matrix.T)
    result = rotated + centroid

    return [(float(x), float(y)) for x, y in result]


def get_polygon_bounds(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Get the bounding box of a polygon."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


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
    return [
        (row, col)
        for row in range(start_row, end_row)
        for col in range(start_col, end_col)
    ]


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


class OBJToSVG:
    def __init__(self, obj_file: str, svg_size_inches: Tuple[float, float], min_distance_inches: float, output_prefix: str):
        self.obj_file = obj_file
        # Store both inch and mm dimensions
        self.svg_width_inches, self.svg_height_inches = svg_size_inches
        self.svg_width = self.svg_width_inches * INCH_TO_MM
        self.svg_height = self.svg_height_inches * INCH_TO_MM
        self.min_distance = min_distance_inches * INCH_TO_MM
        self.output_prefix = output_prefix
        self.vertices: List[List[float]] = []
        self.faces: List[List[int]] = []
        self.projections: List[List[Tuple[float, float]]] = []
        # Blender always exports OBJ files in meters, regardless of scene unit settings.
        # A 60mm cube in Blender will be exported as vertices with coordinates like 0.06 (meters)
        self.input_is_meters = True
        self.edge_labels = {} # Store edge labels
        self.next_label = 1 # Initialize the next label

    def parse_obj(self) -> None:
        """Parses the OBJ file and extracts vertices and flat faces."""
        with open(self.obj_file, "r") as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    self.vertices.append(list(map(float, parts[1:])))
                elif parts[0] == "f":
                    face_indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    self.faces.append(face_indices)

    def project_faces(self) -> None:
        """Projects 3D faces to 2D using orthogonal projection for each face plane."""
        print("\nFace dimensions:")
        edge_map = {} # Dictionary to store edges and their labels
        for face_idx, face in enumerate(self.faces):
            # Get vertices of the face (coordinates are in meters from OBJ)
            face_vertices = np.array([self.vertices[i] for i in face])

            # Convert from meters to mm. Blender exports everything in meters,
            # so we multiply by 1000 to get millimeters (1m = 1000mm)
            if self.input_is_meters:
                face_vertices = face_vertices * 1000

            print(f"\nFace {face_idx + 1} dimensions (mm):")
            # Calculate bounds
            xs_orig = face_vertices[:, 0]
            ys_orig = face_vertices[:, 1]
            zs_orig = face_vertices[:, 2]
            print(
                f"  X range: {min(xs_orig):.2f} to {max(xs_orig):.2f} mm ({max(xs_orig) - min(xs_orig):.2f} mm)"
            )
            print(
                f"  Y range: {min(ys_orig):.2f} to {max(ys_orig):.2f} mm ({max(ys_orig) - min(ys_orig):.2f} mm)"
            )
            print(
                f"  Z range: {min(zs_orig):.2f} to {max(zs_orig):.2f} mm ({max(zs_orig) - min(zs_orig):.2f} mm)"
            )

            # Calculate the normal vector of the face
            v1, v2 = (
                face_vertices[1] - face_vertices[0],
                face_vertices[2] - face_vertices[0],
            )
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            # Project onto appropriate plane
            abs_normal = np.abs(normal)
            if abs_normal[0] >= abs_normal[1] and abs_normal[0] >= abs_normal[2]:
                projection = face_vertices[:, [1, 2]]
                proj_axes = "YZ"
            elif abs_normal[1] >= abs_normal[0] and abs_normal[1] >= abs_normal[2]:
                projection = face_vertices[:, [0, 2]]
                proj_axes = "XZ"
            else:
                projection = face_vertices[:, [0, 1]]
                proj_axes = "XY"

            # Convert to list of tuples
            projection = [(float(x), float(y)) for x, y in projection]

            # Print projected dimensions
            min_x, min_y, max_x, max_y = get_polygon_bounds(projection)
            print(f"  Projected to {proj_axes} plane (mm):")
            print(f"    Width: {max_x - min_x:.2f} mm")
            print(f"    Height: {max_y - min_y:.2f} mm")

            # Generate edge labels
            edges = []
            for i in range(len(projection)):
                p1 = projection[i]
                p2 = projection[(i + 1) % len(projection)]
                
                # Create a key for the edge using the vertex indices
                v1_index = face[i]
                v2_index = face[(i + 1) % len(face)]
                edge_key = tuple(sorted((v1_index, v2_index)))
                
                midpoint_x = (p1[0] + p2[0]) / 2
                midpoint_y = (p1[1] + p2[1]) / 2
                
                if edge_key in edge_map:
                    label = edge_map[edge_key]
                else:
                    label = self.next_label
                    edge_map[edge_key] = label
                    self.next_label += 1
                
                edges.append(((p1, p2), (midpoint_x, midpoint_y), label))
            
            self.projections.append((projection, edges))

        print(f"\nTotal faces projected: {len(self.projections)}")

    def arrange_layout(self) -> List[List[List[Tuple[float, float]]]]:
        """Arranges shapes in an expanding diamond pattern on a grid."""
        shapes = [proj[0] for proj in self.projections] # Extract only the polygon points
        
        # Calculate bounding boxes for all shapes
        bounds = [get_polygon_bounds(shape) for shape in shapes]
        
        # Find the largest bounding box
        max_width = 0
        max_height = 0
        for min_x, min_y, max_x, max_y in bounds:
            max_width = max(max_width, max_x - min_x)
            max_height = max(max_height, max_y - min_y)
        
        # Add margin to the largest bounding box
        cell_size = max(max_width, max_height) + self.min_distance * 2
        
        # Calculate grid dimensions
        num_shapes = len(shapes)
        grid_size = math.ceil(math.sqrt(num_shapes)) * 2 # Ensure enough space for diamond
        grid_width = grid_size * cell_size
        grid_height = grid_size * cell_size
        
        # Calculate SVG dimensions
        svg_width = grid_width + self.min_distance * 2
        svg_height = grid_height + self.min_distance * 2
        
        # Calculate the center of the grid
        center_x = svg_width / 2
        center_y = svg_height / 2
        
        # Create a list to hold the placed shapes
        placed_shapes = []
        
        # Place shapes in an expanding diamond pattern
        x, y = 0, 0
        dx, dy = 0, -1
        for i, shape in enumerate(shapes):
            # Calculate the position of the shape in the grid
            shape_x = center_x + x * cell_size
            shape_y = center_y + y * cell_size
            
            # Calculate the offset to center the shape in the grid cell
            min_x, min_y, max_x, max_y = bounds[i]
            offset_x = shape_x - (min_x + max_x) / 2
            offset_y = shape_y - (min_y + max_y) / 2
            
            # Translate the shape to the correct position
            translated_shape = [(x + offset_x, y + offset_y) for x, y in shape]
            placed_shapes.append(translated_shape)
            
            # Update the grid position
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy
        
        # Return the placed shapes
        return [placed_shapes]

    def export_to_svg(self) -> None:
        """Exports shapes to a single SVG file."""
        pages = self.arrange_layout()
        placed_shapes = pages[0]
        
        output_dir = self.output_prefix
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "sheet1.svg")

        # Calculate SVG dimensions
        all_bounds = [get_polygon_bounds(shape) for shape in placed_shapes]
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        svg_width = max_x - min_x + self.min_distance * 2
        svg_height = max_y - min_y + self.min_distance * 2

        current_page = svgwrite.Drawing(
            filename,
            size=(f"{svg_width}mm", f"{svg_height}mm"),
            viewBox=f"{min_x - self.min_distance} {min_y - self.min_distance} {svg_width} {svg_height}",
        )

        # Add the rectangle representing the original SVG size
        current_page.add(
            current_page.rect(
                insert=(self.min_distance, self.min_distance),
                size=(self.svg_width - self.min_distance * 2, self.svg_height - self.min_distance * 2),
                stroke="red",
                fill="none",
                stroke_width=0.5,
            )
        )
        
        for (projection, edges), placed_projection in zip(self.projections, placed_shapes):
            # Create a group for the polygon and its labels
            group = current_page.g()
            
            # Create the polygon path
            path_data = (
                "M " + " L ".join([f"{x},{y}" for x, y in placed_projection]) + " Z"
            )
            path = current_page.path(
                d=path_data, stroke="black", fill="none", stroke_width=0.5
            )
            group.add(path)
            
            # Add edge labels
            for (p1, p2), (mid_x, mid_y), label in edges:
                # Find the corresponding placed points
                placed_p1_index = projection.index(p1)
                placed_p2_index = projection.index(p2)
                placed_p1 = placed_projection[placed_p1_index]
                placed_p2 = placed_projection[placed_p2_index]
                
                placed_mid_x = (placed_p1[0] + placed_p2[0]) / 2
                placed_mid_y = (placed_p1[1] + placed_p2[1]) / 2
                
                # Calculate the centroid of the polygon
                centroid_x = sum(p[0] for p in placed_projection) / len(placed_projection)
                centroid_y = sum(p[1] for p in placed_projection) / len(placed_projection)
                
                # Calculate the vector from the midpoint to the centroid
                vec_x = centroid_x - placed_mid_x
                vec_y = centroid_y - placed_mid_y
                
                # Normalize the vector
                vec_len = math.sqrt(vec_x**2 + vec_y**2)
                if vec_len > 0:
                    vec_x /= vec_len
                    vec_y /= vec_len
                
                # Move the midpoint slightly towards the centroid
                label_offset = 5 # Adjust this value to control how far the label is moved
                placed_mid_x += vec_x * label_offset
                placed_mid_y += vec_y * label_offset
                
                text = current_page.text(
                    str(label),
                    insert=(placed_mid_x, placed_mid_y),
                    fill="blue",
                    font_size=3,
                    text_anchor="middle",
                    alignment_baseline="middle"
                )
                group.add(text)
            
            # Add the group to the page
            current_page.add(group)

        print(f"Created SVG with {len(placed_shapes)} shapes")
        current_page.save(pretty=True)

    def run(self) -> None:
        """Main method to run all steps."""
        print(f"Processing OBJ file: {self.obj_file}")
        self.parse_obj()
        self.project_faces()
        self.export_to_svg()


# Command-line Usage
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python obj_to_svg.py <input.obj> <svg_width_inches> <svg_height_inches> <min_distance_inches>"
        )
        print("Example: python obj_to_svg.py input.obj 8 8 0.5")
        sys.exit(1)

    input_file = sys.argv[1]
    svg_width_inches = float(sys.argv[2])
    svg_height_inches = float(sys.argv[3])
    min_distance_inches = float(sys.argv[4])
    output_prefix = os.path.splitext(os.path.basename(input_file))[0]

    obj_to_svg = OBJToSVG(
        obj_file=input_file,
        svg_size_inches=(svg_width_inches, svg_height_inches),
        min_distance_inches=min_distance_inches,
        output_prefix=output_prefix,
    )
    obj_to_svg.run()
