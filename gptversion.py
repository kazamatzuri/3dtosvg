import numpy as np
import svgwrite
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import sys
import os
import math
import random
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

# Conversion factor
INCH_TO_MM = 25.4
def debug_shapes(polygons, svg_width, svg_height):
    """Log details of polygons for debugging."""
    for i, poly in enumerate(polygons):
        center = poly['center']
        radius = poly['radius']
        print(f"Shape {i}: Center={center}, Radius={radius}")

        # Ensure shape is within canvas bounds
        if center[0] - radius < 0 or center[0] + radius > svg_width:
            print(f"Shape {i} is out of bounds horizontally!")
        if center[1] - radius < 0 or center[1] + radius > svg_height:
            print(f"Shape {i} is out of bounds vertically!")
def get_polygon_center_and_radius(polygon_points):
    """Calculate the center point and radius of a polygon."""
    x_coords, y_coords = zip(*polygon_points)
    center_x = sum(x_coords) / len(polygon_points)
    center_y = sum(y_coords) / len(polygon_points)

    # Calculate radius as maximum distance from center to any point
    radius = max(math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in polygon_points)
    return (center_x, center_y), radius


def check_collision(points1, points2, min_distance):
    """Check if two polygons are too close."""
    center1, radius1 = get_polygon_center_and_radius(points1)
    center2, radius2 = get_polygon_center_and_radius(points2)

    # Calculate center-to-center distance
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    distance = math.sqrt(dx**2 + dy**2)

    # Check if polygons are too close
    return distance < (radius1 + radius2 + min_distance)

def check_polygon_collision(poly1, poly2, min_distance):
    """Check if two polygons are too close using actual geometry."""
    # Convert to numpy arrays for easier computation
    p1 = np.array(poly1)
    p2 = np.array(poly2)
    
    # Use scipy's distance calculation between point sets
    distances = cdist(p1, p2)
    min_dist = np.min(distances)
    
    # Changed back to < but now SUBTRACT min_distance from comparison
    return min_dist < min_distance

def rotate_polygon(points, angle_degrees):
    """Rotate polygon around its centroid by given angle in degrees."""
    # Convert to numpy array
    points_array = np.array(points)
    
    # Calculate centroid
    centroid = np.mean(points_array, axis=0)
    
    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)
    
    # Create rotation matrix
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Translate to origin, rotate, and translate back
    centered = points_array - centroid
    rotated = np.dot(centered, rot_matrix.T)
    result = rotated + centroid
    
    return [(float(x), float(y)) for x, y in result]

def get_polygon_bounds(points):
    """Get the bounding box of a polygon."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def create_grid(width, height, cell_size):
    """Create a spatial grid for quick collision checks."""
    cols = int(width / cell_size) + 1
    rows = int(height / cell_size) + 1
    return {(i, j): [] for i in range(rows) for j in range(cols)}

def get_grid_cells(bounds, cell_size):
    """Get grid cells that a shape occupies based on its bounds."""
    min_x, min_y, max_x, max_y = bounds
    start_col = max(0, int(min_x / cell_size))
    end_col = int(max_x / cell_size) + 1
    start_row = max(0, int(min_y / cell_size))
    end_row = int(max_y / cell_size) + 1
    return [(row, col) for row in range(start_row, end_row) 
                      for col in range(start_col, end_col)]

def quick_collision_check(bounds1, bounds2, min_distance):
    """Fast AABB collision check before detailed polygon check."""
    min_x1, min_y1, max_x1, max_y1 = bounds1
    min_x2, min_y2, max_x2, max_y2 = bounds2
    return not (max_x1 + min_distance < min_x2 or 
               min_x1 > max_x2 + min_distance or 
               max_y1 + min_distance < min_y2 or 
               min_y1 > max_y2 + min_distance)

class OBJToSVG:
    def __init__(self, obj_file, svg_size_inches, min_distance_inches, output_prefix):
        self.obj_file = obj_file
        # Store both inch and mm dimensions
        self.svg_width_inches, self.svg_height_inches = svg_size_inches
        self.svg_width = self.svg_width_inches * INCH_TO_MM
        self.svg_height = self.svg_height_inches * INCH_TO_MM
        self.min_distance = min_distance_inches * INCH_TO_MM
        self.output_prefix = output_prefix
        self.vertices = []
        self.faces = []
        self.projections = []
        # Blender always exports OBJ files in meters, regardless of scene unit settings.
        # A 60mm cube in Blender will be exported as vertices with coordinates like 0.06 (meters)
        self.input_is_meters = True

    def parse_obj(self):
        """Parses the OBJ file and extracts vertices and flat faces."""
        with open(self.obj_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    self.vertices.append(list(map(float, parts[1:])))
                elif parts[0] == 'f':
                    face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    self.faces.append(face_indices)

    def project_faces(self):
        """Projects 3D faces to 2D using orthogonal projection for each face plane."""
        print("\nFace dimensions:")
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
            print(f"  X range: {min(xs_orig):.2f} to {max(xs_orig):.2f} mm ({max(xs_orig) - min(xs_orig):.2f} mm)")
            print(f"  Y range: {min(ys_orig):.2f} to {max(ys_orig):.2f} mm ({max(ys_orig) - min(ys_orig):.2f} mm)")
            print(f"  Z range: {min(zs_orig):.2f} to {max(zs_orig):.2f} mm ({max(zs_orig) - min(zs_orig):.2f} mm)")
            
            # Calculate the normal vector of the face
            v1, v2 = face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0]
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
            
            self.projections.append(projection)
        
        print(f"\nTotal faces projected: {len(self.projections)}")


    def arrange_layout(self):
        """Force-directed layout for shape packing."""
        pages = []
        remaining_shapes = []
        
        # Pre-calculate and cache bounds for all shapes
        for points in self.projections:
            bounds = get_polygon_bounds(points)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            area = width * height
            remaining_shapes.append((points, bounds, area))
        
        # Sort shapes by area (largest first)
        remaining_shapes.sort(key=lambda x: x[2], reverse=True)
        total_shapes = len(remaining_shapes)
        current_page = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TextColumn("•"),
            TimeRemainingColumn(),
            refresh_per_second=10
        ) as progress:
            packing_task = progress.add_task(f"[cyan]Packing {total_shapes} shapes...", 
                                           total=total_shapes, completed=0)
            
            while remaining_shapes:
                if not current_page:
                    # Start new page
                    current_page = []
                    pages.append(current_page)
                    
                    # Place first shape with proper margin
                    shape, bounds, _ = remaining_shapes.pop(0)
                    # Use single min_distance for margin
                    dx = self.min_distance - bounds[0]
                    dy = self.min_distance - bounds[1]
                    placed_shape = [(x + dx, y + dy) for x, y in shape]
                    current_page.append((placed_shape, get_polygon_bounds(placed_shape)))
                    progress.update(packing_task, advance=1)
                    continue
                
                # Try to place next shape
                best_placement = None
                best_shape_idx = None
                min_energy = float('inf')
                
                # Try each remaining shape
                for shape_idx, (shape, bounds, _) in enumerate(remaining_shapes):
                    shape_width = bounds[2] - bounds[0]
                    shape_height = bounds[3] - bounds[1]
                    
                    # Try different initial positions around existing shapes
                    for placed_shape, placed_bounds in current_page:
                        for angle in range(0, 360, 45):  # Try 8 directions
                            # Calculate radius to ensure minimum separation
                            center_x = (placed_bounds[0] + placed_bounds[2]) / 2
                            center_y = (placed_bounds[1] + placed_bounds[3]) / 2
                            
                            # Calculate radius based on shape sizes plus minimum distance
                            placed_size = max(placed_bounds[2] - placed_bounds[0], 
                                            placed_bounds[3] - placed_bounds[1])
                            new_shape_size = max(shape_width, shape_height)
                            radius = (placed_size + new_shape_size) / 2 + self.min_distance
                            
                            test_x = center_x + radius * math.cos(math.radians(angle))
                            test_y = center_y + radius * math.sin(math.radians(angle))
                            
                            # Try different rotations
                            for rotation in [0, 90, 180, 270]:
                                rotated = rotate_polygon(shape, rotation)
                                rot_bounds = get_polygon_bounds(rotated)
                                
                                dx = test_x - (rot_bounds[2] + rot_bounds[0])/2
                                dy = test_y - (rot_bounds[3] + rot_bounds[1])/2
                                test_shape = [(x + dx, y + dy) for x, y in rotated]
                                test_bounds = get_polygon_bounds(test_shape)
                                
                                # Check page bounds
                                if (test_bounds[0] < self.min_distance or 
                                    test_bounds[1] < self.min_distance or
                                    test_bounds[2] > self.svg_width - self.min_distance or 
                                    test_bounds[3] > self.svg_height - self.min_distance):
                                    continue
                                
                                # Check collisions
                                collision = False
                                for other_shape, _ in current_page:
                                    if check_polygon_collision(test_shape, other_shape, self.min_distance):
                                        collision = True
                                        break
                                
                                if not collision:
                                    # Calculate energy (compactness) of this placement
                                    energy = 0
                                    for other_shape, other_bounds in current_page:
                                        # Add distance between centers
                                        cx1 = (test_bounds[0] + test_bounds[2]) / 2
                                        cy1 = (test_bounds[1] + test_bounds[3]) / 2
                                        cx2 = (other_bounds[0] + other_bounds[2]) / 2
                                        cy2 = (other_bounds[1] + other_bounds[3]) / 2
                                        energy += math.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
                                    
                                    # Prefer placements closer to origin
                                    energy += (test_bounds[0]**2 + test_bounds[1]**2) * 0.1
                                    
                                    if energy < min_energy:
                                        min_energy = energy
                                        best_placement = (test_shape, test_bounds)
                                        best_shape_idx = shape_idx
                
                if best_placement:
                    test_shape, test_bounds = best_placement
                    current_page.append((test_shape, test_bounds))
                    remaining_shapes.pop(best_shape_idx)
                    progress.update(packing_task, advance=1)
                else:
                    # If no placement found, start new page
                    current_page = []
                    progress.update(packing_task, description=f"[cyan]Starting new sheet (packed {total_shapes - len(remaining_shapes)} of {total_shapes} shapes)")
            
            progress.update(packing_task, description=f"[green]Finished packing {total_shapes} shapes into {len(pages)} sheets")
        
        # Convert to final format
        return [[shape for shape, _ in page] for page in pages if page]


    def export_to_svg(self):
        """Exports shapes to SVG files with multiple shapes per page."""
        pages = self.arrange_layout()
        
        output_dir = self.output_prefix
        os.makedirs(output_dir, exist_ok=True)
        
        for page_num, shapes in enumerate(pages, 1):
            filename = os.path.join(output_dir, f"sheet{page_num}.svg")
            current_page = svgwrite.Drawing(
                filename,
                size=(f"{self.svg_width}mm", f"{self.svg_height}mm"),
                viewBox=f"0 0 {self.svg_width} {self.svg_height}"
            )
            
            for projection in shapes:
                path_data = "M " + " L ".join([f"{x},{y}" for x, y in projection]) + " Z"
                path = current_page.path(
                    d=path_data,
                    stroke="black",
                    fill="none",
                    stroke_width=0.5
                )
                current_page.add(path)
            
            print(f"Created SVG for sheet {page_num} with {len(shapes)} shapes")
            current_page.save(pretty=True)


    def run(self):
        """Main method to run all steps."""
        print(f"Processing OBJ file: {self.obj_file}")
        self.parse_obj()
        self.project_faces()
        self.export_to_svg()


# Command-line Usage
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python obj_to_svg.py <input.obj> <svg_width_inches> <svg_height_inches> <min_distance_inches>")
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
        output_prefix=output_prefix
    )
    obj_to_svg.run()
