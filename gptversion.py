import numpy as np
import svgwrite
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import sys
import os
import math
import random

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

class OBJToSVG:
    def __init__(self, obj_file, svg_size_inches, min_distance_inches, output_prefix):
        self.obj_file = obj_file
        # Convert inches to millimeters
        self.svg_width, self.svg_height = [dim * INCH_TO_MM for dim in svg_size_inches]
        self.min_distance = min_distance_inches * INCH_TO_MM
        self.output_prefix = output_prefix
        self.vertices = []
        self.faces = []
        self.projections = []

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
        for face in self.faces:
            # Get vertices of the face (in mm from OBJ)
            face_vertices = np.array([self.vertices[i] for i in face])
            
            # Convert from mm to inches for internal calculations
            face_vertices = face_vertices * INCH_TO_MM
            
            # Calculate the normal vector of the face
            v1, v2 = face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            # Determine the dominant axis of the normal to project onto the appropriate plane
            abs_normal = np.abs(normal)
            if abs_normal[0] >= abs_normal[1] and abs_normal[0] >= abs_normal[2]:  # Dominant X-axis
                projection = face_vertices[:, [1, 2]]  # Project to YZ plane
            elif abs_normal[1] >= abs_normal[0] and abs_normal[1] >= abs_normal[2]:  # Dominant Y-axis
                projection = face_vertices[:, [0, 2]]  # Project to XZ plane
            else:  # Dominant Z-axis
                projection = face_vertices[:, [0, 1]]  # Project to XY plane
            
            # Convert numpy array to list of tuples for consistency
            projection = [(float(x), float(y)) for x, y in projection]
            
            # Keep in inches for layout
            self.projections.append(projection)
        
        print(f"Total faces projected: {len(self.projections)}")


    def arrange_layout(self):
        """Creates a separate page for each shape, centered in the SVG."""
        pages = []
        
        # Convert SVG dimensions back to inches for calculations
        page_width_inches = self.svg_width / INCH_TO_MM
        page_height_inches = self.svg_height / INCH_TO_MM
        
        for projection in self.projections:
            # Get bounds (already in inches)
            min_x, min_y, max_x, max_y = get_polygon_bounds(projection)
            width = max_x - min_x
            height = max_y - min_y
            
            # Calculate center of shape (in inches)
            shape_center_x = (max_x + min_x) / 2
            shape_center_y = (max_y + min_y) / 2
            
            # Calculate center of page (in inches)
            page_center_x = page_width_inches / 2
            page_center_y = page_height_inches / 2
            
            # Calculate translation to center (in inches)
            dx = page_center_x - shape_center_x
            dy = page_center_y - shape_center_y
            
            # Move shape to center (keeping everything in inches)
            centered_points = [(x + dx, y + dy) for x, y in projection]
            pages.append([centered_points])  # Each page contains one centered shape
        
        return pages


    def export_to_svg(self):
        """Exports each shape to its own SVG file in a dedicated directory."""
        pages = self.arrange_layout()
        
        # Create output directory based on input filename
        output_dir = self.output_prefix
        os.makedirs(output_dir, exist_ok=True)
        
        for page_num, shapes in enumerate(pages, 1):
            # Create SVG file in the output directory
            filename = os.path.join(output_dir, f"part{page_num}.svg")
            current_page = svgwrite.Drawing(
                filename,
                size=(f"{self.svg_width}mm", f"{self.svg_height}mm"),
                viewBox=f"0 0 {self.svg_width} {self.svg_height}"
            )
            
            # Add the single centered shape
            projection = shapes[0]  # Only one shape per page
            # Convert inches to mm for SVG output
            projection_mm = [(x * INCH_TO_MM, y * INCH_TO_MM) for x, y in projection]
            path_data = "M " + " L ".join([f"{x},{y}" for x, y in projection_mm]) + " Z"
            path = current_page.path(
                d=path_data,
                stroke="black",
                fill="none",
                stroke_width=0.5  # thin line for laser cutting
            )
            current_page.add(path)
            print(f"Created SVG for part {page_num}")
            
            # Save the SVG
            current_page.save(pretty=True)
            print(f"SVG file saved: {filename}")


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