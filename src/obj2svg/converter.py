import os
import math
from typing import List, Tuple, Dict, Any
import numpy as np
import svgwrite
from freetype import Face
from fontTools.ttLib import TTFont

from .geometry import get_polygon_bounds
from .text import create_number_path

# Conversion factor
INCH_TO_MM = 25.4

class OBJToSVG:
    def __init__(self, obj_file: str, svg_size_inches: Tuple[float, float], min_distance_inches: float, output_prefix: str, edge_labels: bool = False):
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
        self.input_is_meters = True
        self.edge_labels = {}
        self.next_label = 1
        
        self.edge_labels_enabled = edge_labels
        
        # Load the font file
        self.font = TTFont('font/fawesome.otf')
        self.glyph_set = self.font.getGlyphSet()
        self.face = Face('font/fawesome.otf')
        
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
        edge_map = {}
        for face_idx, face in enumerate(self.faces):
            face_vertices = np.array([self.vertices[i] for i in face])

            if self.input_is_meters:
                face_vertices = face_vertices * 1000

            print(f"\nFace {face_idx + 1} dimensions (mm):")
            # Calculate bounds and print dimensions
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

            # Calculate normal and project onto appropriate plane
            v1, v2 = face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

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
        """Arranges shapes in an expanding diamond pattern."""
        shapes = [proj[0] for proj in self.projections]
        bounds = [get_polygon_bounds(shape) for shape in shapes]
        
        max_width = max(max_x - min_x for min_x, _, max_x, _ in bounds)
        max_height = max(max_y - min_y for _, min_y, _, max_y in bounds)
        cell_size = max(max_width, max_height) + self.min_distance * 2
        
        grid_size = math.ceil(math.sqrt(len(shapes))) * 2
        grid_width = grid_size * cell_size
        grid_height = grid_size * cell_size
        
        center_x = grid_width / 2
        center_y = grid_height / 2
        
        placed_shapes = []
        x, y = 0, 0
        dx, dy = 0, -1
        
        for i, shape in enumerate(shapes):
            shape_x = center_x + x * cell_size
            shape_y = center_y + y * cell_size
            
            min_x, min_y, max_x, max_y = bounds[i]
            offset_x = shape_x - (min_x + max_x) / 2
            offset_y = shape_y - (min_y + max_y) / 2
            
            translated_shape = [(x + offset_x, y + offset_y) for x, y in shape]
            placed_shapes.append(translated_shape)
            
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy
        
        return [placed_shapes]

    def export_to_svg(self) -> None:
        """Exports shapes to a single SVG file."""
        pages = self.arrange_layout()
        placed_shapes = pages[0]
        
        output_dir = self.output_prefix
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "sheet1.svg")

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

        for (projection, edges), placed_projection in zip(self.projections, placed_shapes):
            group = current_page.g()
            
            path_data = "M " + " L ".join([f"{x},{y}" for x, y in placed_projection]) + " Z"
            path = current_page.path(d=path_data, stroke="black", fill="none", stroke_width=0.5)
            group.add(path)
            
            if self.edge_labels_enabled:
                for (p1, p2), (mid_x, mid_y), label in edges:
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
                    
                    path_data, transforms = create_number_path(
                        self.face,
                        str(label),
                        placed_mid_x,
                        placed_mid_y,
                        scale=0.05
                    )
                    
                    if path_data:
                        number_group = current_page.g()
                        path = current_page.path(
                            d=path_data,
                            stroke="black",
                            fill="black",
                            stroke_width=0.5
                        )
                        number_group.add(path)
                        
                        if transforms:
                            number_group.attribs['transform'] = transforms
                        
                        group.add(number_group)
            
            current_page.add(group)

        current_page.save(pretty=True)

    def run(self) -> None:
        """Main method to run all steps."""
        print(f"Processing OBJ file: {self.obj_file}")
        self.parse_obj()
        self.project_faces()
        self.export_to_svg() 