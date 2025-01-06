"""SVG export functionality."""

import svgwrite
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from .processor import Face, Edge, Point2D

class SVGExporter:
    """Export faces and labeled edges to SVG format."""
    
    def __init__(self, width_inches: float, height_inches: float, output_file: str, dpi: float = 96.0):
        """Initialize the SVG exporter.
        
        Args:
            width_inches: SVG canvas width in inches
            height_inches: SVG canvas height in inches
            output_file: Path to output SVG file
            dpi: Dots per inch for conversion (default: 96.0)
        """
        self.width_inches = width_inches
        self.height_inches = height_inches
        self.dpi = dpi
        self.width_px = width_inches * dpi
        self.height_px = height_inches * dpi
        self.output_file = output_file
        self.dwg = svgwrite.Drawing(
            output_file,
            size=(f"{width_inches}in", f"{height_inches}in"),
            viewBox=f"0 0 {self.width_px} {self.height_px}"
        )
        
        # Add viewBox and sizing attributes to maintain scale
        self.dwg.attribs['preserveAspectRatio'] = 'xMidYMid meet'
    
    def get_page_filename(self, page_number: int) -> str:
        """Get filename for a specific page."""
        if page_number == 1:
            return self.output_file
            
        # Insert page number before extension
        path = Path(self.output_file)
        return str(path.parent / f"{path.stem}_page_{page_number}{path.suffix}")
        
    def calculate_label_position(self, edge_start: Point2D, edge_end: Point2D, face_center: Point2D) -> tuple:
        """Calculate optimal label position relative to an edge.
        
        Args:
            edge_start: Start point of edge
            edge_end: End point of edge
            face_center: Center point of the face
            
        Returns:
            Tuple of (x, y) coordinates for label placement
        """
        # Calculate edge midpoint
        mid_x = (edge_start.x + edge_end.x) / 2
        mid_y = (edge_start.y + edge_end.y) / 2
        
        # Calculate edge vector
        edge_vec = np.array([edge_end.x - edge_start.x, edge_end.y - edge_start.y])
        edge_length = np.linalg.norm(edge_vec)
        if edge_length < 1e-6:
            return (mid_x, mid_y)
            
        # Calculate normal vector to edge (both directions)
        normal1 = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
        normal2 = -normal1
        
        # Calculate center-to-midpoint vector
        to_center = np.array([face_center.x - mid_x, face_center.y - mid_y])
        
        # Choose normal vector that points more towards face center
        normal = normal1 if np.dot(normal1, to_center) > 0 else normal2
        
        # Position label slightly inside the edge
        offset = min(edge_length * 0.2, 10)  # 20% of edge length or 10 units, whichever is smaller
        label_x = mid_x + normal[0] * offset
        label_y = mid_y + normal[1] * offset
        
        return (label_x, label_y)

    def export_faces_to_pages(self, pages: Dict[int, List[Face]], 
                            edges: Dict[Edge, Edge]) -> None:
        """Export faces and labeled edges to multiple SVG files.
        
        Args:
            pages: Dictionary mapping page numbers to lists of faces
            edges: Dictionary of edges with labels
        """
        # Assign labels to shared edges
        label_counter = 1
        for edge in edges.values():
            if len(edge.faces) > 1 and edge.label is None:
                edge.label = label_counter
                label_counter += 1
                
        # Export each page
        for page_number, faces in pages.items():
            filename = self.get_page_filename(page_number)
            self.dwg = svgwrite.Drawing(
                filename,
                size=(f"{self.width_inches}in", f"{self.height_inches}in"),
                viewBox=f"0 0 {self.width_px} {self.height_px}"
            )
            self.dwg.attribs['preserveAspectRatio'] = 'xMidYMid meet'
            
            # Add page number if not first page
            if page_number > 1:
                self.dwg.add(self.dwg.text(
                    f"Page {page_number}",
                    insert=(10, 20),
                    font_size=12 * (self.dpi/96.0),
                    fill='gray'
                ))
            
            self.export_faces(faces, edges)
            
    def export_faces(self, faces: List[Face], edges: Dict[Edge, Edge]) -> None:
        """Export faces and labeled edges to a single SVG file."""
        # Draw faces
        for face in faces:
            if not face.projected:
                continue
                
            # Calculate face center for label positioning
            center_x = sum(v.x for v in face.projected) / len(face.projected)
            center_y = sum(v.y for v in face.projected) / len(face.projected)
            face_center = Point2D(center_x, center_y)
            
            # Draw face outline
            points = [(v.x, v.y) for v in face.projected]
            self.dwg.add(self.dwg.polygon(
                points,
                fill='none',
                stroke='black',
                stroke_width=self.dpi/96.0  # Scale stroke width with DPI
            ))
            
            # Add labels for shared edges
            for edge in face.edges:
                if edge.label is not None and edge.projected:
                    # Calculate optimal label position
                    label_pos = self.calculate_label_position(
                        edge.projected[0],
                        edge.projected[1],
                        face_center
                    )
                    
                    # Scale font and circle size with DPI
                    font_size = 10 * (self.dpi/96.0)
                    circle_radius = 8 * (self.dpi/96.0)
                    
                    # Add background circle
                    self.dwg.add(self.dwg.circle(
                        center=label_pos,
                        r=circle_radius,
                        fill='white',
                        stroke='none'
                    ))
                    # Add text
                    self.dwg.add(self.dwg.text(
                        str(edge.label),
                        insert=label_pos,
                        font_size=font_size,
                        text_anchor='middle',
                        dominant_baseline='middle'
                    ))
        
        self.dwg.save()