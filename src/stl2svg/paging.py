"""Page management for multi-page SVG export."""

from typing import List, Dict, Tuple
import numpy as np
from .processor import Face, Point2D

class PageManager:
    """Manage distribution of faces across multiple pages."""
    
    def __init__(self, width: float, height: float):
        """Initialize the page manager.
        
        Args:
            width: Page width
            height: Page height
        """
        self.width = width
        self.height = height
        
    def get_face_bounds(self, face: Face) -> Tuple[float, float, float, float]:
        """Get the bounding box of a face.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        xs = [v.x for v in face.projected]
        ys = [v.y for v in face.projected]
        return min(xs), min(ys), max(xs), max(ys)
        
    def face_fits_on_page(self, face: Face, existing_faces: List[Face], 
                         min_distance: float) -> bool:
        """Check if a face fits on the page with existing faces."""
        if not face.projected:
            return False
            
        # Get face bounds
        min_x, min_y, max_x, max_y = self.get_face_bounds(face)
        
        # Check if face is within page bounds
        if (max_x > self.width or max_y > self.height or 
            min_x < 0 or min_y < 0):
            return False
            
        # Check distance to all existing faces
        for other_face in existing_faces:
            other_min_x, other_min_y, other_max_x, other_max_y = self.get_face_bounds(other_face)
            
            # Calculate minimum distance between faces
            dx = max(0, min(abs(min_x - other_max_x), abs(max_x - other_min_x)))
            dy = max(0, min(abs(min_y - other_max_y), abs(max_y - other_min_y)))
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                return False
                
        return True
        
    def distribute_faces(self, faces: List[Face], min_distance: float) -> Dict[int, List[Face]]:
        """Distribute faces across pages using first-fit approach.
        
        Args:
            faces: List of faces to distribute
            min_distance: Minimum distance between faces
            
        Returns:
            Dictionary mapping page numbers to lists of faces
        """
        pages: Dict[int, List[Face]] = {}
        current_page = 1
        pages[current_page] = []
        
        # Sort faces by area (larger faces first)
        sorted_faces = sorted(faces, reverse=True)
        
        # Place each face
        for face in sorted_faces:
            placed = False
            current_page = 1
            
            # Try to fit on existing pages
            while current_page <= len(pages):
                if self.face_fits_on_page(face, pages[current_page], min_distance):
                    pages[current_page].append(face)
                    placed = True
                    break
                current_page += 1
                
            # If face doesn't fit on any existing page, create new page
            if not placed:
                pages[current_page] = [face]
                
        return pages