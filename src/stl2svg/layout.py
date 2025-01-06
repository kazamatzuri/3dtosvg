"""2D layout optimization using force-directed placement."""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple
from .processor import Face, Point2D

class ForceLayout:
    """Force-directed layout for 2D face placement."""
    
    def __init__(self, faces: List[Face], min_distance: float, width: float, height: float):
        """Initialize the force layout.
        
        Args:
            faces: List of faces to arrange
            min_distance: Minimum distance between faces
            width: Available width
            height: Available height
        """
        self.faces = faces
        self.min_distance = min_distance
        self.width = width
        self.height = height
        self.iterations = 100
        self.learning_rate = 0.1
        
    def get_face_center(self, face: Face) -> np.ndarray:
        """Calculate the center of a face."""
        if not face.projected:
            return np.array([0., 0.])
        return np.mean([[v.x, v.y] for v in face.projected], axis=0)
    
    def get_face_size(self, face: Face) -> float:
        """Calculate the characteristic size of a face (max dimension)."""
        if not face.projected:
            return 0.0
        xs = [v.x for v in face.projected]
        ys = [v.y for v in face.projected]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return max(width, height)
    
    def move_face(self, face: Face, offset: np.ndarray) -> None:
        """Move an entire face by an offset."""
        if not face.projected:
            return
            
        # Move all vertices
        for vertex in face.projected:
            vertex.x += offset[0]
            vertex.y += offset[1]
            
        # Move edge projections
        for edge in face.edges:
            if edge.projected:
                edge.projected = (
                    Point2D(edge.projected[0].x + offset[0], edge.projected[0].y + offset[1]),
                    Point2D(edge.projected[1].x + offset[0], edge.projected[1].y + offset[1])
                )
    
    def apply_forces(self) -> None:
        """Apply force-directed layout to position faces."""
        # Initialize random positions scaling by face sizes
        for face in self.faces:
            if not face.projected:
                continue
            size = self.get_face_size(face)
            # Random position that ensures face is fully within bounds
            max_x = self.width - size
            max_y = self.height - size
            if max_x > 0 and max_y > 0:  # Only move if face can fit
                offset = np.array([
                    np.random.uniform(size, max_x),
                    np.random.uniform(size, max_y)
                ]) - self.get_face_center(face)
                self.move_face(face, offset)
        
        for iteration in range(self.iterations):
            # Calculate face centers
            centers = []
            valid_faces = []
            for face in self.faces:
                center = self.get_face_center(face)
                if np.all(np.isfinite(center)):
                    centers.append(center)
                    valid_faces.append(face)
            
            if not centers:
                return
                
            centers = np.array(centers)
            forces = np.zeros_like(centers)
            
            # Calculate repulsive forces between face centers
            tree = cKDTree(centers)
            
            for i, center in enumerate(centers):
                # Get neighbors within potential interaction range
                # Use larger radius to account for face sizes
                face_size = self.get_face_size(valid_faces[i])
                search_radius = self.min_distance * 2 + face_size
                neighbors = tree.query_ball_point(center, search_radius)
                
                for j in neighbors:
                    if i != j:
                        other_size = self.get_face_size(valid_faces[j])
                        
                        # Vector from other face to this face
                        diff = centers[i] - centers[j]
                        distance = np.linalg.norm(diff)
                        
                        # Minimum required distance is sum of face sizes plus min_distance
                        required_distance = (face_size + other_size) / 2 + self.min_distance
                        
                        if distance < required_distance:
                            # Normalize direction and scale force by overlap
                            if distance < 1e-6:  # Prevent division by zero
                                direction = np.random.rand(2) * 2 - 1  # Random direction
                                direction /= np.linalg.norm(direction)
                            else:
                                direction = diff / distance
                            
                            force = direction * (required_distance - distance)
                            forces[i] += force
                            forces[j] -= force
            
            # Apply forces with damping over time
            damping = 1 - (iteration / self.iterations)
            learning_rate = self.learning_rate * damping
            
            # Move faces
            for i, face in enumerate(valid_faces):
                # Calculate new position
                offset = forces[i] * learning_rate
                
                # Ensure face stays in bounds
                center = self.get_face_center(face)
                size = self.get_face_size(face)
                
                # Adjust offset to keep face in bounds
                new_center = center + offset
                new_center[0] = np.clip(new_center[0], size/2, self.width - size/2)
                new_center[1] = np.clip(new_center[1], size/2, self.height - size/2)
                
                # Apply the adjusted movement
                final_offset = new_center - center
                self.move_face(face, final_offset)