"""STL file processing and geometry handling."""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from stl import mesh

@dataclass
class Point2D:
    x: float
    y: float

    def as_tuple(self) -> tuple:
        return (self.x, self.y)

@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.z)

@dataclass
class Edge:
    vertices: Tuple[Point3D, Point3D]
    projected: Optional[Tuple[Point2D, Point2D]] = None
    faces: List[int] = None
    label: Optional[int] = None
    label_position: Optional[Point2D] = None

    def __hash__(self):
        return hash((self.vertices[0].as_tuple(), self.vertices[1].as_tuple()))

    def __eq__(self, other):
        return hash(self) == hash(other)

@dataclass
class Face:
    vertices: List[Point3D]
    normal: np.ndarray
    edges: List[Edge]
    projected: Optional[List[Point2D]] = None
    
    def get_area(self) -> float:
        """Calculate the area of the projected face."""
        if not self.projected:
            return 0.0
        
        # Get bounding box area
        xs = [v.x for v in self.projected]
        ys = [v.y for v in self.projected]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return width * height
        
    def __lt__(self, other: 'Face') -> bool:
        """Compare faces by their projected areas."""
        return self.get_area() < other.get_area()
        
    def __eq__(self, other: object) -> bool:
        """Compare faces for equality."""
        if not isinstance(other, Face):
            return NotImplemented
        return (self.vertices == other.vertices and 
                np.array_equal(self.normal, other.normal) and
                self.edges == other.edges and
                self.projected == other.projected)

class STLProcessor:
    """Process STL files and extract flat faces with edge matching."""
    
    def __init__(self, filename: str, epsilon: float = 1e-6):
        """Initialize the STL processor.
        
        Args:
            filename: Path to the STL file
            epsilon: Tolerance for floating point comparisons
        """
        self.mesh = mesh.Mesh.from_file(filename)
        self.epsilon = epsilon
        self.faces: List[Face] = []
        self.edges: Dict[Edge, Edge] = {}
        
    def process(self) -> None:
        """Process the STL file to extract flat faces and edges."""
        # Extract faces and build edge dictionary
        for i, (vertices, normal) in enumerate(zip(self.mesh.vectors, self.mesh.normals)):
            face_vertices = [Point3D(*v) for v in vertices]
            face_edges = []
            
            # Create edges for this face
            for j in range(3):
                v1, v2 = face_vertices[j], face_vertices[(j + 1) % 3]
                # Ensure consistent vertex order
                if hash((v1.as_tuple(), v2.as_tuple())) > hash((v2.as_tuple(), v1.as_tuple())):
                    v1, v2 = v2, v1
                
                edge = Edge((v1, v2), faces=[i])
                if edge in self.edges:
                    self.edges[edge].faces.append(i)
                else:
                    self.edges[edge] = edge
                face_edges.append(edge)
            
            face = Face(face_vertices, normal, face_edges)
            self.faces.append(face)
    
    def merge_coplanar_faces(self, faces: List[Face]) -> Face:
        """Merge coplanar triangular faces into a single face."""
        # Build set of all vertices
        vertices_dict = {}  # Map vertex tuple to Point3D object
        edge_dict = {}     # Map vertex pair tuples to Edge object
        
        for face in faces:
            # Add vertices
            for vertex in face.vertices:
                vertices_dict[vertex.as_tuple()] = vertex
            
            # Add edges
            for edge in face.edges:
                v1, v2 = edge.vertices
                key = tuple(sorted([v1.as_tuple(), v2.as_tuple()]))
                if key not in edge_dict:
                    edge_dict[key] = edge

        # Function to find next vertex in boundary
        def find_next_vertex(current, used_edges):
            for edge_key, edge in edge_dict.items():
                if edge_key in used_edges:
                    continue
                v1_tuple, v2_tuple = edge_key
                if v1_tuple == current:
                    return v2_tuple, edge_key
                if v2_tuple == current:
                    return v1_tuple, edge_key
            return None, None

        # Find starting vertex (leftmost vertex)
        start_vertex = min(vertices_dict.keys())
        current_vertex = start_vertex
        ordered_vertices = [vertices_dict[start_vertex]]
        used_edges = set()
        boundary_edges = []

        # Walk around boundary
        while True:
            next_vertex, edge_key = find_next_vertex(current_vertex, used_edges)
            if next_vertex is None or len(ordered_vertices) > len(vertices_dict):
                break
                
            if edge_key:
                used_edges.add(edge_key)
                boundary_edges.append(edge_dict[edge_key])
                
            if next_vertex == start_vertex:
                break
                
            ordered_vertices.append(vertices_dict[next_vertex])
            current_vertex = next_vertex

        # Use normal from first face
        normal = faces[0].normal
        
        # Create new face with ordered vertices and edges
        result = Face(vertices=ordered_vertices, normal=normal, edges=boundary_edges)
        
        # Debug output
        print(f"Merged face has {len(result.vertices)} vertices and {len(result.edges)} edges")
        return result

    def identify_flat_faces(self) -> List[Face]:
        """Return merged flat faces."""
        flat_face_groups = []
        processed = set()

        print(f"\nTotal faces in mesh: {len(self.faces)}")
        print("\nNormals of all faces:")
        for i, face in enumerate(self.faces):
            print(f"Face {i}: normal = {face.normal}")
        
        # First, identify all groups of parallel faces
        parallel_groups = []
        processed_parallels = set()
        
        for i, face in enumerate(self.faces):
            if i in processed_parallels:
                continue
                
            current_group = []
            normal = face.normal
            
            # Find all faces with same normal (parallel)
            for j, other in enumerate(self.faces):
                if j in processed_parallels:
                    continue
                    
                # Check if normals are parallel (same or opposite direction)
                normals_parallel = np.allclose(normal, other.normal, atol=self.epsilon) or \
                                 np.allclose(normal, -other.normal, atol=self.epsilon)
                
                if normals_parallel:
                    current_group.append(j)
                    processed_parallels.add(j)
            
            if current_group:
                parallel_groups.append(current_group)

        print(f"\nFound {len(parallel_groups)} groups of parallel faces:")
        for i, group in enumerate(parallel_groups):
            print(f"Group {i}: faces {group}")
        
        # Now process each parallel group to find coplanar faces
        for group_idx, group_indices in enumerate(parallel_groups):
            print(f"\nProcessing parallel group {group_idx}")
            group_faces = [self.faces[i] for i in group_indices]
            normal = group_faces[0].normal
            
            # Sort faces into coplanar sets
            coplanar_sets = []
            used_faces = set()
            
            for i, face1 in enumerate(group_faces):
                if i in used_faces:
                    continue
                    
                current_set = {i}
                point1 = np.array(face1.vertices[0].as_tuple())
                print(f"  Checking face {i} in group")
                
                for j, face2 in enumerate(group_faces):
                    if j in used_faces or j == i:
                        continue
                        
                    point2 = np.array(face2.vertices[0].as_tuple())
                    # Distance from point2 to plane of face1
                    distance = abs(np.dot(normal, point2 - point1))
                    
                    print(f"    Comparing with face {j}, distance = {distance}")
                    if np.allclose(distance, 0, atol=self.epsilon):
                        current_set.add(j)
                        print(f"    Added face {j} to current set")
                
                if current_set:
                    print(f"  Found coplanar set: {current_set}")
                    used_faces.update(current_set)
                    coplanar_faces = [group_faces[i] for i in current_set]
                    if len(coplanar_faces) >= 2:
                        print(f"  Merging {len(coplanar_faces)} faces")
                        merged_face = self.merge_coplanar_faces(coplanar_faces)
                        flat_face_groups.append(merged_face)
                    else:
                        print(f"  Adding single face")
                        flat_face_groups.extend(coplanar_faces)

        print(f"\nFinal number of merged faces: {len(flat_face_groups)}")
        print("Merged face details:")
        for i, face in enumerate(flat_face_groups):
            print(f"Face {i}: {len(face.vertices)} vertices, {len(face.edges)} edges")
            if face.projected:
                print(f"  Projected coordinates: {[(v.x, v.y) for v in face.projected]}")
        return flat_face_groups

    def project_faces(self, faces: List[Face]) -> None:
        """Project faces onto their best-fit 2D plane."""
        if not faces:
            return
            
        # Use first face's normal to define projection plane
        normal = faces[0].normal
        
        # Create projection matrix
        if not np.allclose(normal, [0, 0, 1]):
            # Create orthonormal basis with normal as z-axis
            z_axis = normal / np.linalg.norm(normal)  # Ensure normal is normalized
            
            # Find least aligned axis with normal for x_axis
            alignments = np.abs([np.dot(z_axis, [1, 0, 0]),
                               np.dot(z_axis, [0, 1, 0]),
                               np.dot(z_axis, [0, 0, 1])])
            temp_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]][np.argmin(alignments)])
            
            # Create orthogonal vectors
            y_axis = np.cross(z_axis, temp_axis)
            y_norm = np.linalg.norm(y_axis)
            
            if y_norm < 1e-10:  # If vectors are too closely aligned, try another axis
                temp_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]][(np.argmin(alignments) + 1) % 3])
                y_axis = np.cross(z_axis, temp_axis)
                y_norm = np.linalg.norm(y_axis)
            
            y_axis = y_axis / y_norm
            x_axis = np.cross(y_axis, z_axis)
            x_norm = np.linalg.norm(x_axis)
            
            if x_norm < 1e-10:  # This should never happen if y_axis is properly normalized
                # Fallback to default projection
                projection = np.array([[1, 0, 0], [0, 1, 0]])
            else:
                x_axis = x_axis / x_norm
                projection = np.vstack([x_axis, y_axis])
        else:
            projection = np.array([[1, 0, 0], [0, 1, 0]])
            
        # Project all vertices
        for face in faces:
            vertices = np.array([v.as_tuple() for v in face.vertices])
            projected = vertices @ projection.T
            face.projected = [Point2D(x, y) for x, y in projected]
            
            # Project edges
            for edge in face.edges:
                if edge.projected is None:
                    vertices = np.array([v.as_tuple() for v in edge.vertices])
                    projected = vertices @ projection.T
                    edge.projected = (Point2D(projected[0][0], projected[0][1]),
                                    Point2D(projected[1][0], projected[1][1]))