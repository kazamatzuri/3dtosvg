import unittest
import numpy as np
from stl import mesh
import tempfile
import os
from pathlib import Path
import svgwrite
import xml.etree.ElementTree as ET

# Import from main script
from stl2svg import (
    Point2D, Point3D, Edge, Face, STLProcessor, 
    ForceLayout, SVGExporter, convert_stl_to_svg
)

class TestPoint2D(unittest.TestCase):
    def test_point2d_creation(self):
        p = Point2D(1.0, 2.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        
    def test_point2d_as_tuple(self):
        p = Point2D(1.0, 2.0)
        self.assertEqual(p.as_tuple(), (1.0, 2.0))

class TestPoint3D(unittest.TestCase):
    def test_point3d_creation(self):
        p = Point3D(1.0, 2.0, 3.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.z, 3.0)
        
    def test_point3d_as_tuple(self):
        p = Point3D(1.0, 2.0, 3.0)
        self.assertEqual(p.as_tuple(), (1.0, 2.0, 3.0))

class TestEdge(unittest.TestCase):
    def setUp(self):
        self.p1 = Point3D(0.0, 0.0, 0.0)
        self.p2 = Point3D(1.0, 0.0, 0.0)
        self.edge = Edge((self.p1, self.p2))
        
    def test_edge_creation(self):
        self.assertEqual(self.edge.vertices[0], self.p1)
        self.assertEqual(self.edge.vertices[1], self.p2)
        self.assertIsNone(self.edge.projected)
        self.assertIsNone(self.edge.label)
        
    def test_edge_hash(self):
        edge2 = Edge((self.p1, self.p2))
        self.assertEqual(hash(self.edge), hash(edge2))
        
    def test_edge_equality(self):
        edge2 = Edge((self.p1, self.p2))
        self.assertEqual(self.edge, edge2)

class TestSTLProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary STL file with a simple cube
        self.temp_dir = tempfile.mkdtemp()
        self.stl_path = os.path.join(self.temp_dir, "cube.stl")
        
        # Create a simple cube mesh
        vertices = np.array([
            # Front face
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            # Back face
            [[0, 0, 1], [1, 0, 1], [1, 1, 1]],
            [[0, 0, 1], [1, 1, 1], [0, 1, 1]],
            # Top face
            [[0, 1, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 1]],
            # Bottom face
            [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
            [[0, 0, 0], [1, 0, 1], [0, 0, 1]],
            # Left face
            [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
            # Right face
            [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[1, 0, 0], [1, 1, 1], [1, 0, 1]]
        ])
        
        cube = mesh.Mesh(np.zeros(12, dtype=mesh.Mesh.dtype))
        for i, face in enumerate(vertices):
            for j in range(3):
                cube.vectors[i][j] = face[j]
                
        cube.save(self.stl_path)
        
        self.processor = STLProcessor(self.stl_path)
        
    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
    def test_process(self):
        self.processor.process()
        self.assertEqual(len(self.processor.faces), 12)  # Cube has 12 triangular faces
        
        # Test edge creation
        edges_count = len(self.processor.edges)
        self.assertTrue(edges_count > 0)
        
        # Each edge should be shared by exactly 2 faces
        shared_edges = [edge for edge in self.processor.edges.values() if len(edge.faces) == 2]
        self.assertTrue(len(shared_edges) > 0)
        
    def test_identify_flat_faces(self):
        self.processor.process()
        flat_faces = self.processor.identify_flat_faces()
        
        # Cube should have 6 groups of coplanar faces (2 triangles per face)
        self.assertEqual(len(flat_faces), 6)
        
        # Each group should have 2 faces
        for group in flat_faces:
            self.assertEqual(len(group), 2)
            
    def test_project_faces(self):
        self.processor.process()
        flat_faces = self.processor.identify_flat_faces()
        
        # Test projection of first group
        self.processor.project_faces(flat_faces[0])
        
        # Check that all vertices were projected
        for face in flat_faces[0]:
            self.assertIsNotNone(face.projected)
            self.assertEqual(len(face.projected), 3)  # Triangle has 3 vertices
            
            # Check that projection is 2D
            for point in face.projected:
                self.assertIsInstance(point, Point2D)

class TestForceLayout(unittest.TestCase):
    def setUp(self):
        # Create two simple triangular faces
        v1 = Point3D(0, 0, 0)
        v2 = Point3D(1, 0, 0)
        v3 = Point3D(0, 1, 0)
        
        self.face1 = Face(
            vertices=[v1, v2, v3],
            normal=np.array([0, 0, 1]),
            edges=[],
            projected=[Point2D(0, 0), Point2D(1, 0), Point2D(0, 1)]
        )
        
        v4 = Point3D(2, 0, 0)
        v5 = Point3D(3, 0, 0)
        v6 = Point3D(2, 1, 0)
        
        self.face2 = Face(
            vertices=[v4, v5, v6],
            normal=np.array([0, 0, 1]),
            edges=[],
            projected=[Point2D(2, 0), Point2D(3, 0), Point2D(2, 1)]
        )
        
        self.layout = ForceLayout([self.face1, self.face2], min_distance=1.0, width=10.0, height=10.0)
        
    def test_get_bounding_box(self):
        min_pt, max_pt = self.layout.get_bounding_box(self.face1)
        self.assertEqual(min_pt.x, 0.0)
        self.assertEqual(min_pt.y, 0.0)
        self.assertEqual(max_pt.x, 1.0)
        self.assertEqual(max_pt.y, 1.0)
        
    def test_apply_forces(self):
        # Store initial positions
        initial_positions = [
            [v.x, v.y] for v in self.face1.projected + self.face2.projected
        ]
        
        # Apply forces
        self.layout.apply_forces()
        
        # Check that positions have changed
        final_positions = [
            [v.x, v.y] for v in self.face1.projected + self.face2.projected
        ]
        
        self.assertFalse(np.allclose(initial_positions, final_positions))
        
        # Check that positions are within bounds
        for face in [self.face1, self.face2]:
            for point in face.projected:
                self.assertGreaterEqual(point.x, 0)
                self.assertLessEqual(point.x, 10.0)
                self.assertGreaterEqual(point.y, 0)
                self.assertLessEqual(point.y, 10.0)

class TestSVGExporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.svg_path = os.path.join(self.temp_dir, "test.svg")
        self.exporter = SVGExporter(100.0, 100.0, self.svg_path)
        
        # Create a simple face with a labeled edge
        v1 = Point3D(0, 0, 0)
        v2 = Point3D(1, 0, 0)
        v3 = Point3D(0, 1, 0)
        
        edge = Edge((v1, v2), 
                   projected=(Point2D(10, 10), Point2D(20, 10)),
                   faces=[0, 1],
                   label=1)
        
        self.face = Face(
            vertices=[v1, v2, v3],
            normal=np.array([0, 0, 1]),
            edges=[edge],
            projected=[Point2D(10, 10), Point2D(20, 10), Point2D(10, 20)]
        )
        
    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
    def test_export_faces(self):
        self.exporter.export_faces([self.face], {self.face.edges[0]: self.face.edges[0]})
        
        # Check that SVG file was created
        self.assertTrue(os.path.exists(self.svg_path))
        
        # Parse SVG and check contents
        tree = ET.parse(self.svg_path)
        root = tree.getroot()
        
        # Check for polygon element
        polygons = root.findall(".//{http://www.w3.org/2000/svg}polygon")
        self.assertEqual(len(polygons), 1)
        
        # Check for text element (edge label)
        texts = root.findall(".//{http://www.w3.org/2000/svg}text")
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0].text, "1")

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.stl_path = os.path.join(self.temp_dir, "test.stl")
        self.svg_path = os.path.join(self.temp_dir, "test.svg")
        
        # Create a simple STL file
        vertices = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]]
        ])
        
        simple = mesh.Mesh(np.zeros(2, dtype=mesh.Mesh.dtype))
        for i, face in enumerate(vertices):
            for j in range(3):
                simple.vectors[i][j] = face[j]
                
        simple.save(self.stl_path)
        
    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
    def test_convert_stl_to_svg(self):
        convert_stl_to_svg(self.stl_path, self.svg_path, 100.0, 100.0, 1.0)
        
        # Check that SVG file was created
        self.assertTrue(os.path.exists(self.svg_path))
        
        # Basic validation of SVG contents
        tree = ET.parse(self.svg_path)
        root = tree.getroot()
        
        # Check for expected elements
        polygons = root.findall(".//{http://www.w3.org/2000/svg}polygon")
        self.assertGreater(len(polygons), 0)

if __name__ == '__main__':
    unittest.main()