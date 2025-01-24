import unittest
import os
import subprocess
import shutil
from typing import List, Tuple
from bs4 import BeautifulSoup, Tag


class TestOBJToSVGConverter(unittest.TestCase):
    """Unit tests for the OBJToSVG converter."""

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test environment by copying the triangles.obj file
        and running the converter to generate the SVG.
        """
        cls.test_dir: str = os.path.dirname(os.path.abspath(__file__))
        cls.project_root: str = os.path.abspath(os.path.join(cls.test_dir, '..'))
        cls.obj_file: str = os.path.join(cls.project_root, 'examples', 'triangles.obj')
        cls.output_dir: str = os.path.join(cls.project_root, 'triangles')
        cls.svg_file: str = os.path.join(cls.output_dir, 'sheet1.svg')
        
        # Ensure the output directory is clean
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Run the converter
        try:
            subprocess.run([
                'obj2svg',
                cls.obj_file,
                '8',   # svg_width_inches
                '8',   # svg_height_inches
                '0.5', # min_distance_inches
            ], check=True)
        except subprocess.CalledProcessError as e:
            cls.tearDownClass()
            raise RuntimeError(f"Converter failed: {e}") from e

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Clean up the test environment by removing the output directory.
        """
        # if os.path.exists(cls.output_dir):
        #     shutil.rmtree(cls.output_dir)

    def test_number_of_polygons(self) -> None:
        """
        Test that the SVG contains exactly 3 polygons.
        """
        self.assertTrue(os.path.exists(self.svg_file), f"SVG file {self.svg_file} does not exist.")
        
        with open(self.svg_file, 'r') as file:
            soup: BeautifulSoup = BeautifulSoup(file, 'xml')
        
        polygons: List[Tag] = soup.find_all('path')
        self.assertEqual(len(polygons), 3, "SVG should contain exactly 3 polygons.")

    def test_equilateral_triangles(self) -> None:
        """
        Test that each polygon in the SVG is an equilateral triangle.
        This is done by checking if all sides are approximately equal.
        """
        with open(self.svg_file, 'r') as file:
            soup: BeautifulSoup = BeautifulSoup(file, 'xml')
        
        polygons: List[Tag] = soup.find_all('path')
        tolerance: float = 5.0  # tolerance in millimeters
        
        for idx, poly in enumerate(polygons, start=1):
            d_attr: str = poly.get('d', '')
            self.assertTrue(d_attr, f"Polygon {idx} has no 'd' attribute.")
            
            # Parse the 'd' attribute to extract points
            commands: List[str] = d_attr.strip().split()
            points: List[Tuple[float, float]] = []
            for cmd in commands:
                if ',' in cmd:
                    try:
                        x_str, y_str = cmd.split(',')
                        x: float = float(x_str)
                        y: float = float(y_str)
                        points.append((x, y))
                    except ValueError:
                        self.fail(f"Invalid coordinate format in polygon {idx}: '{cmd}'")
            
            self.assertEqual(len(points), 3, f"Polygon {idx} should have 3 points (excluding 'Z').")
            
            # Calculate side lengths
            side_lengths: List[float] = []
            for i in range(3):
                p1: Tuple[float, float] = points[i]
                p2: Tuple[float, float] = points[(i + 1) % 3]
                length: float = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5
                side_lengths.append(length)
            
            # Check if all sides are approximately equal
            for i in range(1, len(side_lengths)):
                self.assertAlmostEqual(
                    side_lengths[i],
                    side_lengths[0],
                    delta=tolerance,
                    msg=f"Side {i} of polygon {idx} is not approximately equal to side 0."
                )

    def test_shared_edge_labels(self) -> None:
        """
        Test that shared edges have the same label across different polygons.
        """
        with open(self.svg_file, 'r') as file:
            soup: BeautifulSoup = BeautifulSoup(file, 'xml')
        
        labels: List[Tag] = soup.find_all('text')
        edge_labels: dict = {}
        
        for label in labels:
            try:
                x: float = float(label.get('x', '0'))
                y: float = float(label.get('y', '0'))
                label_num: str = label.text.strip()
                edge_labels[(x, y)] = label_num
            except (ValueError, TypeError):
                self.fail(f"Invalid label format: {label}")
        
        # Since each edge should be shared by 2 polygons, ensure each label appears exactly twice
        label_counts: dict = {}
        for label_num in edge_labels.values():
            label_counts[label_num] = label_counts.get(label_num, 0) + 1
        
        for label_num, count in label_counts.items():
            self.assertEqual(count, 2, f"Label {label_num} should appear exactly twice for shared edges.")

    def test_labels_inside_polygons(self) -> None:
        """
        Test that all labels are positioned inside their respective polygons.
        """
        with open(self.svg_file, 'r') as file:
            soup: BeautifulSoup = BeautifulSoup(file, 'xml')
        
        polygons: List[Tag] = soup.find_all('path')
        labels: List[Tag] = soup.find_all('text')
        tolerance: float = 5.0  # tolerance in millimeters
        
        for poly, label in zip(polygons, labels):
            d_attr: str = poly.get('d', '')
            self.assertTrue(d_attr, "Polygon has no 'd' attribute.")
            
            # Parse polygon points
            commands: List[str] = d_attr.strip().split()
            points: List[Tuple[float, float]] = []
            for cmd in commands:
                if ',' in cmd:
                    try:
                        x_str, y_str = cmd.split(',')
                        x: float = float(x_str)
                        y: float = float(y_str)
                        points.append((x, y))
                    except ValueError:
                        self.fail(f"Invalid coordinate format in polygon: '{cmd}'")
            
            if not points:
                self.fail("No points found in polygon for label placement test.")
            
            # Centroid
            centroid_x: float = sum(p[0] for p in points) / len(points)
            centroid_y: float = sum(p[1] for p in points) / len(points)
            
            # Label position
            try:
                label_x: float = float(label.get('x', '0'))
                label_y: float = float(label.get('y', '0'))
            except (ValueError, TypeError):
                self.fail(f"Invalid label coordinates: '{label}'")
            
            # Distance from centroid
            distance: float = ((label_x - centroid_x)**2 + (label_y - centroid_y)**2) ** 0.5
            max_distance: float = max(
                ((p[0] - centroid_x)**2 + (p[1] - centroid_y)**2) ** 0.5 for p in points
            )
            
            self.assertLessEqual(
                distance,
                max_distance - tolerance,
                f"Label at ({label_x}, {label_y}) is not inside the polygon."
            )


if __name__ == '__main__':
    unittest.main() 