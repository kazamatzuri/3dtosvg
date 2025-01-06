"""Convert STL files to SVG with matched edge labeling."""

from stl2svg.core import convert_stl_to_svg
from stl2svg.processor import STLProcessor, Point2D, Point3D, Edge, Face
from stl2svg.layout import ForceLayout
from stl2svg.export import SVGExporter

__version__ = "0.1.0"
__all__ = [
    "convert_stl_to_svg",
    "STLProcessor",
    "ForceLayout",
    "SVGExporter",
    "Point2D",
    "Point3D",
    "Edge",
    "Face",
]