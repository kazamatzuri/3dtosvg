"""Core functionality and CLI interface."""

import argparse
from typing import List
from .processor import Face, STLProcessor
from .layout import ForceLayout
from .export import SVGExporter
from .paging import PageManager

def convert_stl_to_svg(
    stl_file: str,
    svg_file: str,
    width_inches: float,
    height_inches: float,
    min_distance_inches: float,
    single_page: bool = False,
    dpi: float = 96.0
) -> None:
    """Convert STL file to SVG with labeled edges.
    
    Args:
        stl_file: Path to input STL file
        svg_file: Path to output SVG file
        width_inches: SVG canvas width in inches
        height_inches: SVG canvas height in inches
        min_distance_inches: Minimum distance between faces in inches
        single_page: Force output to a single page
        dpi: Dots per inch for conversion (default: 96.0)
    """
    # Convert inches to pixels
    width_px = width_inches * dpi
    height_px = height_inches * dpi
    min_distance_px = min_distance_inches * dpi
    
    # Process STL
    processor = STLProcessor(stl_file)
    processor.process()
    
    # Get flat faces
    flat_faces = processor.identify_flat_faces()
    
    # Project flat faces
    all_processed_faces = []
    for face in flat_faces:
        processor.project_faces([face])
        all_processed_faces.append(face)
    
    # Apply force layout
    layout = ForceLayout(all_processed_faces, min_distance_px, width_px, height_px)
    layout.apply_forces()
    
    # Create SVG(s)
    exporter = SVGExporter(width_inches, height_inches, svg_file, dpi)
    
    if single_page:
        # Export to single page
        exporter.export_faces(all_processed_faces, processor.edges)
    else:
        # Use page manager to distribute faces
        page_manager = PageManager(width_px, height_px)
        pages = page_manager.distribute_faces(all_processed_faces, min_distance_px)
        exporter.export_faces_to_pages(pages, processor.edges)

def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Convert STL files to SVG with matched edge labeling'
    )
    parser.add_argument('stl_file', help='Input STL file path')
    parser.add_argument('svg_file', help='Output SVG file path')
    parser.add_argument(
        '--width',
        type=float,
        default=11.0,
        help='SVG canvas width in inches (default: 11.0 inches - US Letter)'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=8.5,
        help='SVG canvas height in inches (default: 8.5 inches - US Letter)'
    )
    parser.add_argument(
        '--min-distance',
        type=float,
        default=0.25,
        help='Minimum distance between faces in inches (default: 0.25 inches)'
    )
    parser.add_argument(
        '--single-page',
        action='store_true',
        help='Force output to a single page'
    )
    parser.add_argument(
        '--dpi',
        type=float,
        default=96.0,
        help='Dots per inch for rendering (default: 96.0)'
    )
    
    args = parser.parse_args()
    
    convert_stl_to_svg(
        args.stl_file,
        args.svg_file,
        args.width,
        args.height,
        args.min_distance,
        args.single_page,
        args.dpi
    )

if __name__ == '__main__':
    main()