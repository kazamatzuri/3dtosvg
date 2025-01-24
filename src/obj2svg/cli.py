
import os
from .converter import OBJToSVG
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert OBJ files to SVG with optional edge labels.")
    parser.add_argument("input_obj", help="Path to the input OBJ file.")
    parser.add_argument("svg_width_inches", type=float, help="Width of the SVG in inches.")
    parser.add_argument("svg_height_inches", type=float, help="Height of the SVG in inches.")
    parser.add_argument("min_distance_inches", type=float, help="Minimum distance between shapes in inches.")
    parser.add_argument("--edge-labels", action="store_true", help="Enable edge labels in the SVG output.")
    
    args = parser.parse_args()
    
    input_file = args.input_obj
    svg_width_inches = args.svg_width_inches
    svg_height_inches = args.svg_height_inches
    min_distance_inches = args.min_distance_inches
    edge_labels = args.edge_labels
    output_prefix = os.path.splitext(os.path.basename(input_file))[0]

    obj_to_svg = OBJToSVG(
        obj_file=input_file,
        svg_size_inches=(svg_width_inches, svg_height_inches),
        min_distance_inches=min_distance_inches,
        output_prefix=output_prefix,
        edge_labels=edge_labels,
    )
    obj_to_svg.run()

if __name__ == "__main__":
    main() 