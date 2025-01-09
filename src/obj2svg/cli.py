import sys
import os
from .converter import OBJToSVG

def main():
    if len(sys.argv) != 5:
        print(
            "Usage: obj2svg <input.obj> <svg_width_inches> <svg_height_inches> <min_distance_inches>"
        )
        print("Example: obj2svg input.obj 8 8 0.5")
        sys.exit(1)

    input_file = sys.argv[1]
    svg_width_inches = float(sys.argv[2])
    svg_height_inches = float(sys.argv[3])
    min_distance_inches = float(sys.argv[4])
    output_prefix = os.path.splitext(os.path.basename(input_file))[0]

    obj_to_svg = OBJToSVG(
        obj_file=input_file,
        svg_size_inches=(svg_width_inches, svg_height_inches),
        min_distance_inches=min_distance_inches,
        output_prefix=output_prefix,
    )
    obj_to_svg.run()

if __name__ == "__main__":
    main() 