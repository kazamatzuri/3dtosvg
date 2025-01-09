# OBJ to SVG Converter

Convert OBJ files to SVG with matched edge labeling. This tool processes 3D OBJ files and exports flat faces as 2D SVG drawings, with matching edges labeled for assembly.

## Features

- Extract and identify flat faces from OBJ files
- Project 3D faces to 2D space
- Automatic layout with minimum distance constraints
- Edge matching and labeling for assembly
- Outputs a single SVG with all faces arranged in a diamond pattern

## Post-Processing

While this tool arranges faces in a simple diamond pattern, you might want to optimize the layout for CNC cutting. For this purpose, you can use [deepnest.io](https://deepnest.io/), which is a specialized tool for:
- Optimizing part layout to minimize material waste
- Nesting parts efficiently for CNC laser/plasma cutting
- Supporting various input formats including SVG

## Installation

Using Poetry:
```bash
poetry install
```

## Usage

Command line:
```bash
obj2svg <input.obj> <svg_width_inches> <svg_height_inches> <min_distance_inches>
```

Example:
```bash
obj2svg cube.obj 8 8 0.5
```

This will:
1. Process the OBJ file
2. Create a directory named after your input file (e.g., "cube")
3. Generate a "sheet1.svg" file inside that directory
4. Print detailed information about each face's dimensions and projections

Output includes:
- Face dimensions in millimeters (X, Y, Z ranges)
- Projection plane chosen for each face (XY, YZ, or XZ)
- Projected dimensions (width and height)
- Total number of faces processed

## Development

1. Clone the repository
```bash
git clone https://github.com/yourusername/obj2svg.git
cd obj2svg
```

2. Install with Poetry
```bash
poetry install
```

3. Run tests
```bash
poetry run pytest
```

## Dependencies

- Python 3.11 or higher
- svgwrite
- numpy
- scipy
- fonttools
- freetype-py

## License

MIT License