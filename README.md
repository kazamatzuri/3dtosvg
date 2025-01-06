# STL to SVG Converter

Convert STL files to SVG with matched edge labeling. This tool processes 3D STL files and exports flat faces as 2D SVG drawings, with matching edges labeled for assembly.

## Features

- Extract and identify flat faces from STL files
- Project 3D faces to 2D space
- Automatic layout with minimum distance constraints
- Edge matching and labeling for assembly
- Multi-page support for large designs

## Installation

```bash
pip install stl2svg
```

## Usage

Command line:
```bash
stl2svg input.stl output.svg --width 800 --height 600 --min-distance 10
```

Python API:
```python
from stl2svg import convert_stl_to_svg

convert_stl_to_svg(
    stl_file="input.stl",
    svg_file="output.svg",
    width=800,
    height=600,
    min_distance=10
)
```

## Development

1. Clone the repository
```bash
git clone https://github.com/yourusername/stl2svg.git
cd stl2svg
```

2. Install development dependencies
```bash
pip install -e ".[dev]"
```

3. Run tests
```bash
pytest
```

## License

MIT License