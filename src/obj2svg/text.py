from typing import Tuple
from freetype import Face

def create_number_path(face: Face, number: str, x: float, y: float, scale: float = 1.0) -> Tuple[str, str]:
    """Create SVG path for a number using FreeType."""
    try:
        face.set_char_size(1)
        all_paths = []
        current_x = 0
        
        for digit in str(number):
            face.load_char(digit)
            outline = face.glyph.outline
            
            if not outline.points:
                current_x += 30
                continue
                
            y_values = [p[1] for p in outline.points]
            y_max = max(y_values)
            
            outline_points = [(p[0] + current_x, y_max - p[1]) for p in outline.points]
            
            start = 0
            for end in outline.contours:
                points = outline_points[start:end + 1]
                
                path = "M " + f"{points[0][0]},{points[0][1]} "
                
                for i in range(1, len(points)):
                    path += f"L {points[i][0]},{points[i][1]} "
                
                path += "Z"
                all_paths.append(path)
                
                start = end + 1
            
            current_x += 35

        if not all_paths:
            return "", ""

        path_data = " ".join(all_paths)
        transform = f"translate({x},{y}) scale({scale})"
        
        return path_data, transform
        
    except Exception as e:
        print(f"Error creating path for number {number}: {e}")
        return "", "" 