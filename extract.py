#!/usr/bin/env python3
"""
Extract monospace characters from a grid in an image.
Usage: python extract_chars.py input.webp output_dir --grid-x 10 --grid-y 50 --char-width 12.5 --char-height 20 --cols 80 --rows 60
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def extract_characters(
    image_path: str,        # Path to input image (webp or other format)
    output_dir: str,        # Directory to save extracted character images
    grid_x: float,          # X offset to first character (top-left corner)
    grid_y: float,          # Y offset to first character (top-left corner)
    char_width: float,      # Width of each character cell (can be fractional)
    char_height: float,     # Height of each character cell (can be fractional)
    cols: int,              # Number of columns in the grid
    rows: int,              # Number of rows in the grid
) -> None:
    """
    Extract characters from a monospace grid with fractional cell dimensions.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"Extracting {rows * cols} characters...")
    
    char_index = 0
    for row in range(rows):
        for col in range(cols):
            # Calculate character position (using float arithmetic)
            x = grid_x + col * char_width
            y = grid_y + row * char_height
            
            # Round to nearest pixel for extraction
            x_start = int(round(x))
            y_start = int(round(y))
            x_end = x_start + int(char_width) # we need a consistent height and width for further processing
            y_end = y_start + int(char_height)
            
            # Bounds checking
            if y_end > img.shape[0] or x_end > img.shape[1]:
                print(f"Warning: Character at row {row}, col {col} exceeds image bounds, skipping")
                continue
            
            char_img = img[y_start:y_end, x_start:x_end]
            
            # Save character
            # Format: char_NNNNNN_rRR_cCC.png (index, row, col)
            filename = f"char_{char_index:06d}_r{row:02d}_c{col:03d}.png"
            cv2.imwrite(str(output_path / filename), char_img)
            
            char_index += 1
    
    print(f"Extracted {char_index} characters to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract monospace characters from a grid in an image"
    )
    parser.add_argument("input", help="Input image file (webp, png, jpg, etc)")
    parser.add_argument("output_dir", help="Output directory for character images")
    parser.add_argument("--grid-x", type=float, required=True, help="X offset to first character")
    parser.add_argument("--grid-y", type=float, required=True, help="Y offset to first character")
    parser.add_argument("--char-width", type=float, required=True, help="Width of each character cell (can be fractional)")
    parser.add_argument("--char-height", type=float, required=True, help="Height of each character cell (can be fractional)")
    parser.add_argument("--cols", type=int, required=True, help="Number of columns")
    parser.add_argument("--rows", type=int, required=True, help="Number of rows")
    
    args = parser.parse_args()
    
    extract_characters(
        args.input,
        args.output_dir,
        args.grid_x,
        args.grid_y,
        args.char_width,
        args.char_height,
        args.cols,
        args.rows,
    )


if __name__ == "__main__":
    main()
