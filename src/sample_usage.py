#!/usr/bin/env python3
"""
Sample usage of FPSickle
"""

import os
import sys
from fpsickle import FPSickle

def main():
    """
    Sample usage of FPSickle

    This script demonstrates how to use FPSickle class programmatically,
    without using the command-line interface.
    """
    # Check if input file is provided
    if len(sys.argv) < 2:
        print("Usage: python sample_usage.py <path_to_avi_file>")
        return 1

    input_file = sys.argv[1]

    # Default output file is in the same directory as the input file
    output_file = os.path.splitext(input_file)[0] + "_analyzed.mp4"

    print(f"Analyzing video: {input_file}")
    print(f"Output will be saved to: {output_file}")

    try:
        # Create an instance of FPSickle
        analyzer = FPSickle(
            input_file=input_file,
            output_file=output_file,
            window_size=30,  # Analyze FPS every 30 frames
            display=True,    # Display video during processing
            threshold=0.001  # Sensitivity for detecting frame changes
        )

        # Run the analysis
        analyzer.analyze()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
