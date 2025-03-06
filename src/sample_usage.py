#!/usr/bin/env python3
"""
Sample usage of FPSickle
"""

import os
import sys
from fpsickle import FPSickle, MultiVideoFPSAnalyzer

def main():
    """
    Sample usage of FPSickle

    This script demonstrates how to use FPSickle class programmatically,
    without using the command-line interface.
    """
    # Check if input files are provided
    if len(sys.argv) < 2:
        print("Usage: python sample_usage.py <path_to_video_file> [path_to_video_file2 ...]")
        return 1

    input_files = sys.argv[1:]

    # Default output file
    output_file = "output_analyzed.mp4"

    print(f"Analyzing videos: {', '.join(input_files)}")
    print(f"Output will be saved to: {output_file}")

    try:
        if len(input_files) > 1:
            # Multi-video mode
            print("Running in multi-video mode")

            # Create an instance of MultiVideoFPSAnalyzer
            analyzer = MultiVideoFPSAnalyzer(
                output_file=output_file,
                window_size=30,      # Analyze FPS every 30 frames
                display=True,        # Display video during processing
                threshold=0.001      # Sensitivity for detecting frame changes
            )

            # Add each video with a custom label
            for i, input_file in enumerate(input_files):
                filename = os.path.basename(input_file)
                label = f"Video {i+1}: {filename}"
                analyzer.add_video(input_file, label)

            # Run the analysis
            analyzer.analyze()
        else:
            # Single video mode
            print("Running in single video mode")

            # Create an instance of FPSickle
            analyzer = FPSickle(
                input_file=input_files[0],
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
