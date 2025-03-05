#!/usr/bin/env python3
"""
Tests for the FPS Analyser
"""

import sys
import os
import unittest
import numpy as np

# Add the src directory to the path so we can import our module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fps_analyser import FPSAnalyser


class TestFrameDifference(unittest.TestCase):
    """Test the frame difference calculation"""

    def test_identical_frames(self):
        """Test that identical frames have zero difference"""
        # Create a simple test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create an instance of the analyzer with dummy values
        analyzer = FPSAnalyser.__new__(FPSAnalyser)

        # Test difference of identical frames should be 0
        diff = analyzer.calculate_frame_difference(frame, frame)
        self.assertEqual(diff, 0.0)

    def test_different_frames(self):
        """Test that different frames have non-zero difference"""
        # Create two different frames
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Create an instance of the analyzer with dummy values
        analyzer = FPSAnalyser.__new__(FPSAnalyser)

        # Test difference of completely different frames should be 1.0
        diff = analyzer.calculate_frame_difference(frame1, frame2)
        self.assertEqual(diff, 1.0)


if __name__ == '__main__':
    unittest.main()
