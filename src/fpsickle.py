#!/usr/bin/env python3
"""
FPSickle - Analyzes raw video game recordings and estimates real framerate
"""

import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from PIL import Image


class FPSickle:
    def __init__(self, input_file, output_file=None, window_size=30,
                 display=True, threshold=0.001, codec='mp4v'):
        """
        Initialize FPSickle

        Args:
            input_file (str): Path to input AVI video file
            output_file (str, optional): Path to output video file
            window_size (int): Size of the analysis window in frames
            display (bool): Whether to display video during processing
            threshold (float): Threshold for frame difference detection
            codec (str): FourCC codec code for output video (default: mp4v)
        """
        self.input_file = input_file
        self.output_file = output_file
        self.window_size = window_size
        self.display = display
        self.threshold = threshold
        self.codec = codec

        # Check if input file exists and is an AVI file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if not input_file.lower().endswith('.avi'):
            raise ValueError("Only AVI files are supported")

        # Open the video file
        self.cap = cv2.VideoCapture(input_file)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {input_file}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.recorded_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Recorded FPS: {self.recorded_fps}")
        print(f"  Total frames: {self.frame_count}")

        # Initialize variables for analysis
        self.real_fps_values = []
        self.frame_times = []
        self.current_frame = 0

    def calculate_frame_difference(self, frame1, frame2):
        """
        Calculate the difference between two frames

        Args:
            frame1 (numpy.ndarray): First frame
            frame2 (numpy.ndarray): Second frame

        Returns:
            float: Mean absolute difference between frames
        """
        # Check if frames have the same shape
        if frame1.shape != frame2.shape:
            # Resize one to match the other
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff) / 255.0

    def generate_fps_graph(self, width=800, height=200, background_color=(0, 0, 0, 180)):
        """
        Generate a matplotlib graph of the FPS values

        Args:
            width (int): Width of the graph
            height (int): Height of the graph
            background_color (tuple): RGBA background color for the graph

        Returns:
            numpy.ndarray: Image of the graph with alpha channel
        """
        # Create a figure with a specific size
        dpi = 100
        fig = Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        fig.patch.set_facecolor('none')  # Transparent background

        # Add a subplot
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(background_color[3]/255)
        ax.patch.set_facecolor((background_color[0]/255, background_color[1]/255, background_color[2]/255))

        # Plot data
        x = np.array(self.frame_times)
        y = np.array(self.real_fps_values)

        # Add moving average line for smoother graph
        window = min(len(y), 10)
        if window > 0:
            y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
            x_smooth = x[window-1:]
            ax.plot(x_smooth, y_smooth, 'g-', linewidth=2)

        # Plot the actual data points
        ax.plot(x, y, 'b.', alpha=0.3, markersize=3)

        # Set labels and title
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('FPS', color='white')
        ax.set_title(f'Real-time FPS Analysis (Avg: {np.mean(y):.1f} FPS)', color='white')

        # Add grid and set limits
        ax.grid(True, alpha=0.3)
        if len(y) > 0:
            ax.set_ylim([max(0, min(y) - 10), max(y) + 10])

        # Set tick colors
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        # Render the figure to a numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Convert canvas to image
        buf = canvas.buffer_rgba()
        graph_image = np.asarray(buf)

        # Close the figure to free memory
        plt.close(fig)

        return graph_image

    def overlay_graph_on_frame(self, frame, graph, position=(20, 20)):
        """
        Overlay the FPS graph on a video frame

        Args:
            frame (numpy.ndarray): Video frame
            graph (numpy.ndarray): Graph image with alpha channel
            position (tuple): Position for the top-left corner of the graph

        Returns:
            numpy.ndarray: Frame with overlaid graph
        """
        x, y = position
        h, w = graph.shape[:2]

        # Make sure the graph fits within the frame
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            # Resize the graph to fit within the frame
            new_h = min(h, frame.shape[0] - y)
            new_w = min(w, frame.shape[1] - x)
            graph = cv2.resize(graph, (new_w, new_h))
            h, w = graph.shape[:2]

        # Create a region of interest
        roi = frame[y:y+h, x:x+w].copy()

        # Create a properly normalized alpha channel
        alpha_channel = graph[:, :, 3] / 255.0

        # Apply alpha blending manually for each channel
        for c in range(3):  # for each color channel (BGR)
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + graph[:, :, c] * alpha_channel

        # Put the blended image back into the frame
        result = frame.copy()
        result[y:y+h, x:x+w] = roi

        # Add current FPS value text
        if len(self.real_fps_values) > 0:
            current_fps = self.real_fps_values[-1]
            cv2.putText(result, f"FPS: {current_fps:.1f}",
                      (x + w + 10, y + 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return result

    def analyze(self):
        """
        Analyze the video to detect real framerate
        """
        # Initialize the writer if output is specified
        if self.output_file:
            try:
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                out = cv2.VideoWriter(self.output_file, fourcc, self.recorded_fps,
                                     (self.width, self.height))
                if not out.isOpened():
                    print(f"Warning: Could not create output video with codec '{self.codec}'.")
                    print("Trying with default codec...")
                    # Try with default codec
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(self.output_file, fourcc, self.recorded_fps,
                                         (self.width, self.height))
                    if not out.isOpened():
                        print("Failed to create output video. Analysis will proceed without saving output.")
                        out = None
            except Exception as e:
                print(f"Error creating output video: {e}")
                out = None
        else:
            out = None

        # Read the first frame
        ret, prev_frame = self.cap.read()
        if not ret:
            raise IOError("Could not read the first frame")

        # Initialize progress bar
        pbar = tqdm(total=self.frame_count, desc="Analyzing video")

        # Keep track of unique frames in the current window
        unique_frame_count = 1  # Start with 1 to count the first frame
        frames_in_window = 1    # Start with 1 to count the first frame
        self.current_frame = 1  # Start with 1 since we've read the first frame

        # Process each frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame += 1
            frames_in_window += 1
            current_time = self.current_frame / self.recorded_fps

            # Calculate difference with previous frame
            try:
                diff = self.calculate_frame_difference(prev_frame, frame)

                # If the difference is above threshold, it's a new frame
                if diff > self.threshold:
                    unique_frame_count += 1
                    prev_frame = frame.copy()
            except Exception as e:
                print(f"Error calculating frame difference: {e}")
                prev_frame = frame.copy()

            # Every window_size frames, calculate the FPS
            if frames_in_window >= self.window_size:
                # Calculate real FPS over the window
                window_time = frames_in_window / self.recorded_fps
                real_fps = unique_frame_count / window_time

                # Add values to our data
                self.real_fps_values.append(real_fps)
                self.frame_times.append(current_time)

                # Reset counters for next window
                unique_frame_count = 0
                frames_in_window = 0

            try:
                # Generate and overlay the FPS graph
                if len(self.real_fps_values) > 0:
                    graph = self.generate_fps_graph(width=self.width // 3, height=self.height // 4)
                    frame_with_graph = self.overlay_graph_on_frame(frame, graph)
                else:
                    frame_with_graph = frame

                # Write the frame to output video
                if out:
                    out.write(frame_with_graph)

                # Display the frame
                if self.display:
                    cv2.imshow('FPS Analysis', frame_with_graph)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Error processing frame {self.current_frame}: {e}")
                if self.display:
                    cv2.imshow('FPS Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if out:
                    out.write(frame)

            pbar.update(1)

        # Handle last window if it's not complete
        if frames_in_window > 0 and unique_frame_count > 0:
            window_time = frames_in_window / self.recorded_fps
            real_fps = unique_frame_count / window_time
            self.real_fps_values.append(real_fps)
            self.frame_times.append(current_time)

        # Clean up
        pbar.close()
        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        if len(self.real_fps_values) > 0:
            # Display results
            print(f"\nAnalysis results:")
            print(f"  Average FPS: {np.mean(self.real_fps_values):.2f}")
            print(f"  Min FPS: {np.min(self.real_fps_values):.2f}")
            print(f"  Max FPS: {np.max(self.real_fps_values):.2f}")

            if self.output_file:
                print(f"Output video saved to: {self.output_file}")
        else:
            print("No FPS data was collected. Try adjusting the threshold or window size.")


def main():
    """
    Main entry point for FPSickle
    """
    parser = argparse.ArgumentParser(description='Analyze the real framerate of a video')
    parser.add_argument('-i', '--input', required=True, help='Input video file (AVI format)')
    parser.add_argument('-o', '--output', help='Output video file with FPS graph overlay')
    parser.add_argument('-w', '--window', type=int, default=30,
                        help='Analysis window size in frames')
    parser.add_argument('--no-display', action='store_true',
                        help="Don't display video during processing")
    parser.add_argument('-t', '--threshold', type=float, default=0.001,
                        help='Frame difference threshold (default: 0.001)')
    parser.add_argument('-c', '--codec', default='mp4v',
                        help='FourCC codec for output video (default: mp4v)')

    args = parser.parse_args()

    try:
        # Create and run the analyzer
        analyzer = FPSickle(
            input_file=args.input,
            output_file=args.output,
            window_size=args.window,
            display=not args.no_display,
            threshold=args.threshold,
            codec=args.codec
        )

        analyzer.analyze()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
