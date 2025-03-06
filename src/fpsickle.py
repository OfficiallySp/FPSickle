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
    def __init__(
        self,
        input_file,
        output_file=None,
        window_size=30,
        display=True,
        threshold=0.001,
        codec="mp4v",
        label=None,
    ):
        """
        Initialize FPSickle

        Args:
            input_file (str): Path to input video file
            output_file (str, optional): Path to output video file
            window_size (int): Size of the analysis window in frames
            display (bool): Whether to display video during processing
            threshold (float): Threshold for frame difference detection
            codec (str): FourCC codec code for output video (default: mp4v)
            label (str): Label for this video in multi-video analysis
        """
        self.input_file = input_file
        self.output_file = output_file
        self.window_size = window_size
        self.display = display
        self.threshold = threshold
        self.codec = codec
        self.label = label or os.path.basename(input_file)

        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Open the video file
        self.cap = cv2.VideoCapture(input_file)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {input_file}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.recorded_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties for {self.label}:")
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

    def generate_fps_graph(
        self, width=800, height=200, background_color=(0, 0, 0, 180)
    ):
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
        fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor("none")  # Transparent background

        # Add a subplot
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(background_color[3] / 255)
        ax.patch.set_facecolor(
            (
                background_color[0] / 255,
                background_color[1] / 255,
                background_color[2] / 255,
            )
        )

        # Plot data
        x = np.array(self.frame_times)
        y = np.array(self.real_fps_values)

        # Add moving average line for smoother graph
        window = min(len(y), 10)
        if window > 0:
            y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
            x_smooth = x[window - 1 :]
            ax.plot(x_smooth, y_smooth, "g-", linewidth=2)

        # Plot the actual data points
        ax.plot(x, y, "b.", alpha=0.3, markersize=3)

        # Set labels and title
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("FPS", color="white")
        ax.set_title(
            f"Real-time FPS Analysis (Avg: {np.mean(y):.1f} FPS)", color="white"
        )

        # Add grid and set limits
        ax.grid(True, alpha=0.3)
        if len(y) > 0:
            ax.set_ylim([max(0, min(y) - 10), max(y) + 10])

        # Set tick colors
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

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
        roi = frame[y : y + h, x : x + w].copy()

        # Create a properly normalized alpha channel
        alpha_channel = graph[:, :, 3] / 255.0

        # Apply alpha blending manually for each channel
        for c in range(3):  # for each color channel (BGR)
            roi[:, :, c] = (
                roi[:, :, c] * (1 - alpha_channel) + graph[:, :, c] * alpha_channel
            )

        # Put the blended image back into the frame
        result = frame.copy()
        result[y : y + h, x : x + w] = roi

        # Add current FPS value text
        if len(self.real_fps_values) > 0:
            current_fps = self.real_fps_values[-1]
            cv2.putText(
                result,
                f"FPS: {current_fps:.1f}",
                (x + w + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        return result

    def analyze(self):
        """
        Analyze the video to detect real framerate
        """
        # Set window title
        window_title = "FPSickle Analysis"

        # Initialize the writer if output is specified
        if self.output_file:
            try:
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                out = cv2.VideoWriter(
                    self.output_file,
                    fourcc,
                    self.recorded_fps,
                    (self.width, self.height),
                )
                if not out.isOpened():
                    print(
                        f"Warning: Could not create output video with codec '{self.codec}'."
                    )
                    print("Trying with default codec...")
                    # Try with default codec
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        self.output_file,
                        fourcc,
                        self.recorded_fps,
                        (self.width, self.height),
                    )
                    if not out.isOpened():
                        print(
                            "Failed to create output video. Analysis will proceed without saving output."
                        )
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
        frames_in_window = 1  # Start with 1 to count the first frame
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
                    graph = self.generate_fps_graph(
                        width=self.width // 3, height=self.height // 4
                    )
                    frame_with_graph = self.overlay_graph_on_frame(frame, graph)
                else:
                    frame_with_graph = frame

                # Write the frame to output video
                if out:
                    out.write(frame_with_graph)

                # Display the frame
                if self.display:
                    cv2.imshow(window_title, frame_with_graph)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except Exception as e:
                print(f"Error processing frame {self.current_frame}: {e}")
                if self.display:
                    cv2.imshow(window_title, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
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
            print(f"\nAnalysis results for {self.label}:")
            print(f"  Average FPS: {np.mean(self.real_fps_values):.2f}")
            print(f"  Min FPS: {np.min(self.real_fps_values):.2f}")
            print(f"  Max FPS: {np.max(self.real_fps_values):.2f}")

            if self.output_file:
                print(f"Output video saved to: {self.output_file}")
        else:
            print(
                "No FPS data was collected. Try adjusting the threshold or window size."
            )


class MultiVideoFPSAnalyzer:
    """
    Handles multiple video FPS analysis and compositing
    """

    def __init__(
        self,
        output_file=None,
        window_size=30,
        display=True,
        threshold=0.001,
        codec="mp4v",
        grid_layout=None,
    ):
        """
        Initialize MultiVideoFPSAnalyzer

        Args:
            output_file (str, optional): Path to output composite video file
            window_size (int): Size of the analysis window in frames
            display (bool): Whether to display video during processing
            threshold (float): Threshold for frame difference detection
            codec (str): FourCC codec code for output video
            grid_layout (tuple, optional): Grid layout for videos (rows, cols)
        """
        self.analyzers = []
        self.output_file = output_file
        self.window_size = window_size
        self.display = display
        self.threshold = threshold
        self.codec = codec
        self.grid_layout = grid_layout
        self.colors = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]

    def add_video(self, input_file, label=None):
        """
        Add a video to be analyzed

        Args:
            input_file (str): Path to input video file
            label (str, optional): Custom label for this video

        Returns:
            int: Index of the added video
        """
        label = label or f"Video {len(self.analyzers) + 1}"

        analyzer = FPSickle(
            input_file=input_file,
            output_file=None,  # We'll handle the output in this class
            window_size=self.window_size,
            display=False,  # We'll handle display in this class
            threshold=self.threshold,
            codec=self.codec,
            label=label,
        )

        self.analyzers.append(analyzer)
        return len(self.analyzers) - 1

    def determine_grid_layout(self):
        """
        Determine the optimal grid layout for displaying videos

        Returns:
            tuple: (rows, cols) for the grid layout
        """
        if self.grid_layout:
            return self.grid_layout

        n = len(self.analyzers)
        if n <= 1:
            return (1, 1)
        elif n == 2:
            return (1, 2)
        elif n <= 4:
            return (2, 2)
        elif n <= 6:
            return (2, 3)
        elif n <= 9:
            return (3, 3)
        else:
            # Calculate closest grid
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            return (rows, cols)

    def create_composite_frame(self, frames):
        """
        Create a composite frame from multiple video frames

        Args:
            frames (list): List of frames from each video

        Returns:
            numpy.ndarray: Composite frame
        """
        if not frames:
            return None

        # Handle missing frames by using black frames
        for i in range(len(frames)):
            if frames[i] is None:
                # Create a black frame with the same dimensions as the first valid frame
                valid_frames = [f for f in frames if f is not None]
                if valid_frames:
                    frames[i] = np.zeros_like(valid_frames[0])
                else:
                    return None

        # Determine the grid layout
        rows, cols = self.determine_grid_layout()

        # Resize all frames to the same size
        target_height = max(frame.shape[0] for frame in frames) // rows
        target_width = max(frame.shape[1] for frame in frames) // cols

        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (target_width, target_height))
            resized_frames.append(resized)

        # Create the composite frame
        composite = np.zeros(
            (target_height * rows, target_width * cols, 3), dtype=np.uint8
        )

        # Place each frame in the grid
        for i, frame in enumerate(resized_frames):
            if i >= rows * cols:
                break

            r = i // cols
            c = i % cols

            y_start = r * target_height
            y_end = (r + 1) * target_height
            x_start = c * target_width
            x_end = (c + 1) * target_width

            # Add a label to each frame
            if i < len(self.analyzers):
                label = self.analyzers[i].label
                color = self.colors[i % len(self.colors)]
                cv2.putText(
                    frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
                )

                # Add current FPS if available
                if self.analyzers[i].real_fps_values:
                    current_fps = self.analyzers[i].real_fps_values[-1]
                    cv2.putText(
                        frame,
                        f"FPS: {current_fps:.1f}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                    )

            # Place the frame in the composite
            composite[y_start:y_end, x_start:x_end] = frame

        return composite

    def generate_combined_fps_graph(self, width, height):
        """
        Generate a graph showing FPS data from all videos

        Args:
            width (int): Width of the graph
            height (int): Height of the graph

        Returns:
            numpy.ndarray: Graph image with alpha channel
        """
        # Create a figure with a specific size
        dpi = 100
        fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor("none")  # Transparent background

        # Add a subplot
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.7)
        ax.patch.set_facecolor((0, 0, 0))

        # Plot data for each analyzer
        for i, analyzer in enumerate(self.analyzers):
            if len(analyzer.real_fps_values) > 0:
                x = np.array(analyzer.frame_times)
                y = np.array(analyzer.real_fps_values)

                # Get color for this analyzer
                color = self.colors[i % len(self.colors)]
                # Convert BGR to RGB for matplotlib
                rgb_color = (color[2] / 255, color[1] / 255, color[0] / 255)

                # Add moving average line for smoother graph
                window = min(len(y), 10)
                if window > 0:
                    y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
                    x_smooth = x[window - 1 :]
                    ax.plot(
                        x_smooth,
                        y_smooth,
                        color=rgb_color,
                        linewidth=2,
                        label=f"{analyzer.label} (Avg: {np.mean(y):.1f} FPS)",
                    )

                # Plot the actual data points with lower alpha
                ax.plot(x, y, ".", color=rgb_color, alpha=0.3, markersize=3)

        # Set labels and title
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("FPS", color="white")
        ax.set_title("Multi-Video FPS Analysis", color="white")

        # Add legend
        ax.legend(
            loc="lower left",
            facecolor="black",
            framealpha=0.7,
            edgecolor="white",
            labelcolor="white",
        )

        # Add grid and set limits
        ax.grid(True, alpha=0.3)

        # Set y axis limits based on all data
        all_fps_values = []
        for analyzer in self.analyzers:
            all_fps_values.extend(analyzer.real_fps_values)

        if all_fps_values:
            ax.set_ylim([max(0, min(all_fps_values) - 10), max(all_fps_values) + 10])

        # Set tick colors
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        # Render the figure to a numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Convert canvas to image
        buf = canvas.buffer_rgba()
        graph_image = np.asarray(buf)

        # Close the figure to free memory
        plt.close(fig)

        return graph_image

    def analyze(self):
        """
        Analyze all videos and create a composite output
        """
        if not self.analyzers:
            raise ValueError("No videos added for analysis")

        # Set window title
        window_title = "FPSickle Multi-Video Analysis"

        # Determine composite video dimensions
        rows, cols = self.determine_grid_layout()
        max_width = max(analyzer.width for analyzer in self.analyzers)
        max_height = max(analyzer.height for analyzer in self.analyzers)

        composite_width = (max_width // cols) * cols
        composite_height = (max_height // rows) * rows

        # Initialize the writer if output is specified
        if self.output_file:
            try:
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                out = cv2.VideoWriter(
                    self.output_file,
                    fourcc,
                    self.analyzers[0].recorded_fps,
                    (composite_width, composite_height),
                )
                if not out.isOpened():
                    print(
                        f"Warning: Could not create output video with codec '{self.codec}'."
                    )
                    print("Trying with default codec...")
                    # Try with default codec
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        self.output_file,
                        fourcc,
                        self.analyzers[0].recorded_fps,
                        (composite_width, composite_height),
                    )
                    if not out.isOpened():
                        print(
                            "Failed to create output video. Analysis will proceed without saving output."
                        )
                        out = None
            except Exception as e:
                print(f"Error creating output video: {e}")
                out = None
        else:
            out = None

        # Prepare for analysis
        caps = [analyzer.cap for analyzer in self.analyzers]
        prev_frames = []

        # Read the first frame from each video
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                prev_frames.append(frame)
            else:
                prev_frames.append(None)

        # Initialize progress bar using the shortest video
        min_frames = min(analyzer.frame_count for analyzer in self.analyzers)
        pbar = tqdm(total=min_frames, desc="Analyzing videos")

        # Initialize analysis variables for each video
        frames_in_window = [1] * len(self.analyzers)
        unique_frame_counts = [1] * len(self.analyzers)
        current_frames = [1] * len(self.analyzers)

        # Process frames
        while True:
            # Read next frame from each video
            current_frames_list = []
            all_done = True

            for i, cap in enumerate(caps):
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        current_frames_list.append(frame)
                        current_frames[i] += 1
                        frames_in_window[i] += 1
                        all_done = False

                        # Calculate difference with previous frame
                        try:
                            if prev_frames[i] is not None:
                                diff = self.analyzers[i].calculate_frame_difference(
                                    prev_frames[i], frame
                                )

                                # If the difference is above threshold, it's a new frame
                                if diff > self.threshold:
                                    unique_frame_counts[i] += 1
                                    prev_frames[i] = frame.copy()
                        except Exception as e:
                            print(
                                f"Error calculating frame difference for video {i+1}: {e}"
                            )
                            prev_frames[i] = frame.copy()

                        # Every window_size frames, calculate the FPS
                        if frames_in_window[i] >= self.window_size:
                            # Calculate real FPS over the window
                            window_time = (
                                frames_in_window[i] / self.analyzers[i].recorded_fps
                            )
                            real_fps = unique_frame_counts[i] / window_time
                            current_time = (
                                current_frames[i] / self.analyzers[i].recorded_fps
                            )

                            # Add values to our data
                            self.analyzers[i].real_fps_values.append(real_fps)
                            self.analyzers[i].frame_times.append(current_time)

                            # Reset counters for next window
                            unique_frame_counts[i] = 0
                            frames_in_window[i] = 0
                    else:
                        current_frames_list.append(None)
                else:
                    current_frames_list.append(None)

            # If all videos are done, exit the loop
            if all_done:
                break

            # Create composite frame and graph
            try:
                composite_frame = self.create_composite_frame(current_frames_list)

                if composite_frame is not None:
                    # Generate and overlay the FPS graph
                    has_fps_data = any(
                        len(analyzer.real_fps_values) > 0 for analyzer in self.analyzers
                    )

                    if has_fps_data:
                        graph = self.generate_combined_fps_graph(
                            width=composite_width // 2, height=composite_height // 4
                        )

                        # Overlay graph (position it at the bottom right)
                        x = composite_width - graph.shape[1] - 20
                        y = composite_height - graph.shape[0] - 20

                        # Create a region of interest
                        roi = composite_frame[
                            y : y + graph.shape[0], x : x + graph.shape[1]
                        ].copy()

                        # Apply alpha blending
                        alpha_channel = graph[:, :, 3] / 255.0
                        for c in range(3):
                            roi[:, :, c] = (
                                roi[:, :, c] * (1 - alpha_channel)
                                + graph[:, :, c] * alpha_channel
                            )

                        # Put the blended image back into the frame
                        composite_frame[
                            y : y + graph.shape[0], x : x + graph.shape[1]
                        ] = roi

                    # Write the frame to output video
                    if out:
                        out.write(composite_frame)

                    # Display the frame
                    if self.display:
                        cv2.imshow(window_title, composite_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
            except Exception as e:
                print(f"Error creating composite frame: {e}")

            pbar.update(1)

        # Clean up
        pbar.close()
        for cap in caps:
            if cap.isOpened():
                cap.release()

        if out:
            out.release()

        cv2.destroyAllWindows()

        # Print results for each video
        for i, analyzer in enumerate(self.analyzers):
            if len(analyzer.real_fps_values) > 0:
                print(f"\nAnalysis results for {analyzer.label}:")
                print(f"  Average FPS: {np.mean(analyzer.real_fps_values):.2f}")
                print(f"  Min FPS: {np.min(analyzer.real_fps_values):.2f}")
                print(f"  Max FPS: {np.max(analyzer.real_fps_values):.2f}")

        if self.output_file:
            print(f"Output composite video saved to: {self.output_file}")


def main():
    """
    Main entry point for FPSickle
    """
    # Display logo
    logo = r"""
    __________  _____ _      __   __
   / ____/ __ \/ ___/(_)____/ /__/ /__
  / /_  / /_/ /\__ \/ / ___/ //_/ / _ \
 / __/ / ____/___/ / / /__/ ,< / /  __/
/_/   /_/    /____/_/\___/_/|_/_/\___/

    """
    print(logo)
    print("FPSickle - Video Game FPS Analyzer")
    print("------------------------------------")

    parser = argparse.ArgumentParser(description="Analyze the real framerate of videos")
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        required=True,
        help="Input video file(s) (can be specified multiple times)",
    )
    parser.add_argument(
        "-o", "--output", help="Output video file with FPS graph overlay"
    )
    parser.add_argument(
        "-w", "--window", type=int, default=30, help="Analysis window size in frames"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display video during processing",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.001,
        help="Frame difference threshold (default: 0.001)",
    )
    parser.add_argument(
        "-c",
        "--codec",
        default="mp4v",
        help="FourCC codec for output video (default: mp4v)",
    )
    parser.add_argument(
        "-l",
        "--labels",
        action="append",
        help="Labels for input videos (must match number of input files)",
    )
    parser.add_argument("--grid", type=str, help='Grid layout as "rows,cols"')

    args = parser.parse_args()

    try:
        # Get grid layout if specified
        grid_layout = None
        if args.grid:
            try:
                rows, cols = map(int, args.grid.split(","))
                grid_layout = (rows, cols)
            except:
                print("Invalid grid layout. Using automatic layout.")

        # Check if labels are provided and match number of inputs
        labels = args.labels or []
        if labels and len(labels) != len(args.input):
            print(
                f"Warning: Number of labels ({len(labels)}) doesn't match number of input files "
                f"({len(args.input)}). Using default labels."
            )
            labels = []

        # Create multi-video analyzer if more than one input
        if len(args.input) > 1:
            # Create and run the multi-video analyzer
            analyzer = MultiVideoFPSAnalyzer(
                output_file=args.output,
                window_size=args.window,
                display=not args.no_display,
                threshold=args.threshold,
                codec=args.codec,
                grid_layout=grid_layout,
            )

            # Add each video
            for i, input_file in enumerate(args.input):
                label = labels[i] if i < len(labels) else None
                analyzer.add_video(input_file, label)

            analyzer.analyze()
        else:
            # Single video case - use the original FPSickle
            label = labels[0] if labels else None
            analyzer = FPSickle(
                input_file=args.input[0],
                output_file=args.output,
                window_size=args.window,
                display=not args.no_display,
                threshold=args.threshold,
                codec=args.codec,
                label=label,
            )

            analyzer.analyze()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
