# FPSickle Examples

This directory contains example files and documentation for using FPSickle tool.

## Example Usage

To run FPSickle on a sample AVI file:

```bash
python src/fpsickle.py -i path/to/your/recording.avi -o analyzed_output.mp4
```

## Getting Test Files

Since OBS recordings can be large, you'll need to create your own test files. Here's how to capture a short game recording with OBS:

1. Install OBS Studio from [obsproject.com](https://obsproject.com/)
2. Set up OBS to record your game:
   - Add a "Game Capture" source
   - In Settings → Output, set the Recording Format to "AVI"
   - Set the Recording Quality to "Lossless Quality"
   - Make sure "Standard" recording mode is selected
3. Record a short gameplay clip (30 seconds is usually enough for testing)
4. Save the recording to the `data` directory of this project

## Expected Results

After analyzing a video, you'll get:

- A video file with the FPS graph overlay
- Console output with the average, minimum, and maximum FPS values
- A real-time display of the video during analysis (unless `--no-display` is used)

## Fine-tuning Analysis

FPSickle can be tuned using these parameters:

- `-w` / `--window`: Analysis window size in frames (default: 30)
  Smaller values give more frequent updates but may be less accurate

- `-t` / `--threshold`: Frame difference threshold (default: 0.001)
  Lower values are more sensitive to small changes between frames

Example with custom parameters:

```bash
python src/fpsickle.py -i recording.avi -o output.mp4 -w 15 -t 0.0005
```
