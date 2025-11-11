# Examples

This directory contains example scripts demonstrating how to use PyThermal.

## Examples

### 1. `record_thermal.py` - Recording with Multiple Formats

Record thermal camera data with configurable parameters:

```bash
# Record for 10 seconds, save both MP4 and raw data
python examples/record_thermal.py --duration 10 --format both

# Record for 30 seconds, MP4 only
python examples/record_thermal.py --duration 30 --format mp4

# Record for 5 seconds, raw temperature data only
python examples/record_thermal.py --duration 5 --format raw

# Custom FPS and output directory
python examples/record_thermal.py --duration 15 --format both --fps 30 --output-dir my_recordings
```

**Features:**
- Configurable recording duration
- Multiple output formats: MP4 video, raw thermal data, or both
- Temperature arrays saved to NPY files for easy analysis
- Progress updates during recording

**Output files:**
- `thermal_YYYYMMDD_HHMMSS.mp4` - Video file (if format includes mp4)
- `thermal_YYYYMMDD_HHMMSS.tseq` - Raw thermal data (if format includes raw)
- `thermal_YYYYMMDD_HHMMSS.npy` - Temperature arrays in NumPy format

### 2. `live_view.py` - Live View with Display Switching

Display live thermal camera feed with multiple view modes:

```bash
# Basic usage
python examples/live_view.py

# With custom device path
python examples/live_view.py --device /path/to/native/dir
```

**Features:**
- Toggle between view modes:
  - YUYV grayscale view
  - Temperature colorized view (raw values)
  - Temperature Celsius view (actual temperature values)
- Mouse hover to see temperature at cursor position
- Real-time FPS and temperature statistics

**Controls:**
- `q` - Quit
- `t` - Toggle view mode
- Mouse - Hover to see temperature

## Requirements

All examples require:
- PyThermal library installed
- Thermal camera connected and initialized
- Proper USB permissions set up (run `setup.sh`)

For live view examples, an X11 display is required.

