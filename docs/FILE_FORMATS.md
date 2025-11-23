# PyThermal File Formats

This document describes the binary file formats used by PyThermal for storing thermal camera data.

## Table of Contents

- [.tseq Format (Thermal Sequence)](#tseq-format-thermal-sequence)
- [.tframe Format (Thermal Frame)](#tframe-format-thermal-frame)
- [Data Types Reference](#data-types-reference)

---

## .tseq Format (Thermal Sequence)

The `.tseq` format is used for recording sequences of thermal camera frames. It stores raw frame data in a compact binary format optimized for sequential playback.

### File Structure

```
[TSEQ Header]
[Frame 0]
[Frame 1]
[Frame 2]
...
[Frame N]
```

### File Header (6 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | char[4] | Magic number: `"TSEQ"` (0x54 0x53 0x45 0x51) |
| 4 | 1 | uint8 | Format version (currently 1) |
| 5 | 1 | uint8 | Color flag: `0x01` if RGB data included, `0x00` if not |

### Frame Structure

Each frame consists of:

1. **Frame Header** (24 bytes)
2. **YUYV Data** (115,200 bytes)
3. **Temperature Array** (18,432 bytes)
4. **RGB Data** (172,800 bytes, optional, only if color flag is set)

#### Frame Header (24 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 8 | double | Timestamp (seconds since Unix epoch) |
| 8 | 4 | uint32 | Frame sequence number |
| 12 | 4 | float | Minimum temperature (Celsius) |
| 16 | 4 | float | Maximum temperature (Celsius) |
| 20 | 4 | float | Average temperature (Celsius) |

#### YUYV Data (115,200 bytes)

- **Format**: YUYV (YUV 4:2:2 packed)
- **Resolution**: 240 × 240 pixels
- **Bytes per pixel**: 2 bytes
- **Total size**: 240 × 240 × 2 = 115,200 bytes
- **Data type**: `uint8`

YUYV format details:
- **Y**: Luminance (brightness) - 1 byte per pixel
- **U/V**: Chrominance (color) - shared between 2 adjacent pixels
- For every 2 pixels: `[Y0, U, Y1, V]`

#### Temperature Array (18,432 bytes)

- **Resolution**: 96 × 96 pixels
- **Data type**: `uint16` (16-bit unsigned integer)
- **Total size**: 96 × 96 × 2 = 18,432 bytes
- **Layout**: Row-major order (first row, then second row, etc.)
- **Values**: Raw temperature values (need to be mapped to Celsius using min/max from metadata)

#### RGB Data (172,800 bytes, optional)

- **Format**: RGB (Red-Green-Blue)
- **Resolution**: 240 × 240 pixels
- **Channels**: 3 (R, G, B)
- **Bytes per pixel**: 3 bytes
- **Total size**: 240 × 240 × 3 = 172,800 bytes
- **Data type**: `uint8`
- **Layout**: Row-major order, interleaved (R, G, B, R, G, B, ...)
- **Only present if**: Color flag in header is `0x01`

### Frame Size Calculation

- **Without RGB**: 24 + 115,200 + 18,432 = **133,656 bytes**
- **With RGB**: 24 + 115,200 + 18,432 + 172,800 = **306,456 bytes**

### Example Usage

```python
from pythermal import ThermalRecorder

# Record a sequence
recorder = ThermalRecorder(output_dir="recordings", color=True)
recorder.start()
recorder.record_loop(duration=10.0)  # Record for 10 seconds
recorder.stop()
```

### Reading .tseq Files

```python
from pythermal.core import ThermalSequenceReader

reader = ThermalSequenceReader("recordings/thermal_20240101.tseq")
while reader.has_new_frame():
    metadata = reader.get_metadata()
    yuyv = reader.get_yuyv_frame()
    temp_array = reader.get_temperature_array()
    reader.mark_frame_read()
```

---

## .tframe Format (Thermal Frame)

The `.tframe` format is used for storing individual thermal frames with rendered visualization. It includes both the rendered image (with overlay) and raw frame data, making it ideal for screenshots and interactive frame replay.

### File Structure

```
[TFRM Header]
[View Mode String]
[Timestamp]
[Metadata]
[Rendered Image (PNG)]
[YUYV Data]
[Temperature Array]
[RGB Flag]
[RGB Data (optional)]
```

### File Header (5 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | char[4] | Magic number: `"TFRM"` (0x54 0x46 0x52 0x4D) |
| 4 | 1 | uint8 | Format version (currently 1) |

### View Mode String (variable length)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 1 | uint8 | String length (0-255) |
| 1 | N | char[N] | View mode string (UTF-8): `"yuyv"`, `"temperature"`, or `"temperature_celsius"` |

### Timestamp (8 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 8 | double | Timestamp (seconds since Unix epoch) |

### Metadata (16 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | uint32 | Frame sequence number |
| 4 | 4 | float | Minimum temperature (Celsius) |
| 8 | 4 | float | Maximum temperature (Celsius) |
| 12 | 4 | float | Average temperature (Celsius) |

### Rendered Image (variable size)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | uint32 | PNG image data size (bytes) |
| 4 | N | uint8[N] | PNG-encoded image data |

- **Format**: PNG (Portable Network Graphics)
- **Content**: BGR image with overlay (text, statistics)
- **Size**: Variable (depends on image content and compression)
- **Decoding**: Use PNG decoder (e.g., OpenCV `cv2.imdecode()`)
- **Note**: Mouse position/temperature is not stored in the file. The rendered image is independent of mouse position.

### YUYV Data (115,200 bytes)

Same as `.tseq` format:
- **Resolution**: 240 × 240 pixels
- **Format**: YUYV (YUV 4:2:2)
- **Data type**: `uint8`
- **Total size**: 240 × 240 × 2 = 115,200 bytes

### Temperature Array (18,432 bytes)

Same as `.tseq` format:
- **Resolution**: 96 × 96 pixels
- **Data type**: `uint16`
- **Total size**: 96 × 96 × 2 = 18,432 bytes

### RGB Flag (1 byte)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 1 | uint8 | `0x01` if RGB data follows, `0x00` if not |

### RGB Data (172,800 bytes, optional)

Same as `.tseq` format:
- **Resolution**: 240 × 240 pixels
- **Format**: RGB
- **Data type**: `uint8`
- **Total size**: 240 × 240 × 3 = 172,800 bytes
- **Only present if**: RGB flag is `0x01`

### File Size

Variable, depends on:
- View mode string length
- Rendered image size (PNG compressed)
- Whether RGB data is included

Approximate sizes:
- **Minimum** (no RGB, small PNG): ~135 KB
- **With RGB**: ~308 KB
- **Typical** (with RGB, medium PNG): ~350-400 KB

### Example Usage

```python
from pythermal.core import ThermalFrameProcessor
from pythermal import ThermalCapture

# Save a frame as .tframe
capture = ThermalCapture()  # or ThermalCapture("file.tseq")
frame = ThermalFrameProcessor.create_frame_from_capture(capture)

# Get rendered image (with overlay, without mouse position)
rendered_image = get_rendered_image_with_overlay()

# Save as .tframe
ThermalFrameProcessor.write_tframe(
    "screenshot.tframe",
    rendered_image,
    frame,
    view_mode="temperature"
)
```

### Reading .tframe Files

```python
from pythermal.core import ThermalFrameProcessor

# Read .tframe file
data = ThermalFrameProcessor.read_tframe("screenshot.tframe")

if data:
    rendered_image = data['rendered_image']  # BGR image with overlay
    frame = data['frame']  # ThermalFrame object
    view_mode = data['view_mode']  # 'yuyv', 'temperature', etc.
    timestamp = data['timestamp']
    metadata = data['metadata']
```

### Interactive Replay

```python
from pythermal.examples.live_view import EnhancedLiveView

# Load and display .tframe file interactively
viewer = EnhancedLiveView("screenshot.tframe")
viewer.run()  # Opens interactive viewer with mouse temperature display
```

---

## Data Types Reference

### Python struct Format Codes

| Code | C Type | Python Type | Size (bytes) |
|------|--------|-------------|--------------|
| `d` | double | float | 8 |
| `f` | float | float | 4 |
| `I` | unsigned int | int | 4 |
| `H` | unsigned short | int | 2 |
| `c` | char | bytes | 1 |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `WIDTH` | 240 | Image width (pixels) |
| `HEIGHT` | 240 | Image height (pixels) |
| `TEMP_WIDTH` | 96 | Temperature array width (pixels) |
| `TEMP_HEIGHT` | 96 | Temperature array height (pixels) |
| `YUYV_SIZE` | 115,200 | YUYV data size (bytes) |
| `TEMP_SIZE` | 18,432 | Temperature array size (bytes) |
| `RGB_SIZE` | 172,800 | RGB data size (bytes) |
| `FRAME_HEADER_SIZE` | 24 | Frame header size (bytes) |

### Byte Order

All multi-byte values use **little-endian** byte order (standard for x86/x64 architectures).

### Temperature Mapping

Raw temperature values (uint16) in the temperature array need to be mapped to Celsius using the min/max temperatures from the frame metadata:

```python
raw_min = np.min(temp_array)
raw_max = np.max(temp_array)
raw_range = raw_max - raw_min

if raw_range > 0:
    normalized = (raw_value - raw_min) / raw_range
    temp_celsius = min_temp + normalized * (max_temp - min_temp)
else:
    temp_celsius = avg_temp
```

---

## File Format Comparison

| Feature | .tseq | .tframe |
|---------|-------|---------|
| **Purpose** | Sequential recording | Single frame with visualization |
| **Rendered Image** | No | Yes (PNG) |
| **View Mode Info** | No | Yes |
| **File Size** | Fixed per frame | Variable (PNG compression) |
| **Use Case** | Video recording/playback | Screenshots, analysis |
| **RGB Data** | Optional | Optional |
| **Mouse Position** | No | No (frame is independent) |

---

## Version History

### .tseq Format
- **Version 1** (current): Initial format with optional RGB support

### .tframe Format
- **Version 1** (current): Initial format with rendered image and raw frame data

---

## Notes

- Both formats are designed for efficient sequential reading
- `.tseq` files are optimized for streaming playback
- `.tframe` files are optimized for single-frame analysis and visualization
- PNG compression in `.tframe` files provides good compression ratios for rendered images
- All timestamps are in seconds since Unix epoch (January 1, 1970)
- Temperature values are in Celsius

