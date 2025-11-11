# Tests

This directory contains test scripts for PyThermal.

## Test Scripts

### `test_device.py`
Tests the core ThermalDevice functionality:
- Device initialization
- Shared memory access
- Frame streaming

### `test_recording.py`
Tests the ThermalRecorder class:
- Recording functionality
- File output

### `test_live_view.py`
Tests the ThermalLiveView class:
- Initialization
- Method functionality
- Entry point configuration

### `verify_entry_point.py`
Verifies that the command-line entry point is properly configured.

## Running Tests

```bash
# Run all tests
python tests/test_device.py
python tests/test_recording.py
python tests/test_live_view.py
python tests/verify_entry_point.py
```

## Requirements

All tests require:
- PyThermal library (add parent directory to PYTHONPATH or install package)
- Thermal camera connected and initialized
- Proper USB permissions set up

