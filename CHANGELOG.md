# Changelog

All notable changes to PyThermal will be documented in this file.

## [0.3.0] - 2026-01-03

### Added
- **Serial-number-based shared memory naming**: Shared memory segments now use serial numbers (`/dev/shm/yuyv240_shm_[SN]`) instead of device indices for strict device identification
- **Cameras table file**: New CSV table at `~/.pythermal/cameras` for human-readable device mapping
- **Consistent device ordering**: Devices are automatically sorted by serial number to ensure consistent device_index assignment
- **Enhanced device mapping**: Dual storage format (JSON + CSV) for device mappings

### Changed
- **Breaking**: Shared memory naming now uses serial numbers. Old device-index-based names (`yuyv240_shm_1`) are replaced with serial-based names (`yuyv240_shm_EA4782767`)
- Device IDs are now assigned based on sorted serial numbers, ensuring consistent ordering
- Improved device initialization with better error handling and attribute safety checks

### Fixed
- Fixed AttributeError exceptions during cleanup when initialization fails
- Fixed device serial number not being initialized early enough
- Improved robustness of device enumeration and mapping

### Technical Details
- Updated `UsbDemo.cpp` to accept serial number as command-line argument
- Modified `generate_shm_name()` to use serial numbers instead of device indices
- Enhanced `DeviceManager` to normalize device IDs based on sorted serial numbers
- Added `_normalize_device_ids()` method for consistent device ordering

## [0.2.9] - Previous Release

Previous features and bug fixes.

