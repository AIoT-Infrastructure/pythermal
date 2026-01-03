# PyThermal v0.3.0 Release Notes

## 🎉 Major Release: Serial-Number-Based Device Mapping

This release introduces **strict device identification** using USB serial numbers for shared memory naming, ensuring each physical camera always uses the same shared memory segment regardless of USB port or enumeration order.

## ✨ New Features

### Serial-Number-Based Shared Memory Naming
- Shared memory segments now use serial numbers: `/dev/shm/yuyv240_shm_[SN]`
- Each physical camera always gets the same shared memory name
- No confusion even if devices are reconnected in different USB ports
- Example: Camera with serial `EA4782688` → `/dev/shm/yuyv240_shm_EA4782688`

### Cameras CSV Table
- New human-readable table at `~/.pythermal/cameras`
- Easy to view and edit device mappings
- Format:
  ```csv
  serial_number,device_index
  EA4782688,0
  EA4782767,1
  ```

### Consistent Device Ordering
- Devices are automatically sorted by serial number
- Ensures consistent device_index assignment
- Same camera always gets the same device_index

### Enhanced Device Mapping
- Dual storage format: JSON (`device_mapping.json`) + CSV (`cameras`)
- Improved device enumeration and mapping logic
- Better error handling and robustness

## 🔧 Improvements

- Fixed AttributeError exceptions during cleanup
- Improved device initialization with better error handling
- Enhanced device serial number detection and mapping
- Better logging for device identification

## ⚠️ Breaking Changes

**Shared Memory Naming**: 
- **Old**: `/dev/shm/yuyv240_shm_1` (device-index-based)
- **New**: `/dev/shm/yuyv240_shm_EA4782767` (serial-number-based)

If you have existing code that directly accesses shared memory by name, you'll need to update it to use serial numbers or use PyThermal's `get_shm_name()` function.

## 📝 Migration Guide

### For Users
No action required! PyThermal automatically handles the migration. Your existing device mappings will continue to work.

### For Developers
If you're directly accessing shared memory files:
1. Use PyThermal's `get_shm_name(device_index, serial_number)` function
2. Or query the device mapping to get serial numbers
3. Update any hardcoded shared memory paths

## 🔗 Files Changed

- `pythermal/core/device.py` - Enhanced device mapping and serial number handling
- `pythermal/core/device_manager.py` - Serial-number-based ordering and CSV table support
- `pythermal/core/thermal_shared_memory.py` - Updated `get_shm_name()` to use serial numbers
- `hikvision-sdk-builder/demo/armLinux/UsbDemo.cpp` - Serial-number-based shared memory creation
- `hikvision-sdk-builder/demo/armLinux/UsbDemo.h` - Added serial number support

## 📚 Documentation

- Updated README.md with new features
- Added CHANGELOG.md for release tracking
- Enhanced multi-device documentation

## 🐛 Bug Fixes

- Fixed AttributeError: '_device_serial' not initialized
- Fixed AttributeError during cleanup when initialization fails
- Improved error handling in device enumeration

## 🙏 Thanks

Thank you for using PyThermal! This release makes multi-device setups more reliable and easier to manage.

