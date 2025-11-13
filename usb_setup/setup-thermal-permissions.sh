#!/bin/bash

# Thermal Camera Permission Setup Script
# This script sets up proper permissions for accessing thermal cameras without sudo

echo "🔧 Setting up thermal camera permissions..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Please run this script as a regular user, not with sudo"
    echo "   The script will ask for sudo when needed"
    exit 1
fi

# Get the current user
USER=$(whoami)
echo "👤 Setting up permissions for user: $USER"

# Step 1: Add user to plugdev group
echo "📝 Adding user to plugdev group..."
sudo usermod -a -G plugdev $USER

# Step 2: Copy udev rules to system directory
echo "📋 Installing udev rules..."
sudo cp 99-thermal-camera.rules /etc/udev/rules.d/

# Step 3: Set correct permissions on udev rules file
echo "🔐 Setting permissions on udev rules..."
sudo chmod 644 /etc/udev/rules.d/99-thermal-camera.rules

# Step 4: Reload udev rules
echo "🔄 Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

# Step 5: Show current USB devices
echo "📱 Current USB devices:"
lsusb | grep -i thermal || lsusb | grep -i hikvision || lsusb | grep -i hik || echo "   No thermal/hikvision/hik devices found in lsusb output"

# Step 6: Instructions for user
echo ""
echo "✅ Setup completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Disconnect and reconnect your thermal camera"
echo "   2. Log out and log back in (or restart your system)"
echo "   3. Run: ./thermal_recorder"
echo ""
echo "🔍 If it still doesn't work:"
echo "   1. Run 'lsusb' to find your thermal camera's vendor:product ID"
echo "   2. Edit /etc/udev/rules.d/99-thermal-camera.rules"
echo "   3. Replace the vendor/product IDs with your actual values"
echo "   4. Run: sudo udevadm control --reload-rules && sudo udevadm trigger"
echo ""
echo "📞 Common thermal camera vendor IDs:"
echo "   - 2bdf: Hikvision/HIK"
echo "   - 1f3a: Allwinner (some thermal cameras)"
echo "   - Check your specific device with: lsusb | grep -i hik" 