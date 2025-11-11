#!/bin/bash

# HK SDK Setup Script
# This script automates the installation process for the HK thermal camera SDK

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root. Please run as a regular user."
        exit 1
    fi
}

# Function to check if sudo is available
check_sudo() {
    if ! command -v sudo &> /dev/null; then
        print_error "sudo is required but not installed. Please install sudo first."
        exit 1
    fi
}

# Function to update package list
update_packages() {
    print_status "Updating package list..."
    sudo apt update
    print_success "Package list updated"
}

# Function to install cross-compiler
install_cross_compiler() {
    # Detect architecture
    ARCH=$(dpkg --print-architecture)
    
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        print_status "Running on ARM64 - cross-compiler not needed for native builds"
        print_success "Skipping cross-compiler installation"
        return 0
    elif [ "$ARCH" = "amd64" ] || [ "$ARCH" = "x86_64" ]; then
        print_status "Installing ARM cross-compiler for cross-compilation..."
        sudo apt install -y g++-arm-linux-gnueabihf
        print_success "ARM cross-compiler installed"
    else
        print_warning "Unknown architecture $ARCH - skipping cross-compiler installation"
        return 0
    fi
}

# Function to install FFmpeg and development libraries
install_ffmpeg() {
    print_status "Installing FFmpeg and development libraries..."
    
    # Detect architecture
    ARCH=$(dpkg --print-architecture)
    
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        # Native ARM64 installation
        print_status "Installing native ARM64 FFmpeg packages..."
        sudo apt install -y ffmpeg \
                            libavcodec-dev libavformat-dev libavutil-dev \
                            libswscale-dev libswresample-dev \
                            libavfilter-dev libavdevice-dev libpostproc-dev
    elif [ "$ARCH" = "amd64" ] || [ "$ARCH" = "x86_64" ]; then
        # Cross-compilation: install armhf packages
        print_status "Installing ARM cross-compilation FFmpeg packages..."
        
        # Enable multiarch if not already enabled
        if ! dpkg --print-foreign-architectures | grep -q armhf; then
            print_status "Enabling ARMHF multiarch support..."
            sudo dpkg --add-architecture armhf
            sudo apt update
        fi
        
        sudo apt install -y ffmpeg:armhf \
                            libavcodec-dev:armhf libavformat-dev:armhf libavutil-dev:armhf \
                            libswscale-dev:armhf libswresample-dev:armhf \
                            libavfilter-dev:armhf libavdevice-dev:armhf libpostproc-dev:armhf
    else
        # Native installation for other architectures
        print_status "Installing native FFmpeg packages for $ARCH..."
        sudo apt install -y ffmpeg \
                            libavcodec-dev libavformat-dev libavutil-dev \
                            libswscale-dev libswresample-dev \
                            libavfilter-dev libavdevice-dev libpostproc-dev
    fi
    
    print_success "FFmpeg and development libraries installed"
}

# Function to setup USB device permissions
setup_usb_permissions() {
    print_status "Setting up USB device permissions for thermal camera..."
    
    if [[ ! -f "setup-thermal-permissions.sh" ]]; then
        print_error "setup-thermal-permissions.sh not found in current directory"
        exit 1
    fi
    
    chmod +x setup-thermal-permissions.sh
    ./setup-thermal-permissions.sh
    
    print_success "USB device permissions configured"
    print_warning "You will need to disconnect and reconnect your thermal camera"
    print_warning "You should also log out and log back in (or restart) for permissions to take effect"
}

# Function to compile the thermal recorder
compile_thermal_recorder() {
    print_status "Compiling thermal recorder..."
    
    if [[ ! -d "demo/armLinux" ]]; then
        print_error "demo/armLinux directory not found"
        exit 1
    fi
    
    cd demo/armLinux
    make clean && make
    cd ../..
    
    print_success "Thermal recorder compiled successfully"
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    if [[ -f "library/armLinux/thermal_recorder" ]]; then
        print_success "thermal_recorder executable found"
    else
        print_error "thermal_recorder executable not found"
        exit 1
    fi
    
    # Check if executable has proper permissions
    if [[ -x "library/armLinux/thermal_recorder" ]]; then
        print_success "thermal_recorder is executable"
    else
        print_warning "Making thermal_recorder executable..."
        chmod +x library/armLinux/thermal_recorder
    fi
}

# Function to display final instructions
show_final_instructions() {
    echo ""
    echo "==============================================="
    print_success "HK SDK Setup Complete!"
    echo "==============================================="
    echo ""
    print_status "Next Steps:"
    echo "1. Disconnect and reconnect your thermal camera"
    echo "2. Log out and log back in (or restart your system)"
    echo "3. Navigate to the library/armLinux directory:"
    echo "   cd library/armLinux/"
    echo "4. Run the thermal recorder:"
    echo "   ./thermal_recorder"
    echo ""
    print_status "Example commands:"
    echo "• Record for 30 seconds: ./thermal_recorder -s 30"
    echo "• Record for 2 minutes:  ./thermal_recorder -m 2"
    echo "• Show help:             ./thermal_recorder -h"
    echo ""
    print_status "For troubleshooting, refer to the HK_SDK_INSTALL.md file"
}

# Main setup function
main() {
    echo "==============================================="
    echo "HK Thermal Camera SDK Setup Script"
    echo "==============================================="
    echo ""
    
    check_root
    check_sudo
    
    print_status "Starting HK SDK installation..."
    echo ""
    
    # Step 1: Update packages
    update_packages
    echo ""
    
    # Step 2: Install cross-compiler
    install_cross_compiler
    echo ""
    
    # Step 3: Install FFmpeg and libraries
    install_ffmpeg
    echo ""
    
    # Step 4: Setup USB permissions
    setup_usb_permissions
    echo ""
    
    # Step 5: Compile thermal recorder
    compile_thermal_recorder
    echo ""
    
    # Step 6: Verify installation
    verify_installation
    echo ""
    
    # Step 7: Show final instructions
    show_final_instructions
}

# Run main function
main "$@" 