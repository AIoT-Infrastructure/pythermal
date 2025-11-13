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

# Function to detect architecture and set directory names
detect_architecture() {
    # Detect architecture using multiple methods
    if command -v dpkg &> /dev/null; then
        ARCH=$(dpkg --print-architecture)
    elif command -v uname &> /dev/null; then
        ARCH=$(uname -m)
    else
        print_error "Cannot detect architecture. Please install dpkg or ensure uname is available."
        exit 1
    fi
    
    # Normalize architecture names
    case "$ARCH" in
        "amd64"|"x86_64")
            ARCH_DIR="linux64"
            IS_X86_64=true
            IS_ARM=false
            print_status "Detected x86_64 architecture - using linux64 directory"
            ;;
        "arm64"|"aarch64")
            ARCH_DIR="armLinux"
            IS_X86_64=false
            IS_ARM=true
            print_status "Detected ARM64 architecture - using armLinux directory"
            ;;
        "armhf"|"armv7l")
            ARCH_DIR="armLinux"
            IS_X86_64=false
            IS_ARM=true
            print_status "Detected ARM architecture - using armLinux directory"
            ;;
        *)
            print_warning "Unknown architecture: $ARCH - defaulting to armLinux"
            ARCH_DIR="armLinux"
            IS_X86_64=false
            IS_ARM=true
            ;;
    esac
    
    export ARCH_DIR
    export IS_X86_64
    export IS_ARM
}

# Function to update package list
update_packages() {
    print_status "Updating package list..."
    sudo apt update
    print_success "Package list updated"
}

# Function to install cross-compiler
install_cross_compiler() {
    if [ "$IS_ARM" = true ]; then
        print_status "Running on ARM - cross-compiler not needed for native builds"
        print_success "Skipping cross-compiler installation"
        return 0
    elif [ "$IS_X86_64" = true ]; then
        print_status "Running on x86_64 - cross-compiler not needed for native builds"
        print_success "Skipping cross-compiler installation"
        return 0
    else
        print_warning "Unknown architecture - skipping cross-compiler installation"
        return 0
    fi
}

# Function to install FFmpeg and development libraries
install_ffmpeg() {
    print_status "Installing FFmpeg and development libraries..."
    
    if [ "$IS_ARM" = true ] || [ "$IS_X86_64" = true ]; then
        # Native installation for detected architectures
        if [ "$IS_X86_64" = true ]; then
            print_status "Installing native x86_64 FFmpeg packages..."
        else
            print_status "Installing native ARM FFmpeg packages..."
        fi
        
        sudo apt install -y ffmpeg \
                            libavcodec-dev libavformat-dev libavutil-dev \
                            libswscale-dev libswresample-dev \
                            libavfilter-dev libavdevice-dev libpostproc-dev
    else
        # Native installation for other architectures
        print_status "Installing native FFmpeg packages..."
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

# Function to display final instructions
show_final_instructions() {
    echo ""
    echo "==============================================="
    print_success "HK SDK Setup Complete!"
    echo "==============================================="
    echo ""
    print_status "Architecture: $ARCH_DIR"
    echo ""
    print_status "Next Steps:"
    echo "1. Disconnect and reconnect your thermal camera"
    echo "2. Log out and log back in (or restart your system)"
    echo "3. You can now use the PyThermal library with the native binaries"
    echo ""
    print_status "The native binaries are located in: pythermal/_native/$ARCH_DIR/"
    echo ""
    print_status "For troubleshooting, refer to the README.md file"
}

# Main setup function
main() {
    echo "==============================================="
    echo "HK Thermal Camera SDK Setup Script"
    echo "==============================================="
    echo ""
    
    check_root
    check_sudo
    
    # Detect architecture first
    detect_architecture
    echo ""
    
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
    
    # Step 5: Show final instructions
    show_final_instructions
}

# Run main function
main "$@" 