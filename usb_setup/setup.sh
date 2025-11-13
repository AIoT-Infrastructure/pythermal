#!/bin/bash

# Thermal Camera USB Setup Script
# This script sets up USB device permissions for the HK thermal camera

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
    print_success "Thermal Camera USB Setup Complete!"
    echo "==============================================="
    echo ""
    print_status "Next Steps:"
    echo "1. Disconnect and reconnect your thermal camera"
    echo "2. Log out and log back in (or restart your system)"
    echo ""
    print_status "After reconnecting, you should be able to access the thermal camera without sudo"
}

# Main setup function
main() {
    echo "==============================================="
    echo "Thermal Camera USB Setup Script"
    echo "==============================================="
    echo ""
    
    check_root
    check_sudo
    
    print_status "Starting thermal camera USB setup..."
    echo ""
    
    # Setup USB permissions
    setup_usb_permissions
    echo ""
    
    # Show final instructions
    show_final_instructions
}

# Run main function
main "$@" 