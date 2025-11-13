#!/usr/bin/env python3
"""
Custom setuptools commands for pythermal.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from setuptools import Command
from setuptools.command.install import install


class BuildDocsCommand(Command):
    """Custom command to build Sphinx documentation."""
    
    description = 'Build Sphinx documentation'
    user_options = []
    
    def initialize_options(self):
        """Set default values for options."""
        pass
    
    def finalize_options(self):
        """Finalize options."""
        pass
    
    def run(self):
        """Run the documentation build."""
        project_root = Path(__file__).parent.parent
        docs_dir = project_root / 'docs'
        
        if not docs_dir.exists():
            print("Warning: docs directory not found, skipping documentation build")
            return
        
        print("Building Sphinx documentation...")
        try:
            # Change to docs directory and run make html
            result = subprocess.run(
                ['make', 'html'],
                cwd=str(docs_dir),
                check=True,
                capture_output=True,
                text=True
            )
            print("✓ Documentation built successfully")
            print(f"  Output: {docs_dir / 'build' / 'html' / 'index.html'}")
        except subprocess.CalledProcessError as e:
            print(f"Error building documentation: {e}")
            print(e.stdout)
            print(e.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("Error: 'make' command not found. Please install make or build docs manually.")
            print("  Run: cd docs && make html")
            sys.exit(1)


def setup_usb_permissions(project_root=None):
    """Set up USB device permissions for thermal camera.
    
    Args:
        project_root: Optional Path to the project root directory. If not provided,
                     will attempt to find it automatically.
    """
    print("\n" + "="*50)
    print("Setting up USB device permissions for thermal camera...")
    print("="*50)
    
    # Get paths - try to find usb_setup directory
    # Priority: 1) Inside package (for installed packages), 2) Project root (for editable installs)
    usb_setup_dir = None
    udev_rules_file = None
    
    if project_root is None:
        # Strategy 1: Look inside the package directory (for installed packages)
        package_dir = Path(__file__).parent
        package_usb_setup = package_dir / 'usb_setup' / '99-thermal-camera.rules'
        if package_usb_setup.exists():
            usb_setup_dir = package_dir / 'usb_setup'
            udev_rules_file = package_usb_setup
        
        # Strategy 2: Look for setup.py in parent directories (for editable installs)
        if usb_setup_dir is None:
            current_path = Path(__file__).resolve()
            for parent in current_path.parents:
                project_usb_setup = parent / 'usb_setup' / '99-thermal-camera.rules'
                if (parent / 'setup.py').exists() and project_usb_setup.exists():
                    usb_setup_dir = parent / 'usb_setup'
                    udev_rules_file = project_usb_setup
                    project_root = parent
                    break
        
        # Strategy 3: Try current working directory (for development)
        if usb_setup_dir is None:
            cwd = Path.cwd()
            project_usb_setup = cwd / 'usb_setup' / '99-thermal-camera.rules'
            if (cwd / 'setup.py').exists() and project_usb_setup.exists():
                usb_setup_dir = cwd / 'usb_setup'
                udev_rules_file = project_usb_setup
                project_root = cwd
        
        # Strategy 4: Try inside package directory (alternative path for non-editable installs)
        if usb_setup_dir is None:
            package_dir = Path(__file__).parent
            package_usb_setup = package_dir / 'usb_setup' / '99-thermal-camera.rules'
            if package_usb_setup.exists():
                usb_setup_dir = package_dir / 'usb_setup'
                udev_rules_file = package_usb_setup
    else:
        # Project root provided, check both locations
        project_root = Path(project_root)
        # Try project root first (editable installs)
        project_usb_setup = project_root / 'usb_setup' / '99-thermal-camera.rules'
        if project_usb_setup.exists():
            usb_setup_dir = project_root / 'usb_setup'
            udev_rules_file = project_usb_setup
        else:
            # Try package directory
            package_dir = Path(__file__).parent
            package_usb_setup = package_dir / 'usb_setup' / '99-thermal-camera.rules'
            if package_usb_setup.exists():
                usb_setup_dir = package_dir / 'usb_setup'
                udev_rules_file = package_usb_setup
    
    if usb_setup_dir is None or udev_rules_file is None or not udev_rules_file.exists():
        print("Warning: Could not find usb_setup directory with 99-thermal-camera.rules.")
        print("Please ensure the package is installed correctly or run from the project directory.")
        return
    
    # Check if running as root (which is common with pip install)
    is_root = os.geteuid() == 0
    
    try:
        # Step 1: Copy udev rules to system directory
        print("\n📋 Installing udev rules...")
        target_rules = Path('/etc/udev/rules.d/99-thermal-camera.rules')
        
        if is_root:
            # Running as root, can copy directly
            shutil.copy2(udev_rules_file, target_rules)
            os.chmod(target_rules, 0o644)
        else:
            # Need sudo - ensure stdin is connected for password prompt
            subprocess.run(
                ['sudo', 'cp', str(udev_rules_file), str(target_rules)],
                check=True,
                stdin=sys.stdin
            )
            subprocess.run(
                ['sudo', 'chmod', '644', str(target_rules)],
                check=True,
                stdin=sys.stdin
            )
        
        print("✓ udev rules installed")
        
        # Step 2: Add user to plugdev group
        if not is_root:
            print("\n📝 Adding user to plugdev group...")
            username = os.environ.get('SUDO_USER') or os.environ.get('USER') or os.getlogin()
            subprocess.run(
                ['sudo', 'usermod', '-a', '-G', 'plugdev', username],
                check=True,
                stdin=sys.stdin
            )
            print(f"✓ User {username} added to plugdev group")
        else:
            # If running as root, get the original user from SUDO_USER
            username = os.environ.get('SUDO_USER')
            if username:
                print(f"\n📝 Adding user {username} to plugdev group...")
                subprocess.run(
                    ['usermod', '-a', '-G', 'plugdev', username],
                    check=True
                )
                print(f"✓ User {username} added to plugdev group")
            else:
                print("\n⚠️  Running as root without SUDO_USER set.")
                print("   Please manually add your user to plugdev group:")
                print("   sudo usermod -a -G plugdev $USER")
        
        # Step 3: Reload udev rules
        print("\n🔄 Reloading udev rules...")
        if is_root:
            subprocess.run(['udevadm', 'control', '--reload-rules'], check=True)
            subprocess.run(['udevadm', 'trigger'], check=True)
        else:
            subprocess.run(['sudo', 'udevadm', 'control', '--reload-rules'], check=True, stdin=sys.stdin)
            subprocess.run(['sudo', 'udevadm', 'trigger'], check=True, stdin=sys.stdin)
        
        print("✓ udev rules reloaded")
        
        print("\n" + "="*50)
        print("✅ USB device permissions setup complete!")
        print("="*50)
        print("\n📋 Next steps:")
        print("   1. Disconnect and reconnect your thermal camera")
        print("   2. Log out and log back in (or restart your system)")
        print("   3. You should now be able to access the thermal camera without sudo")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error setting up USB permissions: {e}")
        print("You can run 'pythermal-setup-usb' manually to set up USB permissions.")
        sys.exit(1)
    except PermissionError:
        print("\n❌ Permission denied. Please run with sudo:")
        print("   sudo pip install -e .")
        print("Or run the USB setup manually:")
        print("   sudo pythermal-setup-usb")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("You can run 'pythermal-setup-usb' manually to set up USB permissions.")
        sys.exit(1)


class InstallWithUSBSetup(install):
    """Custom install command that sets up USB permissions after installation."""
    
    description = 'Install the package and set up USB device permissions'
    
    def run(self):
        """Run the standard install, then set up USB permissions."""
        # Run the standard install first
        install.run(self)
        
        # Set up USB permissions
        # For editable installs, the source directory is available
        # Try to get it from the distribution metadata
        project_root = None
        try:
            # Get the source directory from the build_py command
            build_py = self.get_finalized_command('build_py')
            if hasattr(build_py, 'package_dir') and build_py.package_dir:
                # For editable installs, package_dir[''] points to the source root
                source_root = build_py.package_dir.get('', '.')
                project_root = Path(source_root).resolve()
                if not (project_root / 'setup.py').exists():
                    project_root = None
        except Exception:
            pass
        
        setup_usb_permissions(project_root=project_root)


def setup_usb_permissions_cli():
    """CLI entry point for manual USB setup."""
    setup_usb_permissions()

