#!/usr/bin/env python3
"""
Setup script for pythermal library
"""

from setuptools import setup, find_packages
from pathlib import Path

# Import custom commands
try:
    from pythermal.commands import BuildDocsCommand, InstallWithUSBSetup
except ImportError:
    # If commands module doesn't exist yet, create dummy commands
    from setuptools import Command
    from setuptools.command.install import install
    
    class BuildDocsCommand(Command):
        description = 'Build Sphinx documentation'
        user_options = []
        def initialize_options(self): pass
        def finalize_options(self): pass
        def run(self): pass
    
    class InstallWithUSBSetup(install):
        description = 'Install the package and set up USB device permissions'
        def run(self): install.run(self)

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Get version from package __init__.py
init_file = Path(__file__).parent / "pythermal" / "__init__.py"
if not init_file.exists():
    raise FileNotFoundError(f"Could not find {init_file}")

version = None
for line in init_file.read_text().splitlines():
    if line.startswith("__version__"):
        version = line.split("=")[1].strip().strip('"').strip("'")
        break

if version is None:
    raise ValueError(f"Could not find __version__ in {init_file}")

# Find all native binaries to include (both architectures)
native_base = Path(__file__).parent / "pythermal" / "_native"
native_files = []

# Include files from both linux64 and armLinux directories
for arch_dir in ["linux64", "armLinux"]:
    arch_path = native_base / arch_dir
    if arch_path.exists():
        for file in arch_path.iterdir():
            if file.is_file():
                native_files.append(str(file.relative_to(Path(__file__).parent)))

setup(
    name="pythermal",
    version=version,
    description="A lightweight Python library for thermal sensing and analytics on Linux platforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ThermalCare Team",
    author_email="yunqiguo@cuhk.edu.hk",
    url="https://github.com/AIoT-Infrastructure/pythermal",
    project_urls={
        "Documentation": "https://aiot-infrastructure.github.io/pythermal/",
        "Source": "https://github.com/AIoT-Infrastructure/pythermal",
        "Bug Tracker": "https://github.com/AIoT-Infrastructure/pythermal/issues",
    },
    packages=find_packages(),
    package_data={
        "pythermal": [
            "_native/linux64/*",
            "_native/armLinux/*",
            "usb_setup/*",
        ],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "yolo": [
            # YOLO v11 detection support (object and pose detection)
            # Install with: pip install pythermal[yolo]
            "ultralytics>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pythermal-preview=pythermal.live_view:main",
            "pythermal-setup-usb=pythermal.commands:setup_usb_permissions_cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    zip_safe=False,  # Required for native binaries
    cmdclass={
        'build_docs': BuildDocsCommand,
        'install': InstallWithUSBSetup,
    },
)

