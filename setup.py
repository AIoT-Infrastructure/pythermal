#!/usr/bin/env python3
"""
Setup script for pythermal library
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Get version from package
version = "0.1.3"

# Find all native binaries to include
package_dir = Path(__file__).parent / "pythermal" / "_native" / "armLinux"
native_files = []
if package_dir.exists():
    for file in package_dir.iterdir():
        if file.is_file():
            native_files.append(str(file.relative_to(Path(__file__).parent)))

setup(
    name="pythermal",
    version=version,
    description="A lightweight Python library for thermal sensing and analytics on ARM Linux platforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ThermalCare Team",
    author_email="yunqiguo@cuhk.edu.hk",
    url="https://github.com/AIoT-Infrastructure/pythermal",
    packages=find_packages(),
    package_data={
        "pythermal": [
            "_native/armLinux/*",
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
    },
    entry_points={
        "console_scripts": [
            "pythermal-preview=pythermal.live_view:main",
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
)

