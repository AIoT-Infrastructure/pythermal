#!/usr/bin/env python3
"""
Build script that builds documentation before building the package.

Usage:
    python build_with_docs.py
    uv build  # Will use this script if configured
"""

import subprocess
import sys
from pathlib import Path


def build_docs():
    """Build Sphinx documentation."""
    project_root = Path(__file__).parent
    docs_dir = project_root / 'docs'
    
    if not docs_dir.exists():
        print("Warning: docs directory not found, skipping documentation build")
        return True
    
    print("Building Sphinx documentation...")
    try:
        result = subprocess.run(
            ['make', 'html'],
            cwd=str(docs_dir),
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Documentation built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False
    except FileNotFoundError:
        print("Warning: 'make' command not found. Skipping documentation build.")
        print("  Install Sphinx and build manually: cd docs && make html")
        return True  # Don't fail the build if make is missing


def main():
    """Main function."""
    # Build docs first
    if not build_docs():
        print("Documentation build failed, but continuing with package build...")
    
    # Then build the package
    print("\nBuilding package with uv...")
    try:
        subprocess.run(['uv', 'build'], check=True)
        print("✓ Package built successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error building package: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'uv' command not found. Please install uv.")
        sys.exit(1)


if __name__ == '__main__':
    main()

