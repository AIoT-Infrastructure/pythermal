#!/usr/bin/env python3
"""
Custom setuptools command to build Sphinx documentation.
"""

import subprocess
import sys
from pathlib import Path
from setuptools import Command


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

