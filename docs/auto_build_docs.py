#!/usr/bin/env python3
"""
Automatic documentation builder and watcher for PyThermal.

This script can:
1. Build documentation once
2. Watch for changes and rebuild automatically
3. Clean and rebuild documentation

Usage:
    python docs/auto_build_docs.py          # Build once
    python docs/auto_build_docs.py --watch   # Watch for changes
    python docs/auto_build_docs.py --clean  # Clean and rebuild
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Optional watchdog imports (only needed for watch mode)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    # Create dummy classes when watchdog is not available
    FileSystemEventHandler = object
    Observer = None


if WATCHDOG_AVAILABLE:
    class DocsRebuildHandler(FileSystemEventHandler):
        """Handler that rebuilds docs when files change."""
        
        def __init__(self, docs_dir):
            self.docs_dir = docs_dir
            self.last_build = 0
            self.debounce_seconds = 2  # Wait 2 seconds after last change before building
            
        def on_modified(self, event):
            """Handle file modification events."""
            if event.is_directory:
                return
            
            # Only watch Python and RST files
            if not (event.src_path.endswith('.py') or event.src_path.endswith('.rst')):
                return
            
            # Debounce: only rebuild if enough time has passed since last change
            current_time = time.time()
            if current_time - self.last_build < self.debounce_seconds:
                return
            
            self.last_build = current_time
            print(f"\n📝 Detected change in: {event.src_path}")
            print("🔄 Rebuilding documentation...")
            self.rebuild_docs()
        
        def rebuild_docs(self):
            """Rebuild the documentation."""
            try:
                result = subprocess.run(
                    ['make', 'html'],
                    cwd=str(self.docs_dir),
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✅ Documentation rebuilt successfully!")
                print(f"   View at: {self.docs_dir / 'build' / 'html' / 'index.html'}\n")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error rebuilding documentation:")
                if e.stderr:
                    print(e.stderr)
                print()
else:
    # Dummy class when watchdog is not available
    class DocsRebuildHandler:
        """Dummy handler when watchdog is not available."""
        pass


def build_docs(docs_dir, clean=False):
    """Build Sphinx documentation.
    
    Args:
        docs_dir: Path to docs directory
        clean: If True, clean before building
    """
    if not docs_dir.exists():
        print(f"❌ Error: docs directory not found: {docs_dir}")
        return False
    
    if clean:
        print("🧹 Cleaning previous build...")
        try:
            subprocess.run(
                ['make', 'clean'],
                cwd=str(docs_dir),
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            pass  # Ignore errors if clean fails
    
    print("📚 Building Sphinx documentation...")
    try:
        result = subprocess.run(
            ['make', 'html'],
            cwd=str(docs_dir),
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Documentation built successfully!")
        print(f"   Output: {docs_dir / 'build' / 'html' / 'index.html'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building documentation:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Error: 'make' command not found.")
        print("   Please install Sphinx: pip install \"sphinx<7.0\" sphinx-rtd-theme sphinxcontrib-napoleon")
        return False


def watch_docs(project_root, docs_dir):
    """Watch for file changes and rebuild docs automatically.
    
    Args:
        project_root: Path to project root
        docs_dir: Path to docs directory
    """
    print("👀 Watching for changes...")
    print("   Press Ctrl+C to stop\n")
    
    # Watch both source code and documentation
    watch_paths = [
        project_root / 'pythermal',  # Source code
        docs_dir / 'source',         # Documentation source
    ]
    
    handler = DocsRebuildHandler(docs_dir)
    observer = Observer()
    
    for path in watch_paths:
        if path.exists():
            observer.schedule(handler, str(path), recursive=True)
            print(f"   Watching: {path}")
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping watcher...")
        observer.stop()
    
    observer.join()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build and watch PyThermal documentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build documentation once
  python docs/auto_build_docs.py

  # Clean and rebuild
  python docs/auto_build_docs.py --clean

  # Watch for changes and auto-rebuild
  python docs/auto_build_docs.py --watch

Note: For watch mode, install watchdog:
  pip install watchdog
        """
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch for file changes and rebuild automatically'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean previous build before building'
    )
    
    args = parser.parse_args()
    
    # Get paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    docs_dir = project_root / 'docs'
    
    if args.watch:
        # Check if watchdog is available
        if not WATCHDOG_AVAILABLE:
            print("❌ Error: 'watchdog' package is required for watch mode.")
            print("   Install with: pip install watchdog")
            sys.exit(1)
        
        # Build once first
        build_docs(docs_dir, clean=args.clean)
        
        # Then start watching
        watch_docs(project_root, docs_dir)
    else:
        # Just build once
        success = build_docs(docs_dir, clean=args.clean)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

