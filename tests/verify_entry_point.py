#!/usr/bin/env python3
"""
Verify entry point configuration for pythermal-preview command
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Try to import the main function
    from pythermal.live_view import main
    
    print("=" * 60)
    print("Entry Point Verification")
    print("=" * 60)
    print()
    print("✓ pythermal.live_view.main is importable")
    print(f"✓ Function type: {type(main)}")
    print(f"✓ Function name: {main.__name__}")
    
    # Check if it's callable
    if callable(main):
        print("✓ Function is callable")
    else:
        print("✗ Function is not callable")
        sys.exit(1)
    
    # Check setup.py entry point configuration
    setup_file = Path(__file__).parent / "setup.py"
    if setup_file.exists():
        content = setup_file.read_text()
        if "pythermal-preview=pythermal.live_view:main" in content:
            print("✓ Entry point configured in setup.py")
            print("  Command: pythermal-preview")
            print("  Module: pythermal.live_view")
            print("  Function: main")
        else:
            print("✗ Entry point not found in setup.py")
            sys.exit(1)
    
    print()
    print("=" * 60)
    print("✓ All entry point checks passed!")
    print("=" * 60)
    print()
    print("After installation, you can use:")
    print("  pythermal-preview")
    print()
    print("Note: Requires X11 display for GUI")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

