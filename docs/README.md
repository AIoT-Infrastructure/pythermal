# PyThermal Documentation

This directory contains the Sphinx documentation for PyThermal.

## Building the Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The generated HTML files will be in `docs/build/html/`.

## Viewing the Documentation

After building, open `docs/build/html/index.html` in your web browser:

```bash
# On Linux/Mac
xdg-open docs/build/html/index.html
# or
open docs/build/html/index.html  # Mac
```

## Documentation Structure

- `source/index.rst` - Main documentation index
- `source/installation.rst` - Installation instructions
- `source/quickstart.rst` - Quick start guide
- `source/api/` - API reference documentation
- `source/examples.rst` - Examples documentation
- `source/architecture.rst` - Architecture documentation
- `YOLO_DETECTION.md` - Comprehensive YOLO v11 detection guide (markdown format)

## Requirements

Documentation build requires:

- sphinx
- sphinx-rtd-theme
- sphinxcontrib-napoleon

Install with:

```bash
pip install sphinx sphinx-rtd-theme sphinxcontrib-napoleon
```

## Rebuilding

To rebuild the documentation after making changes:

```bash
cd docs
make clean
make html
```

