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

## GitHub Pages Deployment

The documentation is automatically built and deployed to GitHub Pages when:
- Changes are pushed to `main` or `master` branch
- Changes are made to files in `docs/` or `pythermal/` directories
- The workflow is manually triggered via GitHub Actions

To enable GitHub Pages:
1. Go to your repository Settings → Pages
2. Under "Source", select "GitHub Actions"
3. The workflow will automatically deploy on the next push

The documentation will be available at: **https://aiot-infrastructure.github.io/pythermal/**

Repository: [https://github.com/AIoT-Infrastructure/pythermal/](https://github.com/AIoT-Infrastructure/pythermal/)

The GitHub Actions workflow is configured in `.github/workflows/docs.yml`.

