# Auto-build Documentation

This project automatically builds Sphinx documentation as part of the build process.

## What Gets Auto-Generated

Sphinx can automatically generate API documentation from your Python docstrings using the `autodoc` extension:

✅ **Auto-generated from code:**
- API reference for all modules, classes, and functions
- Method signatures and parameters
- Class hierarchies and inheritance
- Module-level documentation

⚠️ **Requires manual updates:**
- Quick start guide (`quickstart.rst`)
- Installation instructions (`installation.rst`)
- Examples documentation (`examples.rst`)
- Architecture documentation (`architecture.rst`)
- Custom narrative documentation

## Automatic Documentation Building

### Option 1: Auto-build script with watch mode (Recommended for development)

```bash
# Build once
python docs/auto_build_docs.py

# Clean and rebuild
python docs/auto_build_docs.py --clean

# Watch for changes and auto-rebuild (requires watchdog)
pip install watchdog
python docs/auto_build_docs.py --watch
```

The watch mode monitors:
- `pythermal/` - Source code changes (auto-updates API docs)
- `docs/source/` - Documentation changes (rebuilds docs)

### Option 2: Use the build script (Recommended for releases)

```bash
python build_with_docs.py
```

This will:
1. Build the Sphinx documentation
2. Build the package with `uv build`

### Option 3: Manual build command

You can also build docs separately using setuptools:

```bash
python setup.py build_docs
```

Or directly:

```bash
cd docs && make html
```

### Option 4: GitHub Actions

The GitHub Actions workflow automatically builds documentation before building the package.

## Integration with uv

To automatically build docs when using `uv build`, you can:

1. **Use the wrapper script:**
   ```bash
   python build_with_docs.py
   ```

2. **Or create an alias:**
   ```bash
   alias uv-build='python build_with_docs.py'
   ```

3. **Or manually build docs first:**
   ```bash
   cd docs && make html && cd .. && uv build
   ```

## Development Workflow

For active documentation development:

1. **Start watch mode:**
   ```bash
   python docs/auto_build_docs.py --watch
   ```

2. **Make changes:**
   - Edit Python files → API docs auto-update
   - Edit `.rst` files → Documentation rebuilds

3. **View results:**
   - Open `docs/build/html/index.html` in your browser
   - Refresh to see latest changes

## Documentation Output

Built documentation is available at:
- Local: `docs/build/html/index.html`
- Can be deployed to GitHub Pages or Read the Docs

## Keeping API Docs Updated

The API documentation (`docs/source/api/*.rst`) uses `autodoc` to automatically extract documentation from your code. To ensure API docs stay updated:

1. **Write docstrings** in your Python code following Google or NumPy style
2. **Use type hints** - they'll appear in the generated docs
3. **Run the build** - Sphinx will automatically extract everything

Example docstring format:

```python
def detect_objects(temp_array, min_temp, max_temp):
    """Detect objects based on temperature range.
    
    Args:
        temp_array: Temperature array (96x96, uint16)
        min_temp: Minimum temperature threshold
        max_temp: Maximum temperature threshold
    
    Returns:
        List of DetectedObject instances
    """
    ...
```

This will automatically appear in the API documentation!

