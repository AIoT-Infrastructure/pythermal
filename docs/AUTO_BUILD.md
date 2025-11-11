# Auto-build Documentation

This project automatically builds Sphinx documentation as part of the build process.

## Automatic Documentation Building

### Option 1: Use the build script (Recommended)

```bash
python build_with_docs.py
```

This will:
1. Build the Sphinx documentation
2. Build the package with `uv build`

### Option 2: Manual build command

You can also build docs separately using setuptools:

```bash
python setup.py build_docs
```

Or directly:

```bash
cd docs && make html
```

### Option 3: GitHub Actions

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

## Documentation Output

Built documentation is available at:
- Local: `docs/build/html/index.html`
- Can be deployed to GitHub Pages or Read the Docs

