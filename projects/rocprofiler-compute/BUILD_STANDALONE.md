# Building rocprof-compute Standalone Binary with PyOxidizer

This document describes how to build a standalone binary for rocprof-compute using PyOxidizer.

## Overview

The rocprof-compute project can be built as a standalone binary that bundles Python and all dependencies into a single executable. This was previously done using Nuitka, but has been migrated to PyOxidizer for better performance and maintainability.

## Prerequisites

### Install PyOxidizer

PyOxidizer can be installed via pip or cargo:

```bash
# Using pip (recommended)
pip install pyoxidizer

# Or using cargo (Rust package manager)
cargo install pyoxidizer
```

### System Requirements

- Python 3.9 or higher
- Git (for version tracking)
- CMake 3.19 or higher
- All dependencies listed in `requirements.txt`

## Building the Standalone Binary

### Using CMake (Recommended)

The easiest way to build the standalone binary is through the CMake build system:

```bash
# Create a build directory
mkdir build && cd build

# Configure the project
cmake ..

# Build the standalone binary
cmake --build . --target standalonebinary

# The binary will be created at: build/rocprof-compute.bin
```

### Using PyOxidizer Directly

You can also build directly with PyOxidizer:

```bash
# From the project root directory
pyoxidizer build --release

# The binary will be in: build/x86_64-unknown-linux-gnu/release/install/rocprof-compute
```

## Configuration

The PyOxidizer configuration is defined in `pyoxidizer.bzl` at the project root. This file specifies:

- **Entry Point**: The main `rocprof-compute` script
- **Dependencies**: All packages from `requirements.txt`
- **Resources**: Configuration files, analysis configs, and data files
- **Packaging Policy**: How Python modules and resources are embedded

### Key Configuration Options

The configuration includes:

1. **In-Memory Resources**: Most Python modules are loaded from memory for faster startup
2. **Filesystem Fallback**: Large data files and configurations are stored on the filesystem
3. **Optimization Level**: Set to level 2 for maximum performance
4. **Extension Modules**: All extension modules are included

## Differences from Nuitka

### Advantages of PyOxidizer

1. **Better Performance**: Faster startup times and runtime performance
2. **Simpler Configuration**: Declarative configuration in Starlark (Python-like syntax)
3. **Active Development**: PyOxidizer is actively maintained
4. **Better Resource Handling**: More flexible resource embedding options
5. **No patchelf Required**: PyOxidizer handles RPATH automatically

### Migration Notes

The PyOxidizer build:
- Creates the same output binary name: `rocprof-compute.bin`
- Includes all the same packages and dependencies
- Maintains compatibility with the existing installation structure
- No longer requires `patchelf` for RPATH manipulation

## Troubleshooting

### PyOxidizer Not Found

If you get an error that PyOxidizer is not found:

```bash
pip install pyoxidizer
# Or ensure ~/.local/bin is in your PATH
export PATH="$HOME/.local/bin:$PATH"
```

### Missing Dependencies

Ensure all Python dependencies are installed:

```bash
pip install -r requirements.txt
```

### Build Failures

If the build fails, try:

1. Clean the build directory:
   ```bash
   rm -rf build/
   pyoxidizer build --release
   ```

2. Check PyOxidizer version:
   ```bash
   pyoxidizer --version
   # Should be 0.24.0 or higher
   ```

3. Verify Python version:
   ```bash
   python3 --version
   # Should be 3.9 or higher
   ```

## Testing the Binary

After building, test the standalone binary:

```bash
# Run the binary
./build/rocprof-compute.bin --help

# Test profile mode
./build/rocprof-compute.bin profile --help

# Test analyze mode
./build/rocprof-compute.bin analyze --help
```

## Installation

The standalone binary can be installed manually:

```bash
# Copy to installation directory
sudo cp build/rocprof-compute.bin /opt/rocm/bin/rocprof-compute

# Make executable
sudo chmod +x /opt/rocm/bin/rocprof-compute
```

Or use the CMake install target (which installs the regular Python version, not the standalone binary).

## Additional Resources

- [PyOxidizer Documentation](https://pyoxidizer.readthedocs.io/)
- [PyOxidizer GitHub](https://github.com/indygreg/PyOxidizer)
- [Starlark Language Reference](https://github.com/bazelbuild/starlark)

## Support

For issues related to:
- **PyOxidizer build**: Check the PyOxidizer documentation
- **rocprof-compute functionality**: File an issue in the rocprof-compute repository
- **Missing dependencies**: Verify `requirements.txt` is up to date
