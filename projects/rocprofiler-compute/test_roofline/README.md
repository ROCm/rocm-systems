# Test Roofline C++ Standalone Bundle

This directory contains a standalone version of test_roofline that bundles Python runtime, standard library, and all dependencies. It can run on systems without Python installed.

## Building

**IMPORTANT:** To create a portable bundle, you must build with your system Python (not a virtualenv). This ensures the bundled stdlib doesn't contain venv-specific paths.

Find your system Python:
```bash
which -a python3  # Look for /usr/bin/python3 or similar (NOT a venv path)
```

Build the bundle:
```bash
mkdir build && cd build
cmake -DPython3_EXECUTABLE=/usr/bin/python3 ..
make
```

Replace `/usr/bin/python3` with your actual system Python path if different.

This creates a self-contained bundle (~339MB) in the build directory with:
- `test_roofline_cpp` - Executable with embedded RPATH (102KB)
- `libpython3.10.so.1.0` - Bundled Python library (5.6MB)
- `lib/python3.10/` - Python standard library (~47MB)
- `lib/python3.10/site-packages/` - All Python dependencies (287MB)
  - astunparse, numpy, pandas, plotly, plotext, plotille, dash, sqlalchemy, pyyaml, tabulate, textual
- `test_roofline.py` - Roofline analysis module (7KB)

The bundle includes ALL necessary dependencies to run the roofline analysis without requiring a system Python installation.

## Running

```bash
cd build
./test_roofline_cpp
```

## Distributing

Create a tarball of the build directory:

```bash
cd /path/to/test_roofline
tar czf test_roofline_bundle.tar.gz -C build .
```

On target system (no Python required):

```bash
tar xzf test_roofline_bundle.tar.gz
./test_roofline_cpp
```

## How It Works

- **RPATH=$ORIGIN**: Executable looks for libpython.so in its own directory
- **Py_SetPythonHome()**: Sets Python home to executable directory
- **Py_SetPath()**: Adds site-packages to Python module search path
- **Bundled stdlib & packages**: Everything Python needs is copied locally

## Verification

Check RPATH:
```bash
readelf -d test_roofline_cpp | grep RPATH
```

Check library dependencies:
```bash
ldd test_roofline_cpp
```

## Requirements

Build requirements:
- CMake 3.21+
- g++ with C++17
- System Python 3.x (NOT a virtualenv) with development headers
- pip (for installing dependencies during build)

Runtime requirements (on target system):
- None! The bundle is fully self-contained

## Troubleshooting

### Error: Module imports fail with references to venv paths

If you see errors like `ModuleNotFoundError` with paths containing `/venv/` or other virtualenv directories in the error message, you built with the wrong Python.

**Solution:** Clean and rebuild with system Python:
```bash
cd build
rm -rf *
cmake -DPython3_EXECUTABLE=/usr/bin/python3 ..
make
```

### How to find your system Python

```bash
# List all python3 installations
which -a python3

# Check if a Python is in a venv
/path/to/python3 -c "import sys; print(sys.prefix)"
# If output contains 'venv', 'virtualenv', '.env', etc. - it's NOT system Python
# System Python typically shows: /usr, /usr/local, or similar
```

## How RPATH Works

- `RPATH=$ORIGIN` means "look for shared libraries in the same directory as the executable"
- `$ORIGIN` is a special token that the dynamic linker expands to the executable's directory
- This is the industry standard way to bundle libraries with executables
- No wrapper scripts or LD_LIBRARY_PATH hacks needed
- The executable is fully self-contained

## Why This Works

1. **Compile time**: CMake sets RPATH=$ORIGIN in the executable
2. **Build time**:
   - CMake copies libpython.so next to executable
   - CMake copies Python stdlib to lib/python3.10/
   - pip installs pandas/numpy to lib/python3.10/site-packages/
3. **Run time**:
   - Dynamic linker uses RPATH to find libpython.so in same directory
   - Our code sets Py_SetPythonHome() and Py_SetPath() before Py_Initialize()
   - Python finds stdlib in $PYTHONHOME/lib/python3.10/
   - Python finds packages in site-packages via Py_SetPath()

Everything Just Works™

---

# Windows Build (Embeddable Python Package)

For Windows, use Python's **Embeddable Package** - a pre-configured portable distribution designed for embedding.

## Quick Start

```powershell
# 1. Download embeddable Python from python.org
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip" -OutFile "python-embed.zip"
Expand-Archive python-embed.zip -DestinationPath python-embed

# 2. Bootstrap pip (not included by default)
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
python-embed\python.exe get-pip.py

# 3. Install dependencies
python-embed\python.exe -m pip install -r requirements.txt

# 4. Build
mkdir build && cd build
cmake -G "Visual Studio 16 2019" -A x64 -DPython3_ROOT_DIR=..\python-embed ..
cmake --build . --config Release
```

## What Gets Bundled

```
build/Release/
├── test_roofline_cpp.exe
├── python310.dll              # Python runtime
├── python310.zip              # Compressed stdlib (~3MB)
├── python310._pth             # Path config (tells Python where to find modules)
├── vcruntime140.dll           # VC runtime (included in embeddable package)
├── Lib/site-packages/         # Your packages
└── test_roofline.py
```

## Key Differences from Linux

- **No RPATH**: DLLs found in same directory as .exe
- **Stdlib compressed**: `python310.zip` instead of `lib/python3.10/`
- **Path config**: `python310._pth` file (4 lines) instead of `Py_SetPath()` in C++
- **Fully portable**: No installation, registry, or PATH modification needed

## Distribution

```powershell
Compress-Archive -Path build\Release\* -DestinationPath test_roofline_windows.zip
```

Target machine just extracts and runs - no Python installation required.

**Download:** https://www.python.org/downloads/windows/ (look for "embeddable package")
