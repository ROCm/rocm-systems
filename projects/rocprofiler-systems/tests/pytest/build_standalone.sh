#!/bin/bash
#
# Build standalone pytest executables for rocprofiler-systems tests
#
# This script creates packaging options:
#   1. PyInstaller: Single binary (~50-100MB), no Python needed on target
#   2. PyInstaller+Docker: Uses manylinux for broad glibc compatibility
#   3. Shiv: Python zipapp (~5MB), requires Python on target
#
# Usage:
#   ./build_standalone.sh [--pyinstaller] [--pyinstaller-docker] [--shiv] [--all] [--output-dir DIR]
#
# After building, copy the output to your target machine and run:
#   PyInstaller: ./rocprofsys-tests [pytest args...]
#   Shiv:        python3 rocprofsys-tests.pyz [pytest args...]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/dist"
BUILD_PYINSTALLER=0
BUILD_PYINSTALLER_DOCKER=0
BUILD_SHIV=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pyinstaller)
            BUILD_PYINSTALLER=1
            shift
            ;;
        --pyinstaller-docker)
            BUILD_PYINSTALLER_DOCKER=1
            shift
            ;;
        --shiv)
            BUILD_SHIV=1
            shift
            ;;
        --all)
            BUILD_PYINSTALLER=1
            BUILD_SHIV=1
            shift
            ;;
        --all-docker)
            BUILD_PYINSTALLER_DOCKER=1
            BUILD_SHIV=1
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--pyinstaller] [--pyinstaller-docker] [--shiv] [--all] [--output-dir DIR]"
            echo ""
            echo "Options:"
            echo "  --pyinstaller        Build PyInstaller binary (uses system Python/glibc)"
            echo "  --pyinstaller-docker Build PyInstaller binary in Docker (glibc 2.17+ compatible)"
            echo "  --shiv               Build Shiv zipapp (requires Python on target)"
            echo "  --all                Build pyinstaller + shiv"
            echo "  --all-docker         Build pyinstaller-docker + shiv"
            echo "  --output-dir         Output directory (default: ./dist)"
            echo ""
            echo "NOTE: If you get glibc errors on target, use --pyinstaller-docker or --shiv"
            echo ""
            echo "Examples:"
            echo "  $0 --all                    # Build pyinstaller + shiv"
            echo "  $0 --pyinstaller-docker     # Build compatible binary via Docker"
            echo "  $0 --shiv --output-dir /tmp # Build Shiv to /tmp"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to shiv if nothing specified (safest option)
if [[ $BUILD_PYINSTALLER -eq 0 && $BUILD_PYINSTALLER_DOCKER -eq 0 && $BUILD_SHIV -eq 0 ]]; then
    BUILD_SHIV=1
    echo "No option specified, defaulting to --shiv (most compatible)"
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Building rocprofiler-systems pytest packages"
echo "=============================================="
echo "Source directory: $SCRIPT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create the test runner wrapper script
create_runner_script() {
    cat > "${SCRIPT_DIR}/run_rocprofsys_tests.py" << 'RUNNER_EOF'
#!/usr/bin/env python3
"""
Standalone test runner for rocprofiler-systems pytest tests.

This script is designed to be packaged with PyInstaller or Shiv to create
a standalone executable for running tests on machines with rocprofiler-systems
installed.

Usage:
    ./rocprofsys-tests [pytest options...]

Examples:
    ./rocprofsys-tests                           # Run all tests
    ./rocprofsys-tests -v                        # Verbose output
    ./rocprofsys-tests -k transpose              # Run only transpose tests
    ./rocprofsys-tests --collect-only            # List available tests
    ./rocprofsys-tests test_transpose.py         # Run specific test file

Environment Variables:
    ROCPROFSYS_INSTALL_DIR          - Path to rocprofiler-systems installation
    ROCPROFSYS_BUILD_DIR            - Path to build directory (for development)
    ROCPROFSYS_SOURCE_DIR           - Path to source directory (for development)
    ROCPROFSYS_KEEP_TEST_OUTPUT     - Keep test output on success (ON/OFF, default: ON)
    ROCPROFSYS_USE_ROCPD            - Enable/disable ROCpd validation (ON/OFF, default: ON if available)
    ROCPROFSYS_VALIDATE_PERFETTO    - Enable/disable Perfetto validation (ON/OFF, default: ON if available)
    ROCPROFSYS_TRACE_PROC_SHELL     - Path to trace_processor_shell binary (auto-detected)
    ROCM_PATH                       - Path to ROCm installation (default: /opt/rocm)
    ROCM_LLVM_OBJDUMP               - Path to ROCm's llvm-objdump (default: auto-detected)
"""
import os
import sys

def get_test_dir():
    """Find the tests directory - handles both packaged and development modes."""
    # When packaged with PyInstaller, files are extracted to _MEIPASS
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        test_dir = os.path.join(base_path, 'tests', 'pytest')
        if os.path.isdir(test_dir):
            return test_dir
        # Fallback: tests might be at root level
        test_dir = os.path.join(base_path, 'pytest')
        if os.path.isdir(test_dir):
            return test_dir
        return base_path
    else:
        # Running as regular Python script
        return os.path.dirname(os.path.abspath(__file__))

def main():
    import pytest

    test_dir = get_test_dir()

    # Add test directory to path so imports work
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)

    # Build pytest arguments
    args = list(sys.argv[1:])

    # If no test path specified, use the test directory
    has_test_path = any(
        arg.endswith('.py') or
        os.path.isdir(arg) or
        '::' in arg
        for arg in args if not arg.startswith('-')
    )

    if not has_test_path:
        args.append(test_dir)

    # Print info
    print(f"rocprofiler-systems pytest runner")
    print(f"Test directory: {test_dir}")
    print(f"Arguments: {' '.join(args)}")
    print("-" * 50)

    # Run pytest
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(main())
RUNNER_EOF
    echo "Created: run_rocprofsys_tests.py"
}

# Build with PyInstaller
build_pyinstaller() {
    echo ""
    echo "=== Building PyInstaller standalone binary ==="
    echo ""

    # Check if PyInstaller and required packages are installed
    if ! python3 -c "import PyInstaller" 2>/dev/null; then
        echo "Installing PyInstaller..."
        pip install pyinstaller
    fi

    # Install pytest plugins needed for bundling
    echo "Installing pytest and required plugins..."
    pip install pytest pytest-subtests pytest-timeout pytest-xdist

    # Create spec file for more control
    cat > "${SCRIPT_DIR}/rocprofsys_tests.spec" << SPEC_EOF
# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# Collect all test files and the rocprofsys package
test_dir = '${SCRIPT_DIR}'
datas = []

# Add all Python files from the test directory
for root, dirs, files in os.walk(test_dir):
    # Skip __pycache__ and build directories
    dirs[:] = [d for d in dirs if d not in ('__pycache__', 'dist', 'build')]
    for f in files:
        if f.endswith(('.py', '.txt', '.md', '.json')):
            src = os.path.join(root, f)
            # Compute relative destination
            rel_path = os.path.relpath(root, test_dir)
            if rel_path == '.':
                dst = 'tests/pytest'
            else:
                dst = os.path.join('tests/pytest', rel_path)
            datas.append((src, dst))

a = Analysis(
    ['${SCRIPT_DIR}/run_rocprofsys_tests.py'],
    pathex=['${SCRIPT_DIR}'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'pytest',
        '_pytest',
        '_pytest.assertion',
        '_pytest.config',
        '_pytest.fixtures',
        '_pytest.python',
        'pytest_subtests',
        'pytest_subtests.plugin',
        'pytest_timeout',
        'xdist',
        'rocprofsys',
        'rocprofsys.config',
        'rocprofsys.runners',
        'rocprofsys.validators',
        'rocprofsys.gpu',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='rocprofsys-tests',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
SPEC_EOF

    # Build with PyInstaller
    cd "$SCRIPT_DIR"
    python3 -m PyInstaller \
        --distpath "$OUTPUT_DIR" \
        --workpath "${SCRIPT_DIR}/build/pyinstaller" \
        --clean \
        --noconfirm \
        rocprofsys_tests.spec

    # Cleanup
    rm -f "${SCRIPT_DIR}/rocprofsys_tests.spec"
    rm -rf "${SCRIPT_DIR}/build/pyinstaller"

    echo ""
    echo "PyInstaller build complete!"
    echo "Binary: ${OUTPUT_DIR}/rocprofsys-tests"
    echo "Size: $(du -h "${OUTPUT_DIR}/rocprofsys-tests" | cut -f1)"
}

# Build with PyInstaller in Docker (manylinux for glibc compatibility)
build_pyinstaller_docker() {
    echo ""
    echo "=== Building PyInstaller binary in Docker (manylinux) ==="
    echo ""

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker is not installed or not in PATH"
        echo "Install Docker or use --shiv instead"
        exit 1
    fi

    # Create a temporary build context
    BUILD_CONTEXT=$(mktemp -d)
    trap "rm -rf \"$BUILD_CONTEXT\"" EXIT

    # Copy test files to build context
    cp -r "${SCRIPT_DIR}" "${BUILD_CONTEXT}/pytest"
    cp "${SCRIPT_DIR}/run_rocprofsys_tests.py" "${BUILD_CONTEXT}/" 2>/dev/null || \
        create_runner_script && cp "${SCRIPT_DIR}/run_rocprofsys_tests.py" "${BUILD_CONTEXT}/"

    # Create Dockerfile
    cat > "${BUILD_CONTEXT}/Dockerfile" << 'DOCKERFILE_EOF'
# Use manylinux for broad glibc compatibility (glibc 2.17+)
FROM quay.io/pypa/manylinux2014_x86_64

# Install Python and pip
RUN /opt/python/cp310-cp310/bin/python -m pip install --upgrade pip
RUN /opt/python/cp310-cp310/bin/python -m pip install pyinstaller pytest pytest-subtests pytest-timeout pytest-xdist

# Set Python path
ENV PATH="/opt/python/cp310-cp310/bin:$PATH"

WORKDIR /build

# Copy test files
COPY pytest /build/pytest
COPY run_rocprofsys_tests.py /build/

# Create spec file
RUN cat > /build/rocprofsys_tests.spec << 'SPEC_EOF'
# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

test_dir = '/build/pytest'
datas = []

for root, dirs, files in os.walk(test_dir):
    dirs[:] = [d for d in dirs if d not in ('__pycache__', 'dist', 'build')]
    for f in files:
        if f.endswith(('.py', '.txt', '.md', '.json')):
            src = os.path.join(root, f)
            rel_path = os.path.relpath(root, test_dir)
            if rel_path == '.':
                dst = 'tests/pytest'
            else:
                dst = os.path.join('tests/pytest', rel_path)
            datas.append((src, dst))

a = Analysis(
    ['/build/run_rocprofsys_tests.py'],
    pathex=['/build/pytest'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'pytest', '_pytest', '_pytest.assertion', '_pytest.config',
        '_pytest.fixtures', '_pytest.python',
        'pytest_subtests', 'pytest_subtests.plugin', 'pytest_timeout', 'xdist',
        'rocprofsys', 'rocprofsys.config', 'rocprofsys.runners',
        'rocprofsys.validators', 'rocprofsys.gpu',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name='rocprofsys-tests',
    debug=False,
    strip=True,
    upx=False,
    console=True,
)
SPEC_EOF

# Build
RUN pyinstaller --clean --noconfirm rocprofsys_tests.spec

# Output is in /build/dist/rocprofsys-tests
DOCKERFILE_EOF

    echo "Building Docker image..."
    docker build -t rocprofsys-tests-builder "${BUILD_CONTEXT}"

    echo "Extracting binary from container..."
    CONTAINER_ID=$(docker create rocprofsys-tests-builder)
    docker cp "${CONTAINER_ID}:/build/dist/rocprofsys-tests" "${OUTPUT_DIR}/rocprofsys-tests"
    docker rm "${CONTAINER_ID}"

    # Cleanup
    docker rmi rocprofsys-tests-builder 2>/dev/null || true

    echo ""
    echo "PyInstaller (Docker/manylinux) build complete!"
    echo "Binary: ${OUTPUT_DIR}/rocprofsys-tests"
    echo "Size: $(du -h "${OUTPUT_DIR}/rocprofsys-tests" | cut -f1)"
    echo ""
    echo "This binary is compatible with glibc 2.17+ (RHEL 7, Ubuntu 14.04+, etc.)"
}

# Build simple zipapp (requires pytest on target, but no glibc issues)
build_shiv() {
    echo ""
    echo "=== Building Python zipapp ==="
    echo ""

    # Create a temporary directory
    BUILD_DIR=$(mktemp -d)
    echo "Build directory: $BUILD_DIR"

    # Create the package structure
    mkdir -p "${BUILD_DIR}/rocprofsys"
    mkdir -p "${BUILD_DIR}/tests"

    # Copy the rocprofsys test framework package
    cp -r "${SCRIPT_DIR}/rocprofsys/"* "${BUILD_DIR}/rocprofsys/"

    # Copy test files
    cp "${SCRIPT_DIR}"/test_*.py "${BUILD_DIR}/tests/" 2>/dev/null || true
    cp "${SCRIPT_DIR}/conftest.py" "${BUILD_DIR}/tests/"

    # Ensure __init__.py exists
    touch "${BUILD_DIR}/rocprofsys/__init__.py"
    touch "${BUILD_DIR}/tests/__init__.py"

    # Create __main__.py (entry point for zipapp)
    cat > "${BUILD_DIR}/__main__.py" << 'MAIN_EOF'
#!/usr/bin/env python3
"""
rocprofiler-systems pytest runner - zipapp version

Usage:
    python3 rocprofsys-tests.pyz [pytest options...]

Requirements:
    - pytest must be installed: pip install pytest
    - rocprofiler-systems must be installed on the system

Examples:
    python3 rocprofsys-tests.pyz --collect-only
    python3 rocprofsys-tests.pyz -v
    python3 rocprofsys-tests.pyz -k transpose -v
"""
import os
import sys
import zipfile
import tempfile
import shutil
import atexit

# Global for cleanup
_extract_dir = None

def cleanup():
    """Remove extracted files on exit."""
    global _extract_dir
    if _extract_dir and os.path.isdir(_extract_dir):
        shutil.rmtree(_extract_dir, ignore_errors=True)

def extract_tests():
    """Extract tests from zipapp to temp directory."""
    global _extract_dir

    # Find the zipapp path
    # When running as zipapp, __file__ points inside the zip
    # The zip path is everything before the first component after .pyz
    zipapp_path = None
    for path in sys.path:
        if path.endswith('.pyz') and os.path.isfile(path):
            zipapp_path = path
            break

    if not zipapp_path:
        # Try to find it from __file__
        current = os.path.abspath(__file__)
        while current and not current.endswith('.pyz'):
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        if current.endswith('.pyz'):
            zipapp_path = current

    if not zipapp_path or not os.path.isfile(zipapp_path):
        # Not running from zipapp, use local directory
        return os.path.dirname(os.path.abspath(__file__))

    # Create temp directory and extract
    _extract_dir = tempfile.mkdtemp(prefix='rocprofsys-tests-')
    atexit.register(cleanup)

    with zipfile.ZipFile(zipapp_path, 'r') as zf:
        zf.extractall(_extract_dir)

    return _extract_dir

def main():
    # Check pytest is available
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is not installed")
        print("Please install it: pip install pytest")
        sys.exit(1)

    # Extract tests to temp directory
    app_path = extract_tests()

    # Add app path to sys.path for imports
    if app_path not in sys.path:
        sys.path.insert(0, app_path)

    # Find tests directory
    tests_dir = os.path.join(app_path, 'tests')

    if not os.path.isdir(tests_dir):
        print(f"ERROR: Tests directory not found: {tests_dir}")
        print(f"Contents of {app_path}:")
        for item in os.listdir(app_path):
            print(f"  {item}")
        sys.exit(1)

    # Build pytest arguments
    args = list(sys.argv[1:])

    # Check if user specified a test path
    has_test_path = any(
        (arg.endswith('.py') or os.path.isdir(arg) or '::' in arg)
        for arg in args if not arg.startswith('-')
    )

    if not has_test_path:
        args.append(tests_dir)

    # Print info
    print("=" * 60)
    print("rocprofiler-systems pytest runner")
    print("=" * 60)
    print(f"Tests dir: {tests_dir}")
    print(f"Command: pytest {' '.join(args)}")
    print("=" * 60)
    print()

    # Run pytest
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(main())
MAIN_EOF

    # Create the zipapp (don't use --main since we have __main__.py)
    cd "$BUILD_DIR"
    python3 -m zipapp \
        --python "/usr/bin/env python3" \
        --output "${OUTPUT_DIR}/rocprofsys-tests.pyz" \
        --compress \
        .

    # Make it executable
    chmod +x "${OUTPUT_DIR}/rocprofsys-tests.pyz"

    # Cleanup
    rm -rf "$BUILD_DIR"

    echo ""
    echo "Zipapp build complete!"
    echo "Output: ${OUTPUT_DIR}/rocprofsys-tests.pyz"
    echo "Size: $(du -h "${OUTPUT_DIR}/rocprofsys-tests.pyz" | cut -f1)"
    echo ""
    echo "Requirements on target machine:"
    echo "  - Python 3.8+"
    echo "  - Install dependencies: pip install pytest pytest-subtests pytest-timeout pytest-xdist"
}

# Main build process
create_runner_script

if [[ $BUILD_PYINSTALLER -eq 1 ]]; then
    build_pyinstaller
fi

if [[ $BUILD_PYINSTALLER_DOCKER -eq 1 ]]; then
    build_pyinstaller_docker
fi

if [[ $BUILD_SHIV -eq 1 ]]; then
    build_shiv
fi

# Cleanup runner script
rm -f "${SCRIPT_DIR}/run_rocprofsys_tests.py"

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Output files in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "(no files yet)"
echo ""
echo "=== How to use on target machine ==="
echo ""
echo "1. Copy the binary/zipapp to your target machine"
echo ""
echo "2. Ensure rocprofiler-systems is installed and in PATH, or set:"
echo "   export ROCPROFSYS_INSTALL_DIR=/opt/rocm"
echo ""
echo "3. Run tests:"
if [[ $BUILD_PYINSTALLER -eq 1 || $BUILD_PYINSTALLER_DOCKER -eq 1 ]]; then
    echo "   PyInstaller: ./rocprofsys-tests -v"
fi
if [[ $BUILD_SHIV -eq 1 ]]; then
    echo "   Shiv:        python3 rocprofsys-tests.pyz -v"
    echo "   (Requires: pip install pytest pytest-subtests pytest-timeout pytest-xdist)"
fi
echo ""
echo "4. Common pytest options:"
echo "   -v                  Verbose output"
echo "   -k 'transpose'      Run only tests matching 'transpose'"
echo "   --collect-only      List available tests"
echo "   -x                  Stop on first failure"
echo ""
