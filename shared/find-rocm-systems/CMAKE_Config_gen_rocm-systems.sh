#!/bin/bash
################################################################################
# ROCm Systems CMAKE Config Generation
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
################################################################################

set -e

# Default configuration
BUILD_DIR="build"
INSTALL_PREFIX="/opt/rocm"
BUILD_TYPE="Release"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Configure ROCm Systems

OPTIONS:
    -h, --help              Show this help message
    -b, --build-dir DIR     Build directory (default: build)
    -i, --install-prefix    Installation prefix (default: /opt/rocm)
    -c, --clean             Clean build directory before building
    --rocm-path PATH        Path to ROCm installation (default: /opt/rocm)

EXAMPLES:
    # Clean build
    $0 --clean

EOF
    exit 0
}

# Parse command line arguments
CLEAN_BUILD=0
ROCM_PATH="/opt/rocm"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -i|--install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        -c|--clean)
            echo "CLEAN"
            CLEAN_BUILD=1
            shift
            ;;
        --rocm-path)
            ROCM_PATH="$2"
            shift 2
            ;;
        *)
            print_msg "Unknown option: $1" "$RED"
            usage
            ;;
    esac
done

# Validate build type
case $BUILD_TYPE in
    Debug|Release|RelWithDebInfo)
        ;;
    *)
        print_msg "Invalid build type: $BUILD_TYPE" "$RED"
        exit 1
        ;;
esac

# Clean build directory if requested
if [ $CLEAN_BUILD -eq 1 ]; then
    if [ -d "$BUILD_DIR" ]; then
        print_msg "Cleaning build directory: $BUILD_DIR" "$YELLOW"
        rm -rf "$BUILD_DIR"
	exit 0
    fi
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMake command
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    -DROCM_PATH="$ROCM_PATH"
)

# Configure
print_msg "Configuring ROCm Systems..." "$GREEN"
cmake "${CMAKE_ARGS[@]}" ..

if [ $? -ne 0 ]; then
    print_msg "Configuration failed!" "$RED"
    exit 1
fi

print_msg "Configuration completed successfully!" "$GREEN"
echo ""
print_msg "To install, run:" "$YELLOW"
echo "  cd $BUILD_DIR"
echo "  sudo cmake --install ."

echo ""
print_msg "================================" "$GREEN"
print_msg "Done!" "$GREEN"
print_msg "================================" "$GREEN"


