#!/bin/bash
#===============================================================================
# ROCm Components Build from Source Script
# 
# This script builds rocr-runtime, clr, hip, hip-tests, rocprofiler-sdk, 
# and rocprofiler-systems from source inside a Docker container.
#
# Prerequisites: Docker with GPU support, AMD GPU hardware
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
ROCM_SYSTEMS_REPO="https://github.com/ROCm/rocm-systems.git"
ROCM_INSTALL_PREFIX="/opt/rocm"
WORKSPACE_DIR="/workspace"

#===============================================================================
# Helper Functions
#===============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} ${CYAN}$1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}✖ ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✔ $1${NC}"
}

prompt_continue() {
    echo ""
    echo -e "${YELLOW}────────────────────────────────────────────────────────────────────${NC}"
    read -p "$(echo -e ${CYAN}"Press ENTER to continue or Ctrl+C to abort..."${NC})" 
    echo -e "${YELLOW}────────────────────────────────────────────────────────────────────${NC}"
    echo ""
}

prompt_yes_no() {
    local prompt="$1"
    local default="$2"
    
    if [ "$default" = "y" ]; then
        read -p "$(echo -e ${CYAN}"$prompt [Y/n]: "${NC})" response
        response=${response:-y}
    else
        read -p "$(echo -e ${CYAN}"$prompt [y/N]: "${NC})" response
        response=${response:-n}
    fi
    
    case "$response" in
        [yY][eE][sS]|[yY]) return 0 ;;
        *) return 1 ;;
    esac
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is available"
        return 0
    else
        print_error "$1 is not available"
        return 1
    fi
}

#===============================================================================
# Step 0: Pre-flight Checks
#===============================================================================

preflight_checks() {
    print_header "Step 0: Pre-flight Checks"
    
    echo "Checking environment..."
    echo ""
    
    # Check if running inside container
    if [ -f /.dockerenv ]; then
        print_success "Running inside Docker container"
    else
        print_warning "Not running inside Docker container"
        echo "This script is designed to run inside the rocm/tensorflow Docker container."
        if ! prompt_yes_no "Continue anyway?" "n"; then
            exit 1
        fi
    fi
    
    # Check for GPU
    if [ -e /dev/kfd ]; then
        print_success "AMD GPU device found (/dev/kfd)"
    else
        print_warning "AMD GPU device not found"
    fi
    
    # Check ROCm installation
    if [ -d "$ROCM_INSTALL_PREFIX" ]; then
        print_success "ROCm installation found at $ROCM_INSTALL_PREFIX"
    else
        print_error "ROCm not found at $ROCM_INSTALL_PREFIX"
    fi
    
    # Check basic tools (cmake may not be installed yet)
    check_command make || true
    check_command git
    check_command python3
    
    # Check cmake separately - will be installed in next step if missing
    if command -v cmake &> /dev/null; then
        print_success "cmake is available: $(cmake --version | head -1)"
    else
        print_warning "cmake is not installed - will be installed in the next step"
    fi
    
    echo ""
    echo "Pre-flight checks complete."
    prompt_continue
}

#===============================================================================
# Step 0.5: Install CMake (First Priority)
#===============================================================================

install_cmake_first() {
    print_header "Step 0.5: Install CMake (Required First)"
    
    echo "CMake is required for all subsequent build steps."
    echo "We'll install the latest CMake via pip for best compatibility."
    echo ""
    
    # Check if cmake is already installed
    if command -v cmake &> /dev/null; then
        local cmake_version=$(cmake --version | head -1)
        local cmake_ver_num=$(cmake --version | head -1 | grep -oP '\d+\.\d+' | head -1)
        print_success "CMake is already installed: $cmake_version"
        
        # Check if version is 3.21 or higher
        if [ "$(echo "$cmake_ver_num >= 3.21" | bc 2>/dev/null)" = "1" ] 2>/dev/null; then
            print_success "CMake version is sufficient (>= 3.21)"
            if ! prompt_yes_no "Upgrade to latest cmake anyway?" "n"; then
                echo "Keeping existing cmake installation."
                prompt_continue
                return
            fi
        else
            print_warning "CMake version may be too old. Recommend upgrading."
        fi
    fi
    
    echo "The following will be installed:"
    echo "  - cmake (latest version via pip)"
    echo "  - make (build tool)"
    echo "  - build-essential (compiler toolchain)"
    echo "  - python3-pip (for pip-based cmake install)"
    echo ""
    
    if ! prompt_yes_no "Install CMake and build tools?" "y"; then
        print_warning "Skipping CMake installation"
        print_warning "WARNING: Subsequent build steps will fail without CMake!"
        prompt_continue
        return
    fi
    
    print_step "Updating package lists..."
    apt-get update
    
    print_step "Installing build essentials and pip..."
    apt-get install -y make build-essential python3-pip
    
    print_step "Installing latest CMake via pip..."
    pip install --break-system-packages cmake --upgrade 2>/dev/null || \
    pip install cmake --upgrade
    
    # Ensure pip cmake is in PATH
    export PATH="${HOME}/.local/bin:${PATH}"
    
    # Verify installation
    if command -v cmake &> /dev/null; then
        print_success "CMake installed successfully: $(cmake --version | head -1)"
    else
        print_error "CMake installation failed!"
        echo "Trying apt-get install cmake as fallback..."
        apt-get install -y cmake
    fi
    
    prompt_continue
}

#===============================================================================
# Step 1: Clone rocm-systems Repository
#===============================================================================

clone_rocm_systems() {
    print_header "Step 1: Clone rocm-systems Repository"
    
    cd "$WORKSPACE_DIR"
    
    if [ -d "rocm-systems" ]; then
        print_warning "rocm-systems directory already exists"
        if prompt_yes_no "Remove and re-clone?" "n"; then
            rm -rf rocm-systems
        else
            echo "Using existing rocm-systems directory"
            prompt_continue
            return
        fi
    fi
    
    print_step "Cloning $ROCM_SYSTEMS_REPO..."
    git clone "$ROCM_SYSTEMS_REPO"
    
    print_success "Repository cloned successfully"
    prompt_continue
}

#===============================================================================
# Step 2: Install Common Dependencies
#===============================================================================

install_common_dependencies() {
    print_header "Step 2: Install Common Build Dependencies"
    
    echo "The following packages will be installed:"
    echo ""
    echo "  Required for rocr-runtime:"
    echo "    - cmake (3.7+), g++, make"
    echo "    - libelf-dev"
    echo "    - libdrm-amdgpu-dev (or libdrm-dev)"
    echo "    - pkg-config"
    echo "    - rocm-core"
    echo "    - rocm-llvm-dev (provides LLVM/Clang for ROCm)"
    echo "    - xxd (for shader header generation)"
    echo ""
    echo "  Required for HIP/CLR:"
    echo "    - python3, pip packages (CppHeaderParser)"
    echo ""
    
    if ! prompt_yes_no "Install common dependencies?" "y"; then
        print_warning "Skipping dependency installation"
        return
    fi
    
    print_step "Updating package lists..."
    apt-get update
    
    print_step "Installing build essentials and rocr-runtime dependencies..."
    apt-get install -y \
        cmake \
        g++ \
        make \
        libelf-dev \
        libdrm-amdgpu-dev \
        pkg-config \
        python3 \
        python3-pip \
        xxd
    
    print_step "Installing ROCm LLVM and core packages..."
    echo ""
    print_warning "NOTE: rocm-llvm-dev is required to avoid LLVM version mismatch errors"
    echo "      (System Clang 18 vs ROCm LLVM 20)"
    echo ""
    apt-get install -y \
        rocm-core \
        rocm-llvm-dev
    
    print_step "Installing Python packages..."
    pip3 install CppHeaderParser
    
    # Set CMAKE_PREFIX_PATH for subsequent builds
    export CMAKE_PREFIX_PATH="$ROCM_INSTALL_PREFIX"
    echo ""
    print_success "CMAKE_PREFIX_PATH set to: $CMAKE_PREFIX_PATH"
    
    print_success "Common dependencies installed"
    prompt_continue
}

#===============================================================================
# Step 3: Build ROCR-Runtime
#===============================================================================

build_rocr_runtime() {
    print_header "Step 3: Build ROCR-Runtime (HSA Runtime + libhsakmt)"
    
    echo "This will build:"
    echo "  - libhsakmt (ROCt) - User-mode API for ROCk driver"
    echo "  - HSA Runtime (ROCr) - Core runtime supporting HSA standards"
    echo ""
    echo "Build directory: rocm-systems/projects/rocr-runtime/build"
    echo "Install prefix: $ROCM_INSTALL_PREFIX"
    echo ""
    echo "Required dependencies (should be installed in Step 2):"
    echo "  - cmake 3.7+, g++, libelf-dev, libdrm-amdgpu-dev"
    echo "  - pkg-config, rocm-core, rocm-llvm-dev, xxd"
    echo ""
    
    # Verify critical dependencies
    print_step "Verifying dependencies..."
    
    if ! dpkg -l | grep -q rocm-llvm-dev; then
        print_error "rocm-llvm-dev is not installed!"
        echo "This package is required to avoid LLVM version mismatch errors."
        echo "Run: apt-get install -y rocm-llvm-dev"
        if ! prompt_yes_no "Try to install rocm-llvm-dev now?" "y"; then
            return
        fi
        apt-get update && apt-get install -y rocm-llvm-dev rocm-core
    else
        print_success "rocm-llvm-dev is installed"
    fi
    
    if ! command -v xxd &> /dev/null; then
        print_error "xxd is not installed!"
        echo "Run: apt-get install -y xxd"
        if ! prompt_yes_no "Try to install xxd now?" "y"; then
            return
        fi
        apt-get install -y xxd
    else
        print_success "xxd is installed"
    fi
    
    if ! prompt_yes_no "Build rocr-runtime?" "y"; then
        print_warning "Skipping rocr-runtime build"
        return
    fi
    
    cd "$WORKSPACE_DIR/rocm-systems/projects/rocr-runtime"
    
    # Clean previous build
    if [ -d "build" ]; then
        print_warning "Previous build directory found"
        if prompt_yes_no "Remove and rebuild?" "y"; then
            rm -rf build
        fi
    fi
    
    print_step "Creating build directory..."
    mkdir -p build && cd build
    
    print_step "Configuring with CMake..."
    echo ""
    echo "IMPORTANT: Setting CMAKE_PREFIX_PATH=/opt/rocm to use ROCm's LLVM"
    echo "           This avoids version mismatch with system Clang 18"
    echo ""
    
    cmake \
        -DCMAKE_INSTALL_PREFIX="$ROCM_INSTALL_PREFIX" \
        -DCMAKE_PREFIX_PATH="$ROCM_INSTALL_PREFIX" \
        -DCMAKE_BUILD_TYPE=Release \
        ..
    
    print_step "Building (this may take a while)..."
    make -j$(nproc)
    
    print_step "Installing..."
    make install
    
    # Verify and fix libhsa-runtime64 symlink
    print_step "Verifying libhsa-runtime64.so.1 symlink..."
    HSA_LIB_PATH="$ROCM_INSTALL_PREFIX/lib/libhsa-runtime64.so.1"
    EXPECTED_VERSION="1.18.0"
    
    if [ -L "$HSA_LIB_PATH" ]; then
        CURRENT_LINK=$(readlink "$HSA_LIB_PATH")
        if echo "$CURRENT_LINK" | grep -q "$EXPECTED_VERSION"; then
            print_success "Symlink correctly points to: $CURRENT_LINK"
        else
            print_warning "Symlink points to old version: $CURRENT_LINK"
            echo "Removing old symlink and creating correct one..."
            rm -f "$HSA_LIB_PATH"
            cd "$ROCM_INSTALL_PREFIX/lib"
            ln -sf "libhsa-runtime64.so.$EXPECTED_VERSION" "libhsa-runtime64.so.1"
            print_success "Symlink fixed: libhsa-runtime64.so.1 -> libhsa-runtime64.so.$EXPECTED_VERSION"
        fi
    elif [ -f "$HSA_LIB_PATH" ]; then
        print_warning "$HSA_LIB_PATH exists but is not a symlink - removing and recreating"
        rm -f "$HSA_LIB_PATH"
        cd "$ROCM_INSTALL_PREFIX/lib"
        ln -sf "libhsa-runtime64.so.$EXPECTED_VERSION" "libhsa-runtime64.so.1"
        print_success "Symlink created: libhsa-runtime64.so.1 -> libhsa-runtime64.so.$EXPECTED_VERSION"
    else
        # Create symlink if the target exists
        if [ -f "$ROCM_INSTALL_PREFIX/lib/libhsa-runtime64.so.$EXPECTED_VERSION" ]; then
            cd "$ROCM_INSTALL_PREFIX/lib"
            ln -sf "libhsa-runtime64.so.$EXPECTED_VERSION" "libhsa-runtime64.so.1"
            print_success "Symlink created: libhsa-runtime64.so.1 -> libhsa-runtime64.so.$EXPECTED_VERSION"
        else
            print_warning "$HSA_LIB_PATH does not exist - this may be okay if using different install prefix"
        fi
    fi
    
    print_success "rocr-runtime built and installed successfully"
    prompt_continue
}

#===============================================================================
# Step 4: Build CLR + HIP
#===============================================================================

build_clr_hip() {
    print_header "Step 4: Build CLR (Compute Language Runtime) + HIP"
    
    echo "This will build:"
    echo "  - hipamd - HIP implementation on AMD platform"
    echo "  - rocclr - Compute runtime used by HIP and OpenCL"
    echo ""
    echo "Build directory: rocm-systems/projects/clr/build"
    echo "Install prefix: $ROCM_INSTALL_PREFIX"
    echo ""
    
    if ! prompt_yes_no "Build CLR + HIP?" "y"; then
        print_warning "Skipping CLR + HIP build"
        return
    fi
    
    # Install SIMDe dependency (required for rocclr)
    print_step "Installing SIMDe (SIMD Everywhere) library..."
    apt-get install -y libsimde-dev || {
        print_warning "libsimde-dev not available via apt, installing from source..."
        cd /tmp
        if [ ! -d "simde" ]; then
            git clone https://github.com/simd-everywhere/simde.git
        fi
        cd simde
        mkdir -p build && cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
        make install
        cd "$WORKSPACE_DIR"
    }
    
    # Set HIP_DIR
    export HIP_DIR="$WORKSPACE_DIR/rocm-systems/projects/hip"
    echo "HIP_DIR set to: $HIP_DIR"
    
    cd "$WORKSPACE_DIR/rocm-systems/projects/clr"
    
    # Clean previous build
    if [ -d "build" ]; then
        print_warning "Previous build directory found"
        if prompt_yes_no "Remove and rebuild?" "y"; then
            rm -rf build
        fi
    fi
    
    print_step "Creating build directory..."
    mkdir -p build && cd build
    
    print_step "Configuring with CMake..."
    cmake .. \
        -DCLR_BUILD_HIP=ON \
        -DCLR_BUILD_OCL=OFF \
        -DHIP_COMMON_DIR="$HIP_DIR" \
        -DHIP_PLATFORM=amd \
        -DCMAKE_PREFIX_PATH="$ROCM_INSTALL_PREFIX" \
        -DCMAKE_INSTALL_PREFIX="$ROCM_INSTALL_PREFIX"
    
    print_step "Building (this may take a while)..."
    make -j$(nproc)
    
    print_step "Installing..."
    make install
    
    print_success "CLR + HIP built and installed successfully"
    prompt_continue
}

#===============================================================================
# Step 5: Build ROCprofiler-SDK
#===============================================================================

build_rocprofiler_sdk() {
    print_header "Step 5: Build ROCprofiler-SDK"
    
    echo "This will build rocprofiler-sdk which provides:"
    echo "  - rocprofv3 command-line tool"
    echo "  - GPU profiling and tracing APIs"
    echo ""
    echo "Build directory: rocm-systems/rocprofiler-sdk-build"
    echo "Install prefix: $ROCM_INSTALL_PREFIX"
    echo ""
    
    if ! prompt_yes_no "Build ROCprofiler-SDK?" "y"; then
        print_warning "Skipping ROCprofiler-SDK build"
        return
    fi
    
    # Install SDK-specific dependencies
    print_step "Installing SDK dependencies..."
    apt-get install -y libdw-dev libsqlite3-dev
    
    # Check GCC version
    echo ""
    echo "ROCprofiler-SDK requires GCC 12 or 13."
    echo ""
    
    if command -v g++-13 &> /dev/null; then
        echo "  g++-13 is available"
    fi
    if command -v g++-12 &> /dev/null; then
        echo "  g++-12 is available"
    fi
    
    echo ""
    echo "Current default:"
    g++ --version | head -1
    echo ""
    
    if prompt_yes_no "Set GCC-13 as compiler? (recommended)" "y"; then
        export CXX=/usr/bin/g++-13
        export CC=/usr/bin/gcc-13
        print_success "Using GCC-13"
    elif prompt_yes_no "Set GCC-12 as compiler?" "n"; then
        export CXX=/usr/bin/g++-12
        export CC=/usr/bin/gcc-12
        print_success "Using GCC-12"
    else
        print_warning "Using default compiler"
    fi
    
    cd "$WORKSPACE_DIR/rocm-systems"
    
    # Install Python requirements
    print_step "Installing Python requirements..."
    export PATH="${HOME}/.local/bin:${PATH}"
    python3 -m pip install -r projects/rocprofiler-sdk/requirements.txt
    
    # Clean previous build
    if [ -d "rocprofiler-sdk-build" ]; then
        print_warning "Previous build directory found"
        if prompt_yes_no "Remove and rebuild?" "y"; then
            rm -rf rocprofiler-sdk-build
        fi
    fi
    
    print_step "Configuring with CMake..."
    cmake \
        -B rocprofiler-sdk-build \
        -DCMAKE_INSTALL_PREFIX="$ROCM_INSTALL_PREFIX" \
        -DCMAKE_PREFIX_PATH="$ROCM_INSTALL_PREFIX" \
        projects/rocprofiler-sdk
    
    print_step "Building (this may take a while)..."
    cmake --build rocprofiler-sdk-build --target all --parallel $(nproc)
    
    print_step "Installing..."
    cmake --build rocprofiler-sdk-build --target install
    
    # Add ROCm bin to PATH
    print_step "Adding /opt/rocm/bin to PATH..."
    export PATH="$PATH:/opt/rocm/bin"
    echo ""
    echo "To make this permanent, add to your ~/.bashrc:"
    echo "  export PATH=\$PATH:/opt/rocm/bin"
    
    print_success "ROCprofiler-SDK built and installed successfully"
    prompt_continue
}

#===============================================================================
# Step 6: Build ROCprofiler-Systems
#===============================================================================

build_rocprofiler_systems() {
    print_header "Step 6: Build ROCprofiler-Systems (formerly Omnitrace)"
    
    echo "This will build rocprofiler-systems which provides:"
    echo "  - rocprof-sys-python for Python profiling"
    echo "  - rocprof-sys-sample for call-stack sampling"
    echo "  - rocprof-sys-instrument for binary instrumentation"
    echo "  - Comprehensive CPU+GPU profiling"
    echo ""
    echo "Build directory: rocm-systems/projects/rocprofiler-systems/build"
    echo "Install prefix: /opt/rocprofiler-systems"
    echo ""
    
    print_warning "NOTE: rocprofiler-systems build may have issues with some configurations."
    echo "This step is optional and may fail."
    echo ""
    
    if ! prompt_yes_no "Build ROCprofiler-Systems?" "y"; then
        print_warning "Skipping ROCprofiler-Systems build"
        return
    fi
    
    # Install dependencies
    print_step "Installing dependencies..."
    apt-get install -y \
        autoconf autotools-dev bash-completion bison build-essential \
        bzip2 chrpath curl environment-modules flex gettext git-core gnupg2 \
        gzip iproute2 libiberty-dev libpapi-dev libpfm4-dev libsqlite3-dev libtool \
        locales lsb-release m4 ninja-build python3-pip software-properties-common \
        texinfo unzip wget vim zip zlib1g-dev
    
    print_step "Installing Python packages (perfetto)..."
    pip install --break-system-packages perfetto 2>/dev/null || \
    pip install perfetto
    
    # Ensure latest cmake is available (should already be installed in Step 0.5)
    if ! command -v cmake &> /dev/null; then
        print_step "Installing CMake..."
        pip install --break-system-packages cmake 2>/dev/null || pip install cmake
    fi
    export PATH="${HOME}/.local/bin:${PATH}"
    
    cd "$WORKSPACE_DIR/rocm-systems/projects/rocprofiler-systems"
    
    # Clean previous build if exists
    if [ -d "build/debug" ]; then
        print_warning "Previous debug build directory found"
        if prompt_yes_no "Remove and rebuild?" "y"; then
            rm -rf build/debug
        fi
    fi
    
    print_step "Configuring with CMake debug preset..."
    echo ""
    echo "Using --preset debug which sets:"
    echo "  - CMAKE_BUILD_TYPE=Debug"
    echo "  - ROCPROFSYS_BUILD_DEBUG=ON"
    echo "  - No optimizations (-O0)"
    echo ""
    
    cmake --preset debug
    
    print_step "Building with debug symbols (this may take a while)..."
    cmake --build build/debug --parallel 8
    
    print_step "Installing..."
    cmake --build build/debug --target install
    
    print_step "Setting up environment..."
    if [ -f /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh ]; then
        source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh
        print_success "Environment configured"
    fi
    
    # Post-build verification: Check what HSA library rocprofiler-systems linked against
    print_step "Post-build verification: Checking linked HSA runtime..."
    local EXPECTED_HSA_VERSION="1.18.0"
    
    ROCPROFSYS_LIB=$(find /opt/rocprofiler-systems -name "librocprofiler-systems.so*" 2>/dev/null | head -1)
    if [ -n "$ROCPROFSYS_LIB" ]; then
        echo "Checking: $ROCPROFSYS_LIB"
        HSA_LINKED=$(ldd "$ROCPROFSYS_LIB" 2>/dev/null | grep "libhsa-runtime64" || echo "not found")
        if [ -n "$HSA_LINKED" ]; then
            echo "  $HSA_LINKED"
            if echo "$HSA_LINKED" | grep -q "$EXPECTED_HSA_VERSION"; then
                print_success "rocprofiler-systems correctly linked to HSA runtime $EXPECTED_HSA_VERSION"
            else
                print_warning "rocprofiler-systems may be linked to wrong HSA version"
                echo "  Expected: $EXPECTED_HSA_VERSION"
                echo "  If profiling fails, the symlink check before running rocprofv3 will help fix this"
            fi
        fi
    fi
    
    print_success "ROCprofiler-Systems built and installed"
    prompt_continue
}

#===============================================================================
# Step 7: Create Example Script
#===============================================================================

create_example_script() {
    print_header "Step 7: Create TensorFlow Example Script"
    
    echo "Creating example.py - A simple MNIST training example"
    echo ""
    
    cat > "$WORKSPACE_DIR/example.py" << 'EOF'
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
EOF
    
    print_success "Created $WORKSPACE_DIR/example.py"
    echo ""
    cat "$WORKSPACE_DIR/example.py"
    prompt_continue
}

#===============================================================================
# Step 8: Test with rocprofv3
#===============================================================================

test_rocprofv3() {
    print_header "Step 8: Test Profiling with rocprofv3"
    
    echo "This will test rocprofv3 profiling with the TensorFlow example."
    echo ""
    
    # Ensure /opt/rocm/bin is in PATH
    export PATH="$PATH:/opt/rocm/bin"
    
    # Check if rocprofv3 is available
    if ! command -v rocprofv3 &> /dev/null; then
        print_error "rocprofv3 is not found in PATH"
        echo ""
        echo "Make sure ROCprofiler-SDK was built and installed correctly."
        echo "Try running: which rocprofv3"
        prompt_continue
        return
    fi
    
    print_success "rocprofv3 found: $(which rocprofv3)"
    echo ""
    
    # Check rocprofv3 version
    print_step "Checking rocprofv3 version..."
    rocprofv3 --version 2>&1 || true
    echo ""
    
    # CRITICAL: Verify HSA runtime symlink before running
    print_step "Verifying HSA runtime symlink (CRITICAL for profiling)..."
    HSA_LIB_PATH="$ROCM_INSTALL_PREFIX/lib/libhsa-runtime64.so.1"
    EXPECTED_VERSION="1.18.0"
    
    echo "Checking: $HSA_LIB_PATH"
    
    if [ -L "$HSA_LIB_PATH" ]; then
        CURRENT_LINK=$(readlink "$HSA_LIB_PATH")
        echo "  Current symlink: $HSA_LIB_PATH -> $CURRENT_LINK"
        
        if echo "$CURRENT_LINK" | grep -q "$EXPECTED_VERSION"; then
            print_success "HSA runtime symlink OK: points to $EXPECTED_VERSION"
        else
            print_warning "HSA runtime symlink points to OLD version: $CURRENT_LINK"
            echo "Removing old symlink to prevent linking issues..."
            rm -f "$HSA_LIB_PATH"
            
            # Create correct symlink if the target exists
            if [ -f "$ROCM_INSTALL_PREFIX/lib/libhsa-runtime64.so.$EXPECTED_VERSION" ]; then
                cd "$ROCM_INSTALL_PREFIX/lib"
                ln -sf "libhsa-runtime64.so.$EXPECTED_VERSION" "libhsa-runtime64.so.1"
                cd "$WORKSPACE_DIR"
                print_success "Symlink recreated: libhsa-runtime64.so.1 -> libhsa-runtime64.so.$EXPECTED_VERSION"
            else
                print_warning "libhsa-runtime64.so.$EXPECTED_VERSION not found - run 'Build rocr-runtime' first"
            fi
        fi
    elif [ -f "$HSA_LIB_PATH" ]; then
        print_warning "$HSA_LIB_PATH exists but is not a symlink - removing it"
        rm -f "$HSA_LIB_PATH"
    else
        print_warning "$HSA_LIB_PATH not found - will use system default"
    fi
    echo ""
    
    echo "Command: rocprofv3 -s -o /tmp/rocprof/output -- python3.12 example.py"
    echo ""
    echo "This will:"
    echo "  - Profile the TensorFlow example"
    echo "  - Generate SQLite output at /tmp/rocprof/output_results.db"
    echo "  - Capture HIP/HSA API calls and kernel dispatches"
    echo ""
    
    print_warning "NOTE: If you see a segmentation fault, it may indicate:"
    echo "  - Version mismatch between built and installed ROCm components"
    echo "  - Missing runtime dependencies"
    echo "  - Try using the pre-installed rocprofv3 instead of the built one"
    echo ""
    
    if ! prompt_yes_no "Run rocprofv3 test?" "y"; then
        print_warning "Skipping rocprofv3 test"
        return
    fi
    
    cd "$WORKSPACE_DIR"
    
    # Create all necessary directories
    print_step "Creating output directories..."
    mkdir -p /tmp/rocprof
    mkdir -p /tmp/rocprof/.rocprofv3
    
    # Clean any previous output
    rm -f /tmp/rocprof/output_results.db 2>/dev/null
    
    # Set environment variables
    export ROCPROF_TMPDIR=/tmp/rocprof
    export ROCPROF_OUTPUT_PATH=/tmp/rocprof
    
    print_step "Running rocprofv3..."
    echo ""
    echo -e "${YELLOW}─────────────── rocprofv3 Output ───────────────${NC}"
    
    # Run with error handling
    local exit_code=0
    rocprofv3 -s -o /tmp/rocprof/output -- python3.12 example.py || exit_code=$?
    
    echo -e "${YELLOW}────────────────────────────────────────────────${NC}"
    echo ""
    
    if [ $exit_code -ne 0 ]; then
        print_warning "rocprofv3 exited with code: $exit_code"
        echo ""
        
        if [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ]; then
            print_error "Segmentation fault or crash detected!"
            echo ""
            echo "Possible causes:"
            echo "  1. Version mismatch between ROCm components"
            echo "  2. The built rocprofv3 conflicts with pre-installed ROCm"
            echo ""
            
            # Offer to try pre-installed version
            if [ -f "/opt/rocm/bin/rocprofv3" ]; then
                if prompt_yes_no "Try using pre-installed rocprofv3 at /opt/rocm/bin/rocprofv3?" "y"; then
                    echo ""
                    print_step "Running with pre-installed rocprofv3..."
                    echo ""
                    rm -f /tmp/rocprof/output_results.db 2>/dev/null
                    /opt/rocm/bin/rocprofv3 -s -o /tmp/rocprof/output -- python3.12 example.py || true
                    echo ""
                fi
            fi
        fi
    fi
    
    # Check output
    echo ""
    print_step "Checking output files..."
    ls -la /tmp/rocprof/ 2>/dev/null || echo "  (no files found)"
    echo ""
    
    if [ -f "/tmp/rocprof/output_results.db" ]; then
        print_success "Profiling completed successfully!"
        echo ""
        echo "Output file: /tmp/rocprof/output_results.db"
        ls -lh /tmp/rocprof/output_results.db
    else
        print_warning "Output file not found at /tmp/rocprof/output_results.db"
        echo ""
        echo "This may be due to:"
        echo "  - rocprofv3 crashed before generating output"
        echo "  - Output was written to a different location"
        echo ""
        echo "Check for output files:"
        find /tmp -name "*results.db" 2>/dev/null | head -5 || echo "  No .db files found in /tmp"
        echo ""
        echo "To manually try with pre-installed version:"
        echo "  /opt/rocm/bin/rocprofv3 -s -o /tmp/output -- python3.12 example.py"
    fi
    
    prompt_continue
}

#===============================================================================
# Step 9: Test with rocprof-sys-run
#===============================================================================

test_rocprof_sys() {
    print_header "Step 9: Test Profiling with rocprof-sys-run"
    
    echo "Command: rocprof-sys-run -- python3.12 example.py"
    echo ""
    print_warning "NOTE: This may crash or have issues depending on the build."
    echo "rocprofiler-systems is more complex and may require additional configuration."
    echo ""
    
    if ! prompt_yes_no "Run rocprof-sys-run test?" "y"; then
        print_warning "Skipping rocprof-sys-run test"
        return
    fi
    
    cd "$WORKSPACE_DIR"
    
    # Setup environment if available
    if [ -f /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh ]; then
        source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh
    fi
    
    # Check if rocprof-sys-run is available
    if ! command -v rocprof-sys-run &> /dev/null; then
        print_error "rocprof-sys-run not found in PATH"
        echo "Make sure rocprofiler-systems was built and installed correctly"
        return
    fi
    
    print_step "Running rocprof-sys-run..."
    echo ""
    echo -e "${YELLOW}─────────────── rocprof-sys-run Output ───────────────${NC}"
    
    rocprof-sys-run -- python3.12 example.py || {
        print_warning "rocprof-sys-run exited with non-zero status"
        echo "This is expected if there are configuration issues"
    }
    
    echo -e "${YELLOW}──────────────────────────────────────────────────────${NC}"
    echo ""
    
    prompt_continue
}

#===============================================================================
# Final Summary
#===============================================================================

print_summary() {
    print_header "Build Complete - Summary"
    
    echo "Components built from source:"
    echo ""
    
    # Check each component
    if [ -f "$ROCM_INSTALL_PREFIX/lib/libhsa-runtime64.so" ]; then
        print_success "rocr-runtime (HSA Runtime)"
    else
        print_warning "rocr-runtime - not found"
    fi
    
    if [ -f "$ROCM_INSTALL_PREFIX/lib/libamdhip64.so" ]; then
        print_success "CLR + HIP"
    else
        print_warning "CLR + HIP - not found"
    fi
    
    if command -v rocprofv3 &> /dev/null; then
        print_success "rocprofiler-sdk (rocprofv3)"
    else
        print_warning "rocprofiler-sdk - not found"
    fi
    
    if command -v rocprof-sys-run &> /dev/null; then
        print_success "rocprofiler-systems"
    else
        print_warning "rocprofiler-systems - not found"
    fi
    
    echo ""
    echo "Useful commands:"
    echo ""
    echo "  # Profile with rocprofv3"
    echo "  rocprofv3 -s -o /tmp/output -- python3.12 example.py"
    echo ""
    echo "  # Profile with rocprof-sys-python"
    echo "  rocprof-sys-python -- example.py"
    echo ""
    echo "  # View SQLite results"
    echo "  sqlite3 /tmp/output_results.db '.tables'"
    echo ""
    echo "  # Restart container later"
    echo "  docker start -ai rocm-tf-dev"
    echo ""
}

#===============================================================================
# Main Menu
#===============================================================================

show_menu() {
    print_header "ROCm Build from Source - Main Menu"
    
    echo "Select an option:"
    echo ""
    echo "  1) Run all steps (full build)"
    echo "  2) Pre-flight checks only"
    echo "  3) Install CMake first (required)"
    echo "  4) Clone repository only"
    echo "  5) Install dependencies only"
    echo "  6) Build rocr-runtime only"
    echo "  7) Build CLR + HIP only"
    echo "  8) Build ROCprofiler-SDK only"
    echo "  9) Build ROCprofiler-Systems only"
    echo "  10) Create example and test"
    echo "  0) Exit"
    echo ""
    read -p "$(echo -e ${CYAN}"Enter choice [0-10]: "${NC})" choice
    
    case $choice in
        1) run_all ;;
        2) preflight_checks ;;
        3) install_cmake_first ;;
        4) clone_rocm_systems ;;
        5) install_common_dependencies ;;
        6) build_rocr_runtime ;;
        7) build_clr_hip ;;
        8) build_rocprofiler_sdk ;;
        9) build_rocprofiler_systems ;;
        10) create_example_script && test_rocprofv3 && test_rocprof_sys ;;
        0) exit 0 ;;
        *) print_error "Invalid choice"; show_menu ;;
    esac
    
    show_menu
}

run_all() {
    preflight_checks
    install_cmake_first
    clone_rocm_systems
    install_common_dependencies
    build_rocr_runtime
    build_clr_hip
    build_rocprofiler_sdk
    build_rocprofiler_systems
    create_example_script
    test_rocprofv3
    test_rocprof_sys
    print_summary
}

#===============================================================================
# Entry Point
#===============================================================================

main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     ROCm Components Build from Source                            ║${NC}"
    echo -e "${CYAN}║     rocr-runtime | clr | hip | rocprofiler-sdk | systems         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Check for --all flag
    if [ "$1" = "--all" ] || [ "$1" = "-a" ]; then
        run_all
        exit 0
    fi
    
    show_menu
}

main "$@"

