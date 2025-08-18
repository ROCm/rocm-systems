#!/usr/bin/env bash
set -e

function print_help() {
  echo "Help:"
  echo "--build-hip           : build hip"
  echo "--build-opencl        : build opencl"
  echo "--build-all           : build both opencl and hip"
  echo "--rocm-path=<path>    : pass rocm path for clang search, defaults to /opt/rocm"
  echo "--install-path=<path> : install path, defaults to build/install"
  echo "--build-only          : just build, defaults to off"
}

declare -a input_args=("$@")

DEFAULT_ROCM_PATH="/opt/rocm"
BUILD_HIP=0
BUILD_OPENCL=0
DEFAULT_HIP_HEADER_PATH="$(pwd)/../hip"
CMAKE_OPTIONS=""
DEFAULT_INSTALL_PATH="$(pwd)/build/install"
BUILD_ONLY=0

for arg in "${input_args[@]}"; do
  if [[ $arg == "--help" || $arg == "-h" ]]; then
    print_help
    exit 0
  fi
  if [ $arg == "--build-hip" ]; then
    BUILD_HIP=1
    continue
  fi
  if [ $arg == "--build-opencl" ]; then
    BUILD_OPENCL=1
    continue
  fi
  if [ $arg == "--build-all" ]; then
    BUILD_HIP=1
    BUILD_OPENCL=1
    continue
  fi
  if [ $arg == "--build-only" ]; then
    BUILD_ONLY=1
    continue
  fi
  if [[ $arg == "--rocm-path="* ]]; then
    TEMP_ARG="--rocm-path="
    DEFAULT_ROCM_PATH="${arg/$TEMP_ARG/}"
  fi
  if [[ $arg == "--hip-header="* ]]; then
    TEMP_ARG="--hip-header="
    DEFAULT_HIP_HEADER_PATH="${arg/$TEMP_ARG/}"
  fi
  if [[ $arg == "install-path="* ]]; then
    TEMP_ARG="install-path="
    DEFAULT_INSTALL_PATH="${arg/$TEMP_ARG/}"
  fi
done

CLANG_C_PATH=$DEFAULT_ROCM_PATH"/llvm/bin/clang"
CLANG_CXX_PATH=$DEFAULT_ROCM_PATH"/llvm/bin/clang++"
LLVM_LIBRARY_PATH="$DEFAULT_ROCM_PATH/lib/llvm/lib"

if [ -z "$CLANG_CXX_PATH" ]; then
  echo "Can not find clang at path: $CLANG_CXX_PATH"
  exit 1
fi

if [[ "$BUILD_HIP" -eq 0 && "$BUILD_OPENCL" -eq 0 ]]; then
  echo "pass --build-hip, --build-opencl or --build-all to the script"
  exit 1
fi

if [ $BUILD_HIP -eq 1 ]; then
  CMAKE_OPTIONS+=" -DCLR_BUILD_HIP=ON"
fi
if [ $BUILD_OPENCL -eq 1 ]; then
  CMAKE_OPTIONS+=" -DCLR_BUILD_OCL=ON"
fi
CMAKE_OPTIONS+=" -DHIP_COMMON_DIR=${DEFAULT_HIP_HEADER_PATH}"
CMAKE_OPTIONS+=" -DCMAKE_INSTALL_PREFIX=${DEFAULT_INSTALL_PATH}"

echo "-------------Configure script---------------"
echo "--- ROCm Path:          ${DEFAULT_ROCM_PATH}"
echo "--- Found Clang C++ at: ${CLANG_CXX_PATH}"
echo "--- Found Clang C at:   ${CLANG_C_PATH}"
echo "--- Building HIP:       ${BUILD_HIP}"
echo "--- Building OpenCL:    ${BUILD_OPENCL}"
echo "--- HIP Header arg:     ${DEFAULT_HIP_HEADER_PATH}"
echo "--- Install Path:       ${DEFAULT_INSTALL_PATH}"
echo "--- CMake arg:          ${CMAKE_OPTIONS}"
echo "--- LLVM Library Path:  ${LLVM_LIBRARY_PATH}"
echo "--------------------------------------------"

mkdir -p build
cd build

cmake -DCMAKE_C_COMPILER="$CLANG_C_PATH" \
      -DCMAKE_CXX_COMPILER="$CLANG_CXX_PATH" \
      -DLLVM_LIBRARY_PATH="$LLVM_LIBRARY_PATH" \
      $CMAKE_OPTIONS \
      ..

if [ $BUILD_ONLY -eq 1 ]; then
  make -j$(nproc)
else
  make -j$(nproc) install
fi
