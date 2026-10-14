#!/bin/bash

full=${full:-0}

export HIP_DIR=$(readlink -f ../hip)
echo "HIP_DIR: $HIP_DIR"
mkdir -p build
pushd build

ROCR_DIR=/data/rocr-runtime-install

if [[ ${full} -eq 1 ]]; then
  pip3 install CppHeaderParser
  rm -rf *
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_DIR -D__HIP_ENABLE_PCH=OFF \
    -DCMAKE_PREFIX_PATH="$ROCR_DIR;/opt/rocm" -Dhsa-runtime64_ROOT=$ROCR_DIR \
    -DCUSTOM_HSA_RUNTIME_PATH=$ROCR_DIR \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link=$ROCR_DIR/lib -Wl,-rpath=$ROCR_DIR/lib" \
    -DHIPCC_BIN_DIR=/opt/rocm/bin -DLLVM_ROOT=/opt/rocm/llvm \
    -Dclang=/usr/bin/clang-18
fi

make -j VERBOSE=1
popd
