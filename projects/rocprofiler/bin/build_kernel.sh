#!/bin/sh

TEST_NAME=$1
DST_DIR=$2

if [ -z "$TEST_NAME" ] ; then
  echo "Usage: $0 <test name> <dst dir>"
  echo "  Will look for <test name>.cl and will build <test name>.so dynamic object library"
  exit 1
fi

if [ -z "$DST_DIR" ] ; then
  DST_DIR=$(dirname TEST_NAME)
fi

GFXIP=$(/opt/rocm/bin/rocminfo | grep "amdgcn-amd-amdhsa--" | head -n 1 | sed -n "s/^.*amdgcn-amd-amdhsa--\(\w*\).*$/\1/p")
if [ -z "$GFXIP" ] ; then
  echo "GPU is not found"
  exit 1
fi

OBJ_PREF=$(echo $GFXIP | head -c 4)
OBJ_NAME=$(echo "_$(basename $TEST_NAME)" | sed -e 's/_./\U&\E/g' -e 's/_//g')
OBJ_FILE=${OBJ_PREF}_${OBJ_NAME}.hsaco

/opt/rocm/opencl/bin/x86_64/clang -cl-std=CL2.0 -cl-std=CL2.0 -include /opt/rocm/opencl/include/opencl-c.h -Xclang -mlink-bitcode-file -Xclang /opt/rocm/opencl/lib/x86_64/bitcode/opencl.amdgcn.bc -Xclang -mlink-bitcode-file -Xclang /opt/rocm/opencl/lib/x86_64/bitcode/ockl.amdgcn.bc -target amdgcn-amd-amdhsa -mcpu=$GFXIP -mno-code-object-v3 $TEST_NAME.cl -o $OBJ_FILE

echo "'$OBJ_FILE' is generated for '$GFXIP'"

exit 0
