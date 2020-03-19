#!/bin/sh -x

TEST_NAME=$1
DST_DIR=$2
ROCM_DIR=$3
TGT_LIST=$4

if [ -z "$TEST_NAME" ] ; then
  echo "Usage: $0 <test name> <dst dir>"
  echo "  Will look for <test name>.cl and will build <test name>.so dynamic object library"
  exit 1
fi

if [ -z "$DST_DIR" ] ; then
  DST_DIR=$(dirname TEST_NAME)
fi

if [ -z "$ROCM_DIR" ] ; then
  ROCM_DIR=/opt/rocm
fi

if [ -z "$TGT_LIST" ] ; then
  TGT_LIST=$(/opt/rocm/bin/rocminfo | grep "amdgcn-amd-amdhsa--" | head -n 1 | sed -n "s/^.*amdgcn-amd-amdhsa--\(\w*\).*$/\1/p")
fi

if [ -z "$TGT_LIST" ] ; then
  echo "Error: GPU targets not found"
  exit 1
fi

OCL_VER="2.0"
OCL_DIR=$ROCM_DIR/opencl

LLVM_DIR=$ROCM_DIR/hcc
CLANG=$LLVM_DIR/bin/clang
BITCODE_OPTS="\
  -Xclang -mlink-bitcode-file -Xclang $LLVM_DIR/lib/opencl.amdgcn.bc \
  -Xclang -mlink-bitcode-file -Xclang $LLVM_DIR/lib/ockl.amdgcn.bc \
  -Xclang -mlink-bitcode-file -Xclang $LLVM_DIR/lib/ocml.amdgcn.bc"

for GFXIP in $TGT_LIST ; do
  OBJ_PREF=$GFXIP
  OBJ_NAME=$(echo "_$(basename $TEST_NAME)" | sed -e 's/_./\U&\E/g' -e 's/_//g')
  OBJ_FILE=${OBJ_PREF}_${OBJ_NAME}.hsaco
  $CLANG -cl-std=CL$OCL_VER -include $OCL_DIR/include/opencl-c.h $BITCODE_OPTS -target amdgcn-amd-amdhsa -mcpu=$GFXIP -mno-code-object-v3 $TEST_NAME.cl -o $DST_DIR/$OBJ_FILE
  echo "'$OBJ_FILE' is generated for '$GFXIP'"
done

exit 0
