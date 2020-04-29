#!/bin/sh -x
SO_EXT="hsaco"

TEST_NAME=$1
DST_DIR=$2
ROCM_DIR=$3
TGT_LIST=$4

if [ -z "$TEST_NAME" ] ; then
  echo "Usage: $0 <test name> <dst dir>"
  echo "  Will look for <test name>.cl and will build <test name>.$SO_EXT dynamic code object library"
  exit 1
fi
OBJ_NAME=$(echo "_$(basename $TEST_NAME)" | sed -e 's/_./\U&\E/g' -e 's/_//g')

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

if [ -e $ROCM_DIR/llvm ] ; then
  LLVM_DIR=$ROCM_DIR/llvm
  LIB_DIR=$ROCM_DIR/lib
else
  LLVM_DIR=$ROCM_DIR/hcc
  LIB_DIR=$LLVM_DIR/lib
fi

BC_DIR=$LIB_DIR/bitcode
if [ ! -d "$BC_DIR" ] ; then BC_DIR=$LIB_DIR; fi

CLANG_ROOT=$LLVM_DIR/lib/clang
CLANG_DIR=`ls -d $CLANG_ROOT/* | head -n 1`
if [ "$CLANG_DIR" = "" ] ; then
  echo "Error: LLVM clang library was not found"
  exit 1
fi

BIN_DIR=$LLVM_DIR/bin
INC_DIR=$CLANG_DIR/include
BITCODE_OPTS="\
  -Xclang -mlink-bitcode-file -Xclang $BC_DIR/opencl.amdgcn.bc \
  -Xclang -mlink-bitcode-file -Xclang $BC_DIR/ockl.amdgcn.bc \
  -Xclang -mlink-bitcode-file -Xclang $BC_DIR/ocml.amdgcn.bc"

for GFXIP in $TGT_LIST ; do
  OBJ_PREF=$GFXIP
  OBJ_FILE="${OBJ_PREF}_${OBJ_NAME}.$SO_EXT"
  $BIN_DIR/clang -cl-std=CL$OCL_VER -include $INC_DIR/opencl-c.h $BITCODE_OPTS -target amdgcn-amd-amdhsa -mcpu=$GFXIP -mno-code-object-v3 $TEST_NAME.cl -o $DST_DIR/$OBJ_FILE
  echo "'$OBJ_FILE' generated"
done

exit 0
