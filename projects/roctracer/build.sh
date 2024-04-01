#!/bin/bash -e

################################################################################
# Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
################################################################################

SRC_DIR=`dirname $0`
COMPONENT="roctracer"
ROCM_PATH="${ROCM_PATH:=/opt/rocm}"
LD_RUNPATH_FLAG=" -Wl,--enable-new-dtags -Wl,--rpath,$ROCM_PATH/lib:$ROCM_PATH/lib64"
DEFAULTS=defaults.sh

fatal() {
  echo "$1"
  exit 1
}

umask 022

if [ -e "$DEFAULTS" ] ; then source "$DEFAULTS"; fi

if [ -z "$ROCTRACER_ROOT" ]; then ROCTRACER_ROOT=$SRC_DIR; fi
if [ -z "$BUILD_DIR" ] ; then BUILD_DIR=$PWD; fi
if [ -z "$BUILD_TYPE" ] ; then BUILD_TYPE="release"; fi
if [ -z "$PACKAGE_ROOT" ] ; then PACKAGE_ROOT=$ROCM_PATH; fi
if [ -z "$PACKAGE_PREFIX" ] ; then PACKAGE_PREFIX="$ROCM_PATH/$COMPONENT"; fi
if [ -z "$PREFIX_PATH" ] ; then PREFIX_PATH=$PACKAGE_ROOT; fi
if [ -z "$HIP_VDI" ] ; then HIP_VDI=0; fi
if [ -n "$ROCM_RPATH" ] ; then LD_RUNPATH_FLAG=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"; fi
if [ -z "$GPU_LIST" ] ; then GPU_LIST="gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1100 gfx1101 gfx1102"; fi

ROCTRACER_ROOT=$(cd $ROCTRACER_ROOT && echo $PWD)

if [ "$TO_CLEAN" = "yes" ] ; then rm -rf $BUILD_DIR; fi
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

cmake \
    -DCMAKE_MODULE_PATH=$ROCM_PATH/hip/cmake \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX=$PACKAGE_ROOT \
    -DCPACK_PACKAGING_INSTALL_PREFIX=$PACKAGE_PREFIX \
    -DCPACK_GENERATOR="${CPACKGEN:-"DEB;RPM"}" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LD_RUNPATH_FLAG" \
    -DGPU_TARGETS="$GPU_LIST" \
    -DCPACK_OBJCOPY_EXECUTABLE="${PACKAGE_ROOT}/llvm/bin/llvm-objcopy" \
    -DCPACK_READELF_EXECUTABLE="${PACKAGE_ROOT}/llvm/bin/llvm-readelf" \
    -DCPACK_STRIP_EXECUTABLE="${PACKAGE_ROOT}/llvm/bin/llvm-strip" \
    -DCPACK_OBJDUMP_EXECUTABLE="${PACKAGE_ROOT}/llvm/bin/llvm-objdump" \
     $ROCTRACER_ROOT

make
make mytest
make package

exit 0
