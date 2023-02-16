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

SRC_DIR=$(dirname "$0")
COMPONENT="rocprofiler"
ROCM_PATH="${ROCM_PATH:=/opt/rocm}"
LD_RUNPATH_FLAG=" -Wl,--enable-new-dtags -Wl,--rpath,$ROCM_PATH/lib:$ROCM_PATH/lib64"

usage() {
  echo -e "ROCProfiler Build Script Usage:"
  echo -e "\nTo run ./run.sh PARAMs, PARAMs can be the following:"
  echo -e "-h   | --help               For showing this message"
  echo -e "-b   | --build              For compiling"
  echo -e "-cb  | --clean-build        For full clean build"
  echo -e "-act | --asan-clean-build   For compiling with ASAN library attached"
  exit 1
}

while [ 1 ] ; do
  if [[ "$1" = "-h" || "$1" = "--help" ]] ; then
    usage
    exit 1
  elif [[ "$1" = "-b" || "$1" = "--build" ]] ; then
    TO_CLEAN=no
    shift
  elif [[ "$1" = "-acb" || "$1" = "--asan-clean-build" ]] ; then
    ASAN=True TO_CLEAN=yes
    shift
  elif [[ "$1" = "-cb" || "$1" = "--clean-build" ]] ; then
    TO_CLEAN=yes
    shift
  elif [[ "$1" = "-"* || "$1" = "--"* ]] ; then
    echo -e "Wrong option \"$1\", Please use the following options:\n"
    usage
    exit 1
  else
    break
  fi
done

umask 022

if [ -z "$ROCPROFILER_ROOT" ]; then ROCPROFILER_ROOT=$SRC_DIR; fi
if [ -z "$BUILD_DIR" ] ; then BUILD_DIR=build; fi
if [ -z "$BUILD_TYPE" ] ; then BUILD_TYPE="RelWithDebInfo"; fi
if [ -z "$PACKAGE_ROOT" ] ; then PACKAGE_ROOT=$ROCM_PATH; fi
if [ -z "$PREFIX_PATH" ] ; then PREFIX_PATH=$PACKAGE_ROOT; fi
if [ -z "$HIP_VDI" ] ; then HIP_VDI=0; fi
if [ -n "$ROCM_RPATH" ] ; then LD_RUNPATH_FLAG=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"; fi
if [ -z "$TO_CLEAN" ] ; then TO_CLEAN=yes; fi
if [ -z "$ASAN" ] ; then ASAN=False; fi

ROCPROFILER_ROOT=$(cd $ROCPROFILER_ROOT && echo $PWD)

if [ "$TO_CLEAN" = "yes" ] ; then rm -rf $BUILD_DIR; fi
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_MODULE_PATH=$ROCM_PATH/hip/cmake \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX="$PACKAGE_ROOT" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LD_RUNPATH_FLAG" \
    $ROCPROFILER_ROOT

make -j

exit 0
