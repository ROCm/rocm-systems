#!/bin/bash -x
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
if [ -z "$HIP_PATH" ] ; then export HIP_PATH="$ROCM_PATH/hip"; fi
if [ -z "$BUILD_TYPE" ] ; then BUILD_TYPE="release"; fi
if [ -z "$PACKAGE_ROOT" ] ; then PACKAGE_ROOT=$ROCM_PATH; fi
if [ -z "$PACKAGE_PREFIX" ] ; then PACKAGE_PREFIX="$ROCM_PATH/$COMPONENT"; fi
if [ -z "$PREFIX_PATH" ] ; then PREFIX_PATH=$PACKAGE_ROOT; fi
if [ -z "$HIP_VDI" ] ; then HIP_VDI=0; fi
if [ -z "$HIP_API_STRING" ] ; then HIP_API_STRING=1; fi
if [ -n "$ROCM_RPATH" ] ; then LD_RUNPATH_FLAG=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"; fi

ROCTRACER_ROOT=$(cd $ROCTRACER_ROOT && echo $PWD)

if [ "$TO_CLEAN" = "yes" ] ; then rm -rf $BUILD_DIR; fi
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

cmake \
    -DCMAKE_MODULE_PATH=$ROCTRACER_ROOT/cmake_modules \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX=$PACKAGE_ROOT \
    -DCMAKE_SHARED_LINKER_FLAGS="$LD_RUNPATH_FLAG" \
    -DHIP_VDI="$HIP_VDI" \
    -DHIP_API_STRING="$HIP_API_STRING" \
    -DLIBRARY_TYPE=STATIC \
    $ROCTRACER_ROOT

make

exit 0
