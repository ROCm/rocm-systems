#!/bin/bash -x
SRC_DIR=`dirname $0`
COMPONENT="roctracer"
ROCM_PATH="/opt/rocm"

fatal() {
  echo "$1"
  exit 1
}

umask 022

if [ -z "$ROCTRACER_ROOT" ]; then ROCTRACER_ROOT=$SRC_DIR; fi
if [ -z "$BUILD_DIR" ] ; then BUILD_DIR=$PWD; fi
if [ -z "$HIP_PATH" ] ; then export HIP_PATH="$ROCM_PATH/hip"; fi
if [ -z "$HCC_HOME" ] ; then export HCC_HOME="$ROCM_PATH/hcc"; fi
if [ -z "$BUILD_TYPE" ] ; then BUILD_TYPE="release"; fi
if [ -z "$PACKAGE_ROOT" ] ; then PACKAGE_ROOT=$ROCM_PATH; fi
if [ -z "$PACKAGE_PREFIX" ] ; then PACKAGE_PREFIX="$ROCM_PATH/$COMPONENT"; fi
if [ -z "$PREFIX_PATH" ] ; then PREFIX_PATH=$PACKAGE_ROOT; fi

ROCTRACER_ROOT=$(cd $ROCTRACER_ROOT && echo $PWD)
MAKE_OPTS="-j 8 -C $BUILD_DIR"

mkdir -p $BUILD_DIR
pushd $BUILD_DIR

cmake \
    -DCMAKE_MODULE_PATH=$ROCTRACER_ROOT/cmake_modules \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX=$PACKAGE_ROOT \
    -DCPACK_PACKAGING_INSTALL_PREFIX=$PACKAGE_PREFIX \
    -DCPACK_GENERATOR="DEB;RPM" \
    $ROCTRACER_ROOT
make
make mytest
make package

exit 0
