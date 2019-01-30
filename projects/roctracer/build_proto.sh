#!/bin/bash -x
COMPONENT="roctracer"
ROCM_PATH="/opt/rocm"

fatal() {
  echo "$1"
  exit 1
}

umask 022

if [ -z "$ROCTRACER_ROOT" ]; then export ROCTRACER_ROOT=$PWD; fi
if [ -z "$BUILD_DIR" ] ; then export BUILD_DIR=$PWD; fi
if [ -z "$PACKAGE_ROOT" ] ; then export PACKAGE_ROOT=$ROCM_PATH; fi
if [ -z "$PACKAGE_PREFIX" ] ; then export PACKAGE_PREFIX="$ROCM_PATH/$COMPONENT"; fi

MAKE_OPTS="-j 8 -C $BUILD_DIR"

mkdir -p $BUILD_DIR
pushd $BUILD_DIR
cmake \
    -DCMAKE_MODULE_PATH=$ROCTRACER_ROOT/cmake_modules \
    -DCMAKE_INSTALL_PREFIX=$PACKAGE_ROOT \
    -DCPACK_PACKAGING_INSTALL_PREFIX=$PACKAGE_PREFIX \
    -DCPACK_GENERATOR="DEB;RPM" \
    $ROCTRACER_ROOT/inc/ext
popd

make package

exit 0
