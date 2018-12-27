#!/bin/bash
COMPONENT="roctracer"
ROCM_PATH="/opt/rocm"

fatal() {
  echo "$1"
  exit 1
}

if [ -z "$ROCTRACER_ROOT" ]; then export ROCTRACER_ROOT=$PWD; fi
if [ -z "$BUILD_DIR" ] ; then export BUILD_DIR=$PWD; fi
if [ -z "$HIP_PATH" ] ; then export HIP_PATH="$ROCM_PATH/hip"; fi
if [ -z "$HCC_HOME" ] ; then export HCC_HOME="$ROCM_PATH/hcc"; fi
if [ -z "$BUILD_TYPE" ] ; then export BUILD_TYPR="release"; fi
if [ -z "$PACKAGE_ROOT" ] ; then export PACKAGE_ROOT=$ROCM_PATH; fi
if [ -z "$PACKAGE_PREFIX" ] ; then export PACKAGE_PREFIX="$ROCM_PATH/$COMPONENT"; fi

MAKE_OPTS="-j 8 -C $BUILD_DIR"

mkdir -p $BUILD_DIR
pushd $BUILD_DIR
cmake \
    -DCMAKE_MODULE_PATH=$ROCTRACER_ROOT/cmake_modules \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH="$PACKAGE_ROOT/hsa/include/hsa;$PACKAGE_ROOT/hsa/lib;$PACKAGE_ROOT/libhsakmt/lib" \
    -DCMAKE_INSTALL_PREFIX=$PACKAGE_ROOT \
    -DCPACK_PACKAGING_INSTALL_PREFIX=$PACKAGE_PREFIX \
    -DCPACK_GENERATOR="DEB;RPM" \
    $ROCTRACER_ROOT
popd

make $MAKE_OPTS
make $MAKE_OPTS mytest
make $MAKE_OPTS install
make $MAKE_OPTS package

exit 0
