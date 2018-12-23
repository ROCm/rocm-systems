#!/bin/bash

fatal() {
  echo "$1"
  exit 1
}

if [ -z "$BUILD_DIR" ] ; then fatal "env var BUILD_DIR is not defined"; fi
if [ -z "$HIP_PATH" ] ; then fatal "env var HIP_PATH is not defined"; fi
if [ -z "$HCC_HOME" ] ; then fatal "env var HCC_HOME is not defined"; fi
if [ -z "$ROCTRACER_ROOT" ]; then fatal "env var ROCTRACER_ROOT is not defined"; fi
if [ -z "$BUILD_TYPE" ] ; then fatal "env var BUILD_TYPE is not defined"; fi
if [ -z "$PACKAGE_ROOT" ] ; then fatal "env var PACKAGE_ROOT is not defined"; fi
if [ -z "$PACKAGE_PREFIX" ] ; then fatal "env var PACKAGE_PREFIX is not defined"; fi
if [ -z "$ROCM_RPATH" ] ; then fatal "env var ROCM_RPATH is not defined"; fi

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
