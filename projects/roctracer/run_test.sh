#!/bin/bash
ROCM_PATH="/opt/rocm"

fatal() {
  echo "$1"
  exit 1
}

if [ -z "$BUILD_DIR" ] ; then export BUILD_DIR=$PWD; fi

cd $BUILD_DIR
./run.sh

exit 0
