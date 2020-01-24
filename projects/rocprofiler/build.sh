#!/bin/sh -x
BIN_DIR=`dirname $0`
BLD_DIR=$BIN_DIR/build

export CMAKE_PREFIX_PATH=/opt/rocm/include/hsa:/opt/rocm
rm -rf $BLD_DIR && mkdir $BLD_DIR && cd $BLD_DIR && cmake ..
make -j
make mytest
./run.sh
