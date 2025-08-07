#!/bin/bash

mkdir -p build
cd build
cmake -DROCM_DIR=/opt/rocm ..
make
cd ..

