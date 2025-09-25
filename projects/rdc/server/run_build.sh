#!/bin/bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

mkdir -p build
cd build
cmake -DROCM_DIR=/opt/rocm ..
make
cd ..

