#!/bin/bash

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

SOURCE_DIR=..
BUILD_DIR=.

if [ -n "$1" ]; then SOURCE_DIR=$1; shift; fi
if [ -n "$1" ]; then BUILD_DIR=$1; shift; fi

echo "Testing 'cmake --build ${BUILD_DIR} --target all'..."
cmake --build ${BUILD_DIR} --target all
RET=$?

echo -n "Testing whether ${SOURCE_DIR}/include/timemory is a valid path... "
if [ ! -d ${SOURCE_DIR}/include/timemory ]; then
    RET=1;
fi
echo " Done"

if [ "${RET}" -ne 0 ]; then
    echo "Run from build directory within the source tree"
    exit 1
fi

for i in $(find ${SOURCE_DIR}/include/timemory -type f | egrep '\.(h|hpp|c|cpp)$')
do
    echo -n "Attempting to remove ${i}... "
    rm $i
    cmake --build ${BUILD_DIR} --target all &> /dev/null
    RET=$?
    if [ "${RET}" -ne 0 ]; then
        git checkout ${i} &> /dev/null
        echo "Failed"
    else
        git add ${i} &> /dev/null
        echo "Success"
    fi
done
