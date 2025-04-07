#!/bin/bash

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
