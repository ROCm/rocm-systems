#!/bin/bash
make clean
make
echo -e "\n\n"

echo "# Test 1 : CPU Only"
./EmptyKernelTest 10 1 1 1 0
echo -e "\n\n"

echo "# Test 2 : GPU Only"
./EmptyKernelTest 10 1 1 0 1
echo -e "\n\n"

echo "# Test 3 : CPU and GPU"
./EmptyKernelTest 10 1 1 1 1 0
echo -e "\n\n"

echo "# Test 4 : Outer loop - CPU Only"
./EmptyKernelTest 10 1 1 0 0 1
echo -e "\n\n"
