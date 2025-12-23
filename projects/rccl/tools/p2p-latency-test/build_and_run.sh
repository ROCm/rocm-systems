# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the MIT License.

make clean
make -j

# Example run: test one-way latency between GPU 0 and GPU 1 in both directions.
export HSA_FORCE_FINE_GRAIN_PCIE=1

echo Running p2p_latency_test using GPU pair 0 1
./p2p_latency_test 0 1
echo ""

sleep 1

echo Running p2p_latency_test using GPU pair 1 0
./p2p_latency_test 1 0
echo ""

sleep 1

echo Running ll_latency_test using GPU pair 0 1
./ll_latency_test 0 1
echo ""

sleep 1

echo Running ll_latency_test using GPU pair 1 0
./ll_latency_test 1 0
echo ""
