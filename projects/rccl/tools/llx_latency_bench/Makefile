# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the MIT License.
#
# See LICENSE.txt for license information

ROCM_PATH ?= $(wildcard /opt/rocm)
HIPCC      = $(ROCM_PATH)/bin/hipcc
CXXFLAGS   = -O3 -std=c++17 --offload-arch=native -x hip

BINS = llx_latency_bench_64 llx_latency_bench_128 llx_latency_bench_256

all: $(BINS)

llx_latency_bench_64: llx_latency_bench.c
	$(HIPCC) $(CXXFLAGS) -DLINE_BYTES=64  $< -o $@

llx_latency_bench_128: llx_latency_bench.c
	$(HIPCC) $(CXXFLAGS) -DLINE_BYTES=128 $< -o $@

llx_latency_bench_256: llx_latency_bench.c
	$(HIPCC) $(CXXFLAGS) -DLINE_BYTES=256 $< -o $@

clean:
	rm -f $(BINS)
