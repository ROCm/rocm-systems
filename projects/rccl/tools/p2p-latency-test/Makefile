# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the MIT License.
#
# See LICENSE.txt for license information

ROCM_PATH ?= $(wildcard /opt/rocm)
HIPCC = $(ROCM_PATH)/bin/hipcc

all: p2p_latency_test ll_latency_test

CXXFLAGS = -g -O3

p2p_latency_test: p2p_latency_test.cpp
	@printf "Compiling  %-35s > %s\n" $< $@
	$(HIPCC) $(CXXFLAGS) $^ -o $@

ll_latency_test: ll_latency_test.cpp
	@printf "Compiling  %-35s > %s\n" $< $@
	$(HIPCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -f *.o p2p_latency_test ll_latency_test
