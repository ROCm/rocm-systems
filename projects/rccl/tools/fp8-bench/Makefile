# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

ROCM_PATH ?= /opt/rocm
HIPCC     = $(ROCM_PATH)/bin/hipcc

EXE       = fp8_bench
SRC       = fp8_bench.cpp
CXXFLAGS  = -std=c++17 -O3 --offload-arch=native

all: $(EXE)

$(EXE): $(SRC) fp8_probe.h
	$(HIPCC) $(CXXFLAGS) $(SRC) -o $@

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)

.PHONY: all run clean
