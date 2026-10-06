# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

ROCM_PATH ?= /opt/rocm
HIPCC     = $(ROCM_PATH)/bin/hipcc
PYTHON    ?= python3

EXE       = fp8_bench
SRC       = fp8_bench.cpp
AUTO_H    = fp8_auto_probe.h
CXXFLAGS  = -std=c++17 -O3 --offload-arch=native

# Categories extracted from BuiltinsAMDGPU: cvt (default), or all
AUTO_CATEGORIES ?= cvt

all: $(EXE)

$(AUTO_H): gen_auto_probe.py
	$(PYTHON) gen_auto_probe.py --rocm-path $(ROCM_PATH) \
		--categories $(AUTO_CATEGORIES) -o $@

$(EXE): $(SRC) fp8_probe.h $(AUTO_H)
	$(HIPCC) $(CXXFLAGS) $(SRC) -o $@

run: $(EXE)
	./$(EXE)

# Rebuild auto header against whatever BuiltinsAMDGPU the toolchain ships
regen:
	rm -f $(AUTO_H)
	$(MAKE) $(AUTO_H)

clean:
	rm -f $(EXE) $(AUTO_H)

.PHONY: all run clean regen
