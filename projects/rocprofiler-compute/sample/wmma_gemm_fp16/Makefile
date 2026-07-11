# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# FP16 WMMA GEMM sample for gfx1250 (MI450) metric health profiling.

HIPCC      ?= hipcc
GPU_ARCH   ?= gfx1250
HIPCC_FLAGS = -O2 -g -std=c++17 -Wall -Wextra -fPIC
INCLUDES    = -I.

TARGET      = wmma_gemm_fp16
KERNEL_SRC  = wmma_gemm_fp16.hip
HOST_SRC    = main.cpp

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(KERNEL_SRC) $(HOST_SRC) wmma_gemm_fp16_config.hpp wmma_gemm_fp16.hpp
	@echo "Building $(TARGET) for $(GPU_ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=$(GPU_ARCH) $(INCLUDES) -o $@ $(HOST_SRC) $(KERNEL_SRC)
	@echo "Build complete: $@"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

help:
	@echo "Targets:"
	@echo "  make            Build for gfx1250 (override: GPU_ARCH=gfx1250)"
	@echo "  make run        Build and run with defaults"
	@echo "  make clean      Remove binary"
