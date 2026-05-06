# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AMD GPU Mega Kernel Unit Test - Makefile
#
# Supported offload-arch values (single source for per-arch binaries):
ARCHES := gfx950 gfx942 gfx90a gfx1150 gfx1151 gfx1152 gfx1201 gfx1200
ARCH_BINS := $(addprefix mega_kernel_test_,$(ARCHES))
#
#   CDNA (Compute/Datacenter):
#   - MI250/MI250X (CDNA2): gfx90a
#   - MI300/MI300X (CDNA3): gfx942  
#   - MI350/MI355X (CDNA4): gfx950
#
#   RDNA 3.5+ APU iGPU (gfx115x-class today; additional APUs may use this bucket later):
#   - Strix Point (RDNA 3.5): gfx1150
#   - Strix Halo  (RDNA 3.5): gfx1151
#   - Krackan     (RDNA 3.5): gfx1152
#
#   RDNA4 (Consumer):
#   - RX 9070 XT/9070 (RDNA4): gfx1201
#   - RX 9060 series (RDNA4): gfx1200
#
# Targets:
#   make          - Build for gfx1151 (default; override with GPU_ARCH=...)
#   make gfx950   - Build for MI350 (gfx950/CDNA4)
#   make gfx942   - Build for MI300 (gfx942/CDNA3)
#   make gfx90a   - Build for MI250 (gfx90a/CDNA2)
#   make gfx1150  - Build for Strix Point (gfx1150/RDNA3.5)
#   make gfx1151  - Build for Strix Halo (gfx1151/RDNA3.5)
#   make gfx1152  - Build for Krackan (gfx1152/RDNA3.5)
#   make gfx1201  - Build for RX 9070 XT (gfx1201/RDNA4)
#   make gfx1200  - Build for RX 9060 (gfx1200/RDNA4)
#   make all-arch - Build binaries for all ARCHES
#   make clean    - Remove build artifacts
#   make run      - Build and run the test

# Compiler
HIPCC ?= hipcc

# Default architecture: gfx1151 (Strix Halo / RDNA 3.5)
GPU_ARCH ?= gfx1151

# Common flags
HIPCC_FLAGS = -O2 -g -std=c++17
HIPCC_FLAGS += -Wall -Wextra
HIPCC_FLAGS += -fPIC

# Include paths
INCLUDES = -I.

# Source files
KERNEL_SRC = mega_kernel.hip
HOST_SRC = main.cpp

# Output
DEFAULT_TARGET = mega_kernel_test

# Build rules
.PHONY: all clean run $(ARCHES) all-arch help

all: $(DEFAULT_TARGET)

$(DEFAULT_TARGET): $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for $(GPU_ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=$(GPU_ARCH) $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

# One pattern rule for all per-arch binaries (stem is full token, e.g. gfx950)
$(ARCH_BINS): mega_kernel_test_%: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for $*..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=$* $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

# Symlink convenience: gfx950 -> mega_kernel_test_gfx950, etc.
gfx%: mega_kernel_test_gfx%
	@ln -sf $< mega_kernel_test
	@echo "Linked mega_kernel_test -> $<"

all-arch: $(ARCH_BINS)
	@echo "Built: $(ARCHES)"

# Run the test
run: $(DEFAULT_TARGET)
	@echo ""
	@echo "Running GPU Mega Kernel Test..."
	@echo ""
	./$(DEFAULT_TARGET)

# Clean
clean:
	rm -f mega_kernel_test mega_kernel_test_* *.o *.out

# Help
help:
	@echo "AMD GPU Mega Kernel Unit Test - Build System"
	@echo ""
	@echo "Supported Architectures (ARCHES in Makefile):"
	@echo "  $(ARCHES)"
	@echo ""
	@echo "  CDNA (Compute/Datacenter):"
	@echo "    gfx90a  - MI250/MI250X (CDNA2): FP32/FP64/FP16/BF16, HW FP64 atomics"
	@echo "    gfx942  - MI300/MI300X (CDNA3): + FP8/BF8 (FNUZ-style cvt), TF32, Sparse MFMA"
	@echo "    gfx950  - MI350/MI355X (CDNA4): + FP6/FP4, OCP FP8/BF8 paths where applicable, Async LDS, Dual-Issue"
	@echo ""
	@echo "  RDNA 3.5+ APU iGPU (gfx115x-class):"
	@echo "    gfx1150 - Strix Point: WMMA, TEX load/store tests (no CDNA-style FP8/BF8 cvt)"
	@echo "    gfx1151 - Strix Halo: same as gfx1150"
	@echo "    gfx1152 - Krackan: same as gfx1150"
	@echo ""
	@echo "  RDNA4 (Consumer):"
	@echo "    gfx1201 - RX 9070 XT/9070: FP8/BF8 OCP, Async LDS, Wave32 WMMA path"
	@echo "    gfx1200 - RX 9060 series:  FP8/BF8 OCP, Async LDS, Wave32 WMMA path"
	@echo ""
	@echo "Usage:"
	@echo "  make             - Build for gfx1151 (default; set GPU_ARCH to override)"
	@echo "  make gfx950      - Build for MI350 (gfx950/CDNA4)"
	@echo "  make gfx942      - Build for MI300 (gfx942/CDNA3)"
	@echo "  make gfx90a      - Build for MI250 (gfx90a/CDNA2)"
	@echo "  make gfx1150     - Build for Strix Point (gfx1150)"
	@echo "  make gfx1151     - Build for Strix Halo (gfx1151)"
	@echo "  make gfx1152     - Build for Krackan (gfx1152)"
	@echo "  make gfx1201     - Build for RX 9070 XT (gfx1201/RDNA4)"
	@echo "  make gfx1200     - Build for RX 9060 (gfx1200/RDNA4)"
	@echo "  make all-arch    - Build one binary per ARCHES"
	@echo "  make run         - Build and run the test"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make help        - Show this help message"
	@echo ""
	@echo "Environment Variables:"
	@echo "  GPU_ARCH         - Target GPU architecture (default: gfx1151)"
	@echo "  HIPCC            - HIP compiler path (default: hipcc)"
	@echo ""
	@echo "Example:"
	@echo "  make gfx1201     - Build for RX 9070 XT (RDNA4)"
	@echo "  make gfx942      - Build for MI300"
	@echo "  GPU_ARCH=gfx950 make   # example: MI350 (CDNA4)"
