# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# AMD GPU Mega Kernel Unit Test - Makefile
#
# Supported GPU Architectures:
#   CDNA (Compute/Datacenter):
#   - MI250/MI250X (CDNA2): gfx90a
#   - MI300/MI300X (CDNA3): gfx942  
#   - MI350/MI355X (CDNA4): gfx950
#
#   RDNA 3.5 (APU):
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
#   make all-arch - Build binaries for all supported offload-arch targets
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
.PHONY: all clean run gfx950 gfx942 gfx90a gfx1150 gfx1151 gfx1152 gfx1201 gfx1200 all-arch all-arch-full help

all: $(DEFAULT_TARGET)

$(DEFAULT_TARGET): $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for $(GPU_ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=$(GPU_ARCH) $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

# Explicit architecture-specific build rules
mega_kernel_test_gfx950: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for gfx950 (MI350/CDNA4)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=gfx950 $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

mega_kernel_test_gfx942: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for gfx942 (MI300/CDNA3)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=gfx942 $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

mega_kernel_test_gfx90a: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for gfx90a (MI250/CDNA2)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=gfx90a $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

mega_kernel_test_gfx1150: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for gfx1150 (Strix Point/RDNA3.5)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=gfx1150 $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

mega_kernel_test_gfx1151: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for gfx1151 (Strix Halo/RDNA3.5)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=gfx1151 $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

mega_kernel_test_gfx1152: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for gfx1152 (Krackan/RDNA3.5)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=gfx1152 $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

mega_kernel_test_gfx1201: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for gfx1201 (RX 9070 XT/RDNA4)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=gfx1201 $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

mega_kernel_test_gfx1200: $(KERNEL_SRC) $(HOST_SRC)
	@echo "Building for gfx1200 (RX 9060/RDNA4)..."
	$(HIPCC) $(HIPCC_FLAGS) --offload-arch=gfx1200 $(INCLUDES) -o $@ $^
	@echo "Build complete: $@"

# Architecture-specific targets (using gfx names)
gfx950: mega_kernel_test_gfx950
	@ln -sf mega_kernel_test_gfx950 mega_kernel_test
	@echo "Linked mega_kernel_test -> mega_kernel_test_gfx950"

gfx942: mega_kernel_test_gfx942
	@ln -sf mega_kernel_test_gfx942 mega_kernel_test
	@echo "Linked mega_kernel_test -> mega_kernel_test_gfx942"

gfx90a: mega_kernel_test_gfx90a
	@ln -sf mega_kernel_test_gfx90a mega_kernel_test
	@echo "Linked mega_kernel_test -> mega_kernel_test_gfx90a"

gfx1150: mega_kernel_test_gfx1150
	@ln -sf mega_kernel_test_gfx1150 mega_kernel_test
	@echo "Linked mega_kernel_test -> mega_kernel_test_gfx1150"

gfx1151: mega_kernel_test_gfx1151
	@ln -sf mega_kernel_test_gfx1151 mega_kernel_test
	@echo "Linked mega_kernel_test -> mega_kernel_test_gfx1151"

gfx1152: mega_kernel_test_gfx1152
	@ln -sf mega_kernel_test_gfx1152 mega_kernel_test
	@echo "Linked mega_kernel_test -> mega_kernel_test_gfx1152"

gfx1201: mega_kernel_test_gfx1201
	@ln -sf mega_kernel_test_gfx1201 mega_kernel_test
	@echo "Linked mega_kernel_test -> mega_kernel_test_gfx1201"

gfx1200: mega_kernel_test_gfx1200
	@ln -sf mega_kernel_test_gfx1200 mega_kernel_test
	@echo "Linked mega_kernel_test -> mega_kernel_test_gfx1200"

# Build one binary per listed offload-arch (no symlink)
all-arch: mega_kernel_test_gfx950 mega_kernel_test_gfx942 mega_kernel_test_gfx90a mega_kernel_test_gfx1150 mega_kernel_test_gfx1151 mega_kernel_test_gfx1152 mega_kernel_test_gfx1201 mega_kernel_test_gfx1200
	@echo "Built: gfx950 gfx942 gfx90a gfx1150 gfx1151 gfx1152 gfx1201 gfx1200"

# Build all including experimental (requires newer ROCm)
all-arch-full: mega_kernel_test_gfx950 mega_kernel_test_gfx942 mega_kernel_test_gfx90a mega_kernel_test_gfx1150 mega_kernel_test_gfx1151 mega_kernel_test_gfx1152 mega_kernel_test_gfx1201 mega_kernel_test_gfx1200
	@echo "Built for all architectures: gfx950, gfx942, gfx90a, gfx1150, gfx1151, gfx1152, gfx1201, gfx1200"

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
	@echo "Supported Architectures:"
	@echo ""
	@echo "  CDNA (Compute/Datacenter):"
	@echo "    gfx90a  - MI250/MI250X (CDNA2): FP32/FP64/FP16/BF16, HW FP64 atomics"
	@echo "    gfx942  - MI300/MI300X (CDNA3): + FP8/BF8, TF32, Sparse MFMA"
	@echo "    gfx950  - MI350/MI355X (CDNA4): + FP6/FP4, Async LDS, Dual-Issue"
	@echo ""
	@echo "  RDNA 3.5 (APU iGPU):"
	@echo "    gfx1150 - Strix Point: WMMA, TEX load/store tests (no CDNA-style FP8/BF8 cvt)"
	@echo "    gfx1151 - Strix Halo: same as gfx1150"
	@echo "    gfx1152 - Krackan: same as gfx1150"
	@echo ""
	@echo "  RDNA4 (Consumer):"
	@echo "    gfx1201 - RX 9070 XT/9070: FP8/BF8 OCP, Async LDS, Raytracing"
	@echo "    gfx1200 - RX 9060 series:  FP8/BF8 OCP, Async LDS, Raytracing"
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
	@echo "  make all-arch    - Build one binary per supported offload-arch"
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
