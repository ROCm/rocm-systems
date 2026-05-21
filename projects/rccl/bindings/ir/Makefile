#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Adapted from NVIDIA NCCL ir/Makefile (v2.29.2-1).
#
# See LICENSE.txt for license information
#
# LLVM IR / bitcode generation for RCCL device APIs.
#
# This Makefile compiles a single force-instantiation translation unit
# (nccl_device_wrapper__impl.h) with HIP/clang in device-only mode and
# emits a portable LLVM bitcode file (librccl_device.bc) plus a public
# C-style wrapper header (nccl_device_wrapper.h). The bitcode is intended
# to be linked by downstream code generators (Triton, MLIR, etc.) that
# need RCCL device-side primitives without recompiling RCCL from source.
#
# Usage (standalone):
#     make EMIT_LLVM_IR=1 BITCODE_LIB_ARCH=gfx950
#
# Prerequisites:
#   - ROCm LLVM toolchain (clang++, opt, llvm-dis, llvm-as) under
#     $(ROCM_PATH)/llvm/bin (default ROCM_PATH=/opt/rocm).
#   - The main RCCL build has been run at least once so that
#     $(BUILDDIR)/include/rccl.h (generated from src/nccl.h.in) exists.
#

ifneq ($(EMIT_LLVM_IR), 0)

##### Source #####
LLVM_SRC := nccl_device_wrapper__impl.h

##### Build directories (mirror NCCL/RCCL layout) #####
BUILDDIR ?= $(abspath ../build)
OBJDIR   := $(BUILDDIR)/obj/llvm_ir
LIBDIR   := $(BUILDDIR)/lib
INCDIR   := $(BUILDDIR)/include

##### RCCL source tree (relative to ir/) #####
RCCLSRC  := $(abspath ..)/src

##### ROCm toolchain #####
ROCM_PATH ?= /opt/rocm
ROCM_LLVM ?= $(ROCM_PATH)/llvm

CLANG    ?= $(ROCM_LLVM)/bin/clang++
OPT      ?= $(ROCM_LLVM)/bin/opt
LLVM_DIS ?= $(ROCM_LLVM)/bin/llvm-dis
LLVM_AS  ?= $(ROCM_LLVM)/bin/llvm-as

##### Build configuration #####
# Default GPU architecture; override on the command line, e.g.:
#     make BITCODE_LIB_ARCH=gfx942
BITCODE_LIB_ARCH ?= gfx950

# RCCL device APIs are C++17.
BITCODE_CXX_STD  ?= gnu++17

##### Output files #####
UNOPTIMIZED_BC := $(OBJDIR)/librccl_device.bc.unoptimized
OPTIMIZED_BC   := $(OBJDIR)/librccl_device.bc.optimized
LLVM_IR_FILE   := $(OBJDIR)/librccl_device.ll
FINAL_BC       := $(LIBDIR)/librccl_device.bc

WRAPPER_HEADER := $(INCDIR)/nccl_device_wrapper.h

##### Include paths #####
# RCCL's nccl_device/*.h headers are already HIP-aware (they branch on
# __HIP_PLATFORM_AMD__), so we point clang at the raw sources rather than
# at the CMake hipify staging directory.
RCCL_INCLUDES := \
	-I$(INCDIR) \
	-I$(RCCLSRC)/include \
	-I$(RCCLSRC)/include/nccl_device \
	-I$(RCCLSRC)/device

ROCM_INCLUDES := \
	-I$(ROCM_PATH)/include

##### Compiler flags #####
# HIP device-only compilation. --offload-device-only is the modern
# (target-agnostic) spelling of NCCL's --cuda-device-only.
COMMON_CLANG_FLAGS := \
	-std=$(BITCODE_CXX_STD) \
	-x hip \
	--rocm-path=$(ROCM_PATH) \
	--offload-device-only \
	--offload-arch=$(BITCODE_LIB_ARCH) \
	$(RCCL_INCLUDES) $(ROCM_INCLUDES) \
	-D__clang_llvm_bitcode_lib__ \
	-D__HIP_PLATFORM_AMD__=1

CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS)

##### opt passes #####
# Internalize anything not in the public ncclXxx surface, then inline
# and drop dead globals. Same recipe as NCCL.
OPT_PASSES := --passes='internalize,inline,globaldce' \
	-internalize-public-api-list='*nccl*'

##### Rules #####
$(WRAPPER_HEADER): nccl_device_wrapper.h
	@mkdir -p $(INCDIR)
	@echo "Copying nccl_device_wrapper.h -> $@"
	cp -f $< $@

$(UNOPTIMIZED_BC): $(LLVM_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Generating unoptimized LLVM bitcode for $(BITCODE_LIB_ARCH)..."
	$(CLANG) $(CLANG_FLAGS) $(LLVM_SRC) -o $@

$(OPTIMIZED_BC): $(UNOPTIMIZED_BC)
	@echo "Optimizing LLVM bitcode (internalize/inline/globaldce)..."
	$(OPT) $(OPT_PASSES) $< -o $@

# Round-trip through textual IR to keep the inspection artifact (.ll).
# AMDGPU has no equivalent of NCCL's 'nvvm-reflect-ftz' metadata cleanup,
# so the disassembled IR is emitted verbatim.
$(LLVM_IR_FILE): $(OPTIMIZED_BC)
	@echo "Disassembling bitcode -> textual LLVM IR..."
	$(LLVM_DIS) $< -o $@

$(FINAL_BC): $(LLVM_IR_FILE)
	@mkdir -p $(LIBDIR)
	@echo "Assembling final bitcode -> $@"
	$(LLVM_AS) $< -o $@

llvm_ir: $(FINAL_BC) $(OPTIMIZED_BC) $(WRAPPER_HEADER)
	@echo "RCCL LLVM IR and bitcode generated successfully:"
	@echo "  C++ Standard:      $(BITCODE_CXX_STD)"
	@echo "  GPU Architecture:  $(BITCODE_LIB_ARCH)"
	@echo "  Unoptimized BC:    $(UNOPTIMIZED_BC)"
	@echo "  Optimized BC:      $(OPTIMIZED_BC)"
	@echo "  LLVM IR (textual): $(LLVM_IR_FILE)"
	@echo "  Final BC:          $(FINAL_BC)"
	@echo "  Wrapper Header:    $(WRAPPER_HEADER)"

else  # EMIT_LLVM_IR == 0

llvm_ir:
	@echo "RCCL LLVM IR generation disabled (EMIT_LLVM_IR=0)"

endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR) $(FINAL_BC) $(WRAPPER_HEADER)

.PHONY: llvm_ir build clean
