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
# Usage (standalone, from projects/rccl/bindings/ir):
#     make EMIT_LLVM_IR=1 BITCODE_LIB_ARCH=gfx942
#     make EMIT_LLVM_IR=1 RCCL_ROOT=/path/to/projects/rccl BUILDDIR=/path/to/build
#
# Prerequisites:
#   - ROCm LLVM toolchain (clang++, opt, llvm-dis, llvm-as) under
#     $(ROCM_PATH)/llvm/bin (default ROCM_PATH=/opt/rocm).
#   - The RCCL CMake build has populated the hipify staging directory at
#     $(BUILDDIR)/hipify (same headers librccl.so and CMake llvm_ir use).
#     Example: cmake --build $(BUILDDIR) --target hipify_all
#

ifneq ($(EMIT_LLVM_IR), 0)

##### Source #####
LLVM_SRC := nccl_device_wrapper__impl.h

##### RCCL root (bindings/ir -> projects/rccl) #####
RCCL_ROOT ?= $(abspath ../..)

##### Build directories (mirror RCCL CMake layout) #####
BUILDDIR   ?= $(RCCL_ROOT)/build
HIPIFY_DIR := $(BUILDDIR)/hipify
HIPIFY_INC := $(HIPIFY_DIR)/src/include
OBJDIR     := $(BUILDDIR)/obj/llvm_ir
LIBDIR     := $(BUILDDIR)/lib
INCDIR     := $(BUILDDIR)/include

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
# Match bindings/ir/CMakeLists.txt: compile against the hipify-staged
# headers librccl.so is built from, not raw src/include.
RCCL_INCLUDES := \
	-I$(INCDIR) \
	-I$(HIPIFY_INC) \
	-I$(HIPIFY_INC)/nccl_device \
	-I$(HIPIFY_DIR)/src/device

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
# Internalize everything outside the public surface, then inline and drop
# dead globals. opt's internalize pass takes EXACT symbol names, not globs;
# keep this list in sync with RCCL_DEVICE_BC_PUBLIC_API in CMakeLists.txt
# and the NCCL_IR_EXPORT entries in nccl_device_wrapper.h.
RCCL_DEVICE_BC_PUBLIC_API := ncclGetPeerPointerTeam,ncclCoopAnyInitThread,ncclCoopAnyInitWarp,ncclCoopAnyInitLanes,ncclCoopAnyInitWarpSpan,ncclCoopAnyInitCta,ncclCoopThreadRank,ncclCoopSize,ncclCoopNumThreads,ncclCoopSync,ncclLsaBarrierSessionInit,ncclLsaBarrierSessionArrive,ncclLsaBarrierSessionWait,ncclLsaBarrierSessionSync,ncclGinBarrierSessionInit,ncclGinBarrierSessionSync,ncclBarrierSessionInit,ncclBarrierSessionSync

OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-list=$(RCCL_DEVICE_BC_PUBLIC_API)

##### Preflight #####
.PHONY: check_hipify
check_hipify:
	@if [ ! -d "$(HIPIFY_INC)" ]; then \
		echo "ERROR: hipify staging directory not found: $(HIPIFY_INC)" >&2; \
		echo "Run the RCCL CMake build first to populate it, e.g.:" >&2; \
		echo "  cmake -B $(BUILDDIR) -DEMIT_LLVM_IR=ON $(RCCL_ROOT)" >&2; \
		echo "  cmake --build $(BUILDDIR) --target hipify_all" >&2; \
		exit 1; \
	fi

##### Rules #####
$(WRAPPER_HEADER): nccl_device_wrapper.h
	@mkdir -p $(INCDIR)
	@echo "Copying nccl_device_wrapper.h -> $@"
	cp -f $< $@

$(UNOPTIMIZED_BC): $(LLVM_SRC) check_hipify
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

.PHONY: llvm_ir build clean check_hipify
