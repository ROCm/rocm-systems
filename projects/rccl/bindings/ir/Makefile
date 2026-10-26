#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Adapted from NVIDIA NCCL ir/Makefile (v2.29.2-1).
#
# See LICENSE.txt for more license information
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

##### LTO (deferred-inlining) bitcode artifact #####
# Opt-in: EMIT_LTO_BC=1 additionally emits librccl_device.lto.bc, a bitcode
# variant with no opt internalize/inline/globaldce pass applied, intended
# for downstream AMDGPU LTO consumption (-Xoffload-linker <bc>
# -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false), analogous
# to NCCL's nvcc -dlto --ltoir artifact.
EMIT_LTO_BC ?= 0

# Asking for the variant implies the library it is a variant of: the lto_bc
# rules live inside the EMIT_LLVM_IR block below, and $(FINAL_LTO_BC) already
# carries an order-only dependency on $(FINAL_BC). Without this, a bare
# `make EMIT_LTO_BC=1` reports "disabled (EMIT_LLVM_IR=0)" and silently
# produces nothing. `override` so it also holds for an explicit
# `EMIT_LLVM_IR=0` on the command line, which is a contradictory request.
ifneq ($(EMIT_LTO_BC), 0)
override EMIT_LLVM_IR := 1
endif

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

##### LTO (deferred-inlining) bitcode artifact outputs #####
# See the EMIT_LTO_BC knob at the top of this file.
LTO_BC_RAW   := $(OBJDIR)/librccl_device.lto.bc.raw
LTO_IR_FILE  := $(OBJDIR)/librccl_device.lto.ll
FINAL_LTO_BC := $(LIBDIR)/librccl_device.lto.bc

##### Device API defines #####
# Enable all ReduceCopy type and block-size combinations in generated IR.
NCCL_DEVICE_DEFINES := -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1

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
# Clang supports extended device lambdas but does not define nvcc's feature
# macro; the ReduceCopy device APIs require it.
BASE_CLANG_FLAGS := \
	-std=$(BITCODE_CXX_STD) \
	-x hip \
	--rocm-path=$(ROCM_PATH) \
	--offload-device-only \
	--offload-arch=$(BITCODE_LIB_ARCH) \
	$(RCCL_INCLUDES) $(ROCM_INCLUDES) \
	-D__CUDACC_EXTENDED_LAMBDA__ \
	$(NCCL_DEVICE_DEFINES) \
	-D__HIP_PLATFORM_AMD__=1

# Both artifacts are the bitcode library, so both define
# __clang_llvm_bitcode_lib__. Beyond inlining policy that macro also selects
# the caller-owned-LDS path in nccl_device/coop.h and extern "C" linkage for
# NCCL_IR_EXTERN_C in nccl_device/utility.h; an artifact that omits it gets a
# function-local addrspace(3) barrier array that the AMDGPU group-segment
# sizing pass cannot see through the vtable indirect call. See the macro
# commentary in nccl_device_wrapper.h.
BITCODE_LIB_DEFINE := -D__clang_llvm_bitcode_lib__

# Plain (collapsed) bitcode variant: thunks' callees are always_inline, and
# the opt internalize/inline/globaldce pass below collapses everything
# outside RCCL_DEVICE_BC_PUBLIC_API.
COMMON_CLANG_FLAGS := $(BASE_CLANG_FLAGS) $(BITCODE_LIB_DEFINE)

# Deferred-inlining ("LTO") variant: same library, one modifier on top.
# __RCCL_DEVICE_LTO_BC_LIB__ drops always_inline from the thunks' callees
# (see NCCL_DEVICE_INLINE in nccl_device_wrapper.h) and NO opt pass runs, so
# a downstream AMDGPU LTO link (lld) can inline across the real call site
# with full visibility.
LTO_BC_CLANG_FLAGS := $(BASE_CLANG_FLAGS) $(BITCODE_LIB_DEFINE) -D__RCCL_DEVICE_LTO_BC_LIB__

CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS)
LTO_BC_CLANG_COMPILE_FLAGS := -c -emit-llvm -O1 $(LTO_BC_CLANG_FLAGS)

##### opt passes #####
# Internalize everything outside the public surface, then inline and drop
# dead globals. opt's internalize pass takes EXACT symbol names, not globs;
# keep this list in sync with RCCL_DEVICE_BC_PUBLIC_API in CMakeLists.txt
# and the NCCL_IR_EXPORT entries in nccl_device_wrapper.h.
# ReduceCopy thunks: one per op and type, matching util.h.
RCCL_IR_ALL_TYPES      := I8 U8 I32 U32 I64 U64 F16 F32 F64 BF16
RCCL_IR_MULTIMEM_TYPES := I32 U32 I64 U64 F16 F32 F64 BF16
RCCL_IR_REDUCE_COPY_API := \
	$(foreach op,ncclLsaReduceSum ncclLsaCopy ncclLsaReduceSumCopy ncclLocalReduceSumCopy,$(foreach t,$(RCCL_IR_ALL_TYPES),$(op)_$(t))) \
	$(foreach op,ncclMultimemReduceSum ncclMultimemCopy ncclMultimemReduceSumCopy,$(foreach t,$(RCCL_IR_MULTIMEM_TYPES),$(op)_$(t)))
RCCL_IR_EMPTY :=
RCCL_IR_SPACE := $(RCCL_IR_EMPTY) $(RCCL_IR_EMPTY)
RCCL_IR_COMMA := ,
RCCL_DEVICE_BC_PUBLIC_API := ncclLsaBarrierSession_C_size,ncclGinBarrierSession_C_size,ncclBarrierSession_C_size,ncclDevComm_Rank,ncclDevComm_NRanks,ncclDevComm_LsaRank,ncclDevComm_LsaSize,ncclDevComm_LsaBarrier,ncclDevComm_RailGinBarrier,ncclDevComm_HybridLsaBarrier,ncclDevComm_HybridRailGinBarrier,ncclDevComm_WorldGinBarrier,ncclDevComm_LsaMultimem,ncclGetPeerPointerTeam,ncclCoopAnyInitThread,ncclCoopAnyInitWarp,ncclCoopAnyInitLanes,ncclCoopAnyInitWarpSpan,ncclCoopAnyInitCta,ncclCoopThreadRank,ncclCoopSize,ncclCoopNumThreads,ncclCoopSync,ncclLsaBarrierSessionInit,ncclLsaBarrierSessionArrive,ncclLsaBarrierSessionWait,ncclLsaBarrierSessionSync,ncclLsaBarrierSessionDestroy,ncclGinBarrierSessionInit,ncclGinBarrierSessionSync,ncclGinBarrierSessionDestroy,ncclBarrierSessionInit,ncclBarrierSessionSync,ncclBarrierSessionDestroy,$(subst $(RCCL_IR_SPACE),$(RCCL_IR_COMMA),$(strip $(RCCL_IR_REDUCE_COPY_API)))

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

# check_hipify is order-only (`|`): it is .PHONY, so as a normal prerequisite
# it is always "out of date" and forced a full clang recompile on every make.
# Order-only still runs the check first but does not make this target stale.
$(UNOPTIMIZED_BC): $(LLVM_SRC) $(WRAPPER_HEADER) util.h | check_hipify
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

##### LTO (deferred-inlining) bitcode rules #####
ifneq ($(EMIT_LTO_BC), 0)

# Order-only check_hipify, same reasoning as $(UNOPTIMIZED_BC) above.
$(LTO_BC_RAW): $(LLVM_SRC) $(WRAPPER_HEADER) util.h | check_hipify
	@mkdir -p $(OBJDIR)
	@echo "Generating deferred-inlining (LTO) LLVM bitcode for $(BITCODE_LIB_ARCH)..."
	$(CLANG) $(LTO_BC_CLANG_COMPILE_FLAGS) $(LLVM_SRC) -o $@

# No opt pass: this is the entire point of the LTO variant. The downstream
# AMDGPU LTO link (lld) decides inlining with the real caller visible.
$(LTO_IR_FILE): $(LTO_BC_RAW)
	@echo "Disassembling LTO bitcode -> textual LLVM IR..."
	$(LLVM_DIS) $< -o $@

# Order-only prerequisite on $(FINAL_BC): both rules `mkdir -p $(LIBDIR)`,
# so under `make -j` with both artifacts requested this avoids a concurrent
# mkdir race without forcing a rebuild when only $(FINAL_BC) is stale.
$(FINAL_LTO_BC): $(LTO_IR_FILE) | $(FINAL_BC)
	@mkdir -p $(LIBDIR)
	@echo "Assembling final LTO bitcode -> $@"
	$(LLVM_AS) $< -o $@

lto_bc: $(FINAL_LTO_BC)
	@echo "RCCL deferred-inlining (LTO) bitcode generated successfully:"
	@echo "  Raw LTO BC:        $(LTO_BC_RAW)"
	@echo "  LTO IR (textual):  $(LTO_IR_FILE)"
	@echo "  Final LTO BC:      $(FINAL_LTO_BC)"

else  # EMIT_LTO_BC == 0

lto_bc:
	@echo "RCCL LTO bitcode generation disabled (EMIT_LTO_BC=0)"

endif

else  # EMIT_LLVM_IR == 0

llvm_ir:
	@echo "RCCL LLVM IR generation disabled (EMIT_LLVM_IR=0)"

lto_bc:
	@echo "RCCL LTO bitcode generation disabled (EMIT_LLVM_IR=0)"

endif

build: llvm_ir lto_bc

clean:
	rm -rf $(OBJDIR) $(FINAL_BC) $(FINAL_LTO_BC) $(WRAPPER_HEADER)

.PHONY: llvm_ir lto_bc build clean check_hipify
