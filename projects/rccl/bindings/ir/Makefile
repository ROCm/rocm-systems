#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Adapted from NVIDIA NCCL ir/Makefile (v2.29.2-1).
#
# See LICENSE.txt for more license information
#
# LLVM IR / bitcode generation for RCCL device APIs.
#
# This Makefile compiles a single force-instantiation translation unit
# (nccl_device_ir.cu) with HIP/clang in device-only mode and
# emits a portable LLVM bitcode file (librccl_device.bc) plus the public
# C-style ABI header (nccl_device_ir.h). The bitcode is intended
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
# variant compiled with no LLVM passes and internalized but never inlined,
# intended for downstream AMDGPU LTO consumption (-Xoffload-linker <bc>
# -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false), analogous
# to NCCL's nvcc -dlto --ltoir artifact.
#
# The deferral comes from -Xclang -disable-llvm-passes on the compile, not
# from the __RCCL_DEVICE_LTO_BC_LIB__ macro. Dropping always_inline from the
# callees is not sufficient on its own: LLVM's inliner gives internal,
# single-use callees a "last call to local" bonus and collapses them anyway,
# which is why this artifact was previously byte-equivalent to the plain .bc
# in everything but two out-of-line C++ functions. See LTO_BC_CLANG_COMPILE_FLAGS
# and LTO_OPT_PASSES below for the measurements.
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
LLVM_SRC := nccl_device_ir.cu

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

IR_HEADER := $(INCDIR)/nccl_device_ir.h

##### LTO (deferred-inlining) bitcode artifact outputs #####
# See the EMIT_LTO_BC knob at the top of this file.
LTO_BC_RAW   := $(OBJDIR)/librccl_device.lto.bc.raw
LTO_BC_INT   := $(OBJDIR)/librccl_device.lto.int.bc
LTO_IR_FILE  := $(OBJDIR)/librccl_device.lto.ll
FINAL_LTO_BC := $(LIBDIR)/librccl_device.lto.bc

##### Device API defines #####
# Enable all ReduceCopy type and block-size combinations in generated IR.
NCCL_DEVICE_DEFINES := -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1

##### Include paths #####
# Match bindings/ir/CMakeLists.txt: compile against the hipify-staged
# headers librccl.so is built from, not raw src/include.
#
# $(INCDIR) must come LAST. It is wanted only for the generated rccl.h /
# nccl.h, but CMake's copy_nccl_device_headers target also drops a raw,
# *un-hipified* copy of src/include/nccl_device there. hipify prepends
# `#include "hip/hip_runtime.h"` to every header it processes (that is
# where __forceinline__, __popc, __syncthreads come from) and rewrites
# __CUDACC__ to __HIPCC__; the raw copy has neither. With $(INCDIR)
# first, that copy shadows the hipified tree and the device TU fails to
# compile. Search hipify first so only rccl.h/nccl.h fall through.
RCCL_INCLUDES := \
	-I$(HIPIFY_INC) \
	-I$(HIPIFY_INC)/nccl_device \
	-I$(HIPIFY_DIR)/src/device \
	-I$(INCDIR)

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
# the caller-owned-LDS path in nccl_device/coop.h; an artifact that omits it
# gets a function-local addrspace(3) barrier array that the AMDGPU
# group-segment sizing pass cannot see through the vtable indirect call. See
# the macro commentary in nccl_device_ir.h.
#
# It no longer gates linkage: the public device API in src/include/nccl_device/
# is plain NCCL_DEVICE_INLINE and inlines into the thunks, and the thunks get
# their extern "C" from NCCL_IR_EXPORT plus the prototypes in
# bindings/ir/impl/.
BITCODE_LIB_DEFINE := -D__clang_llvm_bitcode_lib__

# Plain (collapsed) bitcode variant: thunks' callees are always_inline, and
# the opt internalize/inline/globaldce pass below collapses everything
# outside the @llvm.used export surface.
COMMON_CLANG_FLAGS := $(BASE_CLANG_FLAGS) $(BITCODE_LIB_DEFINE)

# Deferred-inlining ("LTO") variant: same library, one modifier on top.
# __RCCL_DEVICE_LTO_BC_LIB__ keys the NCCL_DEVICE_INLINE branch in
# nccl_device_ir.h, mirroring NCCL's __NCCL_DEVICE_LTOIR_LIB__.
LTO_BC_CLANG_FLAGS := $(BASE_CLANG_FLAGS) $(BITCODE_LIB_DEFINE) -D__RCCL_DEVICE_LTO_BC_LIB__

CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS)

# -Xclang -disable-llvm-passes is what actually makes this artifact deferred.
# It is the analogue of NCCL's `nvcc -dlto --ltoir`, which emits NVVM IR
# straight from the frontend and leaves every inlining decision to nvlink.
# Clang runs no LLVM pass at all, so the thunks' callees survive as separate
# functions for the downstream AMDGPU LTO link (lld) to re-decide with the
# real call site visible.
#
# -O1 is deliberate and must NOT be lowered to -O0. Measured on gfx950 /
# ROCm 7.2.4, on a function with the same shape as a real thunk:
#
#   -O0                              -> noinline optnone   (optnone even with
#   -O0 -Xclang -disable-llvm-passes -> noinline optnone    -disable-llvm-passes)
#   -O1                              -> fully optimized body, ~20 inferred
#                                       amdgpu-no-* attributes baked in
#   -O1 -Xclang -disable-llvm-passes -> convergent mustprogress nounwind,
#                                       pristine 53-line body
#
# optnone permanently forbids the link-time optimizer from touching the
# function, which is strictly worse than --ltoir: that artifact is
# unoptimized but still *optimizable*. -O1 sets CodeGenOpts.OptimizationLevel
# so the frontend stamps neither optnone nor noinline, while
# -disable-llvm-passes suppresses the pipeline. Only this pairing works.
LTO_BC_CLANG_COMPILE_FLAGS := -c -emit-llvm -O1 -Xclang -disable-llvm-passes $(LTO_BC_CLANG_FLAGS)

##### opt passes #####
# Internalize everything outside the public surface, then inline and drop
# dead globals.
#
# No -internalize-public-api-list. The export surface is already declared in
# the IR: NCCL_IR_EXPORT carries `used, retain`, so every thunk lands in
# @llvm.used (`appending addrspace(1) global [N x ptr]`), and opt's
# internalize pass treats @llvm.used entries as roots. Verified empirically --
# dropping the flag leaves the artifact byte-for-byte identical in symbol
# content (same exported `T` set, same `define internal` count) to the build
# that passed all 99 names explicitly.
#
# This replaces a hand-maintained 99-name list that was duplicated here and in
# CMakeLists.txt, had to be edited for every new entry point, and silently
# dropped a thunk to internal linkage whenever someone forgot. NCCL's recipe
# spells it "-internalize-public-api-list=nccl*", which does not work at all:
# opt's internalize pass takes EXACT symbol names, never globs, so the literal
# token "nccl*" matches nothing.

OPT_PASSES := --passes=internalize,inline,globaldce

# The LTO variant is OPT_PASSES minus exactly one pass: `inline`. That one
# pass is the whole difference between the two artifacts. internalize and
# globaldce only rewrite linkage and delete unreachable globals; neither
# inlines anything, so neither costs any deferral.
#
# internalize is mandatory, not cosmetic. Without it the artifact leaks
# C++-mangled symbols into its export surface; the plain .bc never showed
# this because its opt pass internalized them as a side effect. Measured on
# the pre-change artifact:
#
#   before: 97 defined T, 1 mangled (_ZN9ncclGin_CC1ERK11ncclDevComm...)
#   after : 96 defined T, 0 mangled, all 96 ncclIr* preserved via @llvm.used
#
# Pristine IR raises the stakes: with no LLVM passes run at compile time,
# every non-static function reachable from the headers would otherwise
# survive as an external mangled symbol.
#
# globaldce is equally mandatory, for a subtler reason. coop.h's
# ncclCoopBcast_{WarpSpan,Cta}_stash() hold function-local __shared__ arrays.
# __clang_llvm_bitcode_lib__ selects the caller-owned-LDS path so nothing
# calls them, but "uncalled" is not "absent": without DCE they survive as
# addrspace(3) globals that the AMDGPU group-segment sizing pass cannot
# account for through the vtable indirect call (HSA exception 0x1016).
# Measured: internalize alone leaves 4 addrspace(3) references,
# internalize,globaldce leaves 0, and the define count only moves
# 4737 -> 4567 -- i.e. it deletes dead code, not deferred callees.
LTO_OPT_PASSES := --passes=internalize,globaldce

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
$(IR_HEADER): nccl_device_ir.h
	@mkdir -p $(INCDIR)
	@echo "Copying nccl_device_ir.h -> $@"
	cp -f $< $@

# check_hipify is order-only (`|`): it is .PHONY, so as a normal prerequisite
# it is always "out of date" and forced a full clang recompile on every make.
# Order-only still runs the check first but does not make this target stale.
$(UNOPTIMIZED_BC): $(LLVM_SRC) $(IR_HEADER) util.h | check_hipify
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

llvm_ir: $(FINAL_BC) $(OPTIMIZED_BC) $(IR_HEADER)
	@echo "RCCL LLVM IR and bitcode generated successfully:"
	@echo "  C++ Standard:      $(BITCODE_CXX_STD)"
	@echo "  GPU Architecture:  $(BITCODE_LIB_ARCH)"
	@echo "  Unoptimized BC:    $(UNOPTIMIZED_BC)"
	@echo "  Optimized BC:      $(OPTIMIZED_BC)"
	@echo "  LLVM IR (textual): $(LLVM_IR_FILE)"
	@echo "  Final BC:          $(FINAL_BC)"
	@echo "  IR Header:         $(IR_HEADER)"

##### LTO (deferred-inlining) bitcode rules #####
ifneq ($(EMIT_LTO_BC), 0)

# Order-only check_hipify, same reasoning as $(UNOPTIMIZED_BC) above.
$(LTO_BC_RAW): $(LLVM_SRC) $(IR_HEADER) util.h | check_hipify
	@mkdir -p $(OBJDIR)
	@echo "Generating deferred-inlining (LTO) LLVM bitcode for $(BITCODE_LIB_ARCH)..."
	$(CLANG) $(LTO_BC_CLANG_COMPILE_FLAGS) $(LLVM_SRC) -o $@

# internalize + globaldce, but NOT inline: cleans the export surface and
# drops dead LDS globals without touching the deferred callees. See
# LTO_OPT_PASSES above.
$(LTO_BC_INT): $(LTO_BC_RAW)
	@echo "Internalizing LTO bitcode (internalize/globaldce)..."
	$(OPT) $(LTO_OPT_PASSES) $< -o $@

# Round-trip through textual IR to keep the inspection artifact (.ll), same
# as the plain path. The downstream AMDGPU LTO link (lld) decides inlining
# with the real caller visible.
$(LTO_IR_FILE): $(LTO_BC_INT)
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
	@echo "  Internalized BC:   $(LTO_BC_INT)"
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
	rm -rf $(OBJDIR) $(FINAL_BC) $(FINAL_LTO_BC) $(IR_HEADER)

.PHONY: llvm_ir lto_bc build clean check_hipify
