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
#     make EMIT_LLVM_IR=0 NCCL_EMIT_LTO_IR=1 LTO_IR_ARCH=gfx942 ltoir
#
# Prerequisites:
#   - ROCm LLVM toolchain (clang++, opt, llvm-dis, llvm-as) under
#     $(ROCM_PATH)/llvm/bin (default ROCM_PATH=/opt/rocm).
#   - The RCCL CMake build has populated the hipify staging directory at
#     $(BUILDDIR)/hipify (same headers librccl.so and CMake llvm_ir use).
#     Example: cmake --build $(BUILDDIR) --target hipify_all
#

# Shared by llvm_ir and ltoir; kept outside the EMIT_LLVM_IR guard.

##### Knob defaults #####
# EMIT_LLVM_IR has no default (standalone `make llvm_ir` works unset); LTO IR is opt-in.
NCCL_EMIT_LTO_IR ?= 0

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

##### Final artifacts #####
# Defined unconditionally so `clean` removes them whatever the knobs are
# set to. The intermediates live under OBJDIR, which clean removes wholesale.
FINAL_BC       := $(LIBDIR)/librccl_device.bc
FINAL_LTO_IR   := $(LIBDIR)/librccl_device.ltoir

WRAPPER_HEADER := $(INCDIR)/nccl_device_wrapper.h

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

##### Public wrapper header (shared by both artifacts) #####
$(WRAPPER_HEADER): nccl_device_wrapper.h
	@mkdir -p $(INCDIR)
	@echo "Copying nccl_device_wrapper.h -> $@"
	cp -f $< $@

##########################################################################
# llvm_ir: librccl_device.bc
##########################################################################

ifneq ($(EMIT_LLVM_IR), 0)

##### Output files #####
UNOPTIMIZED_BC := $(OBJDIR)/librccl_device.bc.unoptimized
OPTIMIZED_BC   := $(OBJDIR)/librccl_device.bc.optimized
LLVM_IR_FILE   := $(OBJDIR)/librccl_device.ll

##### Compiler flags #####
# HIP device-only compilation. --offload-device-only is the modern
# (target-agnostic) spelling of NCCL's --cuda-device-only.
# Clang supports extended device lambdas but does not define nvcc's feature
# macro; the ReduceCopy device APIs require it.
COMMON_CLANG_FLAGS := \
	-std=$(BITCODE_CXX_STD) \
	-x hip \
	--rocm-path=$(ROCM_PATH) \
	--offload-device-only \
	--offload-arch=$(BITCODE_LIB_ARCH) \
	$(RCCL_INCLUDES) $(ROCM_INCLUDES) \
	-D__clang_llvm_bitcode_lib__ \
	-D__CUDACC_EXTENDED_LAMBDA__ \
	$(NCCL_DEVICE_DEFINES) \
	-D__HIP_PLATFORM_AMD__=1

CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS)

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
RCCL_DEVICE_BC_PUBLIC_API := ncclGetPeerPointerTeam,ncclCoopAnyInitThread,ncclCoopAnyInitWarp,ncclCoopAnyInitLanes,ncclCoopAnyInitWarpSpan,ncclCoopAnyInitCta,ncclCoopThreadRank,ncclCoopSize,ncclCoopNumThreads,ncclCoopSync,ncclLsaBarrierSessionInit,ncclLsaBarrierSessionArrive,ncclLsaBarrierSessionWait,ncclLsaBarrierSessionSync,ncclGinBarrierSessionInit,ncclGinBarrierSessionSync,ncclBarrierSessionInit,ncclBarrierSessionSync,$(subst $(RCCL_IR_SPACE),$(RCCL_IR_COMMA),$(strip $(RCCL_IR_REDUCE_COPY_API)))

OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-list=$(RCCL_DEVICE_BC_PUBLIC_API)

##### Rules #####
$(UNOPTIMIZED_BC): $(LLVM_SRC) $(WRAPPER_HEADER) util.h check_hipify
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

# ltoir: librccl_device.ltoir, RCCL's analogue of NCCL's deferred-inlining libnccl_device.ltoir.

ifneq ($(NCCL_EMIT_LTO_IR), 0)

# Mirror NCCL's variable names. NCCL defaults LTO_IR_ARCH to
# BITCODE_LIB_ARCH; it uses a different C++ std for the two artifacts
# (gnu++17 for the .bc, c++17 for the .ltoir), but RCCL has no reason to
# diverge the two, so this defaults to the same std as the .bc.
LTO_IR_ARCH    ?= $(BITCODE_LIB_ARCH)
LTO_IR_CXX_STD ?= $(BITCODE_CXX_STD)

LTO_IR_RAW := $(OBJDIR)/librccl_device.ltoir.raw

##### Compiler flags #####
# AMDGPU analogue of --ltoir: -O1 avoids -O0's optnone, -disable-llvm-passes defers inlining.
LTO_IR_CLANG_FLAGS := -c -emit-llvm -O1 -Xclang -disable-llvm-passes \
	-std=$(LTO_IR_CXX_STD) \
	-x hip \
	--rocm-path=$(ROCM_PATH) \
	--offload-device-only \
	--offload-arch=$(LTO_IR_ARCH) \
	$(RCCL_INCLUDES) $(ROCM_INCLUDES) \
	-D__clang_llvm_bitcode_lib__ \
	-D__NCCL_DEVICE_LTOIR_LIB__ \
	-D__CUDACC_EXTENDED_LAMBDA__ \
	$(NCCL_DEVICE_DEFINES) \
	-D__HIP_PLATFORM_AMD__=1

##### opt passes #####
# internalize only (roots = @llvm.used); inline/globaldce would turn this back into the .bc.
LTO_IR_OPT_PASSES := --passes=internalize

##### Rules #####
# Rebuild until all transitive headers and compiler configuration are tracked.
# The order-only preflight cannot invalidate LTO IR when staged headers or
# LTO_IR_ARCH change, so use the normal phony prerequisite as on llvm_ir.
$(LTO_IR_RAW): $(LLVM_SRC) $(WRAPPER_HEADER) util.h check_hipify
	@mkdir -p $(OBJDIR)
	@echo "Generating LTO IR (deferred inlining) for $(LTO_IR_ARCH)..."
	$(CLANG) $(LTO_IR_CLANG_FLAGS) $(LLVM_SRC) -o $@

# No llvm-dis/llvm-as round-trip and no ftz/align scrubbing on this path:
# NCCL strips nvvm-reflect-ftz metadata and align attributes because libNVVM
# rejects them. Both are NVVM-specific; AMDGPU has no analogue and needs none.
$(FINAL_LTO_IR): $(LTO_IR_RAW)
	@mkdir -p $(LIBDIR)
	@echo "Internalizing LTO IR -> $@"
	$(OPT) $(LTO_IR_OPT_PASSES) $< -o $@

ltoir: $(FINAL_LTO_IR) $(WRAPPER_HEADER)
	@echo "RCCL LTO IR generated successfully:"
	@echo "  C++ Standard:      $(LTO_IR_CXX_STD)"
	@echo "  GPU Architecture:  $(LTO_IR_ARCH)"
	@echo "  Raw LTO IR:        $(LTO_IR_RAW)"
	@echo "  Final LTO IR:      $(FINAL_LTO_IR)"
	@echo "  Wrapper Header:    $(WRAPPER_HEADER)"

else  # NCCL_EMIT_LTO_IR == 0

ltoir:
	@echo "RCCL LTO IR generation disabled (NCCL_EMIT_LTO_IR=0)"

endif

build: llvm_ir ltoir

clean:
	rm -rf $(OBJDIR) $(FINAL_BC) $(FINAL_LTO_IR) $(WRAPPER_HEADER)

.PHONY: llvm_ir ltoir build clean

# Serialize the two IR compiles under -j (order-only, as in NCCL's bindings/ir/Makefile).
ifneq ($(filter-out 0,$(EMIT_LLVM_IR)),)
ifneq ($(NCCL_EMIT_LTO_IR), 0)
$(LTO_IR_RAW): | $(FINAL_BC)
endif
endif
