# Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
ROCM_PATH ?= /opt/rocm
HIPCC = $(ROCM_PATH)/bin/hipcc
HIPIFY_PL_EXE ?= $(ROCM_PATH)/bin/hipify-perl
HIPIFY_PL_FLAGS = -inplace -quiet-warnings

# Detect ROCm version using CMake (reads from rocm-core package or .info/version file)
ROCM_VERSION_STRING := $(shell cmake -DROCM_PATH='$(ROCM_PATH)' -P get_rocm_version.cmake 2>&1 | tail -1)
ifeq ($(ROCM_VERSION_STRING),)
$(error Failed to detect ROCm version. Ensure ROCM_PATH is set correctly and ROCm is installed.)
endif
ROCM_MAJOR := $(shell echo $(ROCM_VERSION_STRING) | cut -d. -f1)
ROCM_MINOR := $(shell echo $(ROCM_VERSION_STRING) | cut -d. -f2)
ROCM_PATCH := $(shell echo $(ROCM_VERSION_STRING) | cut -d. -f3)
ROCM_VERSION := $(shell echo $$((10000 * $(ROCM_MAJOR) + 100 * $(ROCM_MINOR) + $(ROCM_PATCH))))

.DEFAULT_GOAL := all

EXE = topo_expl

# Detect host OS so that headers like src/include/os.h can declare the
# correct ncclSocketDescriptor/ncclAffinity typedefs. The main RCCL build
# does the equivalent in CMakeLists.txt; topo_expl needs the same flag.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
NCCL_OS_DEFINE := -DNCCL_OS_LINUX
else
NCCL_OS_DEFINE := -DNCCL_OS_WINDOWS
endif

CXXFLAGS = -g -ffunction-sections -fdata-sections -DROCM_VERSION=$(ROCM_VERSION) \
           -std=c++17 \
           -Wl,--gc-sections -fgpu-rdc \
           -Iinclude -Itest -Ihipify_rccl/include -Ihipify_rccl/include/plugin \
		   -Ihipify_rccl/include/nccl_device \
           -Ihipify_rccl/src/device/include -Ihipify_rccl/graph -I/opt/rocm/include/ \
           -DTOPO_EXPL -DENABLE_TRACE -DENABLE_LL128 -DNVTX_NO_IMPL -DRCCL_EXPOSE_STATIC \
           $(NCCL_OS_DEFINE) \
           -lpthread

TOPO_EXPL_FILES = src/$(EXE).cpp src/topo_expl_api.cpp src/model.cpp src/topo_expl_impl.cpp src/utils.cpp src/stubs.cc test/topo_expl_tests.cpp

RCCL_FILES = hipify_rccl/graph/topo.cc \
			hipify_rccl/graph/rings.cc \
			hipify_rccl/graph/paths.cc \
			hipify_rccl/graph/trees.cc \
			hipify_rccl/graph/search.cc \
			hipify_rccl/graph/connect.cc \
			hipify_rccl/graph/tuning.cc \
			hipify_rccl/graph/xml.cc \
			hipify_rccl/graph/rome_models.cc \
			hipify_rccl/graph/archinfo.cc \
			../../src/misc/param.cc \
			../../src/misc/nvmlwrap_stub.cc \
			../../src/rccl_wrap.cc \
			hipify_rccl/collectives.cc \
			hipify_rccl/register.cc \
			hipify_rccl/enqueue.cc

files = $(TOPO_EXPL_FILES) $(RCCL_FILES)

FMT_DIR     := third_party/fmt
FMT_INCLUDE := $(FMT_DIR)/include
FMT_HEADER  := $(FMT_INCLUDE)/fmt/format.h
FMT_GIT     := https://github.com/fmtlib/fmt.git

# Probe for <fmt/format.h> using a portable pipe (no <<< here-string)
HAVE_FMT := $(shell echo '#include <fmt/format.h>' | \
               $(HIPCC) -xc++ -std=c++17 -E - >/dev/null 2>&1 && echo yes || echo no)

ifeq ($(HAVE_FMT),no)
  CXXFLAGS += -I$(FMT_INCLUDE)
  NEED_FMT  := 1
endif

ifeq ($(NEED_FMT),1)
$(FMT_HEADER):
	rm -rf third_party/fmt
	@echo ">>> fmt not found; cloning $(FMT_GIT) ..."
	@mkdir -p $(dir $(FMT_DIR))
	@git clone --depth=1  $(FMT_GIT) $(FMT_DIR)
else
$(FMT_HEADER):
endif

all: $(FMT_HEADER) hipify $(EXE)
	@echo "Built with ROCm version: $(ROCM_MAJOR).$(ROCM_MINOR).$(ROCM_PATCH) (ROCM_VERSION=$(ROCM_VERSION))"

$(EXE): $(files)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

hipify:
	rm -rf hipify_rccl
	mkdir -p hipify_rccl/src/device/include hipify_rccl/include/network/unpack hipify_rccl/device/symmetric include/rccl
	# Copy nccl.h.in from rccl source to always get latest API definitions
	cp -a ../../src/nccl.h.in include/nccl.h
	# Also copy to rccl/rccl.h for files that use #include <rccl/rccl.h>
	cp -a ../../src/nccl.h.in include/rccl/rccl.h
	cp -a ../../src/include/ hipify_rccl/
	cp -a ../../src/graph/ hipify_rccl/
	cp -a ../../src/device/*.h hipify_rccl/src/device/include
	cp -a ../../src/device/network/unpack/*.h hipify_rccl/include/network/unpack
	# src/include/sym_kernels.h includes "../device/symmetric/gin_scratch.h"
	# (relative path), so mirror those headers under hipify_rccl/device/symmetric/.
	cp -a ../../src/device/symmetric/*.h hipify_rccl/device/symmetric/
	cp -a ../../src/enqueue.cc hipify_rccl/
	cp -a ../../src/register/register.cc hipify_rccl/
	cp -a ../../src/collectives.cc hipify_rccl/
	cp -a ../../src/misc/archinfo.cc hipify_rccl/graph/
	# Hipify all header and source files (CUDA -> HIP).
	# Walk every subdirectory under hipify_rccl/include/ recursively so that
	# any new subtrees added by future NCCL syncs (e.g. rma/, gin/, ionic/,
	# mlx5/, algorithms/, etc.) are covered automatically.
	# nvtx3/ is intentionally excluded because we swap in a stub via PATCH 2.
	@echo ">>> Hipifying RCCL header files..."
	find hipify_rccl/include -type f -name '*.h' -not -path 'hipify_rccl/include/nvtx3/*' \
	    -exec $(HIPIFY_PL_EXE) $(HIPIFY_PL_FLAGS) {} +
	$(HIPIFY_PL_EXE) $(HIPIFY_PL_FLAGS) hipify_rccl/src/device/include/*.h
	# PATCH 1: Fix device includes - remove "device/" prefix since files are in same directory
	sed -i 's|#include "device/|#include "|g' hipify_rccl/src/device/include/*.h
	sed -i 's|#include "device/|#include "|g' hipify_rccl/include/*.h
	sed -i 's|#include "device/|#include "|g' hipify_rccl/*.cc
	# PATCH 2: Use NVTX stubs - topo_expl doesn't need full NVTX implementation
	cp hipify_rccl/include/nvtx_stub.h hipify_rccl/include/nvtx.h
	bash ../../cmake/scripts/add_unroll.sh "hipify_rccl/src/device/include/common.h"
	@echo ">>> Hipifying RCCL source files..."
	$(HIPIFY_PL_EXE) $(HIPIFY_PL_FLAGS) hipify_rccl/graph/*
	$(HIPIFY_PL_EXE) $(HIPIFY_PL_FLAGS) hipify_rccl/include/network/unpack/*
	$(HIPIFY_PL_EXE) $(HIPIFY_PL_FLAGS) hipify_rccl/*.cc
	# PATCH 3: Generate *_tmp.h files - these are build-time generated headers in rccl
	cp hipify_rccl/include/nccl_device/core.h hipify_rccl/include/nccl_device/core_tmp.h
	cp hipify_rccl/include/nccl_device/comm.h hipify_rccl/include/nccl_device/comm_tmp.h
	cp hipify_rccl/include/nccl_device/gin.h hipify_rccl/include/nccl_device/gin_tmp.h
	# PATCH 4: Add missing NVTX stub macros for topo_expl build
	echo '#define NCCL_NVTX3_FUNC_RANGE do {} while(0)' >> hipify_rccl/include/nvtx.h
	echo 'inline void initNvtxRegisteredEnums() {}' >> hipify_rccl/include/nvtx.h
	# PATCH 5: Add nccl_tuner.h include to get NCCL_NUM_ALGORITHMS and NCCL_NUM_PROTOCOLS macros
	sed -i '/#include "core.h"/a #include "plugin/nccl_tuner.h"' hipify_rccl/include/rccl_common.h
	# PATCH 7: hipify-perl rewrites __CUDACC__ -> __HIPCC__ in nccl_device/*.h.
	# Restore the original __CUDACC__ guard so device-only blocks are skipped
	# on the topo_expl host pass (no -x hip device compile here). This keeps
	# caller-side guards aligned with the device-API *declarations* in
	# nccl_device/core.h that are also gated on __CUDACC__.
	find hipify_rccl/include/nccl_device -type f -name '*.h' \
	    -exec sed -i 's|__HIPCC__|__CUDACC__|g' {} +
	# PATCH 7b: nccl_device/hip_compat.h binds NCCL_CHECK_CUDACC to
	# NCCL_DEVICE_COMPILE, which is true under defined(__HIP_PLATFORM_AMD__)
	# on both host and device passes. Without this patch, callers under
	# `#if NCCL_CHECK_CUDACC` (e.g. gin_scratch__types.h, ll_a2a__funcs.h,
	# barrier__funcs.h) would reference device-API declarations from
	# nccl_device/core.h that were skipped by PATCH 7's __CUDACC__ guard,
	# producing 'undeclared identifier' errors. Force NCCL_CHECK_CUDACC to
	# track __CUDACC__ so both sides are uniformly skipped on the host build.
	sed -i 's|^#define NCCL_CHECK_CUDACC NCCL_DEVICE_COMPILE$$|#define NCCL_CHECK_CUDACC __CUDACC__|' \
	    hipify_rccl/include/nccl_device/hip_compat.h
	# PATCH 6: Guard RCCL_EXPOSE_STATIC so -DRCCL_EXPOSE_STATIC from CXXFLAGS does not trigger redefinition warning
	sed -i 's|^#define RCCL_EXPOSE_STATIC \(.*\)|#ifndef RCCL_EXPOSE_STATIC\n#define RCCL_EXPOSE_STATIC \1\n#endif|' hipify_rccl/include/rccl_vars.h
	@echo ">>> Hipify complete."

clean:
	rm -rf hipify_rccl third_party include/nccl.h
	rm -f *.o $(EXE)

.PHONY: all clean hipify
