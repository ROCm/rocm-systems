# Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
ROCM_PATH ?= /opt/rocm
HIPCC = $(ROCM_PATH)/bin/hipcc

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
CXXFLAGS = -g -ffunction-sections -fdata-sections -DROCM_VERSION=$(ROCM_VERSION) -Wl,--gc-sections -fgpu-rdc -Iinclude -Ihipify_rccl/include -Ihipify_rccl/include/plugin -Ihipify_rccl/src/device/include  -Ihipify_rccl/graph -I/opt/rocm/include/ -DTOPO_EXPL -DENABLE_TRACE -DENABLE_LL128 -DNVTX_NO_IMPL -DRCCL_EXPOSE_STATIC -lpthread

files = $(EXE).cpp model.cpp utils.cpp stubs.cc hipify_rccl/graph/topo.cc hipify_rccl/graph/rings.cc hipify_rccl/graph/paths.cc hipify_rccl/graph/trees.cc ../../src/misc/param.cc \
	hipify_rccl/graph/search.cc hipify_rccl/graph/connect.cc hipify_rccl/graph/tuning.cc hipify_rccl/graph/xml.cc ../../src/misc/nvmlwrap_stub.cc hipify_rccl/graph/rome_models.cc hipify_rccl/graph/archinfo.cc \
	hipify_rccl/collectives.cc hipify_rccl/register.cc hipify_rccl/enqueue.cc ../../src/rccl_wrap.cc

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
# ---------------------------------------------------------------------------



all: $(FMT_HEADER) hipify $(EXE)
	@echo "Built with ROCm version: $(ROCM_MAJOR).$(ROCM_MINOR).$(ROCM_PATCH) (ROCM_VERSION=$(ROCM_VERSION))"

$(EXE): $(files)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

hipify:
	rm -rf hipify_rccl
	mkdir -p hipify_rccl/src/device/include hipify_rccl/include/network/unpack include/rccl
	# Copy nccl.h.in from rccl source to always get latest API definitions
	cp -a ../../src/nccl.h.in include/nccl.h
	# Also copy to rccl/rccl.h for files that use #include <rccl/rccl.h>
	cp -a ../../src/nccl.h.in include/rccl/rccl.h
	cp -a ../../src/include/ hipify_rccl/
	cp -a ../../src/graph/ hipify_rccl/
	cp -a ../../src/device/*.h hipify_rccl/src/device/include
	cp -a ../../src/device/network/unpack/*.h hipify_rccl/include/network/unpack
	cp -a ../../src/enqueue.cc hipify_rccl/
	cp -a ../../src/register/register.cc hipify_rccl/
	cp -a ../../src/collectives.cc hipify_rccl/
	cp -a ../../src/misc/archinfo.cc hipify_rccl/graph/
	# Hipify all header and source files (CUDA -> HIP)
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/latency_profiler/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/plugin/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/latency_profiler/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/src/device/include/*.h
	# PATCH 1: Fix device includes - remove "device/" prefix since files are in same directory
	sed -i 's|#include "device/|#include "|g' hipify_rccl/src/device/include/*.h
	sed -i 's|#include "device/|#include "|g' hipify_rccl/include/*.h
	# PATCH 2: Use NVTX stubs - topo_expl doesn't need full NVTX implementation
	cp hipify_rccl/include/nvtx_stub.h hipify_rccl/include/nvtx.h
	bash ../../cmake/scripts/add_unroll.sh "hipify_rccl/src/device/include/common.h"
	hipify-perl -inplace -quiet-warnings hipify_rccl/graph/*
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/network/unpack/*
	hipify-perl -inplace -quiet-warnings hipify_rccl/*.cc
	# PATCH 3: Generate *_tmp.h files - these are build-time generated headers in rccl
	cp hipify_rccl/include/nccl_device/core.h hipify_rccl/include/nccl_device/core_tmp.h
	cp hipify_rccl/include/nccl_device/comm.h hipify_rccl/include/nccl_device/comm_tmp.h
	# PATCH 4: Add missing NVTX stub macros for topo_expl build
	echo '#define NCCL_NVTX3_FUNC_RANGE do {} while(0)' >> hipify_rccl/include/nvtx.h
	echo 'inline void initNvtxRegisteredEnums() {}' >> hipify_rccl/include/nvtx.h
	# PATCH 5: Add nccl_tuner.h include to get NCCL_NUM_ALGORITHMS and NCCL_NUM_PROTOCOLS macros
	sed -i '/#include "core.h"/a #include "plugin/nccl_tuner.h"' hipify_rccl/include/rccl_common.h

clean:
	rm -rf hipify_rccl third_party include/nccl.h
	rm -f *.o $(EXE)
