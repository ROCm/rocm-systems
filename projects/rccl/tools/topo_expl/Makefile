# Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
HIP_PATH ?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH = ../../..
endif
HIPCC = $(HIP_PATH)/bin/hipcc

.DEFAULT_GOAL := all

EXE = topo_expl
CXXFLAGS = -g -ffunction-sections -fdata-sections -Wl,--gc-sections -fgpu-rdc -Iinclude -Ihipify_rccl/include -Ihipify_rccl/include/plugin -Ihipify_rccl/src/device/include  -Ihipify_rccl/graph -I/opt/rocm/include/ -DTOPO_EXPL -DENABLE_TRACE -DENABLE_LL128 -DNVTX_NO_IMPL -DRCCL_EXPOSE_STATIC -lpthread

files = $(EXE).cpp model.cpp utils.cpp hipify_rccl/graph/topo.cc hipify_rccl/graph/rings.cc hipify_rccl/graph/paths.cc hipify_rccl/graph/trees.cc ../../src/misc/param.cc \
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

$(EXE): $(files)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

hipify:
	rm -rf hipify_rccl
	mkdir -p hipify_rccl/src/device/include hipify_rccl/include/network/unpack
	cp -a ../../src/include/ hipify_rccl/
	cp -a ../../src/graph/ hipify_rccl/
	cp -a ../../src/device/*.h hipify_rccl/src/device/include
	cp -a ../../src/device/network/unpack/*.h hipify_rccl/include/network/unpack
	cp -a ../../src/enqueue.cc hipify_rccl/
	cp -a ../../src/register/register.cc hipify_rccl/
	cp -a ../../src/collectives.cc hipify_rccl/
	cp -a ../../src/misc/archinfo.cc hipify_rccl/graph/
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/latency_profiler/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/plugin/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/latency_profiler/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/src/device/include/*.h
	bash ../../cmake/scripts/add_unroll.sh "hipify_rccl/src/device/include/common.h"
	hipify-perl -inplace -quiet-warnings hipify_rccl/graph/*
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/network/unpack/*
	hipify-perl -inplace -quiet-warnings hipify_rccl/*.cc

clean:
	rm -rf hipify_rccl third_party
	rm -f *.o $(EXE)
