#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#
.DEFAULT_GOAL: build
ROCM_PATH ?= /opt/rocm
CXX = $(ROCM_PATH)/bin/amdclang++
BUILDDIR ?= .
HIPIFY_PL_EXE ?= $(ROCM_PATH)/bin/hipify-perl
HIPIFY_DIR := hipify-profiler

SRC_FILES := $(wildcard *.cc)
HIPIFY_SRC := $(addprefix $(HIPIFY_DIR)/,$(SRC_FILES))

build: ${BUILDDIR}/librccl-profiler-example.so

${BUILDDIR}/librccl-profiler-example.so: $(HIPIFY_SRC)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CXX) -D__HIP_PLATFORM_AMD__ -I$(HIPIFY_DIR) -I$(HIPIFY_DIR)/nccl -I$(ROCM_PATH)/include -fPIC -shared -o $@ $^

$(HIPIFY_DIR)/%.cc: %.cc
	@mkdir -p $(HIPIFY_DIR)/nccl
	@cp *.cc *.h $(HIPIFY_DIR)/
	@cp nccl/*.h $(HIPIFY_DIR)/nccl/
	@$(HIPIFY_PL_EXE) -inplace -quiet-warnings $(HIPIFY_DIR)/*.cc $(HIPIFY_DIR)/*.h

clean:
	rm -rf ${BUILDDIR}/librccl-profiler-example.so $(HIPIFY_DIR)