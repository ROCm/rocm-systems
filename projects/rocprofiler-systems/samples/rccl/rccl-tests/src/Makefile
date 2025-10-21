#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
# Modifications are Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#

ROCM_PATH ?= /opt/rocm
MPI_HOME ?= /usr/lib/x86_64-linux-gnu
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0
NCCL_HOME ?= ""
CUSTOM_RCCL_LIB ?= ""

HIPCC ?= $(ROCM_PATH)/bin/amdclang++
HIPCONFIG = $(ROCM_PATH)/bin/hipconfig
CXX = $(HIPCC)

HIPCUFLAGS := -std=c++14
LDFLAGS    :=
HIPLDFLAGS :=

HIP_VERSION = $(strip $(shell which $(HIPCONFIG) >/dev/null && $(HIPCONFIG) --version))
HIP_MAJOR = $(shell echo $(HIP_VERSION) | cut -d "." -f 1)
HIP_MINOR = $(shell echo $(HIP_VERSION) | cut -d "." -f 2)

# Better define GPU_TARGETS in your environment to the minimal set
# of archs to reduce compile time.
# Currently, supports gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201
ifndef GPU_TARGETS
GPU_TARGETS = gfx906 gfx908 gfx90a
  ifeq ($(shell test "0$(HIP_MAJOR)" -eq 6; echo $$?),0)
    # Include gfx942 support if we're using ROCm 6.0 or above
    GPU_TARGETS += gfx942
    ifeq ($(shell test "0$(HIP_MINOR)" -ge 5; echo $$?),0)
    # Include gfx950 support if we're using ROCm 6.5 or above
    GPU_TARGETS += gfx950
    endif
  endif
GPU_TARGETS += gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201
endif

GPU_TARGETS_FLAGS = $(foreach target,$(GPU_TARGETS),"--offload-arch=$(target)")

#CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
#CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
#CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
#
## Better define NVCC_GENCODE in your environment to the minimal set
## of archs to reduce compile time.
#ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 12 -a "0$(CUDA_MINOR)" -ge 8 -o "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
## Include Blackwell support if we're using CUDA12.8 or above
#NVCC_GENCODE ?=	-gencode=arch=compute_80,code=sm_80 \
#		-gencode=arch=compute_90,code=sm_90 \
#		-gencode=arch=compute_100,code=sm_100 \
#		-gencode=arch=compute_120,code=sm_120 \
#		-gencode=arch=compute_120,code=compute_120
#else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 12; echo $$?),0)
#NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
#                -gencode=arch=compute_61,code=sm_61 \
#                -gencode=arch=compute_70,code=sm_70 \
#		-gencode=arch=compute_80,code=sm_80 \
#		-gencode=arch=compute_90,code=sm_90 \
#		-gencode=arch=compute_90,code=compute_90
#else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
#NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
#                -gencode=arch=compute_61,code=sm_61 \
#                -gencode=arch=compute_70,code=sm_70 \
#		-gencode=arch=compute_80,code=sm_80 \
#		-gencode=arch=compute_80,code=compute_80
#else
#NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
#                -gencode=arch=compute_50,code=sm_50 \
#                -gencode=arch=compute_60,code=sm_60 \
#                -gencode=arch=compute_61,code=sm_61 \
#                -gencode=arch=compute_70,code=sm_70 \
#                -gencode=arch=compute_70,code=compute_70
#endif

ifneq ($(NCCL_HOME), "")
HIPCUFLAGS += -I$(NCCL_HOME)/ -I$(NCCL_HOME)/include
HIPLDFLAGS += -Wl,-rpath,$(NCCL_HOME) -L$(NCCL_HOME) -L$(NCCL_HOME)/lib
endif

HIPCUFLAGS += -I$(ROCM_PATH)/include
HIPCUFLAGS += -I$(ROCM_PATH)/include/hip
HIPCUFLAGS += -x hip -D__HIP_PLATFORM_AMD__ -D__HIPCC__ $(GPU_TARGETS_FLAGS)
LDFLAGS    += -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt
ifneq ($(CUSTOM_RCCL_LIB), "")
HIPLDFLAGS += -L$(CUSTOM_RCCL_LIB)
endif
HIPLDFLAGS += -L$(ROCM_PATH)/lib -lhsa-runtime64 -lamdhip64 -lstdc++ -lrt -pthread

ifeq ($(DEBUG), 0)
HIPCUFLAGS += -O3
else
HIPCUFLAGS += -O0 -g -ggdb3
endif

ifeq ($(VERBOSE), 0)
.SILENT:
endif

.PHONY: build clean

BUILDDIR ?= ../build
HIPIFY_DIR ?= $(BUILDDIR)/hipify

.PRECIOUS: $(HIPIFY_DIR)/%.cpp $(HIPIFY_DIR)/%.h

ifeq ($(MPI), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/openmpi -I${MPI_HOME}/openmpi/include -I${MPI_HOME}/openmpi/include/openmpi
HIPLDFLAGS += -L${MPI_HOME}/lib -L${MPI_HOME}/openmpi/lib -lmpi
else ifeq ($(MPICH), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/mpich/include -I/usr/include/x86_64-linux-gnu/mpich
HIPLDFLAGS += -L${MPI_HOME}/lib -L${MPI_HOME}/mpich/lib -lmpich
endif

LIBRARIES += rccl
HIPLDFLAGS += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cpp)
OBJ_FILES := $(SRC_FILES:%.cpp=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv alltoallv
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

GIT_VERSION_FILE := ${DST_DIR}/src/git_version.cpp
GIT_REV          := $(shell git log --pretty=format:'%h' -n 1)
GIT_DIFF         := $(shell git diff --quiet --exit-code || echo +)
GIT_BRANCH       := $(shell git rev-parse --abbrev-ref HEAD)

build: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/verifiable
include ../verifiable/verifiable.mk

# Rule to create git_version.cpp
$(GIT_VERSION_FILE):
	@mkdir -p ${DST_DIR}/src
	@echo 'const char* rcclTestsGitHash = "$(GIT_BRANCH):$(GIT_REV)$(GIT_DIFF)";' > $@

${HIPIFY_DIR}/%.cpp: %.cpp
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	hipify-perl -quiet-warnings $< > $@

${HIPIFY_DIR}/%.h: %.h
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	hipify-perl -quiet-warnings $< > $@

${DST_DIR}/%.o: ${HIPIFY_DIR}/%.cpp ${HIPIFY_DIR}/common.h $(TEST_VERIFIABLE_HDRS) $(GIT_VERSION_FILE)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) $(HIPCUFLAGS) -I. -c -o $@ $<"
	$(HIPCC) $(HIPCUFLAGS) -I. -c -o $@ $<

${DST_DIR}/timer.o: timer.cc timer.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -o $@ -c timer.cc

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_OBJS) $(DST_DIR)/src/git_version.cpp
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $^ $(HIPLDFLAGS)"
	$(HIPCC) -o $@ $^ $(HIPLDFLAGS)

