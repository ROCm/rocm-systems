#
# Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
# Modifications are Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#
include common.mk

ROCM_PATH ?= /opt/rocm
MPI_HOME ?= /usr/lib/x86_64-linux-gnu
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0
NCCL_HOME ?= ""
CUSTOM_RCCL_LIB ?= ""

HIPCC ?= $(ROCM_PATH)/llvm/bin/amdclang++
HIPCONFIG = $(ROCM_PATH)/bin/hipconfig
HIPIFY_PL_EXE=$(ROCM_PATH)/bin/hipify-perl
HIPIFY_PL_FLAGS = -quiet-warnings
CXX = $(HIPCC)

HIPCUFLAGS := -std=c++14
LDFLAGS    :=
HIPLDFLAGS :=

# Clang compiler fails to auto detect device lib path in some ROCK builds
DEVICE_LIB_PATH := $(if $(wildcard $(ROCM_PATH)/amdgcn/bitcode),,$(if $(wildcard $(ROCM_PATH)/llvm/amdgcn/bitcode),--rocm-device-lib-path=$(ROCM_PATH)/llvm/amdgcn/bitcode))

# Set to 1 to enable MPI support (multi-process/multi-node)
MPI ?= 0
# Set to 1 to enable device API builds (requires NCCL >= 2.28)
ENABLE_DEVICE_API ?= 0
# e.g. Set to _mpi when using MPI=1
NAME_SUFFIX ?=
# Set to 1 to create and use libverifiable.so to reduce binary size
DSO ?= 0

HIP_VERSION = $(strip $(shell which $(HIPCONFIG) >/dev/null && $(HIPCONFIG) --version))
HIP_MAJOR = $(shell echo $(HIP_VERSION) | cut -d "." -f 1)
HIP_MINOR = $(shell echo $(HIP_VERSION) | cut -d "." -f 2)

# Better define GPU_TARGETS in your environment to the minimal set
# of archs to reduce compile time.
# Currently, supports gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1102,gfx1151,gfx1200,gfx1201,gfx1250
ifndef GPU_TARGETS
GPU_TARGETS = gfx906 gfx908 gfx90a
  ifeq ($(shell test "0$(HIP_MAJOR)" -ge 7; echo $$?),0)
    GPU_TARGETS += gfx942 gfx950
    # gfx1250 - HIP compatibility is not yet fixed, currently gated by HIP version >= 7.2.0
    # as ROCm 7.12.0 is based on HIP 7.2.0
    ifeq ($(shell test "0$(HIP_MINOR)" -ge 2; echo $$?),0)
      GPU_TARGETS += gfx1250
    endif
  else ifeq ($(shell test "0$(HIP_MAJOR)" -eq 6; echo $$?),0)
    # Include gfx942 support if we're using ROCm 6.0 or above
    GPU_TARGETS += gfx942
    ifeq ($(shell test "0$(HIP_MINOR)" -ge 5; echo $$?),0)
    # Include gfx950 support if we're using ROCm 6.5 or above
    GPU_TARGETS += gfx950
    endif
  endif
GPU_TARGETS += gfx1030 gfx1100 gfx1101 gfx1102 gfx1151 gfx1200 gfx1201
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
HIPCUFLAGS += -x hip -D__HIP_PLATFORM_AMD__ -D__HIPCC__ $(DEVICE_LIB_PATH) $(GPU_TARGETS_FLAGS)
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

.PRECIOUS: $(HIPIFY_DIR)/%.cu.cpp $(HIPIFY_DIR)/%.h

ifeq ($(MPI), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/openmpi -I${MPI_HOME}/openmpi/include -I${MPI_HOME}/openmpi/include/openmpi
HIPLDFLAGS += -L${MPI_HOME}/lib -L${MPI_HOME}/openmpi/lib -lmpi
else ifeq ($(MPICH), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/mpich/include -I/usr/include/x86_64-linux-gnu/mpich
HIPLDFLAGS += -L${MPI_HOME}/lib -L${MPI_HOME}/mpich/lib -lmpich
endif

ifeq ($(ENABLE_DEVICE_API), 1)
HIPCUFLAGS += -DENABLE_DEVICE_API
endif

# Enable LLVM source-based coverage instrumentation on both host and device
# sides. The patched clang driver in llvm-decouple auto-links the per-arch
# device profile runtime (libclang_rt.profile.a) and force-links the host-side
# drain symbol (__llvm_profile_hip_collect_device_data), which registers an
# atexit handler to walk loaded ELF code objects via HSA introspection.
ifeq ($(ENABLE_DEVICE_COVERAGE), 1)
HIPCUFLAGS += -fprofile-instr-generate -fcoverage-mapping
HIPLDFLAGS += -fprofile-generate

# Toolchains that do NOT auto-link the amdgcn device profile runtime fail each
# -c HIP embedded device link with "undefined symbol: __llvm_profile_instrument_gpu"
# / __llvm_profile_raw_version. Locate the per-arch libclang_rt.profile.a and
# forward it to the offload linker, mirroring rccl's cmake/DeviceLinker.cmake.
DEVICE_RESOURCE_DIR := $(shell $(HIPCC) --target=amdgcn-amd-amdhsa -print-resource-dir 2>/dev/null)
DEVICE_PROFILE_RT := $(firstword $(wildcard \
  $(DEVICE_RESOURCE_DIR)/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a \
  $(DEVICE_RESOURCE_DIR)/lib/linux/libclang_rt.profile-amdgcn.a))
ifeq ($(DEVICE_PROFILE_RT),)
$(error ENABLE_DEVICE_COVERAGE=1 but no amdgcn libclang_rt.profile.a found under \
$(DEVICE_RESOURCE_DIR); rebuild compiler-rt profile for the amdgcn device target)
endif
# Empty, retained __llvm_prf_{names,cnts,data,vnds} sections so the linker always
# defines __start/__stop___llvm_prf_* for every device offload link, even when a
# device object references those boundaries but provides none of the sections
# itself (e.g. collector). Generic amdgcn bitcode (no mcpu), so each per-arch
# device link LTO-compiles it.
DEVICE_PROFILE_ANCHOR := $(BUILDDIR)/prf_section_anchor.bc
$(shell mkdir -p $(BUILDDIR); [ -f $(DEVICE_PROFILE_ANCHOR) ] || printf \
  '__attribute__((used, retain, section("__llvm_prf_names"))) static char __rccl_prf_names_anchor[0];\n__attribute__((used, retain, section("__llvm_prf_cnts"))) static char __rccl_prf_cnts_anchor[0];\n__attribute__((used, retain, section("__llvm_prf_data"))) static char __rccl_prf_data_anchor[0];\n__attribute__((used, retain, section("__llvm_prf_vnds"))) static char __rccl_prf_vnds_anchor[0];\n' \
  | $(HIPCC) --target=amdgcn-amd-amdhsa -c -emit-llvm -O1 -x c - -o $(DEVICE_PROFILE_ANCHOR))
# amdgpu-internalize-symbols (auto-injected into the offload link) would internalize
# and dead-strip the hidden __llvm_profile_instrument_gpu that lives in the profile
# RT's separate bitcode LTO module; disable it so the runtime symbol resolves.
# Apply to both the per-.o embedded device link (HIPCUFLAGS) and the final
# executable offload link (HIPLDFLAGS), which re-links all device objects.
COVERAGE_DEVICE_LINK_FLAGS := \
  -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
  -Xoffload-linker $(DEVICE_PROFILE_ANCHOR) \
  -Xoffload-linker $(DEVICE_PROFILE_RT)
HIPCUFLAGS += $(COVERAGE_DEVICE_LINK_FLAGS)
HIPLDFLAGS += $(COVERAGE_DEVICE_LINK_FLAGS)
endif

LIBRARIES += rccl dl
HIPLDFLAGS += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv alltoallv hypercube all_reduce_bias
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf${NAME_SUFFIX})

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

${HIPIFY_DIR}/%.cu.cpp: %.cu
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	${HIPIFY_PL_EXE} ${HIPIFY_PL_FLAGS} $< > $@

${HIPIFY_DIR}/%.h: %.h
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	${HIPIFY_PL_EXE} ${HIPIFY_PL_FLAGS} $< > $@

.PRECIOUS: ${DST_DIR}/%.o

${DST_DIR}/%.o: ${HIPIFY_DIR}/%.cu.cpp ${HIPIFY_DIR}/common.h ${HIPIFY_DIR}/util.h ${HIPIFY_DIR}/collector.h ${HIPIFY_DIR}/rccl_vector_types.h ${HIPIFY_DIR}/multimem_ops.h $(TEST_VERIFIABLE_HDRS) $(GIT_VERSION_FILE)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) $(HIPCUFLAGS) -I. -c -o $@ $<"
	$(HIPCC) $(HIPCUFLAGS) -I. -I${HIPIFY_DIR} -c -o $@ $<

${DST_DIR}/%$(NAME_SUFFIX).o: ${HIPIFY_DIR}/%.cu.cpp ${HIPIFY_DIR}/common.h ${HIPIFY_DIR}/util.h ${HIPIFY_DIR}/collector.h ${HIPIFY_DIR}/rccl_vector_types.h ${HIPIFY_DIR}/multimem_ops.h $(TEST_VERIFIABLE_HDRS) $(GIT_VERSION_FILE)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) $(HIPCUFLAGS) -I. -c -o $@ $<"
	$(HIPCC) $(HIPCUFLAGS) -I. -I${HIPIFY_DIR} -c -o $@ $<

${DST_DIR}/timer.o: timer.cc timer.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -o $@ -c $<

ifeq ($(DSO), 1)
${DST_DIR}/%_perf$(NAME_SUFFIX): ${DST_DIR}/%.o ${DST_DIR}/common$(NAME_SUFFIX).o ${DST_DIR}/util$(NAME_SUFFIX).o ${DST_DIR}/collector.o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_LIBS) $(DST_DIR)/src/git_version.cpp
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $^ $(HIPLDFLAGS)"
	$(HIPCC) -o $@ $^ $(HIPLDFLAGS) -L$(TEST_VERIFIABLE_BUILDDIR) -lverifiable -Xlinker "--enable-new-dtags" -Xlinker "-rpath,\$$ORIGIN:\$$ORIGIN/verifiable"
else
${DST_DIR}/%_perf$(NAME_SUFFIX):${DST_DIR}/%.o ${DST_DIR}/common$(NAME_SUFFIX).o ${DST_DIR}/util$(NAME_SUFFIX).o ${DST_DIR}/collector.o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_OBJS) $(DST_DIR)/src/git_version.cpp
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $^ $(HIPLDFLAGS)"
	$(HIPCC) -o $@ $^ $(HIPLDFLAGS)
endif

clean_intermediates:
	rm -f ${DST_DIR}/*.o $(TEST_VERIFIABLE_OBJS)

