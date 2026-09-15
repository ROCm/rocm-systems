# SPDX-License-Identifier: Apache-2.0
#
# Common build rules for the RCCL examples.
# Ported from the NVIDIA NCCL examples (docs/examples) to use HIP/RCCL.

ROCM_PATH ?= /opt/rocm
RCCL_HOME ?= $(ROCM_PATH)
PREFIX    ?= /usr/local

# Examples carried verbatim from upstream NCCL keep their CUDA sources so that
# NCCL syncs apply cleanly; those .cu files are rewritten to HIP at build time
# (as librccl itself does) and compiled out of HIPIFY_DIR.
HIPIFY     ?= $(ROCM_PATH)/bin/hipify-perl
HIPIFY_DIR ?= .hipify

HIPCC ?= $(ROCM_PATH)/bin/hipcc
# Force CXX to hipcc (Make sets CXX=g++ by default, so we override).
CXX := $(HIPCC)

# Default GPU architectures - override with `make GPU_TARGETS="gfx942"` etc.
GPU_TARGETS ?= gfx90a gfx942 gfx1100
OFFLOAD_ARCH_FLAGS := $(foreach a,$(GPU_TARGETS),--offload-arch=$(a))

CXXFLAGS  ?= -O2 -g -std=c++17 -Wall
CXXFLAGS  += -D__HIP_PLATFORM_AMD__ $(OFFLOAD_ARCH_FLAGS)

# RCCL_HOME is searched before ROCM_PATH so a custom RCCL wins over the one
# shipped in ROCm. It may name an install prefix or a CMake build tree, where
# librccl.so sits at the top level rather than under lib/, so search both.
INCLUDES  := -I$(RCCL_HOME)/include -I$(ROCM_PATH)/include
LIBRARIES := -L$(RCCL_HOME)/lib -L$(RCCL_HOME) -L$(ROCM_PATH)/lib
LDFLAGS   := -lrccl -lamdhip64 -Wl,-rpath,$(RCCL_HOME)/lib -Wl,-rpath,$(RCCL_HOME) -Wl,-rpath,$(ROCM_PATH)/lib

# MPI configuration
ifeq ($(MPI), 1)
ifdef MPI_HOME
MPICXX ?= $(MPI_HOME)/bin/mpicxx
MPIRUN ?= $(MPI_HOME)/bin/mpirun
INCLUDES  += -I$(MPI_HOME)/include
LIBRARIES += -L$(MPI_HOME)/lib
else
MPICXX ?= mpicxx
MPIRUN ?= mpirun
endif
# Force the MPI compiler wrapper to use hipcc instead of g++ so HIP flags work.
export OMPI_CXX := $(HIPCC)
export MPICH_CXX := $(HIPCC)
CXXFLAGS += -DMPI_SUPPORT
LDFLAGS  += -lmpi
endif

# Hipify and compile rules for upstream .cu sources.
$(HIPIFY_DIR)/%.cu.cpp: %.cu
	@mkdir -p $(dir $@)
	$(HIPIFY) -experimental -quiet-warnings $< -o $@.tmp && mv -f $@.tmp $@

$(HIPIFY_DIR)/%.cu.o: $(HIPIFY_DIR)/%.cu.cpp
ifeq ($(MPI),1)
	$(MPICXX) $(CXXFLAGS) -x hip $(INCLUDES) -c $< -o $@
else
	$(CXX) $(CXXFLAGS) -x hip $(INCLUDES) -c $< -o $@
endif

# Keep the hipified source so it is not deleted as a make intermediate.
.PRECIOUS: $(HIPIFY_DIR)/%.cu.cpp
