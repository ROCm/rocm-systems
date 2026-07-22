# SPDX-License-Identifier: Apache-2.0
#
# Common build rules for the RCCL examples.
# Ported from the NVIDIA NCCL examples (docs/examples) to use HIP/RCCL.

ROCM_PATH ?= /opt/rocm
RCCL_HOME ?= $(ROCM_PATH)
PREFIX    ?= /usr/local

HIPCC ?= $(ROCM_PATH)/bin/hipcc
# Force CXX to hipcc (Make sets CXX=g++ by default, so we override).
CXX := $(HIPCC)

# Default GPU architectures - override with `make GPU_TARGETS="gfx942"` etc.
GPU_TARGETS ?= gfx90a gfx942 gfx1100
OFFLOAD_ARCH_FLAGS := $(foreach a,$(GPU_TARGETS),--offload-arch=$(a))

CXXFLAGS  ?= -O2 -g -std=c++17 -Wall
CXXFLAGS  += -D__HIP_PLATFORM_AMD__ $(OFFLOAD_ARCH_FLAGS)

INCLUDES  := -I$(ROCM_PATH)/include -I$(RCCL_HOME)/include
LIBRARIES := -L$(ROCM_PATH)/lib -L$(RCCL_HOME)/lib
LDFLAGS   := -lrccl -lamdhip64 -Wl,-rpath,$(RCCL_HOME)/lib -Wl,-rpath,$(ROCM_PATH)/lib

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
