#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
# Modifications are Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#

#include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
DST_DIR := $(BUILDDIR)/test/verifiable

ROCM_PATH ?= /opt/rocm
MPI_HOME ?= /usr/lib/x86_64-linux-gnu
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0
NCCL_HOME ?= ""

HIPCC = $(ROCM_PATH)/bin/amdclang++
CXX = $(HIPCC)

HIPCUFLAGS := -std=c++14
LDFLAGS    :=
HIPLDFLAGS :=

ifneq ($(NCCL_HOME), "")
HIPCUFLAGS += -I$(NCCL_HOME)/ -I$(NCCL_HOME)/include
HIPLDFLAGS   += -Wl,-rpath,$(NCCL_HOME) -L$(NCCL_HOME) -L$(NCCL_HOME)/lib
endif

HIPCUFLAGS += -I$(ROCM_PATH)/include
HIPCUFLAGS += -I$(ROCM_PATH)/include/hip
LDFLAGS    += -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt
HIPLDFLAGS += $(CUSTOM_RCCL_LIB) -L$(ROCM_PATH)/lib -lhsa-runtime64 -lamdhip64 -lstdc++ -lrt

ifeq ($(DEBUG), 0)
HIPCUFLAGS += -O3
else
HIPCUFLAGS += -O0 -g -ggdb3
endif

ifeq ($(VERBOSE), 0)
.SILENT:
endif

ifeq ($(MPI), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/openmpi -I${MPI_HOME}/openmpi/include -I${MPI_HOME}/openmpi/include/openmpi
HIPLDFLAGS += -L${MPI_HOME}/lib -L${MPI_HOME}/openmpi/lib -lmpi
else ifeq ($(MPICH), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/mpich/include -I/usr/include/x86_64-linux-gnu/mpich
HIPLDFLAGS += -L${MPI_HOME}/lib -L${MPI_HOME}/mpich/lib -lmpich
endif

LIBRARIES += rccl
HIPLDFLAGS += $(LIBRARIES:%=-l%)

all: $(DST_DIR)/verifiable.o $(DST_DIR)/self_test

clean:
	rm -rf $(DST_DIR)

TEST_VERIFIABLE_SRCDIR := .
TEST_VERIFIABLE_BUILDDIR := $(DST_DIR)
include verifiable.mk

self_test: $(DST_DIR)/self_test

$(DST_DIR)/self_test: verifiable.cpp verifiable.h
	@printf "Linking  %s\n" $@
	@mkdir -p $(DST_DIR)
	$(HIPCC) -o $@ $(HIPCUFLAGS) -DSELF_TEST=1 verifiable.cpp $(HIPLDFLAGS)
