#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#
# Builds rccl_ab_perf. The binary does not link against RCCL; both libraries
# under comparison are dlopen()ed at runtime.

ROCM_PATH ?= /opt/rocm
MPI ?= 1
MPI_HOME ?= /usr/lib/x86_64-linux-gnu
BUILDDIR ?= ../../build
override BUILDDIR := $(abspath $(BUILDDIR))

CXX ?= g++
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17 -Wall -D__HIP_PLATFORM_AMD__ -I$(ROCM_PATH)/include
LDFLAGS += -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -ldl

ifeq ($(MPI), 1)
CXXFLAGS += -DMPI_SUPPORT -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX \
            -I$(MPI_HOME)/include -I$(MPI_HOME)/include/openmpi \
            -I$(MPI_HOME)/openmpi/include -I$(MPI_HOME)/openmpi/include/openmpi
LDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/openmpi/lib -Wl,-rpath,$(MPI_HOME)/lib \
           -Wl,-rpath,$(MPI_HOME)/openmpi/lib -lmpi
else ifeq ($(MPICH), 1)
CXXFLAGS += -DMPI_SUPPORT -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX \
            -I$(MPI_HOME)/include -I$(MPI_HOME)/mpich/include \
            -I/usr/include/x86_64-linux-gnu/mpich
LDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/mpich/lib -lmpich
endif

TARGET := $(BUILDDIR)/rccl_ab_perf

.PHONY: all clean
all: $(TARGET)

$(TARGET): rccl_ab_perf.cc Makefile
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
