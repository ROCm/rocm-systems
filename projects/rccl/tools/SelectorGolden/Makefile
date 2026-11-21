# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

ROCM_DIR ?= /opt/rocm
RCCL_DIR ?= ../../build/release
MPI_DIR  ?= /opt/ompi

INCLUDES = -I$(MPI_DIR)/include -I$(RCCL_DIR)/include
LDFLAGS  = -L$(MPI_DIR)/lib -L$(RCCL_DIR) -lmpi -lrccl -Wl,-rpath,$(abspath $(RCCL_DIR))

all: SelectorGolden

SelectorGolden: SelectorGolden.cpp
	$(ROCM_DIR)/bin/hipcc -std=c++17 -O2 -Wall $< -o $@ $(INCLUDES) $(LDFLAGS)

clean:
	rm -f SelectorGolden

.PHONY: all clean
