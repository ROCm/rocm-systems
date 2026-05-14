# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

# Set to where RCCL is installed
RCCL_INSTALL=../../build/release

HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=GraphBench
CXXFLAGS = -std=c++11 -O3 -I../../src/include -I$(RCCL_INSTALL)/include -L$(RCCL_INSTALL) -lrccl

all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

test: $(EXE)
	LD_LIBRARY_PATH=$(RCCL_INSTALL) RCCL_ENABLE_HIPGRAPH=1 ./$(EXE)

testInfo: $(EXE)
	NCCL_DEBUG=INFO LD_LIBRARY_PATH=$(RCCL_INSTALL) RCCL_ENABLE_HIPGRAPH=1 ./$(EXE)
clean:
	rm -f *.o $(EXE)
