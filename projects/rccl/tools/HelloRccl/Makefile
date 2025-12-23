# Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

# Set to where RCCL is installed
RCCL_INSTALL=../../build/release

HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=HelloRccl
CXXFLAGS = -std=c++11 -O3 -I../../src/include -I$(RCCL_INSTALL) -L$(RCCL_INSTALL) -lrccl

all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o $(EXE)
