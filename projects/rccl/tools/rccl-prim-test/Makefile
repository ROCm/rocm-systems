# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=rccl_prim_test
CXXFLAGS = -O3 -g -I/opt/rocm/rocrand/include

all: $(EXE)

$(EXE): rccl_prim_test.cpp
	$(HIPCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -f *.o $(EXE)
