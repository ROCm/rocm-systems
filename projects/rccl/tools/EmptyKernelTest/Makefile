
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda

HIPCC=$(ROCM_PATH)/bin/hipcc
NVCC=$(CUDA_PATH)/bin/nvcc

# Compile TransferBenchCuda if nvcc detected
ifeq ("$(shell test -e $(NVCC) && echo found)", "found")
	EXE=./EmptyKernelTestCuda
else
	EXE=./EmptyKernelTest
endif

all: $(EXE)

./EmptyKernelTest:  EmptyKernelTest.cpp
	$(HIPCC) EmptyKernelTest.cpp -o EmptyKernelTest

./EmptyKernelTestCuda: EmptyKernelTest.cpp
	$(NVCC) EmptyKernelTest.cpp -x cu -o EmptyKernelTestCuda


clean:
	rm -f ./EmptyKernelTest ./EmptyKernelTestCuda

