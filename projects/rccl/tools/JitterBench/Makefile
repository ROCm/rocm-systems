ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda
HIPCC     = $(ROCM_PATH)/bin/hipcc
NVCC      = $(CUDA_PATH)/bin/nvcc

CCFLAGS   = -O3 -lhsa-runtime64 -fopenmp -lnuma
NVFLAGS   = -O3  -x cu -lnuma -Xcompiler -fopenmp -gencode=arch=compute_90,code=sm_90

ifneq ("$(MPI_DIR)", "")
MPIFLAGS = -DMPI_SUPPORT -I$(MPI_DIR)/include -L$(MPI_DIR)/lib -lmpi
else
MPIFLAGS =
endif

all: JitterBench

JitterBench: JitterBench.cpp Common.hpp Timeline.hpp
ifeq ("$(shell test -e $(NVCC) && echo found)", "found")
	$(NVCC) $(NVFLAGS) $(MPIFLAGS) $< -o $@
else
	$(HIPCC) $(CCFLAGS) $(MPIFLAGS) $< -o $@
endif

clean:
	rm -f ./JitterBench
