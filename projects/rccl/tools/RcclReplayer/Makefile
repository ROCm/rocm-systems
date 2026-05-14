ROCM_DIR ?= /opt/rocm
RCCL_DIR ?= ../../build/release
MPI_DIR  ?= /opt/ompi

INCLUDES = -I$(MPI_DIR)/include -I$(RCCL_DIR)/include -I../../src/include
LDFLAGS  = -L$(MPI_DIR)/lib -L$(RCCL_DIR) -lmpi -lrccl

main: rcclReplayer.cpp
	$(ROCM_DIR)/bin/hipcc -std=c++20 rcclReplayer.cpp -O0 -g -o rcclReplayer $(INCLUDES) $(LDFLAGS)

clean:
	rm -f ./rcclReplayer
