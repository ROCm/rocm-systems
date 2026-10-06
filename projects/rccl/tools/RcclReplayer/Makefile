ROCM_DIR ?= /opt/rocm
RCCL_DIR ?= ../../build/release
MPI_DIR  ?= /opt/ompi

INCLUDES = -I$(MPI_DIR)/include -I$(RCCL_DIR)/include -I../../src/include
LDFLAGS  = -L$(MPI_DIR)/lib -L$(RCCL_DIR) -lmpi -lrccl

main: rcclReplayer.cpp rcclReplayer.hpp replay_log_name.hpp
	$(ROCM_DIR)/bin/hipcc -std=c++20 rcclReplayer.cpp -O0 -g -o rcclReplayer $(INCLUDES) $(LDFLAGS)

test: test_replay_log_name.cpp replay_log_name.hpp
	$(CXX) -std=c++17 -Wall -o test_replay_log_name test_replay_log_name.cpp
	./test_replay_log_name

clean:
	rm -f ./rcclReplayer ./test_replay_log_name

.PHONY: test clean
