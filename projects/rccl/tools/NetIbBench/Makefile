# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE.txt for license information

MPICXX ?= mpicxx

RCCL_ROOT ?= ../..
RCCL_BUILD ?= $(RCCL_ROOT)/build/release

INCLUDES = -I./include \
           -I$(RCCL_ROOT)/src/include \
           -I$(RCCL_ROOT)/src/include/plugin

CXXFLAGS = -g -O3 -std=c++14 -Wall -Wextra -Wno-unused-parameter $(INCLUDES) -DOMPI_SKIP_MPICXX
LDFLAGS  = -ldl -lpthread

NET_OBJ_DIR = $(RCCL_BUILD)/src/CMakeFiles/rccl.dir/__/hipify/src/plugin/net
NET_OBJS = $(NET_OBJ_DIR)/net_v11.cc.o \
           $(NET_OBJ_DIR)/net_v10.cc.o \
           $(NET_OBJ_DIR)/net_v9.cc.o  \
           $(NET_OBJ_DIR)/net_v8.cc.o  \
           $(NET_OBJ_DIR)/net_v7.cc.o  \
           $(NET_OBJ_DIR)/net_v6.cc.o

all: alltoall_bench

nccl_stubs.o: nccl_stubs.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

alltoall_bench: alltoall_bench.cpp include/net_plugin_loader.h include/benchmark_utils.h nccl_stubs.o
	$(MPICXX) $(CXXFLAGS) $< nccl_stubs.o $(NET_OBJS) -o $@ $(LDFLAGS)

clean:
	rm -f alltoall_bench nccl_stubs.o

.PHONY: all clean
