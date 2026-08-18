#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#
RCCL_HOME:=../../build/release
HIP_HOME:=/opt/rocm
INC:= -I$(RCCL_HOME)/include/  -I$(HIP_HOME)/include/ -D__HIP_PLATFORM_AMD__ -Inccl
PLUGIN_SO:=librccl-tuner.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c
	$(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

clean:
	rm -f $(PLUGIN_SO)
