# *************************************************************************
# Copyright (c) 2025-2026 Broadcom. The term "Broadcom" refers solely to the Broadcom Inc. corporate affiliate that distributes this software. All rights reserved.
# 
# See LICENSE.txt for license information
# ************************************************************************

RCCL_HOME:=../..
RCCL_BUILD:=$(RCCL_HOME)/build
RCCL_SRC:=$(RCCL_HOME)/src
RCCL_HIP:=$(RCCL_BUILD)/hipify

INC:= -I. -I$(RCCL_BUILD)/_deps/fmt-src/include -I$(RCCL_BUILD)/hipify/src/include/plugin -I$(RCCL_BUILD)/hipify/src/include -I$(RCCL_BUILD)/hipify/src  -I$(RCCL_SRC)/include -I$(RCCL_BUILD)/include
INC+= -I$(RCCL_HIP)/src/device -I$(RCCL_HIP)/gensrc

SHOW_UNDEF_SYMBOLS:= # -Wl,-z,defs
LIBS:= -L$(RCCL_BUILD) -lrccl

CC:= /opt/rocm/bin/hipcc

PLUGIN_SO:=librccl-net-bnxt.so

OBJECTS=\
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/misc/socket.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/misc/ibvwrap.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/misc/utils.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/misc/param.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/misc/ibvsymbols.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/debug.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/graph/xml.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/misc/archinfo.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/misc/rocm_smi_wrap.cc.o \
$(RCCL_BUILD)/CMakeFiles/rccl.dir/hipify/src/misc/alt_rsmi.cc.o \




default: $(PLUGIN_SO)

$(PLUGIN_SO): net_ib_bnxt.cc
        # $(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^
	$(CC) -O3 $(INC) -DROCM_VERSION=70002 -fPIC -shared -o $@ $(OBJECTS) -Wl,-soname,$(PLUGIN_SO) $(SHOW_UNDEF_SYMBOLS) -Wl,--error-limit,0 $(LIBS) $^

clean:
	rm -f $(PLUGIN_SO)
