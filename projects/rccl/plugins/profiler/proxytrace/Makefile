# RCCL proxy-trace profiler plugin (librccl-profiler-proxytrace.so)
RCCL_ROOT := ../../..
# nccl.h is generated into the build tree; point RCCL_BUILD at a configured build dir.
RCCL_BUILD ?= $(RCCL_ROOT)/build/release
ROCM_PATH ?= /opt/rocm
CXX ?= $(ROCM_PATH)/bin/amdclang++
PLUGIN_SO := librccl-profiler-proxytrace.so
INC := -I$(RCCL_ROOT)/src/include/plugin -I$(RCCL_ROOT)/src/include -I$(RCCL_BUILD)/include -I$(ROCM_PATH)/include -I$(CURDIR)
CXXFLAGS := -fPIC -shared -O2 -g -std=c++17 -Wall -Wextra -DRCCL_PROXYTRACE_PLUGIN_BUILD -DFMT_HEADER_ONLY=1 -DNCCL_OS_LINUX -D__HIP_PLATFORM_AMD__

SRC := proxytrace_plugin.cc $(RCCL_ROOT)/src/misc/proxy_trace/proxy_trace.cc

all: $(PLUGIN_SO)

$(PLUGIN_SO): proxytrace_plugin.cc proxytrace_plugin_shim.h $(RCCL_ROOT)/src/misc/proxy_trace/proxy_trace.cc proxytrace_plugin_log.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $(SRC) -lfmt

clean:
	rm -f $(PLUGIN_SO)

.PHONY: all clean
