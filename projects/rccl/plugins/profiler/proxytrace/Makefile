# RCCL proxy-trace profiler plugin (librccl-profiler-proxytrace.so)
RCCL_ROOT := ../../..
ROCM_PATH ?= /opt/rocm
ifeq ($(origin CXX),default)
CXX := $(ROCM_PATH)/bin/amdclang++
endif
PLUGIN_SO := librccl-profiler-proxytrace.so
# The profiler ABI comes from RCCL's own headers rather than a vendored copy, so
# it cannot drift. nccl/ must stay first: it supplies the minimal os.h that
# keeps RCCL's internal os.h (and nccl.h) out of the build.
INC := -I$(CURDIR) -I$(CURDIR)/nccl -I$(RCCL_ROOT)/src/include/plugin -I$(ROCM_PATH)/include
CXXFLAGS := -fPIC -shared -O2 -g -std=c++17 -Wall -Wextra -DRCCL_PROXYTRACE_PLUGIN_BUILD -DNCCL_OS_LINUX -D__HIP_PLATFORM_AMD__

SRC := proxytrace_plugin.cc proxy_trace.cc

all: $(PLUGIN_SO)

$(PLUGIN_SO): proxytrace_plugin.cc proxy_trace.cc proxy_trace.h proxytrace_plugin_shim.h proxytrace_plugin_log.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $(SRC)

clean:
	rm -f $(PLUGIN_SO)

.PHONY: all clean
