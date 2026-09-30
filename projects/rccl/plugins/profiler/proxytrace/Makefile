# RCCL proxy-trace profiler plugin (librccl-profiler-proxytrace.so)
RCCL_ROOT := ../../..
ROCM_PATH ?= /opt/rocm
ifeq ($(origin CXX),default)
CXX := $(ROCM_PATH)/bin/amdclang++
endif
PLUGIN_SO := librccl-profiler-proxytrace.so
# ProxyTrace is compiled from the core source so the plugin and librccl cannot
# disagree on the trace format.
CORE_SRC := $(RCCL_ROOT)/src/misc/proxy_trace/proxy_trace.cc
# The profiler ABI comes from RCCL's own headers rather than a vendored copy, so
# it cannot drift. nccl/ must stay ahead of src/include: it supplies the minimal
# os.h that keeps RCCL's internal os.h (and nccl.h) out of the build.
INC := -I$(CURDIR) -I$(CURDIR)/nccl -I$(RCCL_ROOT)/src/include/plugin -I$(RCCL_ROOT)/src/include -I$(ROCM_PATH)/include
CXXFLAGS := -fPIC -O2 -g -std=c++17 -Wall -Wextra -DRCCL_PROXYTRACE_PLUGIN_BUILD -DNCCL_OS_LINUX -D__HIP_PLATFORM_AMD__
LDFLAGS := -shared
# Emit a .d alongside each object so edits to any header reached by an #include
# rebuild the plugin. -MP adds phony targets for those headers, so a removed or
# renamed header does not leave a stale rule that breaks the next build.
DEPFLAGS := -MMD -MP

OBJ := proxytrace_plugin.o proxy_trace.o
DEP := $(OBJ:.o=.d)

all: $(PLUGIN_SO)

$(PLUGIN_SO): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^

proxytrace_plugin.o: proxytrace_plugin.cc
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c -o $@ $<

proxy_trace.o: $(CORE_SRC)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c -o $@ $<

clean:
	rm -f $(PLUGIN_SO) $(OBJ) $(DEP)

-include $(DEP)

.PHONY: all clean
