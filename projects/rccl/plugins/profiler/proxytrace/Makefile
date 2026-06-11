# RCCL proxy-trace profiler plugin (librccl-profiler-proxytrace.so)
RCCL_ROOT := ../../..
CXX ?= g++
PLUGIN_SO := librccl-profiler-proxytrace.so
INC := -I$(RCCL_ROOT)/src/include/plugin -I$(RCCL_ROOT)/src/include -I$(CURDIR)
CXXFLAGS := -fPIC -shared -O2 -g -std=c++17 -Wall -Wextra -DRCCL_PROXYTRACE_PLUGIN_BUILD -DFMT_HEADER_ONLY=1

SRC := proxytrace_plugin.cc $(RCCL_ROOT)/src/misc/proxy_trace/proxy_trace.cc

all: $(PLUGIN_SO)

$(PLUGIN_SO): proxytrace_plugin.cc proxytrace_plugin_shim.h $(RCCL_ROOT)/src/misc/proxy_trace/proxy_trace.cc proxytrace_plugin_log.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $(SRC) -lfmt

clean:
	rm -f $(PLUGIN_SO)

.PHONY: all clean
