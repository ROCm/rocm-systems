# RCCL proxy-trace profiler plugin (librccl-profiler-proxytrace.so)
RCCL_ROOT := ../../..
CXX ?= g++
PLUGIN_SO := librccl-profiler-proxytrace.so
INC := -I$(RCCL_ROOT)/src/include/plugin -I$(RCCL_ROOT)/src/include \
	-I$(RCCL_ROOT)/external/fmt/include -I$(CURDIR)
CXXFLAGS := -fPIC -O2 -g -std=c++17 -Wall -Wextra -DRCCL_PROXYTRACE_PLUGIN_BUILD -DFMT_HEADER_ONLY=1
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

proxy_trace.o: $(RCCL_ROOT)/src/misc/proxy_trace/proxy_trace.cc
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c -o $@ $<

clean:
	rm -f $(PLUGIN_SO) $(OBJ) $(DEP)

-include $(DEP)

.PHONY: all clean
