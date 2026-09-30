RCCL_ROOT := ../../..
INC := -I. -I$(RCCL_ROOT)/src/include/plugin -I$(RCCL_ROOT)/src/include
PLUGIN_SO := librccl-profiler-accl.so

CXX := g++
CXXFLAGS := -std=c++14 -O2 -fPIC -shared -Wall -Wextra -DNDEBUG

ifeq ($(DEBUG), 1)
CXXFLAGS += -g2 -ggdb3 -rdynamic -funwind-tables -fno-omit-frame-pointer
endif

SOURCES := accl_profiler.cc
LDFLAGS := -lpthread

all: $(PLUGIN_SO)

$(PLUGIN_SO): $(SOURCES) accl_profiler.h accl_shim.h
	@echo "Compiling to create $@ from $^"
	$(CXX) $(INC) $(CXXFLAGS) -o $@ -Wl,-soname,$(PLUGIN_SO) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(PLUGIN_SO)

.PHONY: all clean
