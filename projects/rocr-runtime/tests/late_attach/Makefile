# Makefile for HSA Queue Intercept Late-Attach Integration Test
#
# Builds:
#   late_attach_test      - HIP test application (uses hipcc, NO link to injector)
#   libtrace_injector.so  - Shared library loaded via dlopen at runtime
#
# Usage:
#   make          - build both targets
#   make test     - build and run the test
#   make clean    - remove build artifacts

ROCM ?= /opt/rocm
HIPCC ?= $(ROCM)/bin/hipcc
CXX ?= g++

# HIP test app: built with hipcc, links HSA for queue creation, dlopen for injector
HIP_FLAGS = -std=c++17 -g -O2 -I$(ROCM)/include
HIP_LDFLAGS = -L$(ROCM)/lib -lhsa-runtime64 -ldl -lpthread -Wl,-rpath,$(ROCM)/lib

# Injector shared library: built with g++, links only against HSA runtime
INJ_CXXFLAGS = -std=c++17 -g -O2 -fPIC -I$(ROCM)/include
INJ_LDFLAGS = -shared -L$(ROCM)/lib -lhsa-runtime64 -Wl,-rpath,$(ROCM)/lib

.PHONY: all test clean

all: late_attach_test libtrace_injector.so

# The test binary uses HIP and raw HSA, but does NOT link libtrace_injector
late_attach_test: late_attach_test.cpp
	$(HIPCC) $(HIP_FLAGS) -o $@ $< $(HIP_LDFLAGS)

# The injector is a standalone shared library linked against HSA runtime
libtrace_injector.so: trace_injector.cpp
	$(CXX) $(INJ_CXXFLAGS) -o $@ $< $(INJ_LDFLAGS)

test: all
	@echo ""
	@echo "=========================================="
	@echo "Running late-attach integration test"
	@echo "=========================================="
	@./late_attach_test

clean:
	rm -f late_attach_test libtrace_injector.so
