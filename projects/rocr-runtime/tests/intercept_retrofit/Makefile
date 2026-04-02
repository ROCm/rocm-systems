# Makefile for HSA Queue Intercept Retrofit Tests
ROCM ?= /opt/rocm
CXX ?= g++
CXXFLAGS = -std=c++17 -g -O0 -Wall -I$(ROCM)/include -DAMD_INTERNAL_BUILD
LDFLAGS = -L$(ROCM)/lib -lhsa-runtime64 -lpthread -Wl,-rpath,$(ROCM)/lib

TESTS = test_basic_attach_detach test_lifecycle_concurrency test_hw_state_separation

all: $(TESTS)

test_basic_attach_detach: test_basic_attach_detach.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_lifecycle_concurrency: test_lifecycle_concurrency.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_hw_state_separation: test_hw_state_separation.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test: $(TESTS)
	@echo ""; echo "=== test_basic_attach_detach ===" ; ./test_basic_attach_detach || true
	@echo ""; echo "=== test_lifecycle_concurrency ===" ; ./test_lifecycle_concurrency || true
	@echo ""; echo "=== test_hw_state_separation ===" ; ./test_hw_state_separation || true

clean:
	rm -f $(TESTS)

.PHONY: all test clean
