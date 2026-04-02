# Makefile for HSA Queue Intercept Retrofit Tests
ROCM ?= /opt/rocm
CXX ?= g++
CXXFLAGS = -std=c++17 -g -O0 -Wall -I$(ROCM)/include -DAMD_INTERNAL_BUILD
LDFLAGS = -L$(ROCM)/lib -lhsa-runtime64 -lpthread -Wl,-rpath,$(ROCM)/lib

TESTS = test_basic_attach_detach test_lifecycle_concurrency test_hw_state_separation
STRESS_TESTS = test_stress_rapid_cycle test_stress_concurrent test_stress_memory

all: $(TESTS) $(STRESS_TESTS)

test_basic_attach_detach: test_basic_attach_detach.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_lifecycle_concurrency: test_lifecycle_concurrency.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_hw_state_separation: test_hw_state_separation.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_stress_rapid_cycle: test_stress_rapid_cycle.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_stress_concurrent: test_stress_concurrent.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_stress_memory: test_stress_memory.cpp test_helpers.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test: $(TESTS)
	@echo ""; echo "=== test_basic_attach_detach ===" ; ./test_basic_attach_detach || true
	@echo ""; echo "=== test_lifecycle_concurrency ===" ; ./test_lifecycle_concurrency || true
	@echo ""; echo "=== test_hw_state_separation ===" ; ./test_hw_state_separation || true

stress: $(STRESS_TESTS)
	@echo ""; echo "=== test_stress_rapid_cycle ===" ; ./test_stress_rapid_cycle || true
	@echo ""; echo "=== test_stress_concurrent ===" ; ./test_stress_concurrent || true
	@echo ""; echo "=== test_stress_memory ===" ; ./test_stress_memory || true

clean:
	rm -f $(TESTS) $(STRESS_TESTS)

.PHONY: all test stress clean
