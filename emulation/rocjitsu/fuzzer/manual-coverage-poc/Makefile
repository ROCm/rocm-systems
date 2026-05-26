HIPCC ?= hipcc
AFL_CXX ?= ../../Tools/AFLplusplus/afl-clang-fast++
CXX ?= /opt/rocm-7.0.0/lib/llvm/bin/clang++
HIP_PATH ?= $(shell hipconfig --path 2>/dev/null || echo /opt/rocm)
GPU_ARCH ?= native
BUILD_DIR ?= build

CXXFLAGS ?= -O2 -g -std=c++17
HOST_INCLUDES := -I$(HIP_PATH)/include
HOST_DEFINES := -D__HIP_PLATFORM_AMD__
HOST_LDFLAGS := -L$(HIP_PATH)/lib -lamdhip64

HSACO := $(BUILD_DIR)/manual_device_coverage.hsaco
STANDALONE_BIN := $(BUILD_DIR)/manual_coverage_standalone
AFL_BIN := $(BUILD_DIR)/manual_coverage_afl

.PHONY: all standalone afl clean

all: standalone

standalone: $(STANDALONE_BIN) $(HSACO)

afl: $(AFL_BIN) $(HSACO)

$(BUILD_DIR):
	mkdir -p $@

$(HSACO): device_kernel.hip | $(BUILD_DIR)
	$(HIPCC) --genco --offload-arch=$(GPU_ARCH) -O2 $< -o $@

$(STANDALONE_BIN): host_afl_driver.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HOST_DEFINES) $(HOST_INCLUDES) $< $(HOST_LDFLAGS) -o $@

$(AFL_BIN): host_afl_driver.cpp | $(BUILD_DIR)
	@if [ ! -x "$(AFL_CXX)" ]; then \
	  echo "AFL compiler wrapper not found: $(AFL_CXX)"; \
	  echo "Build AFL++ first or pass AFL_CXX=/path/to/afl-clang-fast++"; \
	  exit 1; \
	fi
	$(AFL_CXX) $(CXXFLAGS) $(HOST_DEFINES) $(HOST_INCLUDES) $< $(HOST_LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)
