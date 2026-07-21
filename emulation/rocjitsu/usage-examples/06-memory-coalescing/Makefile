HIPCC ?= hipcc
ARCH ?= gfx950
CXXFLAGS = -std=c++20 -Wall -O2
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG ?= ../../configs/amdgpu_cdna4_kmd.json

.PHONY: all run run-profile clean help

all: $(BUILD_DIR)/memory_pattern

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/memory_pattern: src/memory_pattern.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) -o $@ $<

# Plain functional run (application output only).
run: $(BUILD_DIR)/memory_pattern
	$(ROCJITSU_BIN) --config $(CONFIG) -- $<

# Run with the hook profiler. The profiler group emits HOOK_PROFILE to stderr
# and does NOT honor RJ_SINKS=file/RJ_SINK_DIR, so we tee stderr into
# logs/profile.log ourselves to keep both the console output and a saved copy.
run-profile: $(BUILD_DIR)/memory_pattern
	@mkdir -p logs
	RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1 \
		$(ROCJITSU_BIN) --config $(CONFIG) -- $< 2>&1 | tee logs/profile.log
	@echo "Profile saved to logs/profile.log"

clean:
	rm -rf $(BUILD_DIR) logs

help:
	@echo "Memory access profiling example - targets:"
	@echo "  make              - build the memory_pattern binary"
	@echo "  make run          - plain functional run (app output only)"
	@echo "  make run-profile  - run with hook profiler; tees output to logs/profile.log"
	@echo "  make clean        - remove build/ and logs/"
	@echo ""
	@echo "Note: rocjitsu does NOT model memory coalescing/bandwidth. The profiler"
	@echo "counts emulator hook invocations and CPU time, not GPU performance."
