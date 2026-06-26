HIPCC ?= hipcc
ARCH ?= gfx950
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG ?= ../../configs/amdgpu_cdna4_kmd.json

all: $(BUILD_DIR)/memory_pattern

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/memory_pattern: src/memory_pattern.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $<

run: $(BUILD_DIR)/memory_pattern
	$(ROCJITSU_BIN) --config $(CONFIG) -- $<

run-profile: $(BUILD_DIR)/memory_pattern
	@mkdir -p logs
	RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1 RJ_SINKS=file RJ_SINK_DIR=logs \
		$(ROCJITSU_BIN) --config $(CONFIG) -- $<

clean:
	rm -rf $(BUILD_DIR) logs

.PHONY: all run run-profile clean