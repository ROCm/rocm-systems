HIPCC ?= hipcc
ARCH ?= gfx950
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG ?= ../../configs/amdgpu_cdna4_kmd.json

all: $(BUILD_DIR)/global_race $(BUILD_DIR)/global_fixed

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: src/%.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $<

run-race: $(BUILD_DIR)/global_race
	RJ_SINKS=race_detector $(ROCJITSU_BIN) --config $(CONFIG) -- $<

run-fixed: $(BUILD_DIR)/global_fixed
	RJ_SINKS=race_detector $(ROCJITSU_BIN) --config $(CONFIG) -- $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run-race run-fixed clean