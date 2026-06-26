HIPCC ?= hipcc
ARCH ?= gfx950
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG ?= ../../configs/amdgpu_cdna4_kmd.json

all: $(BUILD_DIR)/crash_example $(BUILD_DIR)/crash_fixed

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: src/%.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $<

run-crash: $(BUILD_DIR)/crash_example
	RJ_LOG=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $< || true

run-fixed: $(BUILD_DIR)/crash_fixed
	$(ROCJITSU_BIN) --config $(CONFIG) -- $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run-crash run-fixed clean