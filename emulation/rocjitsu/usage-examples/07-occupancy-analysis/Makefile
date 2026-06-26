HIPCC ?= hipcc
ARCH ?= gfx950
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG ?= ../../configs/amdgpu_cdna4_kmd.json

all: $(BUILD_DIR)/occupancy

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/occupancy: src/occupancy.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $<

run: $(BUILD_DIR)/occupancy
	RJ_LOG=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run clean