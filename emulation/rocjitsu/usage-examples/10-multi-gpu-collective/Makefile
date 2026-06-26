HIPCC ?= hipcc
ARCH ?= gfx950
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG_2GPU ?= ../../configs/amdgpu_cdna4_kmd_2gpu.json

all: $(BUILD_DIR)/multi_gpu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/multi_gpu: src/multi_gpu.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $<

run-2gpu: $(BUILD_DIR)/multi_gpu
	$(ROCJITSU_BIN) --daemon --config $(CONFIG_2GPU) -- $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run-2gpu clean