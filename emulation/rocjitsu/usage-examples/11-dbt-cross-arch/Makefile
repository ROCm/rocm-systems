HIPCC ?= hipcc
ARCH ?= gfx950
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG_CDNA3 ?= ../../configs/amdgpu_cdna3_kmd.json

all: $(BUILD_DIR)/dbt_example

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/dbt_example: src/dbt_example.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) -o $@ $<
	@echo "Compiled for $(ARCH)"

run-dbt: $(BUILD_DIR)/dbt_example
	@echo "Running with DBT: gfx950 -> gfx942"
	RJ_DBT_LOG=1 $(ROCJITSU_BIN) --config $(CONFIG_CDNA3) -- $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run-dbt clean