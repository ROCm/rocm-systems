HIPCC ?= hipcc
ARCH ?= gfx950
CXXFLAGS = -std=c++20 -Wall -O2
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG ?= ../../configs/amdgpu_cdna4_kmd.json

.PHONY: all run run-quiet clean help

all: $(BUILD_DIR)/occupancy

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/occupancy: src/occupancy.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) -o $@ $<

# Run with RJ_LOG so the dispatch metadata (vgprs=) can be cross-checked against
# the alloc_vgprs the program computes.
run: $(BUILD_DIR)/occupancy
	RJ_LOG=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $<

# Run without dispatch logging (host occupancy report only).
run-quiet: $(BUILD_DIR)/occupancy
	$(ROCJITSU_BIN) --config $(CONFIG) -- $<

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Occupancy analysis example - targets:"
	@echo "  make            - build the occupancy binary"
	@echo "  make run        - run with RJ_LOG=1 (dispatch vgprs + host occupancy calc)"
	@echo "  make run-quiet  - run without RJ_LOG (host occupancy report only)"
	@echo "  make clean      - remove build/"
	@echo ""
	@echo "Note: rocjitsu reports register usage (vgprs) but does NOT compute"
	@echo "occupancy. The occupancy figures are a host-side calculation using the"
	@echo "CU limits in the config. LDS-limited occupancy is not shown (RJ_LOG"
	@echo "does not surface LDS usage)."
