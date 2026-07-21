HIPCC ?= hipcc
ARCH ?= gfx950
CXXFLAGS = -std=c++20 -Wall -O2
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG ?= ../../configs/amdgpu_cdna4_kmd.json

.PHONY: all run-race run-race-no-detector run-fixed compare clean help

all: $(BUILD_DIR)/global_race $(BUILD_DIR)/global_fixed

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: src/%.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) -o $@ $<

# Buggy version under the race detector. NOTE: RJ_RACE=1 does NOT report this
# inter-block global race; the host golden check is what catches it.
run-race: $(BUILD_DIR)/global_race
	RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $< || true

# Buggy version without the detector (identical result — proves the detector
# adds nothing for this class of bug).
run-race-no-detector: $(BUILD_DIR)/global_race
	$(ROCJITSU_BIN) --config $(CONFIG) -- $< || true

# Fixed version (atomicAdd). Host check passes; detector still silent.
run-fixed: $(BUILD_DIR)/global_fixed
	RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $<

compare: $(BUILD_DIR)/global_race $(BUILD_DIR)/global_fixed
	@echo "--- BUGGY (non-atomic) ---"
	@RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(BUILD_DIR)/global_race 2>&1 | grep -E "Verification|sum|Lost" || true
	@echo "--- FIXED (atomicAdd) ---"
	@RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(BUILD_DIR)/global_fixed 2>&1 | grep -E "Verification|sum"

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Global Memory Race Example - targets:"
	@echo "  make                    - build both versions"
	@echo "  make run-race           - buggy + RJ_RACE=1 (detector stays silent; host detects)"
	@echo "  make run-race-no-detector - buggy without detector (same result)"
	@echo "  make run-fixed          - atomicAdd fix (host check passes)"
	@echo "  make compare            - buggy vs fixed side-by-side"
	@echo ""
	@echo "Key point: RJ_RACE=1 does NOT flag inter-block global races."
	@echo "Use it for intra-workgroup LDS/VGPR/SGPR hazards (see Example 04)."
