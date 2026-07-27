HIPCC ?= hipcc
ARCH ?= gfx950
CXXFLAGS = -std=c++20 -O2
BUILD_DIR = build
ROCJITSU_ROOT = ../..
ROCJITSU_BIN ?= $(ROCJITSU_ROOT)/build/tools/rocjitsu/rocjitsu
DBT_TRANSLATE ?= $(ROCJITSU_ROOT)/build/tools/rj_dbt_translate
CONFIG_CDNA4 ?= $(ROCJITSU_ROOT)/configs/amdgpu_cdna4_kmd.json

INPUT_ISA ?= gfx950

.PHONY: all list translate translate-reverse translate-rdna translate-co compare run-gfx950 clean help

all: $(BUILD_DIR)/dbt_example

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/dbt_example: src/dbt_example.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) -o $@ $<
	@echo "Compiled for $(ARCH)"

# A gfx942 build, used as the input for the reverse (MI300 -> MI350) direction.
$(BUILD_DIR)/dbt_example_gfx942: src/dbt_example.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=gfx942 $(CXXFLAGS) -o $@ $<

# List the code objects bundled in the binary (should show $(INPUT_ISA)).
list: $(BUILD_DIR)/dbt_example
	$(DBT_TRANSLATE) $(BUILD_DIR)/dbt_example --list-code-objects

# Down-level: gfx950 -> gfx942 (MI350 -> MI300). Translates cleanly; mostly identity.
translate: $(BUILD_DIR)/dbt_example
	$(DBT_TRANSLATE) $(BUILD_DIR)/dbt_example \
		--input-target $(INPUT_ISA) --output-target gfx942 --output-mode diff

# Up-level: gfx942 -> gfx950 (MI300 -> MI350). Newer ISA is a near-superset, so
# every instruction is copied verbatim -- a no-op translation.
translate-reverse: $(BUILD_DIR)/dbt_example_gfx942
	$(DBT_TRANSLATE) $(BUILD_DIR)/dbt_example_gfx942 \
		--input-target gfx942 --output-target gfx950 --output-mode diff

# Divergent pair: gfx950 -> gfx1200 (CDNA -> RDNA). Shows real substitution /
# lowering, then hits a current translator coverage gap (v_add_u32_e32 EXPAND),
# so it exits non-zero -- that's expected here.
translate-rdna: $(BUILD_DIR)/dbt_example
	-$(DBT_TRANSLATE) $(BUILD_DIR)/dbt_example \
		--input-target $(INPUT_ISA) --output-target gfx1200 --output-mode diff

# Side-by-side: translation effort depends on ISA distance AND direction.
compare: $(BUILD_DIR)/dbt_example $(BUILD_DIR)/dbt_example_gfx942
	@echo "=== gfx942 -> gfx950  (MI300 -> MI350, up-level, CDNA3->CDNA4) ==="
	@$(DBT_TRANSLATE) $(BUILD_DIR)/dbt_example_gfx942 --input-target gfx942 \
		--output-target gfx950 --output-mode diff 2>&1 \
		| grep -E "rj_dbt_translate:|actions:|errors:" || true
	@echo ""
	@echo "=== gfx950 -> gfx942  (MI350 -> MI300, down-level, CDNA4->CDNA3) ==="
	@$(DBT_TRANSLATE) $(BUILD_DIR)/dbt_example --input-target $(INPUT_ISA) \
		--output-target gfx942 --output-mode diff 2>&1 \
		| grep -E "rj_dbt_translate:|actions:|errors:" || true
	@echo ""
	@echo "=== gfx950 -> gfx1200 (MI350 -> Radeon RX 9070, cross-family, CDNA->RDNA) ==="
	@$(DBT_TRANSLATE) $(BUILD_DIR)/dbt_example --input-target $(INPUT_ISA) \
		--output-target gfx1200 --output-mode diff 2>&1 \
		| grep -E "rj_dbt_translate:|actions:|errors:|error:" || true

# Write the translated gfx942 code object to build/dbt_example.gfx942.co
translate-co: $(BUILD_DIR)/dbt_example
	$(DBT_TRANSLATE) $(BUILD_DIR)/dbt_example \
		--input-target $(INPUT_ISA) --output-target gfx942 \
		--output-mode code-object > $(BUILD_DIR)/dbt_example.gfx942.co
	@echo "Wrote $(BUILD_DIR)/dbt_example.gfx942.co"

# Sanity: run the gfx950 binary on its native (emulated) arch under rocjitsu.
run-gfx950: $(BUILD_DIR)/dbt_example
	$(ROCJITSU_BIN) --config $(CONFIG_CDNA4) -- $(BUILD_DIR)/dbt_example

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "DBT cross-arch example - targets:"
	@echo "  make             - compile the kernel for $(ARCH)"
	@echo "  make list        - list code objects bundled in the binary"
	@echo "  make translate   - gfx950 -> gfx942 diff (MI350->MI300, down-level; mostly identity)"
	@echo "  make translate-reverse - gfx942 -> gfx950 diff (MI300->MI350, up-level; pure copy)"
	@echo "  make translate-rdna - gfx950 -> gfx1200 diff (CDNA->RDNA; real xlate, hits a gap)"
	@echo "  make compare     - all three side-by-side (effort scales with ISA distance + direction)"
	@echo "  make translate-co - write the translated gfx942 code object"
	@echo "  make run-gfx950  - run the binary on its native arch under rocjitsu"
	@echo "  make clean       - remove build/"
	@echo ""
	@echo "Translate-and-RUN cross-arch is an HSA-level feature (HSA_TOOLS_LIB +"
	@echo "RJ_DBT_TARGET_ISA); it does not work for HIP apps. See the README."
