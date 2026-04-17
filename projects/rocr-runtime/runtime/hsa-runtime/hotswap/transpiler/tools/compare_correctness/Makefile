# Build compare_correctness + libsalmon_intercept.so + per-ISA kernel .co files.
#
# Required variables:
#   ROCR_BUILD   path to a Salmon-enabled ROCR build (contains
#                rocr/lib/libhsa-runtime64.so with Salmon support).
#
# Usage:
#   make ROCR_BUILD=$HOME/rocm-systems/projects/rocr-runtime/build
#   make clean
#
# Outputs:
#   compare_correctness             — the harness binary
#   libsalmon_intercept.so          — LD_PRELOAD shim copied/built from tests/
#   kernels/build/<kernel>.<isa>.co — per-ISA unbundled code objects

ROCR_BUILD ?=
ifeq ($(strip $(ROCR_BUILD)),)
$(error ROCR_BUILD is not set. Point it at a Salmon-enabled ROCR build root)
endif

ROCM           ?= /opt/rocm-7.2.1
HIPCC          ?= hipcc
CXX            ?= g++
BUNDLER        := $(ROCM)/lib/llvm/bin/clang-offload-bundler

# Where the smoke_test_compare_transpilers tool's copy of salmon_intercept.cpp lives.
INTERCEPT_SRC  := ../../tests/salmon_intercept.cpp

KERNELS        := vecadd block_sum_shfl lane_swap \
                  cvt_pkrtz cvt_pk_f16 bfm_b32 swap_b32 mov_b64 cvt_f32_bf16 \
                  v_add_lshl_u32 v_bfe_i32 s_bfe_i32 s_bitset0_b32 \
                  s_bitset0_b64
ISAS           := gfx942 gfx1250
KERNEL_BUILD   := kernels/build

CO_FILES       := $(foreach k,$(KERNELS),$(foreach a,$(ISAS),$(KERNEL_BUILD)/$(k).$(a).co))

WARN           := -Wall -Wextra -Wno-unused-parameter
OPT            := -O2
CXXFLAGS       := -std=c++17 $(WARN) $(OPT)
HIPFLAGS       := $(CXXFLAGS)

.PHONY: all kernels clean help
all: compare_correctness libsalmon_intercept.so kernels

help:
	@echo "Targets: all | compare_correctness | libsalmon_intercept.so | kernels | clean"

# ─────────────────────────────────────────────────────────────────────────────
# Harness binary.  Compiled with hipcc so HIP headers and hip::host are
# picked up automatically, and -rpath'd at the Salmon-enabled ROCR so
# the produced binary loads THAT libhsa-runtime64.so (not a system one).
# ─────────────────────────────────────────────────────────────────────────────

# We explicitly link libhsa-runtime64 so that rocr_salmon_patch_elf is
# globally resolvable — the LD_PRELOAD shim uses dlsym(RTLD_DEFAULT, ...)
# to find it.  Without this, HIP dlopens the runtime with local scope
# and the symbol is not visible to the shim.
compare_correctness: compare_correctness.cpp
	$(HIPCC) $(HIPFLAGS) $< -o $@ \
	  -Wl,-rpath,$(ROCR_BUILD)/rocr/lib \
	  -L$(ROCR_BUILD)/rocr/lib \
	  -lhsa-runtime64

# ─────────────────────────────────────────────────────────────────────────────
# LD_PRELOAD shim.  Patches gfx1250 ELF e_flags so HIP will accept the
# binary; the real ISA stays in the msgpack metadata and triggers ROCR's
# hotswap hook.
# ─────────────────────────────────────────────────────────────────────────────

libsalmon_intercept.so: $(INTERCEPT_SRC)
	$(CXX) -shared -fPIC -std=c++17 $(OPT) -o $@ $< -ldl

# ─────────────────────────────────────────────────────────────────────────────
# Per-ISA kernel compilation.
#
# hipcc --genco produces a clang-offload-bundle; we then unbundle to get
# the raw ELF code object for a given ISA.
# ─────────────────────────────────────────────────────────────────────────────

kernels: $(CO_FILES)

$(KERNEL_BUILD):
	mkdir -p $@

# Intermediate bundled .co files.  Keep them as $(isa).bundle.co to
# disambiguate — the offload-bundler's unbundle step needs the matching
# target triple, which we also embed in the filename for grep-ability.
$(KERNEL_BUILD)/%.gfx942.bundle.co: kernels/%.hip | $(KERNEL_BUILD)
	$(HIPCC) --genco --offload-arch=gfx942 -o $@ $<

$(KERNEL_BUILD)/%.gfx1250.bundle.co: kernels/%.hip | $(KERNEL_BUILD)
	$(HIPCC) --genco --offload-arch=gfx1250 -o $@ $<

$(KERNEL_BUILD)/%.gfx942.co: $(KERNEL_BUILD)/%.gfx942.bundle.co
	$(BUNDLER) --type=o --input=$< --output=$@ \
	  --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --unbundle

$(KERNEL_BUILD)/%.gfx1250.co: $(KERNEL_BUILD)/%.gfx1250.bundle.co
	$(BUNDLER) --type=o --input=$< --output=$@ \
	  --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 --unbundle

clean:
	rm -f compare_correctness libsalmon_intercept.so
	rm -rf $(KERNEL_BUILD)
