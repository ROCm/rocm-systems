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

# Triton AOT compilation environment.  Override at the command line if your
# Triton tree lives elsewhere.  The venv must have Triton + PyYAML installed;
# the PYTHONPATH entries expose the in-tree triton/python/ sources (which
# have the AMD backend not present in pip's `triton` package).
TRITON_VENV       ?= $(HOME)/rocm-systems/triton/.venv
# triton/python is the in-tree `triton` package.  third_party/amd/python
# has the AMD backend bits the pip-installed Triton lacks.
# triton/python/triton_kernels is the nested `triton_kernels` library
# (pyproject.toml-packaged but not installed in the venv); recipes that
# invoke real GPT-OSS kernels (`sum_bitmatrix_rows_u32.py`,
# `_topk_forward` wrappers, etc.) import from it.
TRITON_PYTHONPATH ?= $(HOME)/rocm-systems/triton/python:$(HOME)/rocm-systems/triton/third_party/amd/python:$(HOME)/rocm-systems/triton/python/triton_kernels
TRITON_PY         := env PYTHONPATH=$(TRITON_PYTHONPATH) $(TRITON_VENV)/bin/python

# Where the smoke_test_compare_transpilers tool's copy of salmon_intercept.cpp lives.
INTERCEPT_SRC  := ../../tests/salmon_intercept.cpp

KERNELS        := vecadd block_sum_shfl lane_swap \
                  cvt_pkrtz cvt_pk_f16 bfm_b32 swap_b32 mov_b64 cvt_f32_bf16 \
                  v_add_lshl_u32 v_bfe_i32 s_bfe_i32 s_bitset0_b32 \
                  s_bitset0_b64 v_bfi_b32 v_cmp_cndmask_sgpr \
                  v_cmpx_gt_i32 s_and_saveexec_b32 \
                  c4_lane_dep_cmpx \
                  wmma_f32_16x16x4_gemm \
                  mubuf_store_b32 s_set_vgpr_msb \
                  canary_readlane_last_lane \
                  canary_ds_swizzle_swap1 \
                  canary_ds_swizzle_quad_perm \
                  canary_cvt_scale_pk8_bf16_fp4

# Triton recipes.  Each name must correspond to a kernels/triton/<name>.py
# file that defines a RECIPES list (see kernels/triton/README.md).
#
# `vecadd_f16` is hand-authored.  `corpus_*` recipes are thin wrappers
# around upstream Triton kernels pulled by kernels/triton/_corpus/pull.py
# (see kernels/triton/_corpus/README.md).  Building a corpus_* recipe
# without first running pull.py fails fast with a helpful message.
# `canary_*` recipes are hand-authored minimal kernels that each pin
# down one wave-size obstruction class from
# `hotswap/docs/gpt-oss-derisking.md §1` — bpermute, DPP, permlane —
# so a salmon regression on any class surfaces from a unit-sized
# recipe rather than only from GPT-OSS-scale kernels where triage
# is hard.
TRITON_KERNELS := vecadd_f16 corpus_add_fp32 corpus_asin_fp32 \
                  corpus_softmax_fp32 corpus_layernorm_fp32 \
                  canary_bpermute_scan_fp32 \
                  canary_dpp_reduce_fp32 \
                  canary_dpp_compound_add_fp32 \
                  canary_dpp_compound_add_u32 \
                  canary_permlanex16_rowmax_fp32 \
                  canary_bitmatrix_composite \
                  rmsnorm_fp32 \
                  swiglu_fp32 \
                  rope_fp32 \
                  matmul_fp16 \
                  matmul_fp16_16x16 \
                  sum_bitmatrix_rows_u32 \
                  sum_bitmatrix_rows_u32_nw4 \
                  downcast_to_mxfp_fp32 \
                  downcast_to_mxfp_bf16 \
                  topk_forward_bf16

# Both KERNELS and TRITON_KERNELS feed the same %.gfx942.co / %.gfx1250.co
# pattern rules.  If a name appears in both lists, Make's rule-selection is
# undefined — it picks one source (HIP .hip or Triton .py) and silently
# uses it for *all* runs of that name.  Catch that loudly at parse time
# rather than producing a confusing build artifact.
_NAME_COLLISIONS := $(filter $(KERNELS),$(TRITON_KERNELS))
ifneq ($(strip $(_NAME_COLLISIONS)),)
$(error Recipe name(s) appear in both KERNELS and TRITON_KERNELS: \
$(_NAME_COLLISIONS). Rename one to avoid pattern-rule ambiguity)
endif

ISAS           := gfx942 gfx1250
KERNEL_BUILD   := kernels/build

CO_FILES         := $(foreach k,$(KERNELS),$(foreach a,$(ISAS),$(KERNEL_BUILD)/$(k).$(a).co))
TRITON_CO_FILES  := $(foreach k,$(TRITON_KERNELS),$(foreach a,$(ISAS),$(KERNEL_BUILD)/$(k).$(a).co))
TRITON_SIDECARS  := $(foreach k,$(TRITON_KERNELS),$(KERNEL_BUILD)/$(k).sidecar.json)

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

kernels: $(CO_FILES) $(TRITON_CO_FILES) $(TRITON_SIDECARS)

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

# ─────────────────────────────────────────────────────────────────────────────
# Triton AOT compilation.
#
# A single invocation of aot_compile.py produces *three* files per recipe —
# the two .co files and the sidecar JSON — so we use a grouped target (&:)
# to tell Make those outputs share one invocation.  The prerequisites also
# include the AOT script itself, so edits to the script rebuild all Triton
# kernels without a `make clean`.
# ─────────────────────────────────────────────────────────────────────────────

TRITON_AOT := kernels/triton/aot_compile.py

$(KERNEL_BUILD)/%.gfx942.co $(KERNEL_BUILD)/%.gfx1250.co $(KERNEL_BUILD)/%.sidecar.json &: \
    kernels/triton/%.py $(TRITON_AOT) | $(KERNEL_BUILD)
	$(TRITON_PY) $(TRITON_AOT) --kernel-file=$< --out-dir=$(KERNEL_BUILD)

clean:
	rm -f compare_correctness libsalmon_intercept.so
	rm -rf $(KERNEL_BUILD)
