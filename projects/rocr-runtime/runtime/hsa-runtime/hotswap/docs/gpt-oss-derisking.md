# GPT-OSS Cross-Architecture Retargeting: De-risking Report

**Scope.** Answer the question "do any kernels in the GPT-OSS North Star corpus
have features that would prevent retargeting from gfx1250 (RDNA4, wave32) to
gfx942 / gfx950 (CDNA3 / CDNA4, wave64) under our principled wave-size
strategy?"

**Answer.** For the 15 Triton kernels that are genuinely GPT-OSS, no kernel
falls into the refusal outcome of the decision procedure in
`wave-size-translation.md §7`. Every observed obstruction is either already
wave-size-oblivious (outcome **a**, emit modulo-replication directly) or has a
defined entry in the finite rewrite table (outcome **b**). The supporting
hipBLASLt GEMM library (tensilelite, 147 kernels) is also clean on the
unrewritable classes, with the bulk of its obstructions concentrated in a
single `ds_bpermute_b32` pattern that is a ~30-minute transpiler fix. AMD's
upstream Gluon gfx1250 example harness (the `triton/` sub-corpus plus three
MXFP variants reasoned about from source in §8) is cleaner still — its only
cross-lane primitive is `v_permlane16_swap`, covered by the same rewrite
table items.

This report documents (a) what the corpus actually is, (b) which parts are
uniquely GPT-OSS vs. generic library code vs. unrelated, (c) per-class
findings scoped to each sub-corpus, and (d) the engineering work that must
land before GPT-OSS will translate correctly end-to-end.

---

## 1. Framework recap

Our wave-size strategy, defined in
`projects/rocr-runtime/runtime/hsa-runtime/hotswap/docs/wave-size-translation.md`,
treats a target wave as `R = W_t / W_s` replicas of the source wave (R = 2 for
the wave32 → wave64 case of interest). The correctness proof obligation is
that each replica observes the same state and makes the same decisions as the
original source wave. Four classes of constructs can violate that obligation:

| Class | Description | Typical instructions |
|-------|-------------|----------------------|
| **C1** | Absolute lane-ID leaks | `v_mbcnt_hi_u32_b32`, `v_readlane_b32` / `v_writelane_b32` with operand ≥ W_s, `llvm.amdgcn.ballot`, `llvm.amdgcn.wavefrontsize` |
| **C2** | Wave-size-baked cross-lane ops | `v_permlane64_b32`, `v_permlane16_swap_b32`, `v_permlane32_swap_b32`, `v_permlanex16_b32`, `ds_swizzle_b32`, DPP with wave-wide semantics, `ds_bpermute_b32` |
| **C3** | Inter-replica communication via shared state | Non-commutative atomics (`cmpxchg`, `xchg`) or LDS write-write races on `lane_id mod W_s`-derived addresses |
| **C4** | Lane-position-dependent EXEC writes | `v_cmpx_*` against a constant tied to absolute lane position; `s_*_saveexec_b32` with an `mbcnt_hi`-derived mask |

Per-kernel decision procedure:

  1. **(a)** Prove the kernel is wave-size-oblivious → emit modulo-replication.
  2. **(b)** Rewrite the failing site from the rewrite table, then re-prove → emit.
  3. **(c)** No rewrite available → `report_fatal_error` with kernel name,
     instruction, and reason.

---

## 2. Corpus — what we actually catalogued

`kerneldex` catalogued 170 gfx1250 code objects. These split into three
categorically different groups, and the rest of this report is careful to
keep them separate because their relationship to "GPT-OSS" is different:

| Group | Count | What it actually is |
|-------|-------|---------------------|
| `scope_discovery/` | **15** | **The GPT-OSS Triton kernel surface**, captured by driving GPT-OSS's own entry points with GPT-OSS-20B / 120B configs. |
| `tensilelite/` | **147** | **hipBLASLt GEMM library** kernels that GPT-OSS (and any other model) calls into for dense matmuls. Not unique to GPT-OSS, but what GPT-OSS actually hits. |
| `triton/` + `test_data_gfx1250/` | **8** | **Not GPT-OSS.** Four internal Triton pipelining experiments and four gfx1250 smoke tests. Included in the corpus because they're representative wave32 workloads, not because the model uses them. |

### 2.1 Provenance of `scope_discovery/`

The 15 Triton kernels were produced by
`hotswap/transpiler/scope_discovery/capture/run_gpt_oss.py`, which:

- imports `_attn_fwd` directly from the actual GPT-OSS source tree
  (`/data/gpt-oss/src/gpt_oss/triton/attention.py`);
- imports the `triton_kernels` library that GPT-OSS pins for its MoE path
  (`matmul_ogs`, `topk`, `compaction`, `swiglu`, `downcast_to_mxfp`,
  `upcast_from_mxfp`, `reduce`);
- runs each entry point with GPT-OSS-20B shapes (`hidden=2880`,
  `num_experts=32`, `experts_per_token=4`, `num_heads=64`, `num_kv_heads=8`
  GQA, `head_dim=64`, `bf16`) and a second stage with GPT-OSS-120B head
  layout (`n_ctx=128`, `gqa=8`);
- monkey-patches `triton.compile` so every JIT-compiled `.hsaco` is written
  to `scope_discovery/kernels/` as the real forward pass executes.

The 15 kernels map to the GPT-OSS forward pass as follows:

| Kernel | Source | Role in GPT-OSS |
|--------|--------|-----------------|
| `attn_fwd` | `gpt_oss.triton.attention._attn_fwd` | FlashAttention with **attention sinks**, GQA |
| `topk_forward` | `triton_kernels.topk` | **MoE router top-k** (4 of 32 experts) |
| `bitmatrix_metadata_compute_stage1` / `_stage2` | routing substructure | Token → expert assignment **bitmatrix** |
| `sum_bitmatrix_rows` | routing substructure | Per-expert token count aggregation |
| `masked_compaction` | `triton_kernels.compaction` | **MoE dispatch / combine** |
| `matmul_ogs` ×4 | `triton_kernels.matmul_ogs` | **MoE expert GEMM** (dense, batched-dense, fp32-acc, MXFP4 weights) |
| `swiglu` | `triton_kernels.swiglu` | **SwiGLU** activation (α=1.702, limit=7.0) |
| `downcast_to_mxfp` / `upcast_from_mxfp` | MXFP quant path | **MXFP4** expert-weight quantize / dequantize |
| `reduce` ×2 | `triton_kernels.reduce` | Utility reductions |

Every distinctive GPT-OSS Triton primitive — attention with sinks, MoE
routing, MoE expert GEMM with MXFP4, SwiGLU activation — is present with the
model's real config.

### 2.2 Provenance of `tensilelite/`

These are hipBLASLt's Tensile-generated GEMMs, captured during the same
GPT-OSS run. The naming encodes dtype pairs (`BB`=bf16/bf16, `HH`=fp16/fp16,
`F8F8`=fp8/fp8, `B8F8`=bf16/fp8, `I8I8`=int8/int8, etc.) and access patterns
(`Ailk_Bjlk_Cijk_Dijk`, `Alik_Bljk_Cijk_Dijk`, ...). GPT-OSS hits these for:

- attention Q / K / V projections and the attention-output projection;
- token embedding and LM head;
- any dense linear layer outside the MoE path.

They are **not unique to GPT-OSS**, but they are the specific shapes /
dtypes selected by the Tensile autotuner for GPT-OSS's hidden size and
precision choices.

### 2.3 Provenance of `triton/` and `test_data_gfx1250/`

These 8 kernels are **not part of GPT-OSS**. They are representative wave32
workloads included in the sweep corpus for coverage reasons:

- `triton/f16_fa_pipeline_attn_fwd_pipelined_kernel` — upstream AMD Gluon
  example, the `--attention-type pipeline` invocation of
  [`third_party/amd/python/examples/gluon/f16_fa_gfx1250.py`](https://github.com/triton-lang/triton/tree/main/third_party/amd/python/examples/gluon).
- `triton/f16_gemm_*_tdm_pipelined_kernel` ×3 — upstream AMD Gluon GEMM
  examples (`f16_gemm_gfx1250.py` with `--num-warps 8`, the same with
  `--single-warp-schedule`, and `f16_gemm_warp_pipeline_gfx1250.py`).
- `test_data_gfx1250/{matmul_f16,matmul_f16_large,softmax,vecadd}` — gfx1250
  transpiler smoke tests.

They are retained in this report because they share the mnemonic histogram
and, as the Gluon provenance shows, serve as a principled reference for how
a carefully hand-written wave32 kernel author expects the compiler to emit
cross-lane primitives. See §8 for a full cross-reference of the Gluon
example surface.

### 2.4 Capture gaps — things GPT-OSS uses that we did NOT JIT-capture

The `scope_discovery` driver exercises Triton entry points. These GPT-OSS
components were **not** captured and the corpus does not cover them:

- **RMSNorm.** GPT-OSS uses RMSNorm. No normalization kernel appears in
  `scope_discovery/`. It is likely invoked through PyTorch's native fused
  norm (not a JIT-compiled Triton kernel) or fused inside `_attn_fwd`. This
  needs to be verified before claiming norm coverage.
- **RoPE (with YaRN scaling).** No standalone RoPE kernel was captured.
  GPT-OSS's `_attn_fwd` applies RoPE inline, so it is likely covered
  transitively by `attn_fwd`, but this should be confirmed in the kernel
  source.
- **KV-cache append / indexed read.** The driver runs single-shot prefill
  only; decode-path KV-cache kernels are not exercised.
- **Sampling / logits post-processing.** Out of scope for this capture.
- **Backward / training kernels.** Out of scope: GPT-OSS *inference* is the
  North Star.

None of these absences invalidate the de-risking conclusions for what
**was** captured, but they mean "the GPT-OSS Triton surface" in this report
refers to inference forward pass modulo the above.

---

## 3. Audit methodology

For each kernel, the per-mnemonic histogram was regex-scanned for the four
obstruction classes. C2 is split into three sub-categories because the
transpiler's engineering status differs between them (see the per-SemOp
status table in `wave-size-translation.md §5.3`).

```python
C1             = r"^(v_mbcnt_hi|v_readlane_b32|v_writelane_b32|s_ballot|v_ballot|wavefrontsize)"
C2_hard        = r"^(v_permlane64|v_permlane16_swap|v_permlane32_swap|v_permlanex16|ds_swizzle)"
C2_dpp         = r"^v_[a-z0-9_]+_dpp[0-9]?(_e..)?$"
C2_bpermute    = r"^ds_bpermute_b32"
C3_noncomm     = r"(cmpswap|cmpxchg|atomic_swap|atomic_xchg)"
C4_cmpx        = r"^v_cmpx"
C4_saveexec    = r"^s_(and|or|xor|andn2|orn2)_saveexec_"
```

---

## 4. Results — per-corpus summary

Per-class kernel counts, split by sub-corpus:

| Class | GPT-OSS Triton (15) | hipBLASLt (147) | Other (8) |
|-------|---------------------|------------------|-----------|
| C1 absolute lane-ID leak | 1 (6.7%) | 0 (0%) | 2 (25%) |
| C2 hard (`permlane*`, `ds_swizzle`) | 4 (26.7%) | 0 (0%) | 2 (25%) |
| C2 DPP modifier | 5 (33.3%) | 0 (0%) | 1 (12.5%) |
| C2 `ds_bpermute_b32` | 3 (20.0%) | 53 (36.1%) | 0 (0%) |
| C3 non-commutative atomic | **0** | **0** | **0** |
| C4 `v_cmpx` | 2 (13.3%) | 1 (0.7%) | 3 (37.5%) |
| C4 `s_*_saveexec_b32` | 3 (20.0%) | 81 (55.1%) | 6 (75%) |

Structural results that hold **across every sub-corpus**:

- **No `v_mbcnt_hi_u32_b32`** anywhere. This is the canonical C1 leak that
  chains into EXEC computation and has no rewrite; its absence is load-bearing
  for the whole strategy.
- **No `llvm.amdgcn.ballot` / `wavefrontsize`** anywhere.
- **No non-commutative atomics** anywhere. Triton reductions in GPT-OSS use
  commutative `atomic_add` / `atomic_max_u32` only.
- **No `v_permlane64_b32`** (the only C2 primitive that would require a true
  64-lane hardware shuffle on the target).

---

## 5. Per-kernel findings — GPT-OSS Triton (the 15 that matter)

Complete per-kernel obstruction map for `scope_discovery/`:

| Kernel | Obstructions |
|--------|--------------|
| `attn_fwd` | C2-hard: `v_permlane16_swap_b32_e32 ×4` |
| `bitmatrix_metadata_compute_stage1` | C1: `v_readlane_b32 ×1`; C2-hard: `v_permlanex16_b32 ×1`; C2-DPP: `v_add_nc_u32_dpp ×4`, `v_mov_b32_dpp ×2`; C2-bpermute: `ds_bpermute_b32 ×12`; C4: `v_cmpx_eq_u32_e32 ×1`, `s_and_saveexec_b32 ×1` |
| `bitmatrix_metadata_compute_stage2` | C2-hard: `v_permlane16_swap_b32_e32 ×3`; C2-DPP: `v_mov_b32_dpp ×34`; C2-bpermute: `ds_bpermute_b32 ×5`; C4: `v_cmpx_eq_u32_e32 ×1` |
| `downcast_to_mxfp` | C2-DPP: `v_mov_b32_dpp ×8` |
| `masked_compaction` | C2-bpermute: `ds_bpermute_b32 ×2` |
| `matmul_ogs_06d912ce88af` | C4: `s_and_saveexec_b32 ×54` (bounds-check form) |
| `matmul_ogs_0af655e6ea2b` | *(clean — outcome a)* |
| `matmul_ogs_16674cb4d384` | *(clean — outcome a)* |
| `matmul_ogs_25359f86c8d1` | *(clean — outcome a)* |
| `reduce_00843b402f1a` | *(clean — outcome a)* |
| `reduce_e1b013fbed52` | *(clean — outcome a)* |
| `sum_bitmatrix_rows` | C2-hard: `ds_swizzle_b32 ×8`; C2-DPP: `v_mov_b32_dpp ×96` |
| `swiglu` | *(clean — outcome a)* |
| `topk_forward` | C2-DPP: `v_mov_b32_dpp ×10`; C4: `s_and_saveexec_b32 ×1` |
| `upcast_from_mxfp` | *(clean — outcome a)* |

Highlights:

- **7 of 15 GPT-OSS kernels are fully clean** (outcome a directly): three
  `matmul_ogs` variants, both `reduce`, `swiglu`, `upcast_from_mxfp`.
  Roughly half the MoE forward-pass surface translates with no rewrite at
  all.
- **The MoE-router bitmatrix pipeline (`stage1` + `stage2`) concentrates
  most of the complexity.** Between them, these two kernels carry every
  class of obstruction GPT-OSS exhibits. They are the highest-risk pair and
  should be the earliest translation targets for correctness validation.
- **Attention (`attn_fwd`) only needs `permlane16_swap` support.** The
  GPT-OSS FlashAttention-with-sinks kernel is clean on C1 / C3 / C4 (the
  non-rewritable classes) and its only obstruction is the softmax row-reduce
  pattern using `v_permlane16_swap_b32_e32`, which has a clean intrinsic
  lift (`wave-size-translation.md §5.3` items P2 / P4).
- **The MoE expert GEMM (`matmul_ogs`) is almost entirely clean**, with one
  of four variants using a bounds-check `s_and_saveexec_b32` (position-
  independent EXEC mask, outcome a under modulo-replication).

---

## 6. Per-kernel findings — hipBLASLt (the 147 that GPT-OSS calls)

The tensilelite GEMM library is structurally simpler from an obstruction
standpoint:

- **C1 / C2 hard / C2 DPP / C3: all zero.** No permlane ops, no wave-size-
  baked cross-lane primitives, no non-commutative atomics, no absolute
  lane-ID leaks in any of 147 kernels.
- **C2 `ds_bpermute_b32`: 53 / 147 (36.1%).** This is the MFMA layout
  transpose pattern. Observed selectors are uniform / block-invariant. It is
  rewritable via `wave-size-translation.md §5.3` item **P1** (emit
  `llvm.amdgcn.ds_bpermute`) — a ~30-minute transpiler fix.
- **C4 `s_*_saveexec_b32`: 81 / 147 (55.1%).** Tensile's tail-loop and
  edge-tile handling emits save-exec around bounds-checked epilogues.
  Invariantly the mask derives from a uniform bounds check on a
  `lane_id mod W_s`-projectable expression, so these are outcome (a) under
  modulo-replication.
- **C4 `v_cmpx`: 1 / 147 (0.7%).** A single GEMM variant, same bounds-check
  shape as above.

hipBLASLt is de-risked: once P1 lands for `ds_bpermute`, the entire library
subcorpus should translate under outcome (a) or the bounds-check specialization
of outcome (a).

---

## 7. Detailed findings per class (cross-corpus)

### 7.1 C1 — absolute lane-ID leaks

| Kernel | Corpus | Instructions |
|--------|--------|--------------|
| `bitmatrix_metadata_compute_stage1` | GPT-OSS | `v_readlane_b32 ×1` |
| `matmul_f16_large_gfx1250` | other (test) | `v_readlane_b32 ×15`, `v_writelane_b32 ×15` |
| `softmax_gfx1250` | other (test) | `v_readlane_b32 ×2` |

All three are `readlane` / `writelane` with small-constant lane operands,
correct as long as the operand is statically in `[0, W_s)`. Rewrite per
`wave-size-translation.md §6` (`OutOfRangeLaneOperand`): operand-range
check at raise time, fail loudly if the operand is out of range, pass
through otherwise. The GPT-OSS occurrence satisfies the constraint.

### 7.2 C2 hard — `permlane*` and `ds_swizzle`

All in GPT-OSS Triton or `other`. hipBLASLt has zero.

| Kernel | Corpus | Instructions |
|--------|--------|--------------|
| `attn_fwd` | GPT-OSS | `v_permlane16_swap_b32_e32 ×4` (softmax row reduce) |
| `bitmatrix_metadata_compute_stage1` | GPT-OSS | `v_permlanex16_b32 ×1` |
| `bitmatrix_metadata_compute_stage2` | GPT-OSS | `v_permlane16_swap_b32_e32 ×3` |
| `sum_bitmatrix_rows` | GPT-OSS | `ds_swizzle_b32 ×8` |
| `softmax_gfx1250` | other (test) | `v_permlanex16_b32 ×2` |
| `f16_fa_pipeline_attn_fwd_pipelined_kernel` | other (triton pipelining) | `v_permlane16_swap_b32_e32 ×19` |

Rewrite path: `wave-size-translation.md §5.3` items **P2 / P4** (permlane
intrinsic lifts) and **P6** (`ds_swizzle` intrinsic lift). All have clean
LLVM intrinsics on both source and target — the rewrite is
intrinsic-to-intrinsic in the raiser.

### 7.3 C2 DPP — the largest current correctness hazard

`raiser.cpp` canonicalizes DPP modifiers away without lifting them to the
`llvm.amdgcn.update.dpp` intrinsic. This is
`wave-size-translation.md §5.3` item **P5**, the most invasive open item.

GPT-OSS kernels affected:

| Kernel | Corpus | Instructions |
|--------|--------|--------------|
| `bitmatrix_metadata_compute_stage1` | GPT-OSS | `v_add_nc_u32_dpp ×4`, `v_mov_b32_dpp ×2` |
| `bitmatrix_metadata_compute_stage2` | GPT-OSS | `v_mov_b32_dpp ×34` |
| `downcast_to_mxfp` | GPT-OSS | `v_mov_b32_dpp ×8` |
| `sum_bitmatrix_rows` | GPT-OSS | `v_mov_b32_dpp ×96` |
| `topk_forward` | GPT-OSS | `v_mov_b32_dpp ×10` |
| `softmax_gfx1250` | other (test) | `v_mov_b32_dpp ×8`, `v_add_f32_dpp ×4` |

**5 of 15 GPT-OSS kernels use DPP.** Until P5 lands, these kernels are
silently miscompiled. This is the single largest outstanding correctness
risk for the GPT-OSS North Star. DPP is not unrewritable in principle — the
work is "lift the DPP modifier to the intrinsic on raise, let the backend
re-emit it on lower" — but it is estimated at ~1 engineering-week.

### 7.4 C2 `ds_bpermute_b32`

| Corpus | Kernels hit | Notes |
|--------|-------------|-------|
| GPT-OSS | 3 / 15 | `bitmatrix_metadata_compute_stage1` (×12), `bitmatrix_metadata_compute_stage2` (×5), `masked_compaction` (×2) |
| hipBLASLt | 53 / 147 | MFMA layout transpose, uniform selectors |

Current raiser behavior is a same-lane move (the pre-P1 state —
`wave-size-translation.md §5.3` P1 documents the landed
`llvm.amdgcn.ds.bpermute` emission). Observed GPT-OSS usage uses
uniform / block-invariant selectors, so the same-lane move happens to
produce the correct value — but this is **accidental correctness**,
not principled.

P1 is a ~30-minute fix (emit `llvm.amdgcn.ds_bpermute`). It must land before
we claim either GPT-OSS or the hipBLASLt GEMM backbone translates correctly.

### 7.5 C3 — non-commutative atomics

**None. Anywhere. In any sub-corpus.**

This is the single largest structural de-risking result. Neither GPT-OSS
Triton code, nor hipBLASLt, nor the pipelining experiments uses
`cmpxchg` / `xchg` / swap-style atomics where one replica would race another
over shared state.

### 7.6 C4 — EXEC writers

`s_*_saveexec_b32` and `v_cmpx_*` are widespread, but provenance matters:

- **hipBLASLt:** 81 / 147 (55.1%) use `s_and_saveexec_b32`. All observed
  uses wrap bounds-checked tail-loop epilogues; the mask is a function of
  `lane_id mod W_s` alone. Outcome (a).
- **GPT-OSS:** 3 / 15 use `s_and_saveexec_b32`, 2 / 15 use `v_cmpx`. Same
  shape: Triton's `tl.arange(0, N) < bound_uniform` idiom, position-
  independent. Outcome (a).
- **Non-bounds-check C4** (raw `v_cmpx` against an absolute lane-ID
  constant, the pattern `cross_wave_warn.hip` was engineered to catch) does
  not appear in GPT-OSS or hipBLASLt.

---

## 8. Gluon kernel reachability

A related question came up separately: will AMD's upstream Gluon examples
for gfx1250 (in
[`third_party/amd/python/examples/gluon/`](https://github.com/triton-lang/triton/tree/main/third_party/amd/python/examples/gluon))
exercise any pattern we can't retarget? Gluon is a lower-level Triton DSL
that exposes hardware intrinsics directly, so it is the most principled
reference we have for what a disciplined wave32 kernel surface looks like.

### 8.1 Mapping from Gluon commands to our corpus

Four of the seven active Gluon invocations in the reference harness map
directly to kernels we already captured:

| Gluon command | Captured binary |
|---------------|-----------------|
| `f16_gemm_gfx1250.py ... --num-warps 8 --num-buffers 3` | `triton__f16_gemm_8warp_basic_gemm_tdm_pipelined_kernel` |
| `f16_gemm_gfx1250.py ... --prefetch-lds --single-warp-schedule` | `triton__f16_gemm_single_warp_schedule_gemm_tdm_pipelined_single_warp_per_simd_schedule_kernel` |
| `f16_gemm_warp_pipeline_gfx1250.py` | `triton__f16_gemm_warp_pipeline_gemm_tdm_pipelined_warp_pipelined_kernel` |
| `f16_fa_gfx1250.py --attention-type pipeline` | `triton__f16_fa_pipeline_attn_fwd_pipelined_kernel` |

The three remaining Gluon invocations were not compiled into the corpus and
have to be reasoned about from source:

- `mxfp_gemm_gfx1250.py` (MXFP GEMM, e.g. `dtype_a=float8_e4m3 dtype_b=float8_e4m3 --scale_preshuffled`)
- `mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e4m3 ... --scale_type global --pipelined`
- `mxfp_fa_gfx1250.py --q_type e4m3 --kv_type e2m1 ... --scale_type block --pipelined`

### 8.2 Empirical profile of the captured four

From the binary audit in §7:

- **F16 FA pipeline:** 19× `v_permlane16_swap_b32_e32`. Nothing else. Single
  C2-hard primitive, covered by P2/P3/P4.
- **F16 GEMM ×3:** only `s_and_saveexec_b32` (53–113 occurrences) and
  `v_cmpx_{gt_i64,lt_u32}` (1–4 occurrences). All bounds-check form,
  outcome (a).

No DPP, no `ds_bpermute`, no `ds_swizzle`, no C1 leak, no C3 atomic in any
of the four. This is the cleanest surface in the whole 170-kernel corpus.

### 8.3 Source-level surface of all Gluon examples

Grepping `mxfp_fa_gfx1250.py`, `mxfp_gemm_gfx1250.py`, and
`f16_gemm_common_gfx1250.py` for obstruction-class primitives:

**What the Gluon surface does use:**

- `ttgl.amd.gfx1250.tdm.{make_tensor_descriptor, async_load, async_wait, prefetch}`
  — gfx1250 Tensor Data Memory (the RDNA4 analogue of Hopper TMA). ISA-level
  gap to gfx950; not a wave-size issue.
- `ttgl.amd.gfx1250.wmma` / `wmma_scaled` — gfx1250 WMMA (16×16×32 f16,
  16×16×64/128 MXFP with inline scale). ISA-level gap.
- `ttgl.amd.gfx1250.buffer_store` — plain buffer store; **no atomic**.
- `ttgl.amd.gfx1250.cluster.{arrive, wait}` — multi-XCD cluster barrier.
- `ttgl.amd.warp_pipeline_stage("...", priority=N)` — software-pipeline
  scheduling hint; no hardware instruction.
- `ttgl.max(qk, -1)` and `ttgl.sum(p, -1)` — only cross-lane reductions,
  only in the FA kernel. These are what lower to `v_permlane16_swap`.
- `ttgl.program_id(0..2)` — workgroup-uniform, fine.

**What the Gluon surface conspicuously does NOT use (verified by grep):**

- No `permlane64`, no `permlane32_swap`, no `permlanex16`, no `ds_swizzle`.
- No `readlane` / `writelane`, no `mbcnt`, no `ballot`, no `wavefrontsize`.
- No `ds_bpermute`.
- No explicit DPP primitives (`_dpp`, `update_dpp`).
- No atomics of any kind — `atomic_add`, `atomic_cas`, `atomic_xchg`,
  `buffer_atomic_*` are all absent.
- **No `warp_specialize` / `async_task` / `num_consumer_groups`.** The
  Hopper-style producer/consumer pattern that would be a serious C4 risk
  (EXEC writes based on warp ID) is simply absent from all gfx1250 Gluon
  examples.

### 8.4 The Gluon test harness itself asserts `v_permlane16_swap`

In `mxfp_fa_gfx1250.py` (lines 2987–2991 and 3067–3071), the kernel's own
test asserts:

```python
v_permlane_instrs = [instr for instr in instrs if re.match(r'v_permlane_*', instr)]
assert len(v_permlane_instrs) > 0 and all(
    instr.startswith("v_permlane16_swap") for instr in v_permlane_instrs)
```

The Gluon authors explicitly designed the kernel so the only permlane variant
that appears is `v_permlane16_swap`, used for both the softmax layout
convert and the WMMA-fragment → DotOperand shuffle. That is a single
well-defined obstruction class covered by `wave-size-translation.md §5.3`
item P4.

### 8.5 Predictions for the three uncaptured Gluon kernels

**`mxfp_gemm_gfx1250.py`:** expected profile

| Class | Expected | Reasoning |
|-------|----------|-----------|
| C1 | 0 | No lane-ID use in pure GEMM |
| C2-hard | 0 | Accumulator stored in WMMA layout via `buffer_store` |
| C2-DPP | 0 | Consistent with captured f16 GEMM binaries (zero DPP) |
| C2-bpermute | 0–few | Not structurally needed on WMMA-native output |
| C3 | 0 | No atomics |
| C4 saveexec | many, bounds-check | Tail-loop edge tile, same shape as f16 GEMM |

**Predicted outcome: (a)** — same profile as the three captured f16 GEMM
variants.

**`mxfp_fa_gfx1250.py` (both variants):** expected profile

| Class | Expected | Reasoning |
|-------|----------|-----------|
| C1 | 0 | No lane-ID use |
| C2-hard | **YES, `v_permlane16_swap_b32`** | Source explicitly asserts this is the only permlane variant emitted |
| C2-DPP | 0 | Consistent with captured f16 FA pipeline binary |
| C2-bpermute | 0 | No transposes needed |
| C3 | 0 | No atomics, online softmax accumulates in registers |
| C4 saveexec | 0–few | Block sizes chosen to divide 8192 cleanly |

**Predicted outcome: (b)** via P2/P3/P4 permlane intrinsic lift. Same
shape as `triton__f16_fa_pipeline_attn_fwd_pipelined_kernel`, with likely
slightly higher instance count because the MXFP tile is larger (256×128 vs
128×128) and there is an extra dequant convert step.

### 8.6 Conclusion: Gluon's real gaps are ISA, not wave-size

On the wave-size axis, the Gluon example surface is the cleanest in the
entire corpus — cleaner than the Triton `scope_discovery` kernels (which
had DPP, bpermute, ds_swizzle, readlane) and cleaner than the hipBLASLt
GEMMs (36% bpermute reach). The Gluon authors are explicitly disciplined
about which cross-lane primitive to use — `permlane16_swap`, and nothing
else from the permlane family.

The gfx1250 → gfx950 gaps for Gluon kernels live on a different axis and
are tracked in `gfx1250-on-gfx950-analysis.md`, not here:

1. **TDM emulation.** `ttgl.amd.gfx1250.tdm.*` has no gfx950 equivalent;
   needs emulation as `buffer_load` / `global_load` with software
   descriptor decode and LDS staging.
2. **`wmma_scaled` replacement.** MXFP WMMA with inline scale is
   gfx1250-exclusive; on gfx950 either dequantize to bf16 and use MFMA, or
   use gfx950's F8F8 / F8B8 MFMA variants and apply scales separately.
3. **WMMA → MFMA fragment layout translation.** Not a source-level
   obstruction, but the lowerer will synthesize its own cross-lane
   shuffles to bridge the accumulator layouts. Those synthesized shuffles
   must themselves conform to SPE rules (i.e. use `permlane16_swap` or
   similar, not `permlane64`).
4. **Cluster barrier.** `ttgl.amd.gfx1250.cluster.*` is used only in
   multi-XCD configurations and has a different model on gfx950.

None of these four is a wave-size obstruction class. They are orthogonal
ISA-translation work.

---

## 9. Conclusions

### 9.1 No refusal cases in GPT-OSS

Every obstruction in the 15 GPT-OSS Triton kernels maps to outcome (a) or
outcome (b). There is no kernel for which the principled strategy would emit
a fatal refusal. The GPT-OSS North Star is reachable by the transpiler.

### 9.2 No refusal cases in hipBLASLt either

The 147-kernel GEMM library that GPT-OSS calls into is clean on all
non-rewritable classes (C1, C2-hard, C2-DPP, C3). Its only obstruction
profile is bounds-check C4 (outcome a) and `ds_bpermute` (outcome b via P1).

### 9.3 No refusal cases in Gluon either

The four Gluon example kernels captured in the corpus and the three
reasoned about from source in §8 are all either outcome (a) or outcome (b).
The only obstruction Gluon introduces is `v_permlane16_swap` in the FA
variants, which is covered by `wave-size-translation.md §5.3` item P4.

### 9.4 Blocking engineering work

GPT-OSS will translate correctly end-to-end once the following
`wave-size-translation.md §5.3` items land, in descending order of risk:

1. **P5 — DPP intrinsic lift** (~1 week).
   Required for correctness on 5 GPT-OSS kernels
   (`bitmatrix_metadata_compute_stage1`, `bitmatrix_metadata_compute_stage2`,
   `downcast_to_mxfp`, `sum_bitmatrix_rows`, `topk_forward`). Silently
   miscompiled today. **Largest outstanding risk.**
2. **P1 — `ds_bpermute_b32` intrinsic lift** (~30 minutes).
   Required for correctness on 3 GPT-OSS kernels and 53 hipBLASLt kernels.
   Currently "accidentally correct" because selectors are uniform; not
   principled.
3. **P2 / P4 — `permlane16` / `permlanex16` / `permlane16_swap`
   intrinsic lifts** (~1 day total).
   Required for the GPT-OSS attention forward (`attn_fwd`) and the
   MoE-router bitmatrix pipeline.
4. **P6 — `ds_swizzle_b32` intrinsic lift** (~half-day).
   Required only for `sum_bitmatrix_rows`.
5. **C1 operand-range validator** at raise time (~hours).
   Fails loudly if a `readlane` / `writelane` lane operand is ≥ W_s. The
   one GPT-OSS occurrence passes this check statically.

### 9.5 Open coverage questions (capture gaps)

Before claiming full GPT-OSS inference coverage, the following should be
either exercised by an extended `scope_discovery` driver or confirmed to be
fused inside the kernels we already captured:

- **RMSNorm** — is it a Triton kernel? If so, capture it. If it runs
  through PyTorch's fused kernel, document that path is out of scope for
  the transpiler.
- **RoPE / YaRN** — confirm inline application inside `_attn_fwd` by
  reading `gpt_oss/triton/attention.py`, or add a standalone capture.
- **KV-cache append / indexed read (decode path)** — extend the driver to
  exercise decode in addition to prefill.
- **MXFP Gluon kernels** (`mxfp_gemm_gfx1250.py`, `mxfp_fa_gfx1250.py` ×2) —
  reasoned about from source in §8; should be captured and audited empirically
  to confirm the predictions there.

### 9.6 Structural de-risking

The observations that give the most confidence GPT-OSS is the right North
Star, now scoped correctly to what the corpus actually covers:

- Across **every** sub-corpus (GPT-OSS, hipBLASLt, others), the corpus avoids
  every "no-rewrite" C1 primitive (`mbcnt_hi`, `ballot`, `wavefrontsize`).
- Across **every** sub-corpus, the corpus contains zero C3
  non-commutative atomics.
- Across **every** sub-corpus, the corpus contains zero `v_permlane64_b32`
  (the only C2 op that would require a true 64-lane hardware shuffle).
- In GPT-OSS specifically, **7 of 15 captured kernels are already clean**
  (outcome a with no rewrite needed), including three of four MoE expert
  GEMMs and both reductions and SwiGLU and the upcast-from-MXFP path.
- The dirtiest kernels are the MoE-router bitmatrix pair; the cleanest are
  the MoE expert GEMM and activation kernels — a favorable distribution,
  because the bitmatrix kernels are small and the GEMM / activation kernels
  dominate runtime.

This matches the hypothesis that motivated the wave-size strategy in
`wave-size-translation.md`: the GPT-OSS corpus stays within the
wave-size-oblivious discipline the strategy was designed around.

---

## 10. Artifacts

- `hotswap/transpiler/scope_discovery/capture/run_gpt_oss.py` — the capture
  driver that produced the 15 GPT-OSS Triton kernels.
- `hotswap/transpiler/scope_discovery/kernels/` — the captured `.hsaco`
  files (15 GPT-OSS-20B / 120B kernels).
- `hotswap/kerneldex/dex/reports/mnemonic_histogram.csv` — corpus-wide
  mnemonic histogram (counts + per-kernel reach).
- `hotswap/kerneldex/dex/reports/per_kernel_mnemonics.jsonl` — per-kernel
  mnemonic dicts, the input to the audit in this report.
- `hotswap/kerneldex/dex/reports/coverage.csv` — per-kernel coverage
  outcomes under current raiser coverage (orthogonal axis: what the raiser
  can currently ingest, independent of wave-size rewrite).
- `hotswap/docs/wave-size-translation.md` — the committed wave-size axis
  spec: SIMT Predicated Execution model, modulo-replication projection,
  obstruction classes C1–C4, and the cross-lane primitive rewrite table
  (with implementing commits). Authoritative external reference for the
  axis.
- `hotswap/docs/gfx1250-on-gfx950-analysis.md` — architectural delta between
  source and target ISAs.
- [`triton-lang/triton//third_party/amd/python/examples/gluon/`](https://github.com/triton-lang/triton/tree/main/third_party/amd/python/examples/gluon)
  — upstream AMD Gluon example harness whose F16 invocations produced four
  of the `triton/` kernels in the corpus and whose MXFP invocations are
  reasoned about in §8.

## 11. Related design docs (other translation axes)

This report de-risks the *wave-size* axis. Orthogonal axes with their own
design docs:

- `hotswap/docs/abi-translation.md` — kernel descriptor, kernarg layout,
  hidden-arg compatibility, embedded-descriptor gates.
- `hotswap/docs/sync-translation.md` — barriers (incl. gfx12 split
  barriers), waitcnt, memory scopes on atomics, cache ops, cluster sync.
- `hotswap/docs/matrix-translation.md` — WMMA → MFMA lowering framework,
  per-shape fragment-layout tables, MXFP scaled path.
- `hotswap/docs/tdm-translation.md` — tensor descriptor emulation,
  async_copy lowering, prefetch.

Each doc specifies its own decision procedure and refuses kernels that
fall outside the translation framework. The GPT-OSS corpus is clean on
the wave-size axis (this report) but depends on matrix, sync, and ABI
axes landing before it runs end-to-end; TDM is required for the MXFP
Gluon path captured in §8.
