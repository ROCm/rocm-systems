# Triton recipes for `compare_correctness`

This directory holds Triton-authored kernels that the
`compare_correctness` harness runs through the native / legacy / salmon
three-mode comparison, alongside the hand-written HIP recipes in
`../`.

Each `<name>.py` here becomes **one recipe** in the harness:

- `<name>.py` is AOT-compiled to two code objects,
  `kernels/build/<name>.gfx942.co` and `kernels/build/<name>.gfx1250.co`.
- A JSON "sidecar" `kernels/build/<name>.sidecar.json` records the
  signature, constexprs, shape sweep, launch grid, inputs/outputs, the
  comparator tolerance, and — most importantly — the per-arch kernarg
  layout extracted from the `.co`'s AMDGPU metadata.
- At run time the harness scans `kernels/build/*.sidecar.json`, turns
  each into a `Recipe`, and the shared `tritonDispatch` packs kernargs
  at the exact offsets the code object expects.  **No C++ edits are
  needed per Triton kernel.**

## Why native-as-gold?

HIP recipes in this tool use a hand-written **CPU reference** as
ground truth, which makes the native GPU run part of the measurement
(a mismatch on native is a finding of its own).  That design doesn't
scale to Triton kernels: writing a correct CPU reference for flash
attention, mixed-precision GEMM, or fused softmax is its own small
research project, and we'd end up shipping bugs in the reference.

Instead, Triton recipes declare `goldSource = NativeExecution` and use
the output of the **native gfx942 run** as the gold.  The transpiled
legacy / salmon runs (loading the gfx1250 code object through the ROCR
hotswap hook) are judged against that.  This trades the CPU-grounded
self-check on the native column for the ability to throw arbitrary
Triton kernels at the harness; the trust boundary moves to "hipcc +
Triton on gfx942 compute the right answer".  If the native gfx942 run
fails to materialise, the harness marks the whole `(recipe, shape)`
row as `gold-missing` and skips the transpiled modes for that row —
nothing meaningful can be said about them without a gold.

## Prerequisites

Compilation is done with the in-tree Triton at `$HOME/rocm-systems/triton`.
The Makefile points to:

```
TRITON_VENV       = $(HOME)/rocm-systems/triton/.venv
TRITON_PYTHONPATH = $(HOME)/rocm-systems/triton/python:$(HOME)/rocm-systems/triton/third_party/amd/python
```

Override on the `make` command line if your Triton tree lives
elsewhere.  The venv needs Triton itself and `PyYAML` (shipped with
the repo's Triton venv; used by `aot_compile.py` to parse the
`llvm-readelf --notes` output).  An `llvm-readelf` binary is located
via `$LLVM_READELF`, `/opt/rocm*/lib/llvm/bin/llvm-readelf`, or
`$PATH`, in that order.

## Adding a recipe

1. Drop `<name>.py` here.  The file stem is the recipe name and must
   match the `name:` field of its recipe entry.
2. In `<name>.py`:
   ```python
   import triton
   import triton.language as tl

   @triton.jit
   def my_kernel(...):
       ...

   RECIPES = [{
       "name": "<name>",                    # must match file stem
       "kernel_fn": my_kernel,              # the @triton.jit object
       "kernel_symbol": "my_kernel",        # hipModuleGetFunction symbol
       "signature": { ... },                # Triton type strings
       "constexprs": { ... },
       "num_warps": 4,
       "shape_dim": "N",                    # single scalar shape param
       "default_shapes": [1024, 4096, ...], # values the harness sweeps
       "grid": {"x": "...", "y": "1", "z": "1"},
       "inputs":  [{"name": ..., "dtype": ..., "elems": ...}, ...],
       "outputs": [{"name": ..., "dtype": ..., "elems": ...}],
       "comparator": {"kind": "abs", "tol": 1e-2},
   }]
   ```
3. Add `<name>` to `$(TRITON_KERNELS)` in the Makefile.
4. `make ROCR_BUILD=$HOME/rocm-systems/projects/rocr-runtime/build`.

See `vecadd_f16.py` for a complete working example.

## Recipe schema

| field              | type                          | notes                                                                   |
|--------------------|-------------------------------|-------------------------------------------------------------------------|
| `name`              | `str`                         | unique; must equal the file stem                                        |
| `kernel_fn`         | Triton `@jit` object          | the actual kernel to compile                                            |
| `kernel_symbol`     | `str`                         | symbol looked up via `hipModuleGetFunction`                             |
| `signature`         | `OrderedDict[str, str]`       | Triton signature (`"*fp16"`, `"i32"`, …) in kernel argument order       |
| `constexprs`        | `dict[str, int]`              | Triton `tl.constexpr` values (e.g. `{"BLOCK_SIZE": 1024}`).  Every key MUST appear in the kernel signature — Triton's `ASTSource` rejects dangling constexpr names. |
| `harness_constants` | `dict[str, int]`              | optional integer constants visible only to the harness's expression evaluator (`elems`, `grid`, `scalar_args`).  Use this for sizing values that are *not* kernel parameters (e.g. an `M` row count when the kernel reads its row id from `tl.program_id(0)`), or for runtime kernarg slots that must keep their by-value entry instead of being baked in by `constexprs`.  Names must not collide with `shape_dim` or `constexprs`. |
| `num_warps`         | `int`                         | threads/block = `num_warps * warp_size`                                 |
| `shape_dim`         | `str`                         | scalar sig arg the harness sweeps (Phase 1)                             |
| `default_shapes`    | `list[int]`                   | values of `shape_dim` to sweep                                          |
| `grid`              | `{"x", "y", "z": str}`        | expressions in `shape_dim` + `constexprs` + `harness_constants` + `ceil_div(a,b)` |
| `scalar_args`       | `dict[str, str\|number]`      | optional per-recipe overrides for non-pointer sig args.  Integer-typed sig args take a *string expression* evaluated against the harness scope; floating-point-typed sig args take a *numeric literal* packed at the right width (`fp16`/`bf16`/`fp32`/`fp64`).  Sig args not listed here fall back to auto-resolve from `shape_dim` + `constexprs` + `harness_constants` by name. |
| `inputs`            | `list[{name, dtype, elems, range_lo?, range_hi?}]` | buffers filled with deterministic RNG, bound to pointer sig args   |
| `outputs`           | `list[{name, dtype, elems, comparator?}]` | buffers read back from the device and compared, optionally with their own comparator |
| `comparator`        | `{"kind": "abs"\|"rel"\|"rel-rms", "tol": float}` | recipe-level default comparator                            |

`elems`, `grid`, and the integer entries of `scalar_args` are
**expressions** evaluated at run time against
`shape_dim + constexprs + harness_constants`.  Supported: integer
literals, identifiers (`N`, `BLOCK_SIZE`, `M`, …), infix `+ - * /`,
parens, and a single built-in function `ceil_div(a, b)`.  Anything
else triggers a loud error in the harness — the evaluator
intentionally refuses to silently guess.

### Supported dtypes

| dtype  | bytes | input filler                              | comparator decoding                          |
|--------|-------|-------------------------------------------|----------------------------------------------|
| `fp16` | 2     | uniform `[range_lo, range_hi)`            | full IEEE 754 binary16 → fp64 (no FTZ)       |
| `bf16` | 2     | uniform `[range_lo, range_hi)`            | bf16 → fp32 → fp64 (top-16-bits reinflate)   |
| `fp32` | 4     | uniform `[range_lo, range_hi)`            | fp32 → fp64                                  |
| `fp64` | 8     | uniform `[range_lo, range_hi)`            | native fp64                                  |
| `i32`  | 4     | uniform u32 reinterpreted via `memcpy`    | exact integer compare                        |
| `i64`  | 8     | uniform u64 reinterpreted via `memcpy`    | exact integer compare                        |

`range_lo` / `range_hi` default to `[-1.0, 1.0)` for floats and are
ignored for integer dtypes (the integer fillers always span the full
unsigned range and reinterpret-cast bitwise — useful for index
buffers).  Pin the range tighter when the kernel restricts its inputs
(asin needs `[-1, 1]`, log needs `(0, ∞)`, etc.); otherwise you'll be
chasing NaN-vs-NaN compares instead of real findings.

### Comparator kinds

| kind      | judgement                                                                  |
|-----------|----------------------------------------------------------------------------|
| `abs`     | `\|gold − actual\| ≤ tol` per element                                      |
| `rel`     | `\|gold − actual\| ≤ tol · max(1, \|gold\|)` per element                   |
| `rel-rms` | `sqrt(mean((gold − actual)²)) ≤ tol · sqrt(mean(gold²))` over the buffer   |

`rel-rms` is intended for reduction kernels (softmax, layer-norm,
attention) where wave-size differences (wave32 vs wave64) shuffle the
order of the partial sums — pointwise relative error then exceeds any
sane elementwise tolerance, but the bulk-RMS norm stays small.  It's a
buffer-level verdict so a single bad output element won't drag the
whole grid down, but it will also miss localised correctness
regressions; use the elementwise comparators (`abs` / `rel`) wherever
the kernel is supposed to be elementwise-stable.

NaN / inf handling is uniform across the elementwise comparators:
NaN-vs-NaN is a match (both paths agree there's no defined value
here), `+inf`-vs-`+inf` and `−inf`-vs-`−inf` are matches (kernels
routinely saturate to ±inf), and any cross-class comparison
(NaN-vs-finite, +inf-vs-−inf, finite-vs-inf) is a hard mismatch.

## Sidecar schema (produced by `aot_compile.py`)

```
{
  "name":              "<recipe name>",
  "kernel_symbol":     "<ELF symbol>",
  "num_warps":         <int>,
  "signature":         [{"name":..., "type":...}, ...],    # order matters
  "constexprs":        { ... },
  "harness_constants": { ... },                            # optional
  "shape":             {"dim": "<name>", "values": [<int>, ...]},
  "grid":              {"x":"<expr>", "y":"<expr>", "z":"<expr>"},
  "scalar_args":       { "<sig_arg>": "<expr>"|<number>, ... },  # optional
  "inputs":            [{"name":..., "dtype":..., "elems":"<expr>",
                         "range_lo": <float>, "range_hi": <float>}, ...],
  "outputs":           [{"name":..., "dtype":..., "elems":"<expr>",
                         "comparator": {...}?  /* per-output override */ }, ...],
  "comparator":        {"kind": "abs"|"rel"|"rel-rms", "tol": <float>},
  "metadata": {
    "gfx942":  { "kernarg_segment_size":       <int>,
                 "group_segment_fixed_size":   <int>,   # static LDS (always 0 for Triton)
                 "private_segment_fixed_size": <int>,   # scratch; harness refuses launch if != 0
                 "max_flat_workgroup_size":    <int>,
                 "shared_mem_bytes":           <int>,   # dynamic LDS (compiled.metadata.shared)
                 "args": [{"offset":..., "size":..., "value_kind":...}, ...] },
    "gfx1250": { ... }
  }
}
```

`harness_constants` is round-tripped through the sidecar (the C++
harness needs it to evaluate `elems` / `grid` / `scalar_args`) but
deliberately not forwarded to Triton's compiler — these are
harness-side identifiers, not Triton constexprs.  Likewise
`scalar_args` is consumed by the C++ dispatch (which packs each
referenced sig arg into the kernarg buffer at the right offset and
width); Triton sees only the kernel signature itself.

`metadata.<arch>.shared_mem_bytes` is sourced from
`compiled.metadata.shared` at AOT time — the per-block *dynamic* LDS
(shared memory) Triton's AMD backend reserves for reductions, softmax
scratch, and similar cross-wave accumulators.  It is **not** the same
thing as `group_segment_fixed_size` (static LDS baked into the kernel
descriptor, always 0 for Triton), and it is **not** discoverable from
the HSACO alone; the dispatch must pass it verbatim as
`hipModuleLaunchKernel`'s `dynamicSharedMemBytes` argument or any
reduction-bearing kernel silently returns zero output (the reduction
path stores into a non-existent LDS allocation).  Elementwise kernels
(`corpus_add_fp32`, `corpus_asin_fp32`) naturally have
`shared_mem_bytes = 0`; reduction kernels (`corpus_layernorm_fp32`,
`corpus_softmax_fp32`) have `shared_mem_bytes > 0` and implicitly
regression-test the plumbing — if dispatch ever reverts to passing 0
their native lane goes from `gold` to `WRONG`/`CRASH` on the first
run.

The `metadata.<arch>.args` list is the kernarg layout for that arch,
copied verbatim from the code object's `.note.amdgpu_metadata`.  The
harness iterates the `signature` and the `args` list in lockstep —
pointer sig args must line up with `global_buffer` slots and scalar
sig args with `by_value` slots, or the dispatch dies with a clear
producer/consumer-drift error.  Any trailing metadata entries beyond
the last sig arg (Triton's implicit `global_scratch_base` etc.) are
left zero-initialised in the kernarg buffer; Triton expects that when
no scratch is required.

## Limitations (Phase 1)

- **One scalar shape dim per recipe.**  The harness sweeps a single
  named parameter (for vecadd, `N`).  Multi-dim shapes like matmul's
  `(M, N, K)` need a sidecar-schema extension (probably
  `"shape": {"dims": [...], "combinations": [...]}`) and a per-recipe
  shape-label column in the report.  Workaround for fixed secondary
  dims: declare them in `harness_constants` (see
  `corpus_layernorm_fp32` using `M = 16`).
- **No CLI filter for Triton shapes.**  `--shape=N=1024` isn't wired
  up yet.  `--recipe=<name>` narrows to a single recipe as usual, and
  the recipe's full shape sweep runs.
- **One recipe per `.py` file.**  The AOT script enforces this
  because the Makefile's `%` pattern is the file stem and producing N
  recipes from one file would conflict with the grouped target rule.
  Split files if you want multiple recipes.
- **Scalar arg coverage.**  Pointer (`*<dtype>`), integer (`i32` /
  `i64`), and floating-point (`fp16` / `bf16` / `fp32` / `fp64`)
  scalar sig args are all supported.  Integer scalars take an
  expression in `scalar_args` (or auto-resolve from the harness scope
  by name); float scalars take a numeric literal in `scalar_args`.
  Vector / struct sig args are not packed (Triton itself doesn't
  expose them at this layer).

### Triton compile options (`num_warps` / `num_stages`)

`num_warps` is a required recipe field; `num_stages` is optional and
only the recipes that need to pin a specific pipelining depth
(today: the `matmul_fp16*` recipes) should set it explicitly.
Both are plumbed to Triton via `triton_compile(src, target=target,
options={...})` — **not** via `ASTSource.attrs`, which silently
discards unknown keys and caused every recipe's declared
`num_warps` to fall through to Triton's default (4) regardless of
what the recipe asked for.  If a future schema extension lands a
third compile-option (e.g. `waves_per_eu`), follow the same pattern
in `aot_compile.py`'s `compile_for_target`.

The matmul recipes are the case study for why this matters: at
`BM = BN = BK = 32` / `num_warps = 1`, `num_stages = 1` emits
`v_permlane16_swap_b32 × 8` + `v_wmma × 4`, while `num_stages = 2`
(Triton's default) emits `0 × permlane16_swap` + `8 × v_wmma` —
same kernel source, different cross-lane surface, because
`num_stages ≥ 2` stages each K-iteration through LDS and the
WMMA-fragment shuffle rides on the LDS writes rather than on a
cross-lane primitive.

## Canaries: one recipe per wave-size obstruction class

`canary_*.py` recipes are hand-authored minimal kernels, one per
wave-size obstruction class from `hotswap/docs/gpt-oss-derisking.md §1`.
Each one is ≤ 40 lines of Triton, is authored so the gfx1250 build
**deliberately emits exactly one class of cross-lane primitive**, and
runs in milliseconds.  The point is triage speed: when salmon regresses,
the question is "which C-class broke?" — a canary answers that in
seconds, versus hours of teasing it out of a full GPT-OSS kernel that
touches several classes at once.

Committed canaries (confirmed-emitted instruction counts from
`llvm-objdump --mcpu=gfx1250` against the built `.gfx1250.co`):

| recipe                              | C-class       | instruction forced on gfx1250 | `§5.3` item | covers |
|-------------------------------------|---------------|-------------------------------|-------------|--------|
| `canary_bpermute_scan_fp32`         | C2-bpermute   | `ds_bpermute_b32` × 20        | P1 (landed at `d9bfd99626`) | `_masked_compaction`, `_bitmatrix_metadata_compute_{stage1,stage2}`, 53 of 147 hipBLASLt GEMMs — WRONG under salmon, but NOT from a missing P1 lift; see finding #1 below |
| `canary_dpp_reduce_fp32`            | C2-DPP        | `v_mov_b32_dpp` × 4           | P5          | all five GPT-OSS DPP-using kernels emit `v_mov_b32_dpp` as their dominant DPP opcode |
| `canary_dpp_compound_add_fp32`      | C2-DPP        | `v_add_f32_dpp` × 4 + `v_permlanex16_b32` × 1 | P5   | `_bitmatrix_metadata_compute_stage1`'s `v_add_nc_u32_dpp` pattern (the second of the two DPP handler paths) |
| `canary_permlanex16_rowmax_fp32`    | — (see note)  | `v_permlanex16_b32` × 1 + `v_dual_max_num_f32` × 2 | — | see attribution update in verdict table: the failure is `v_dual_max_num_f32`, not `permlanex16` |

Current salmon verdicts (gfx942 host, gfx1250 build through the
salmon IR raiser, four-shape sweep each — snapshot, rerun from the
Makefile directory for a fresh picture):

| recipe                              | native   | legacy                   | salmon                          |
|-------------------------------------|----------|--------------------------|---------------------------------|
| `canary_bpermute_scan_fp32`         | 4 / 4    | 4 / 4 crash (MCContext UNREACHABLE, same signature as every `triton_corpus_runner` legacy failure) | **4 / 4 refused** (loud `cross-wave-predicate-chain` + `WorkitemIdPredicateChain (Class 5)` diagnostic from the narrow-O1 classifier in `transpiler/c5_predicate_chain_classifier.{hpp,cpp}` — landed 2026-04-21 per `hotswap/docs/modrep-predicate-chain.md §5 O1`). Pre-landing was **4 / 4 WRONG** (max\|err\| 4.2 → 21.9, growing roughly as `sqrt(N/BLOCK_SIZE)`). The classifier catches the Kogge-Stone scan-stage guards (`icmp ult i32 K, tid` with K ∈ {1, 3, 7, 15}, all ≤ W_s-1); see finding #1 below for the full attribution chain. |
| `canary_dpp_reduce_fp32`            | 4 / 4    | 4 / 4 crash (same MCContext UNREACHABLE)                    | **4 / 4 match** |
| `canary_dpp_compound_add_fp32`      | 4 / 4    | 4 / 4 crash (same MCContext UNREACHABLE)                    | **4 / 4 match** |
| `canary_permlanex16_rowmax_fp32`    | 4 / 4    | 4 / 4 crash (same MCContext UNREACHABLE)                    | **4 / 4 match** (was CRASH until `handle_vopd.cpp` gained a `ttmp<N>` source branch; see `lit_tests/vopd_extra_subops` for the regression guard) |

Three findings the canary set forced into the open:

1. **`ds_bpermute_b32` is NOT silently miscompiled — P1 (the
   intrinsic lift) is already landed.**  An earlier revision of this
   finding attributed the `canary_bpermute_scan_fp32` miscompile to
   a missing P1 intrinsic lift per issue #13.  That attribution is
   refuted:

   - `wave-size-translation.md §5.3` (line 777) lists
     `ds_bpermute_b32 (C2)` in the **Landed — emits inline**
     category; §9 line 882 cites the implementing commit
     `d9bfd99626` (newer than issue #13's tree `2873d140b0`).
   - `handle_ds.cpp:608` contains the handler: reads both operands
     from the MC inst, emits `llvm.amdgcn.ds.bpermute`, writes the
     gather result back to the destination VGPR.
   - Running `raise_cli` against the original issue #13 repro
     (`lane_swap.hip`) now **loudly refuses** with
     `FAIL lane_swap -> v_cmpx_gt_i32 [cross-wave-lane-predicated-exec]`
     — G1's classifier refuses the kernel at raise time because of
     its `if (tid >= n) return;` bounds check (a C4 obstruction),
     not because of bpermute.  Graduated from "silent miscompile"
     to "principled refusal" per the §8 fail-loudly contract.
   - `canary_bpermute_scan_fp32` raises cleanly and emits 20×
     `ds_bpermute_b32` (one per per-lane-element × per-scan-stage)
     on the gfx942 output HSACO.  Yet the numerical output differs
     from the Triton-native-compiled gfx942 gold with `max|err|`
     in the range 4.2 → 21.9, climbing with N.

   The residual miscompile is therefore not in the P1 lift but
   somewhere else — specifically in the **interaction between
   Triton's kernel-emitted predicate chains and modulo-replication
   on cross-widened wave64**.  Triton's gfx1250 cumsum lowering
   uses scan predicates computed on `workitem.id.x()` (`tid`)
   rather than the source-wave `mbcnt` lane id; under cross-
   widening the target's `tid` ranges `[0, W_t)` rather than
   `[0, W_s)`, so replica lanes evaluate the scan-stage guards
   on a larger set than the source kernel was written for.  Full
   diagnosis, evidence table, and four fix options in
   [`hotswap/docs/modrep-predicate-chain.md`](../../../../../docs/modrep-predicate-chain.md)
   — the class also covers the `rmsnorm_fp32` / `swiglu_fp32` /
   `corpus_layernorm_fp32` sentinel-leak failures (finding #4
   below).

2. **`§7.3`'s "P5 is the largest risk" is narrower than stated.**
   Both `v_mov_b32_dpp` and the compound `v_add_f32_dpp` path pass
   under salmon.  The pre-authoring probe also established that gfx1250
   emits **no** wave-size-dependent DPP modifiers at all
   (`bcast15` / `bcast31` / `wave_shl` / `wave_shr` / `row_share`
   were gfx9/10 patterns dropped in gfx11+) — every DPP form Triton
   emits on gfx1250 is wave-invariant by ISA construction.  So P5's
   remaining risk surface is whichever DPP-using GPT-OSS kernels
   emit patterns *not* covered by the two DPP canaries here; in
   practice that likely means the raiser's existing DPP handling
   covers the full gfx1250 DPP ISA, and the `§5.3` P5 item is a
   completeness audit, not an open miscompile.  Re-running the
   bit-exact DPP-tree check on each of the 5 GPT-OSS DPP-users
   individually (via a status-like sweep, as their shims land) is
   the cheapest way to verify.

3. **`permlanex16` is NOT the `permlanex16_rowmax` crash cause.**
   Instruction-level diff across the four canaries: the
   `compound_add` canary passes with `v_permlanex16 × 1`, while
   `permlanex16_rowmax` crashes with `v_permlanex16 × 1` *and*
   `v_dual_max_num_f32 × 2`.  The only passing canary with any
   `v_dual_*` has either `v_dual_add_nc_u32` or `v_dual_lshlrev_b32`
   — never `v_dual_max_num_f32`.  The crash therefore attributes to
   **the `v_dual_max_num_f32` opcode decode path** (a gfx11+
   dual-issue variant of IEEE-754 max, not a wave-size-translation
   concern), not to the `permlane*` intrinsic lift the canary was
   named for.  The `permlanex16_rowmax` file is kept but should be
   treated as a `v_dual_max_num_f32` canary until split; the P2 / P4
   rewrite class is already exercised (and *passing*) via
   `canary_dpp_compound_add_fp32`.

4. **The gfx11+ IEEE-754-2019 "num" opcode family is NOT the gap**
   — refuting a prior hypothesis that came out of disassembly
   pattern matching.  Investigation established:

   - `v_max_num_f32` / `v_min_num_f32` / `v_med3_num_f32` /
     `v_minmax_num_f32` are gfx12 **assembly mnemonic aliases** for
     pre-existing LLVM pseudos (`V_MAX_F32`, `V_MIN_F32`,
     `V_MED3_F32`, `V_MINMAX_F32`, all in
     `VOP3Instructions.td` under `VOP3_Realtriple_with_name_gfx12`).
     The raiser's `opcode_map.cpp` kCanonTable already contains
     these pseudos and `handle_valu.cpp` has the handlers.  `raise_cli`
     successfully raises 325 / 325 swiglu instructions and 389 / 389
     rmsnorm instructions with no "unsupported opcode" output.
   - The apparent `v_min_num_f64 × 7` in rmsnorm's disassembly (and
     `× 2` in rope's) is a **disassembler false positive**: offset
     0x1C10 in rmsnorm's gfx1250.co contains raw bytes
     `0xcc284020` — opcode 0xCC28 is `V_PK_MUL_F32` in LLVM's
     `VOP3P_Real_gfx12<0x28>` table, a 64-bit VOP3P instruction.
     `llvm-objdump --mcpu=gfx1250` splits the 8-byte instruction
     into two 4-byte decodes: one undecoded (`.long 0xcc284020`)
     and one incidental `v_min_num_f64_e32` false match on the
     instruction's second dword.  The raiser correctly decodes the
     full 64-bit VOP3P; the IR has the expected 40× `fmul <2 x float>`
     / 8× `call @llvm.fma.v2f32` pattern, zero fp64 ops.
   - Swiglu's raised IR shows clean `llvm.minnum.f32` / `llvm.maxnum.f32`
     for the `v_med3_num_f32` clamps (specifically
     `minnum(7.0, maxnum(-7.0, x))` for `tl.clamp(x, -7.0, 7.0)`),
     so the "num" handler is semantically correct.

   The silent miscompiles therefore live **in the IR semantics, not
   in opcode coverage**.  The four salmon-WRONG Triton recipes
   (`canary_bpermute_scan_fp32`, `rmsnorm_fp32`, `swiglu_fp32`,
   `corpus_layernorm_fp32`) have been narrowed to a single shared
   class:

   > **Kernel-emitted predicate chains that read
   > `llvm.amdgcn.workitem.id.x()` without an AND-mask by
   > `W_s − 1`, under cross-widening with `W_t > W_s`.**  The
   > replica lanes evaluate the predicate on `[0, W_t)` while the
   > source compile baked `[0, W_s)` into the kernel, so replicas
   > take the wrong branch / write to wrong slots.  Diagnostic
   > signatures: `canary_bpermute_scan_fp32` shows partial-scan-
   > drift; `rmsnorm_fp32` / `swiglu_fp32` / `corpus_layernorm_fp32`
   > show `actual = −2.87352e-16` (= fp32 `0xA5A5A5A5`, the raiser's
   > VGPR-init sentinel) in their wrong output slots — entire
   > destination slots go unwritten under the replicas' aliased
   > addressing.

   The class is orthogonal to the existing C1–C4 obstruction axes
   (see [`hotswap/docs/modrep-predicate-chain.md`](../../../../../docs/modrep-predicate-chain.md) §3
   for the per-class falsification), passes G1, and needed a new
   design surface.

   **Status update (2026-04-21).** `modrep-predicate-chain.md §5 O1`
   has landed as a narrow classifier — the first principled outcome
   from that doc's four-option table. The "four salmon-WRONG
   Triton recipes" framing above no longer matches reality:

   - `canary_bpermute_scan_fp32`: silent-WRONG → **loud-refused**
     under the new `CrossWavePredicateChain` / C5 diagnostic. Its
     Kogge-Stone scan-stage guards match the narrow-O1 signature
     (compile-time K ∈ {1, 3, 7, 15}, all ≤ W_s-1). One
     silent-miscompile → principled refusal.
   - `rmsnorm_fp32`: now **4 / 4 match** (orthogonal commit — most
     likely `v_div_scale_f32` / `v_rsq_f32` handler tightening —
     fixed it between the original evidence collection and the
     narrow-O1 landing; unrelated to the C5 classifier).
   - `swiglu_fp32`, `corpus_layernorm_fp32`: stay **4 / 4 WRONG**.
     Their bug class is NOT predicate-chain (their icmps compare
     `tid` against a dynamic kernarg, not a compile-time constant;
     structurally identical to the passing `vecadd_f16` shape).
     Tracked in `modrep-predicate-chain.md §4.3 / §6.4` as
     orthogonal classes pending a single-element mechanism trace.

   The post-landing compare_correctness sweep confirms the narrow-
   O1 classifier refuses exactly `canary_bpermute_scan_fp32`,
   leaves every currently-passing baseline green
   (`canary_dpp_compound_add_fp32`, `rope_fp32`, `vecadd_f16`,
   `corpus_add_fp32`, `corpus_asin_fp32`, `canary_dpp_reduce_fp32`,
   `canary_permlanex16_rowmax_fp32`), and does not affect the
   recipes whose miscompile is outside its scope. Net: +1
   principled refusal, 0 regressions.

   §5 O2 (mask rewrite) from the design doc is explicitly deferred
   — IR inspection (`modrep-predicate-chain.md §5 O1 narrowing`)
   showed its shape is not semantically correct for the remaining
   failing recipes. A future design iteration resurrects it only
   if a single-element mechanism trace produces a principled
   rewrite shape.

   Other non-matching recipes that are NOT in this class:

   | site                                | verdict | attribution |
   |-------------------------------------|---------|-------------|
   | `canary_permlanex16_rowmax_fp32`    | ~~CRASH~~ match | **fixed** — `handle_vopd.cpp` now recognises `ttmp<N>` and `vcc_lo` sources in the VOPD sub-op parser (previously `v_dual_mov_b32 v0, ttmp9 :: v_dual_mov_b32 v1, s0` failed decomposition because `ttmp9` didn't match any parse branch).  Regression guard in `lit_tests/vopd_extra_subops` pins both new source shapes. |
   | `corpus_softmax_fp32`               | CRASH   | originally framed as "same MODREP-predicate-chain class as the four WRONG recipes", but *already* refused loudly via the §5.6.3 writelane/readlane safety net (graduated to default-on in `transpiler: graduate writelane/readlane cross-lane rewrite to default-on`). The narrow-O1 classifier that landed does NOT additionally match softmax (its kernel-level icmps compare `tid` against dynamic kernargs, not compile-time K ≤ W_s-1); the refusal attribution therefore stays on the existing writelane-safety-net diagnostic, not the new C5 path. |
   | `matmul_fp16*`                      | WRONG   | WMMA accumulator lowering — issue #3; distinct bug from the MODREP-predicate-chain class. |

5. **WMMA lowering is broken — distinct from the "num" opcode gap.**
   Both `matmul_fp16` (`BM=BN=BK=32`) and `matmul_fp16_16x16`
   (`BM=BN=BK=16`) produce the same WRONG output signature under
   salmon: the accumulator is pinned to a per-shape constant (e.g.
   `actual = −0.022049` at M=128+ in both recipes) instead of
   accumulating across K iterations.  The two recipes emit *different*
   C2-hard cross-lane primitives (permlane16_swap vs ds_swizzle)
   for the WMMA-fragment shuffle, yet exhibit identical failure
   shape, which points at the WMMA → MFMA translation in
   `wmma_lowering.cpp` itself rather than at either cross-lane
   primitive.  This is **Issue #3 (WMMA translation completeness)**
   from the project tracker, now with a Triton-side repro pair that
   isolates WMMA from the shuffle operands.  Fix scope: the MFMA
   destination layout when reading back from the accumulator
   fragment, most likely the specific
   `v_wmma_f32_16x16x32_f16 → v_mfma_f32_16x16x16_f16`
   sub-sequence the current lowering produces.

Each canary's docstring explains the authoring intent: what instruction
the gfx1250 build is supposed to emit, why the specific tile / row /
group size was chosen to trigger it, and what a regression looks like
in the output.  The gfx942 build is the gold — on that arch the same
kernel typically lowers through a DPP-only tree or a straight global
load / store, so the native gold is "what the correct answer is" and
salmon's job is to re-lower the gfx1250 cross-lane path into whatever
gfx942 needs.

Adding a canary is the right move whenever the de-risking audit
surfaces a new obstruction class (C1 absolute lane-ID leak, C3
non-commutative atomic, C4 lane-position EXEC write) that GPT-OSS
actually reaches and salmon does not yet handle.  Keep them minimal:
one cross-lane primitive per recipe, no comparator slack that a silent
miscompile could hide under, and a docstring that names the `§5.3`
item the canary pins down.

Deliberately **not** a canary today:

- **C2 `permlane16_swap`** — now covered by `matmul_fp16` (see
  GPT-OSS primitives section below), which emits
  `v_permlane16_swap_b32 × 8` at `BM = BN = BK = 32` / num_warps=1 /
  num_stages=1.  Still WRONG under salmon, but the attribution is
  now cleanly separated: the matmul WMMA accumulator is what's
  broken (finding #5 below), not the `permlane*` lift path — which
  the passing `canary_dpp_compound_add_fp32` (emits
  `v_permlanex16_b32 × 1`) already demonstrates is handled.
- **C2 `permlane64` / `permlane32_swap`** — zero uses across the
  entire 170-kernel corpus (`gpt-oss-derisking.md §§4, 7.2`).  Not a
  canary because there is no class-reach to canary against.
- **C2 `ds_swizzle_b32`** — now covered by `matmul_fp16_16x16`,
  which emits `ds_swizzle_b32 × 2` at `BM = BN = BK = 16` /
  num_warps=1 / num_stages=1.  Same salmon verdict (WRONG) and
  same signature as `matmul_fp16`, attribution-wise pointing at the
  WMMA itself rather than the shuffle primitive (finding #5 below).
- **gfx11+ dual-issue VOP3 decode gaps** (e.g.
  `v_dual_max_num_f32`, surfaced by `canary_permlanex16_rowmax_fp32`
  above) — **not** one of the `§1` obstruction classes (it's an ISA
  coverage gap, not a wave-size translation concern), but has joined
  the canary set de facto because one recipe happens to emit it.
  The raiser's dual-issue VOP3 decode surface is a principled
  follow-up once the ISA coverage issue's sweep gets around to it;
  no new canary needed beyond what's already firing the failure.
- **C1 absolute lane-ID leak** — appears once (`v_readlane_b32 ×1`)
  in `_bitmatrix_metadata_compute_stage1`, with a small-constant
  operand that the `OutOfRangeLaneOperand` check in §7's decision
  procedure passes statically.  A canary for this class is cheap in
  principle (a Triton kernel that embeds an explicit lane-ID read),
  but the idiom is uncommon enough in Triton that a miscompile here
  would not surface before a dedicated test.
- **C3 non-commutative atomic** — zero uses across the entire 170-
  kernel corpus (`gpt-oss-derisking.md §7.5`).  Not a canary because
  there is no class-reach to canary against.
- **C4 lane-position EXEC write** — partially covered today by
  `vecadd_f16`'s ragged-shape mask, which exercises
  `s_and_saveexec_b32` on a bounds-check expression (outcome a, the
  same idiom hipBLASLt's 81 / 147 saveexec uses hit).  The class that
  would need a new canary — raw `v_cmpx` against an absolute-lane-ID
  constant — does not appear in GPT-OSS or hipBLASLt
  (`gpt-oss-derisking.md §7.6`).

## GPT-OSS primitives: hand-authored minimal kernels per Tier-1 op

Recipes in this section are hand-authored Triton kernels that mirror
the `triton_kernels` surface GPT-OSS actually calls into — one
recipe per Tier-1 op from issue #6 (`layernorm`, `rmsnorm`,
`attention`, `MoE router`, `scaled matmul`) that we can express with
the current harness schema.  They are not captures of GPT-OSS's
compiled kernels; they are the smallest correct Triton program that
produces the same IR shape as the production kernel, with GPT-OSS's
actual numerical parameters where those matter (e.g. SwiGLU's
`alpha = 1.702`, `limit = 7.0`).  The idea is to close the coverage
gap identified by `hotswap/docs/gpt-oss-derisking.md §2.4`
(RMSNorm / RoPE / KV-cache not in the scope-discovery capture)
without waiting on a full re-capture pass.

Each primitive's docstring explains the formula, cites the upstream
`triton_kernels` source when relevant, and records the current
salmon verdict.  When a live `.hsaco` of the real GPT-OSS kernel
lands from a future scope-discovery run, either the shim graduates
to a corpus-style recipe next to the captured kernel, or this
recipe stays as the "minimal reproducer" next to the richer
production version.

Committed GPT-OSS primitives (confirmed-emitted key instructions
and current salmon verdicts — rerun the Makefile from
`compare_correctness/` for a fresh picture):

| recipe                    | primitive                                           | key gfx1250 cross-lane / WMMA / "num" opcodes                                                        | native | legacy               | salmon                                                                                                                      |
|---------------------------|-----------------------------------------------------|-------------------------------------------------------------------------------------------------------|--------|----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `rmsnorm_fp32`            | RMSNorm                                             | `v_add_f32_dpp × 4`, `v_permlanex16 × 1`, `v_sqrt_f32 × 1`, `v_min_num_f64 × 7`                      | 4 / 4  | 4 / 4 crash (legacy) | **4 / 4 match** (fixed by an orthogonal commit between finding #4's original framing and the 2026-04-21 narrow-O1 landing; the `actual / ref ≈ 0.55` signature no longer reproduces. See modrep-predicate-chain.md §5 / §6 for the current status.) |
| `swiglu_fp32`             | SwiGLU (α=1.702, limit=7.0)                         | `v_exp_f32 × 8`, `v_rcp_f32 × 8`, `v_fma_f32 × 15`, `v_dual_max_num_f32 × 8`, `v_med3_num_f32 × 8`    | 4 / 4  | 4 / 4 crash (legacy) | **4 / 4 WRONG** (`max\|err\| ≈ 56 ≈ limit × 8`) — see finding #4 below                                                      |
| `rope_fp32`               | RoPE (half-rotation, precomputed cos/sin)           | purely elementwise (fma / mul / strided load)                                                          | 4 / 4  | 4 / 4 crash (legacy) | **4 / 4 match**                                                                                                             |
| `matmul_fp16`             | fp16 × fp16 → fp16 GEMM (fp32 acc), 32×32×32 tiles   | `v_wmma × 4`, `v_permlane16_swap_b32 × 8`                                                              | 5 / 5  | 5 / 5 crash (legacy, "8 unsupported") | **5 / 5 WRONG** (accumulator pinned to a per-shape constant — consistent with broken WMMA → MFMA translation) — see finding #5 below |
| `matmul_fp16_16x16`       | fp16 × fp16 → fp16 GEMM (fp32 acc), 16×16×16 tiles   | `v_wmma × 1`, `ds_swizzle_b32 × 2`                                                                     | 5 / 5  | 5 / 5 crash (legacy) | **5 / 5 WRONG** (same signature as matmul_fp16 — `actual = −0.022049` constant)                                             |

Triage value of the mixed-verdict set:

- `rope_fp32` passing proves salmon's elementwise-math +
  strided-load lowering is clean end-to-end.  Anything that fails
  on an elementwise-math-only kernel would be a broader regression
  this recipe surfaces.
- `rmsnorm_fp32` and `swiglu_fp32` failing was predicted by
  `gpt-oss-derisking.md §9.2`'s P5-class concern.  Disassembly
  narrows the blocker to the gfx11+ `v_*_num_*` opcode family
  (finding #4 below), not to wave-size translation per se.
- `matmul_fp16*` failing re-opens **Issue #3 (WMMA translation
  completeness)** from the project tracker — the current
  `wmma_lowering.cpp` path handles `v_wmma_f32_16x16x32_f16` at a
  structural level, but both of our square-GEMM configs produce a
  constant accumulator under salmon, with the same `−0.022049`
  signature regardless of tile size.  The `ds_swizzle` vs
  `permlane16_swap` delta between the two matmul recipes cleanly
  separates the two C2-hard primitives: neither is a crash, both
  produce the same wrong-output shape, which points at the WMMA
  lowering itself as the unified fix target (not the cross-lane
  shuffle each emits).

For the two existing corpus reduction recipes (in the next section),
the same pattern holds: `corpus_softmax_fp32` and
`corpus_layernorm_fp32` both emit the `permlanex16 + DPP` pair
(same as the passing `canary_dpp_compound_add_fp32`) yet fail under
salmon — consistent with the `v_*_num_*` gap (softmax via
`v_exp_f32`'s canonicalisation; layernorm via the rsqrt path, same
site as rmsnorm) rather than any cross-lane primitive.

Currently missing from this section (Tier-1 roadmap):

- **MoE routing** (topk, bitmatrix, compaction) — high-risk C2-DPP
  + C2-bpermute use-sites, but authoring a faithful minimal version
  is non-trivial.  Still TBD.
- **Attention (FlashAttention-style)** — the upstream `_attn_fwd`
  is already pulled into `_corpus/extracted/`.  Wiring it needs a
  multi-dim shape sweep extension to the schema (today: one swept
  dim — attention wants `(BATCH, HEADS, SEQ, HEAD_DIM)`).
- **Scaled matmul (MXFP4)** — needs `uint8` dtype in the harness
  (or `*mxfp4` sig-type support) and `tl.dot_scaled`.  Highest-
  value remaining recipe for the GPT-OSS MoE expert GEMM.

## Corpus: stress-testing salmon with upstream Triton kernels

The `_corpus/` directory pulls and wires up Triton kernels from the
public ecosystem so we can probe salmon coverage on real-world IR
patterns instead of only what we hand-author here.  This is meant for
breadth ("what does salmon choke on?"), not as a substitute for the
hand-written HIP recipes (which test specific lane / wave-size /
cross-lane semantics).

### Pulling

The corpus is **not committed** — `_corpus/.gitignore` excludes
everything under `upstream/` and `extracted/`, plus `INDEX.json`.
Pull it explicitly:

```
cd kernels/triton/_corpus
python3 pull.py        # or:  python3 pull.py --status   to inspect state
```

`pull.py` is manifest-driven (`MANIFEST = [CorpusEntry(...), ...]` at
the top of the file).  Each entry pins a specific upstream URL **and
its sha256**, fetches it once, verifies the digest, and uses Python's
`ast` module to extract just the named `@triton.jit` function bodies
into self-contained files under `extracted/<fn>.py`.  A change in the
upstream file will trip the digest check and refuse to extract — the
manifest pin is a load-bearing piece of reproducibility, not a
formality.  Bumping a pin is a deliberate, reviewable change.

We extract into isolated files because the upstream tutorials run
host-side benchmark code at import time (PyTorch, matplotlib,
`torch.testing.assert_close`); we only want the device kernel
definition.

### Wiring a corpus entry into a recipe

Once `pull.py` has run, an extracted kernel is just a Python function.
A "shim recipe" loads it via the helper `load_corpus_jit` and wraps it
in the same `RECIPES = [{...}]` schema as any other Triton recipe in
this directory.  Four worked examples covering the common shapes of
upstream kernel are committed:

- `corpus_add_fp32.py` — pure pointer + integer scalars, no extras.
  Use this as the template for a one-shot elementwise kernel.
- `corpus_asin_fp32.py` — same shape as add, but pins `range_lo` /
  `range_hi` to `[-1, 1)` to keep `asin` defined.  Demonstrates that
  range tuning is a per-recipe authoring decision.
- `corpus_softmax_fp32.py` — `harness_constants` for a fixed second
  dim (`n_rows`), `scalar_args` for runtime-derived integer strides,
  and a `rel-rms` reduction comparator.  This is the template for any
  upstream kernel where strides are computed from the swept shape.
- `corpus_layernorm_fp32.py` — `harness_constants` for a sizing-only
  dim (`M` is not a kernel argument), `scalar_args` mixing an integer
  expression (`stride = N`) with a float literal (`eps = 1e-5`), and
  three outputs of mixed sizes with per-output comparators.  This is
  the template for kernels that fuse a reduction with elementwise
  outputs.

The shim is what gives the upstream kernel concrete shape sweeps,
input ranges, comparator tolerances, and a place in the `Makefile`'s
`TRITON_KERNELS` list.

Two design points are worth calling out:

1. **The shim, not the upstream file, is committed.**  Upstream
   kernels live behind a digest-checked download; the shim recipe
   sitting next to it in the repo is what defines correctness for
   *our* harness (which inputs, which tolerance, which RMS norm vs.
   pointwise compare).  Rebuilding the corpus is a pull + make, not a
   git checkout.
2. **Range tuning matters.**  Upstream kernels often assume well-
   behaved inputs (asin: `[-1, 1]`; log: `(0, ∞)`; softmax: avoids
   overflow by subtracting max).  Pin `range_lo` / `range_hi` in the
   shim accordingly — running asin on `[-1, 1)` of the *uniform* RNG
   gives essentially zero NaNs, but bumping it to `[-1.001, 1.001)`
   silently turns half your output into NaN-vs-NaN compares and makes
   real findings invisible.

### Status report

`status.py` reports the state of the corpus in two modes:

```
cd kernels/triton/_corpus
python3 status.py          # static: pulled? wrapper exists? built?
python3 status.py --run    # also invokes compare_correctness per recipe
```

Static mode is fast and works without a built tool.  `--run` requires
`compare_correctness` to have been built (and `libsalmon_intercept.so`
to be next to it for the salmon column to do anything meaningful).
Both modes emit a Markdown table; pipe into a PR description if you
want a quick salmon-coverage snapshot.

The `salmon` column distinguishes:

- `ALL_MATCH` — every shape's salmon run produced output that matched
  the gold under the recipe's comparator.
- `MISMATCH` — at least one shape produced output that decoded
  successfully but exceeded the tolerance.  This is the interesting
  case for arithmetic correctness.
- `CRASH` — at least one shape's salmon child died (signal, non-zero
  exit, or no output).  Almost always a missing instruction lowering;
  the harness's `Failures` section has the stderr tail.
- `NOT_BUILT` / `NOT_WIRED` / `SKIPPED` — administrative states (no
  sidecar yet, no shim recipe yet, or `--run` was not passed).

### Current corpus status

This is the snapshot from `python3 status.py --run` on gfx942 with
the in-tree salmon at the time of writing.  Run the script yourself
for an up-to-date view; the table is committed only as a starting
point and will rot quickly as salmon gains lowerings.

| entry                                | function                | wrapper                 | salmon       | one-liner                                                                 |
|--------------------------------------|-------------------------|-------------------------|--------------|---------------------------------------------------------------------------|
| triton-tutorial-01-vector-add        | `add_kernel`            | `corpus_add_fp32`       | `ALL_MATCH`  | elementwise fp32 add — clean baseline                                     |
| triton-tutorial-02-fused-softmax     | `softmax_kernel`        | `corpus_softmax_fp32`   | `CRASH`      | row-wise softmax with masking; salmon segfaults inside the reduction      |
| triton-tutorial-05-layer-norm        | `_layer_norm_fwd_fused` | `corpus_layernorm_fp32` | `CRASH`      | salmon hangs (harness `SIGKILL` after `COMPARE_CORRECTNESS_CHILD_TIMEOUT_S`) |
| triton-tutorial-07-extern-functions  | `asin_kernel`           | `corpus_asin_fp32`      | `CRASH`      | salmon trips on `v_bfi_b32` from libdevice asin (real coverage gap)       |

All four upstream tutorials are now wired through to `compare_correctness`
end-to-end.  The harness's plumbing is no longer the limiting factor:
softmax / layer-norm / asin all reach native gold and then expose
genuine salmon coverage gaps (segfault, hang, missing instruction
lowering) — exactly the kind of breadth the corpus is meant to
surface.  The shims for softmax and layer-norm exercised the schema
work that landed alongside this status update: `harness_constants`
for non-kernel-arg sizing values (e.g. layer-norm's `M`), and
`scalar_args` for runtime-derived integer scalars (strides) and
floating-point scalar literals (`eps`).  The harness's per-child
`SIGKILL` timeout is what kept the layer-norm hang from blocking the
rest of the report; tune via `COMPARE_CORRECTNESS_CHILD_TIMEOUT_S`
if you're investigating a specific salmon hang and want it to run
longer before being cut off.

### Adding a new corpus entry

1. Append a `CorpusEntry(...)` to `MANIFEST` in `_corpus/pull.py`.
   Pin the URL to a specific commit SHA (no `master`, no `main`),
   pre-compute the sha256 of the file at that pin, and list the exact
   `@triton.jit` function names you want extracted.
2. `python3 pull.py` to fetch + verify + extract.
3. Drop a `corpus_<name>.py` shim next to the existing ones, calling
   `load_corpus_jit("<fn_name>")` for the kernel function.
4. Add `corpus_<name>` to `$(TRITON_KERNELS)` in the Makefile.
5. Build and re-run `status.py --run` to see where salmon lands.

If the upstream kernel needs schema features we don't yet have
(today: multi-dim sweeps that genuinely vary more than one shape
axis, vector / struct sig args), leave the shim out for now and add
a note to the entry's `notes:` field explaining what's missing —
`status.py` will surface it as `NOT_WIRED`, which is preferable to a
half-working shim that quietly skips compares.  Common single-dim
gotchas now have idioms instead: extra fixed dims go in
`harness_constants` (see layer-norm's `M`), runtime-derived integer
scalars and float-scalar literals go in `scalar_args` (see softmax's
strides and layer-norm's `eps`).
