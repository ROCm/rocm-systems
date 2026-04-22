# `const_in_probe` — value-independent constant-input probe generator

A one-script tool that generates a `_const_in` sibling recipe
from any existing `compare_correctness` Triton recipe.  The
sibling replaces random input ranges with a near-constant value
across every element; for a deterministic kernel, native output
becomes analytically predictable, and any salmon deviation
carries VALUE-INDEPENDENT structural information about which
specific lifter path misbehaves.

Sibling to `aiter_corpus_runner/`, `triton_corpus_runner/`,
`tensile_gold_verify/` under `transpiler/tools/`.  Operates on
recipe files under
`transpiler/tools/compare_correctness/kernels/triton/`; does not
itself depend on `compare_correctness/`'s C++ harness.

## Methodology + reference case

The const-input probe approach and what deviation patterns
indicate live in `hotswap/docs/learnings.md` entries dated
2026-04-22:

* `V_FMA_MIX inline-constant narrow-half` — the reference bug
  (two days of false leads on a noisy random-input signal; the
  const-input probe pinned it in 5 minutes once applied).
* `Diagnostic technique — value-independent constant-input
  probes` — the general methodology, mechanisation notes,
  deviation-pattern → suspect-class table.

## Usage

From anywhere (the script takes absolute / relative paths):

```bash
python3 \
  transpiler/tools/const_in_probe/mk_const_in_probe.py \
  transpiler/tools/compare_correctness/kernels/triton/<recipe>.py \
  [--const 1.0]
```

Writes `<recipe>_const_in.py` next to the base recipe.  The
generator prints next-step commands for registering, building,
and running the sibling under `compare_correctness`.

## When to use

* **Always** when landing a new direct-invocation recipe whose
  kernel contains `tl.sum` / `tl.max` / `tl.cumsum` / any
  cross-lane reduction.
* **Always** when landing a new recipe whose output is
  elementwise-deterministic in the input (broadcasts, pointwise
  ops).
* **Optional** for kernels with recurrences / control flow
  whose closed-form output-under-constant-input is non-trivial —
  the probe still produces a value-independent verdict, but the
  predicted output requires manual computation.

## When the probe does NOT help

* Kernels whose output is discrete / order-sensitive (sort,
  top-k index selection, argmax).  Constant input makes every
  element tied; the output becomes IMPLEMENTATION-DEFINED, and
  salmon/native divergence is a false positive.
* Kernels whose output depends on data statistics (softmax
  temperature, variance / stddev).  Constant input collapses
  the statistic to a degenerate value.

For these, use the probe on a narrower sub-shape (bisect the
kernel to expose just the deterministic stage) — see the
`topk_forward_bisect_*` family in
`compare_correctness/kernels/triton/` for the pattern.

## Principled-review notes

Audit flags identified at creation time and their disposition:

* **Flat-layout assumption for recipe files.**  `_infer_base_module`
  assumes recipes live directly under `kernels/triton/` and are
  importable by stem name.  Subdirectory recipes (`_corpus/`) are
  extracted kernel sources, not recipes, and are rejected at
  generation time by `_validate_base_shape` with an actionable
  diagnostic.
* **Single-recipe-per-file assumption.**  Empirically verified
  on the current corpus (2026-04-22).  Multi-recipe files fail
  loudly at generation time with an actionable "split before
  generating" diagnostic rather than silently cloning only
  `RECIPES[0]`.
* **Hard-coded constant-range epsilon `1e-3`.**  At bf16 / fp16
  ULP around magnitude 1.0 this epsilon is well below the ULP,
  so every input quantises to the same narrow-float value.  For
  fp32 / fp64 bases needing sub-ULP epsilons, edit
  `_clone_const_in` directly — the methodology depends on
  single-constant-across-inputs, not the specific epsilon.
* **No `--force` / overwrite flag.**  Intentional.  If you need
  to regenerate, `rm` the existing sibling explicitly — surfaces
  the case where a hand-tuned sibling would be overwritten.
* **No CI gate requiring a `_const_in` sibling per recipe.**
  The generator ships here; the policy (which recipes are
  exempt, does it run under `compare_correctness` only, etc.)
  is a separate PR.
