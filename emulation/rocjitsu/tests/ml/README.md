# ML workload tests

Model-level correctness workloads that run unmodified ROCm python stacks on top
of the emulator. Unlike the C++ tests under `tests/`, these need a torch build
whose code objects cover the emulated architecture, so they are run by hand (or
by CI lanes that provide such a wheel) rather than registered with CTest.

## `gpt_oss_kernels.py`

A single-file correctness suite for every distinct GPU kernel that vLLM
dispatches for `openai/gpt-oss-20b`: embedding gather, RMSNorm (plain and
residual-fused), the QKV/O/router/LM-head GEMMs, YaRN NeoX rotary embedding,
paged KV writes, attention with learned softmax sinks in both the full-causal
and sliding-window forms, top-k routing, MXFP4 dequantisation, clamped
SwiGLU-OAI, the MoE GEMM, greedy sampling, and one integrated transformer
block.

Each case has a float64 CPU reference written in the same file from the model
definition, so a shared bug cannot cancel out. Cases whose kernel vLLM runs
through Triton on ROCm carry a Triton implementation as well as the torch-eager
one, so a codegen-dependent failure is isolated to the implementation that has
it.

```bash
# emulated MI355X (gfx950)
rocjitsu --config configs/gfx950_mi355x_kmd.json -- \
    python tests/ml/gpt_oss_kernels.py

# emulated MI455X (gfx1250)
rocjitsu --config configs/gfx1250_mi455x.json -- \
    python tests/ml/gpt_oss_kernels.py

# attribute a failure: the identical code on CPU torch
python tests/ml/gpt_oss_kernels.py --device cpu
```

`--device cpu` is the first thing to run when a case fails. It executes the
same device arm on plain CPU torch; a case that fails there indicts the suite's
own reference or tolerance, not the emulator.

Other flags: `--size {tiny,small,model}` picks the shape profile (`model` is
gpt-oss-20b's real widths and takes hours under functional emulation),
`--kernel`/`--impl` filter cases, `--json` writes a machine-readable report,
and `--with-vllm` adds a cross-check against vLLM's own custom ops where vLLM
is installed.

### Comparing two architectures

Each case records a SHA-256 of its raw device output, and `--compare` diffs two
reports:

```bash
python tests/ml/gpt_oss_kernels.py --compare gfx950.json gfx1250.json
```

Comparing bytes is far more sensitive than comparing per-case error magnitudes,
which only notice a divergence that happens to move the worst element. It found
the BF16 rounding defect fixed in `0d62681384e`.

It is a lead generator, not a verdict. Two kinds of difference are legitimate
and will show up:

- **Different instruction selection.** gfx1250 has no `V_CVT_PK_BF16_F32`, so
  LLVM fuses `fptrunc(a * b)` into `v_fma_mixlo_bf16 v, a, b, 0`. That computes
  `a * b + 0.0`, and IEEE says `(-0.0) + (+0.0)` is `+0.0`, so a product that is
  negative zero on gfx950 is positive zero on gfx1250. Real gfx1250 silicon
  does the same. It shows up in `mxfp4_dequant`, whose value table contains
  `-0.0`, as ~5% of elements differing while both arms still match the float64
  reference exactly.
- **Different software stacks.** The two targets need different ROCm wheels, so
  the library GEMMs are not the same kernels and differ in the last bit. The
  Triton attention cases differ in 2 elements out of 8192 by one ulp, in
  opposite directions.

So treat a byte difference as something to explain. A difference concentrated in
one direction, or one that grows with the reduction length, is worth chasing; a
scattering of one-ulp differences in both directions is not.

### Choosing a python environment

The two targets need different wheels, because ROCm nightly wheels are built
per architecture and no single index ships both: use a torch whose
`rocm_sdk_libraries_*` covers gfx950 for the CDNA4 config and one covering
gfx1250 for the CDNA5 config. Running a gfx1250-only wheel against the gfx950
config fails at the first kernel launch with `hipErrorInvalidImage`, which is a
wheel coverage problem and not an emulator fault.
