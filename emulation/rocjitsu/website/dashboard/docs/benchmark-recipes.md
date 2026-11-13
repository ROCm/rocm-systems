# Performance benchmark recipes

The [corpus suite TOML](https://github.com/ROCm/rocjitsu-test-corpus/tree/f3d8c49d7f0ead0f7369eda577cd46d73aa54a93/benchmarks/suites)
at revision `f3d8c49d7f0ead0f7369eda577cd46d73aa54a93` is the source of truth, pinned by the
[benchmark workflow](../../../../../.github/workflows/rocjitsu-benchmarks.yml).
Update this reference when changing that pin or either recipe.

Target cells list only enabled policies: `- Default` and/or `- Single`.
An empty cell means the workload is not enabled for that target. Target columns
with no enabled entries are omitted. Overrides replace the shared policy list;
CLI filters can narrow selection. Enablement is recipe eligibility, not a
successful measurement or support on every device configuration. Coverage
describes intended mechanisms, not measured instruction coverage.

Both recipes use zero warmups, one sample and a 1200-second per-cell timeout.
No execution or elapsed timings are shown.

## performance-v1

[performance-v1.toml](https://github.com/ROCm/rocjitsu-test-corpus/blob/f3d8c49d7f0ead0f7369eda577cd46d73aa54a93/benchmarks/suites/performance-v1.toml) — **20 workloads; 38 enabled target/policy combinations.**

| Benchmark suite | Workload | Coverage | gfx950 | gfx1250 |
| --- | --- | --- | --- | --- |
| GPT-OSS | Windowed attention<br>`gpt_oss.window_batch4_seq3072` | Causal/windowed attention, matrix arithmetic, softmax and masking. | - Default | - Default |
| GPT-OSS | Full causal attention<br>`gpt_oss.causal_seq2560` | Causal/windowed attention, matrix arithmetic, softmax and masking. | - Default<br>- Single | - Default<br>- Single |
| Triton | FP16 persistent matmul<br>`triton.persistent_1024x4096x4096` | FP16 matrix arithmetic and persistent tile scheduling. | - Default<br>- Single | - Default<br>- Single |
| Triton | FP16 grouped GEMM<br>`triton.grouped_4x2048` | FP16 matrix arithmetic, pointer arrays and multi-problem scheduling. | - Default<br>- Single | - Default<br>- Single |
| DeepSeek | FP8 MLP first projection<br>`deepseek.mlp_w1_m64` | FP8 matrix projections and block scaling. | - Default<br>- Single |  |
| DeepSeek | FP8 MLP first projection<br>`deepseek.mlp_w1_m256` | FP8 matrix projections and block scaling. |  | - Default<br>- Single |
| DeepSeek | FP8 MLP second projection<br>`deepseek.mlp_w2_m64` | FP8 matrix projections and block scaling. | - Default |  |
| DeepSeek | FP8 MLP second projection<br>`deepseek.mlp_w2_m256` | FP8 matrix projections and block scaling. |  | - Default |
| TensileLite | BF16 Stream-K 4096x4096x4096<br>`tensile.bf16_streamk_4096x4096x4096` | BF16 matrix arithmetic and Stream-K work partitioning. | - Default<br>- Single |  |
| TensileLite | BF16 subtile 4096x4096x4096<br>`tensile.bf16_subtile_4096x4096x4096` | BF16 matrix arithmetic with a subtile layout. |  | - Default<br>- Single |
| TensileLite | MXFP8 subtile 4096x4096x4096<br>`tensile.mxfp8_subtile_4096x4096x4096` | MXFP8 matrix arithmetic, subtile layout and operand scaling. | - Default |  |
| TensileLite | MXFP4 Stream-K 4096x4096x4096<br>`tensile.mxfp4_streamk_4096x4096x4096` | MXFP4 matrix arithmetic, Stream-K work partitioning and operand scaling. |  | - Default |
| DeepSeek | FP8 MLP first projection, 3072 tokens<br>`deepseek.mlp_w1_m3072` | FP8 matrix projections and block scaling. |  | - Default |
| DeepSeek | FP8 MLP second projection, 3072 tokens<br>`deepseek.mlp_w2_m3072` | FP8 matrix projections and block scaling. |  | - Default |
| Triton | FP16 grouped GEMM, 4 groups of 3584x3584x3584<br>`triton.grouped_4x3584` | FP16 matrix arithmetic, pointer arrays and multi-problem scheduling. |  | - Default |
| Triton | FP16 persistent matmul, 8192 x 8192 x 4096<br>`triton.persistent_8192x8192x4096` | FP16 matrix arithmetic and persistent tile scheduling. |  | - Default |
| Triton | FP32 softmax<br>`triton.softmax_131072x2048` | Exponentiation, row reductions and normalization. | - Default | - Default |
| Triton | FP32 layer normalization<br>`triton.layernorm_131072x2048` | Mean/variance reductions and reciprocal square root. | - Default | - Default |
| DeepSeek | BF16 activation quantization to FP8<br>`deepseek.act_quant_32768x8192` | BF16-to-FP8 activation quantization and block-scale writes. | - Default | - Default |
| DeepSeek | FP8 weight dequantization to FP32<br>`deepseek.weight_dequant_32768x8192` | FP8-to-FP32 weight dequantization and block-scale reads. | - Default | - Default |

## performance-v2 — v1 + RDNA3/RDNA4 additions

[performance-v2.toml](https://github.com/ROCm/rocjitsu-test-corpus/blob/f3d8c49d7f0ead0f7369eda577cd46d73aa54a93/benchmarks/suites/performance-v2.toml)
preserves all **38 v1 target/policy combinations** and appends **28 combinations**
for **RDNA3 (`gfx1100`) and RDNA4 (`gfx1201`)**. The complete recipe has
**31 workload definitions and 66 enabled combinations**.

**The table below shows only the additions**, not the v1 rows again.
These additions cover 13 workloads: 11 new definitions plus the existing
grouped/persistent GEMM IDs enabled on RDNA with default-only overrides.
The v1 CDNA selections remain unchanged. Full-Tensile FP16 WMMA is not included.

| Benchmark suite | Workload | Coverage | gfx1100 | gfx1201 |
| --- | --- | --- | --- | --- |
| Triton | FP16 persistent matmul<br>`triton.persistent_1024x4096x4096` | FP16 matrix arithmetic and persistent tile scheduling. | - Default | - Default |
| Triton | FP16 grouped GEMM<br>`triton.grouped_4x2048` | FP16 matrix arithmetic, pointer arrays and multi-problem scheduling. | - Default | - Default |
| Triton | FP32 softmax, 65536 x 2048<br>`triton.softmax_65536x2048` | Exponentiation, row reductions and normalization. | - Default | - Default |
| Triton | FP32 layer normalization, 65536 x 2048<br>`triton.layernorm_65536x2048` | Mean/variance reductions and reciprocal square root. | - Default | - Default |
| Gluon | FP32 1D memory copy<br>`gluon.copy_fp32_100663296` | Coalesced global-memory loads/stores and masking. | - Single | - Single |
| Gluon | FP32 2D strided layout-converting copy<br>`gluon.copy_2d_fp32_16384x8192` | Strided addressing, layout conversion, LDS exchange and barriers. | - Default | - Default |
| Tensile | FP32 VALU GEMM<br>`full_tensile.valu_3072x2048x1024` | FP32 vector FMA, LDS reuse and contraction addressing; not WMMA. | - Default | - Default |
| Tensile | FP32 two-summation tensor contraction<br>`full_tensile.tensor_2048x2048x16x64` | Independent summation loops and multidimensional addressing. | - Default | - Default |
| Tensile | FP32 split-K GEMM, GSU3<br>`full_tensile.splitk_3072x2048x1536_gsu3` | Beta initialization, atomic split reduction and helper/main launch ordering. | - Default | - Default |
| Tensile | FP32 pointer-array batched GEMM<br>`full_tensile.pointer_8x1024x1024x1024` | Device-pointer indirection and uniform matrix batching; not arbitrary grouped GEMM. | - Default | - Default |
| llama.cpp | FP32 RoPE, independent-vector batching<br>`llama.rope_128x4096x32` | Positional rotation and trigonometric arithmetic; synthetic independent-vector batching. | - Default | - Default |
| llama.cpp | Q4_0 decode matrix-vector product<br>`llama.matvec_q4_4096x32768` | Q4_0 weight decoding, Q8_1 activation packing and single-RHS dot products. | - Default<br>- Single | - Default<br>- Single |
| llama.cpp | Q8_0 embedding gather<br>`llama.gather_q8_4096x1024x4096` | Indexed embedding reads and Q8_0 dequantization; repeated and dispersed indices. | - Default | - Default |
