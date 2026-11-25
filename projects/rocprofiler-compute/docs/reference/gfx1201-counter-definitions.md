# gfx1201 counter definitions

ROCm Compute Profiler supplies its own SDK configuration through
`ROCPROFILER_METRICS_PATH`. The SDK does not merge an existing override file
with its installed definitions. This configuration retains the complete
87-definition gfx1201 contract from the SDK before the architecture cleanup.

The restoration adds 63 missing architecture definitions. Existing definitions
for other architectures and the firmware restrictions remain intact.
`FETCH_SIZE` includes 32-, 64-, 128- and 256-byte packets.

## References

- [SDK contract before #9541](https://github.com/ROCm/rocm-systems/blob/e92aa6ee8c01f991c9da08ea386330a6374e0f4a/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/config.yaml).
- [Architecture cleanup #9541](https://github.com/ROCm/rocm-systems/commit/9d29a1da312470ebfb1cc3d862e607d9a1960559).
- [Navi48 event-ID correction #158](https://github.com/ROCm/rocm-systems/commit/5dbb3b2f8083ef21f1a4a86c4adcfe4222d39076).
- [256-byte FETCH_SIZE correction #8022](https://github.com/ROCm/rocm-systems/commit/e5f2ddbdb0662b45b9f1ea9eada90cbb7984023c).

## Additional experimental coverage

Another 16 raw selectors and their 16 sums have no gfx1201 definition in the
historical SDK snapshot. They cover GL1C, GL2C, SQ/SQC and TCP requests.
The feature continues to use the existing experimental gfx1201 profile path.

[The gfx1201 reference implementation](https://github.com/qiangpan2/rocm-systems/blob/76e2aa6b63ec70675f7e6c2066d80e6e7a27cd4b/projects/rocprofiler-compute/src/rocprof_compute_soc/profile_configs/sdk_config.yaml)
and the [soc24 selector catalog](https://github.com/ROCm/rocm-systems/blob/6f441646398273bb16a065a3b3798d95f7a95563/projects/aqlprofile/linux/soc24_enum.h)
provide the available public references. Seven GL1C/GL2C selectors still need
a public chip-specific IP/RDL reference: GL1C_GL2_REQ_READ_64B,
GL1C_REQ_READ_32B, GL1C_REQ_READ_128B, GL1C_REQ_WRITE_32B,
GL2C_READ_32_REQ, GL2C_READ_64_REQ and GL2C_READ_128_REQ.

## Runtime validation (2026-11-25)

The author supplied 41 counter-control results: 33 pass their stated models,
6 match their control references, and 2 remain zero and unproven. The rows
come from different controls. Accounting relations use paired samples within
each control; means from different rows are not a combined dispatch.

Bounds establish only the stated bounds. Payload and instruction models are
conditional on their listed shapes, wave modes, cache policies and controls.
These observations add empirical evidence without replacing chip-specific
IP/RDL confirmation.

| Counter | Theory / validation reference | Measured mean | Scoped result |
| --- | --- | --- | --- |
| `GL1C_BUSY` | 0 ≤ BUSY ≤ CYCLE, on each dispatch | 861,448.033333333 | PASS — bound only |
| `GL1C_CYCLE` | 32 × GRBM_GUI_ACTIVE, on each dispatch | 1,925,357.86666667 | PASS — clock relation |
| `GL2C_BUSY` | 0 ≤ BUSY ≤ CYCLE, on each dispatch | 601,549.566666667 | PASS — bound only |
| `GL2C_CYCLE` | 32 × GRBM_GUI_ACTIVE, on each dispatch | 1,925,357.86666667 | PASS — clock relation |
| `GL2C_EA_RDREQ` | EA128 + EA64 + EA256, under the tested EA32=0 path | 2,177.93333333333 | PASS — accounting only |
| `GL2C_EA_RDREQ_128B` | Cold A/B payload: (16×1×4×4,096 + 1×4×4,096) / 128 = 2,176 | 2,176 | PASS — conditional payload model |
| `GL2C_EA_RDREQ_256B` | Scalar-A cached control: Δ = A bytes / 256 = 1,024 | 1,025.93333333333 | PASS — controlled delta |
| `GL2C_EA_RDREQ_32B` | M=8, output stride=32 floats: one 32B request per atomic/wave; G=4,096 | 4,096 | PASS — conditional atomic model |
| `GL2C_EA_RDREQ_64B` | Uncached scalar-A: 1 request/wave × 4,096 | 4,096 | PASS — conditional scalar model |
| `GL2C_EA_WRREQ` | Uncached output; 16 repeated stores/wave × 4,096 = 65,536 | 65,536 | PASS — controlled store count |
| `GL2C_EA_WRREQ_64B` | M=16 → 64 output bytes/batch; 16 × 4,096 = 65,536 | 65,536 | PASS — conditional packet count |
| `GL2C_EA_WRREQ_STALL` | Selector 122 stall cycles: max ≤ GUI cycles; burst is nonzero, paced stores are zero at the same 65,536 writes; MC evaluated as an EA alias. | 10,126.3 | PASS — alias / paced-stall model |
| `GL2C_MC_WRREQ_STALL` | Selector 122 stall cycles: max ≤ GUI cycles; burst is nonzero, paced stores are zero at the same 65,536 writes; MC evaluated as an EA alias. | 10,126.3 | PASS — alias / paced-stall model |
| `GL2C_READ_128_REQ` | One wave: 2 data address groups + 55 instruction reads = 57; alpha=0 gives 0 data + 50 instruction reads. | 57 | PASS — controlled read attribution |
| `SQC_DCACHE_DUPLICATE_MISSES` | One wave: 3 queued header requests + 2 queued tail requests = 5 duplicates; serial waits produce 0. | 5 | PASS — controlled scalar-request model |
| `SQC_ICACHE_MISSES` | 32 WGPs × (55 baseline lines + 32,768 × 4B / 128B) = 34,528; four smaller cold-prefix controls also match. | 34,528 | PASS — conditional cold-code model |
| `SQC_TC_INST_REQ` | 32 WGPs × (55 baseline lines + 32,768 × 4B / 128B) = 34,528; four smaller cold-prefix controls also match. | 34,528 | PASS — conditional cold-code model |
| `SQ_ACCUM_PREV` | Selector 1 ordered after resident-wave level 7: integral = WAVE_CYCLES + WAVES = BUSY_CYCLES − WAVES; paired dispatches match. | 15,983.8666666667 | PASS — ordered wave-level integral model |
| `SQ_INSTS_VEC32_LEVEL_LDS` | Serialized LDS prefix: sampled level is 0 or 1; start once while idle and do not reset during activity. | 0.798245614035088 | PASS — continuous live-state model |
| `SQ_INSTS_WAVE32` | Add 16 v_mov instructions/wave: Δ=16×4,096=65,536 | 806,910 | PASS — controlled instruction delta |
| `SQ_INSTS_WAVE32_LDS` | Add ds_store_b32 + ds_load_b32: Δ=2×4,096=8,192 | 151,552 | PASS — controlled instruction delta |
| `SQ_INSTS_WAVE32_VALU` | Add 16 v_mov instructions/wave: Δ=65,536 | 524,286 | PASS — controlled instruction delta |
| `SQ_INST_CYCLES_VMEM` | Replace one load by four: Δ=3×TA baseline=405,504 | 827,392 | PASS — calibrated issue delta |
| `SQ_INST_LEVEL_LDS` | Serialized LDS prefix: sampled level is 0 or 1; start once while idle and do not reset during activity. | 0.798245614035088 | PASS — continuous live-state model |
| `SQ_WAIT_ANY` | 0 ≤ WAIT_INST_ANY ≤ WAIT_ANY ≤ WAVE_CYCLES, same dispatch | 14,527,707.5333333 | PASS — bound only |
| `SQ_WAIT_INST_ANY` | 0 ≤ WAIT_INST_ANY ≤ WAIT_ANY ≤ WAVE_CYCLES, same dispatch | 9,124,469 | PASS — bound only |
| `SQ_WAVE32_INSTS` | Wave32 marker: baseline +16×4,096 = 2,969,598 | 2,969,598 | PASS — wave-mode / delta |
| `SQ_WAVE64_INSTS` | Wave64 marker: baseline +16×4,096 = 2,985,982 | 2,985,982 | PASS — wave-mode / delta |
| `SQ_WAVE_CYCLES` | Wave-level integral minus admitted waves; aligned GPU clock telemetry agrees within 1.4%; exported instance sum behaves as full cycles. | 16,531.2333333333 | PASS — summed wave-cycle model |
| `TA_BUFFER_LOAD_WAVEFRONTS` | Four loads per original load: 4×baseline 135,168 = 540,672 | 540,672 | PASS — controlled expansion |
| `TA_BUFFER_STORE_WAVEFRONTS` | 16 stores per original store: 16×4,096 = 65,536 | 65,536 | PASS — controlled expansion |
| `TA_TA_BUSY` | 64 TA instances: 0 ≤ min ≤ SUM/64 ≤ max ≤ GUI_ACTIVE; Perf_Windowing unsupported, requiring GPU isolation. | 2,944,367.83333333 | PASS — per-instance clock-bound model |
| `GL2C_READ_32_REQ` | GL2C/78: SECTORS − 2×READ64 − 4×READ128 = 31,232 − 2×1,024 − 4×7,040 = 1,024; same completed-read sample. | 1,024 | PASS — positive response / request and sector accounting |
| `GL1C_GL2_REQ_READ_64B` | Baseline 64; +1 scalar load/wave: 64 + 4,096 = 4,160 | 4,160 | Match |
| `GL1C_REQ_READ_128B` | GL2C_READ_128_REQ − SQC_TC_INST_REQ, same dispatch | 7,808 | Match |
| `GL2C_READ_64_REQ` | Baseline 64; scalar-A adds G=4,096 → 4,160 | 4,160 | Match |
| `SQC_DCACHE_MISSES` | Baseline 64; cold scalar-A adds G=4,096 → 4,160 | 4,160 | Match |
| `TCP_REQ_READ` | Byte-load control: 4×2 active groups×4,096 = 32,768 | 32,768 | Match |
| `TCP_REQ_WRITE` | 16 repeated stores ×4,096 =65,536 | 65,536 | Match |
| `GL1C_REQ_READ_32B` | Selector 36; counting enabled on all eight shader arrays, guards contain only READ128; no documented positive 32B producer. | 0 | Zero, unproven |
| `GL1C_REQ_WRITE_32B` | Selector 42; counting gate captures 8,192 R8 writes, guards classify all as WRITE256; 32B remains untriggered. | 0 | Zero, unproven |

### Interpretation limits

- `SQ_ACCUM_PREV` uses selector 1 ordered after resident-wave level 7. Its
  integral relation is specific to that control configuration.
- `SQ_INST_LEVEL_LDS` and `SQ_INSTS_VEC32_LEVEL_LDS` use continuously sampled
  live state: start while idle and avoid resets during activity.
- The validated `SQ_WAVE_CYCLES` instance sum behaves as full cycles in the
  tested controls. The single-wave rate agrees with aligned clock telemetry
  within 1.4%.
- EA and MC write-stall names select 122 in this configuration. The controls
  evaluated MC as an expression alias of EA; burst and paced responses
  distinguish stall duration from the unchanged write count.
- TA clock bounds require GPU isolation because Perf_Windowing is unsupported.
- `GL1C_REQ_READ_32B` and `GL1C_REQ_WRITE_32B` remain zero and unproven. The
  enabled counting gates and positive size guards do not establish a positive
  producer for their 32B classes.
