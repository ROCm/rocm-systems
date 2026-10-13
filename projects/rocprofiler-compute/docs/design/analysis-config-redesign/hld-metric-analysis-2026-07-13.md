# Metric Analysis — 2026-07-13

> **Scope:** Static analysis of `src/rocprof_compute_soc/analysis_configs/` (122 YAML files, 7 architectures) and `src/rocprof_compute_soc/profile_configs/sets/`. Results inform the Layer 2 schema design (FR2.1 — what attributes make up a complete metric definition).
>
> **Validity:** These findings reflect the codebase state on 2026-07-13. Results may no longer be accurate if analysis configs have been added, removed, or restructured since this date.

---

## Step 1 — Field inventory

Total metric rows loaded: **2,862** across 824 unique metric names.

### Metric-level fields

| Field | Occurrences | Notes |
|---|---|---|
| `unit` | 2,344 | Present in most metrics |
| `avg` | 1,565 | Paired with `min`/`max` |
| `min` | 1,565 | |
| `max` | 1,565 | |
| `value` | 1,142 | Alternative to `avg/min/max` — single aggregate |
| `peak` | 328 | Optional — performance ceiling for comparison |
| `pct_of_peak` | 245 | Optional — `peak` as percentage |
| `xfer` | 72 | Fabric stall table-specific |
| `coherency` | 72 | Fabric stall table-specific |
| `expr` | 71 | Fabric stall table-specific |
| `::_1` | 60 | Unusual — likely a parsing artifact |
| `units` | 42 | Typo variant of `unit` — documentation drift |
| `coll_level` | 12 | Accumulation-level collection (e.g., `SQ_LEVEL_WAVES`) |
| `type` | 12 | Fabric write stall table-specific |
| `transaction` | 12 | Fabric write stall table-specific |
| `read req` | 12 | Fabric write stall table-specific |
| `write req` | 12 | Fabric write stall table-specific |
| `atomic req` | 12 | Fabric write stall table-specific |
| `std dev` | 6 | Statistical field — rare |
| `ea read/write stall - *` | 6 each | EA stall breakdown fields |

### Table-level fields

| Field | Occurrences | Notes |
|---|---|---|
| `header` | 2,862 | Always present — column name mapping |
| `cli_style` | 735 | Render type for CLI output |
| `tui_style` | 735 | Render type for TUI output |
| `comparable` | 462 | Marks table as non-comparable across runs |
| `style` | 12 | Variant of `cli_style` — rare |

---

## Step 2 — Consistency matrix

327 of 824 metric names appear in more than one architecture.

| Classification | Count | Meaning for Layer 2 schema |
|---|---|---|
| **Pure duplicate** — identical across all fields | 213 | Safe single definition with `archs:` list |
| **Case A** — same description, different formula | 59 | One metric, needs `implementations:` block |
| **Both description and formula drift** | 39 | Ambiguous — needs manual classification |
| **Unit drift** — same name, different unit | 32 | Single `unit:` declaration eliminates this |
| **Aggregation field drift** — `avg/min/max` vs `value` | 21 | Schema must handle both aggregation modes |
| **Description drift, same formula** | 16 | Documentation inconsistency |
| **Peak partial** — `peak` present in some archs only | 12 | `peak` must be optional in schema |

### Examples

**Pure duplicates (3 shown)**
- `Active CUs (deprecated)`
- `VALU Utilization`
- `LDS Bank Conflicts/Access`

**Case A — same description, different formula (3 shown)**
- `vL1D Cache BW` — all CDNA archs, different counter expressions per arch
- `L2 Cache BW` — all CDNA archs
- `L2-Fabric Write BW` — all CDNA archs

**Description drift, same formula (3 shown)**
- `SALU Utilization`
- `VALU Active Threads`
- `Theoretical LDS Bandwidth`

**Both description and formula drift (3 shown)** *(need manual review)*
- `VALU FLOPs`
- `MFMA FLOPs (BF16)`
- `MFMA FLOPs (F16)`

**Unit drift (5 shown)**

| Metric | gfx908 | gfx90a–gfx950 | gfx115x |
|---|---|---|---|
| `VALU FLOPs` | `GFLOP/s` | `GFLOPs` | `GFLOP/s` |
| `MFMA FLOPs (BF16)` | `GFLOP/s` | `GFLOPs` | — |
| `MFMA FLOPs (F16)` | `GFLOP/s` | `GFLOPs` | — |
| `MFMA FLOPs (F32)` | `GFLOP/s` | `GFLOPs` | — |
| `MFMA FLOPs (F64)` | `GFLOP/s` | `GFLOPs` | — |

**Aggregation field drift (3 shown)**

| Metric | gfx908 | gfx90a–gfx950 |
|---|---|---|
| `MFMA Utilization` | `value` only | `avg/min/max` |
| `SALU Utilization` | `avg/min/max` | `avg/min/max` *(consistent within CDNA — drift is from gfx115x `value`)* |
| `VALU Utilization` | same pattern | |

---

## Step 3 — Additional combinations

### Same description, different name

43 descriptions are shared by more than one metric name — these are likely the same metric with inconsistent naming across files.

**Examples:**
- `['MFMA IOPs (INT8)', 'MFMA IOPs (Int8)']` — capitalisation inconsistency
- `['Cache Hit', 'Hit Rate', 'L2 Cache Hit Rate', 'L2 Hit']` — four names for the same concept
- `['Fabric Rd Lat', 'L2-Fabric Read Latency', 'Read Latency']` — three names for the same concept

### Metrics appearing in more than one panel

61 metrics appear in multiple panels within the same architecture. This means the "one definition per metric" rule must handle cross-panel references — a metric cannot be tied to a single panel.

**Examples (metric: panel ids)**
- `VALU FLOPs`: panels 200, 400, 1100
- `MFMA FLOPs (BF16)`: panels 200, 400, 1100
- `MFMA FLOPs (F16)`: panels 200, 400, 1100
- `MFMA FLOPs (F32)`: panels 200, 400, 1100
- `MFMA FLOPs (F64)`: panels 200, 400, 1100

### Special characters in metric names

95 metric names contain `()`, `/`, `%`, or `:`. A separate stable `id` field is required — ids cannot be derived mechanically from display names.

Examples: `MFMA FLOPs (BF16)`, `MFMA FLOPs (F16)`, `L2-Fabric Write BW`, `vL1D Hit Rate (%)`

### Missing descriptions

- **308 / 2,862 rows (10%)** have no description entry.
- Description length: min=25 chars, max=658 chars, median=154 chars.
- `description` is optional today; whether it should be required in the new schema is a design decision.

### `$variable` references in formulas

Full inventory of hardware topology variables used across all metric formulas:

| Variable | References | Meaning |
|---|---|---|
| `$denom` | 3,207 | Normalization denominator |
| `$normUnit` | 1,082 | Normalization unit label |
| `$GRBM_GUI_ACTIVE_PER_XCD` | 566 | Active cycles per XCD |
| `$cu_per_gpu` | 542 | CU count |
| `$max_sclk` | 155 | Peak clock speed |
| `$se_per_gpu` | 114 | Shader engine count |
| `$total_l2_chan` | 93 | Total L2 channels |
| `$GRBM_SPI_BUSY_PER_XCD` | 93 | SPI busy cycles |
| `$wave_size` | 62 | Wavefront size |
| `$lds_banks_per_cu` | 44 | LDS banks per CU |
| `$sqc_per_gpu` | 38 | SQC count |
| `$pipes_per_gpu` | 21 | Pipes per GPU |
| `$numActiveCUs` | 18 | Active CU count (runtime) |
| `$hbmBandwidth` | 18 | Peak HBM bandwidth |
| `$GRBM_COUNT_PER_XCD` | 18 | GRBM count per XCD |
| `$max_waves_per_cu` | 7 | Max wavefronts per CU |
| `$FP32Flops_empirical_peak` | 7 | Empirical FP32 peak |
| `$FP64Flops_empirical_peak` | 7 | Empirical FP64 peak |
| `$L2Bw_empirical_peak` | 7 | Empirical L2 bandwidth peak |
| `$L1Bw_empirical_peak` | 7 | Empirical L1 bandwidth peak |

### Optional fields

| Field | Metric count | Notes |
|---|---|---|
| `pct_of_peak` | 50 | Optional — percentage of empirical peak |
| `peak` | partial in 12 metrics | Optional — absent for some archs |
| `coll_level` | 12 | Optional — accumulation level (e.g., `SQ_LEVEL_WAVES`) |

### Unusual/table-type-specific fields

These fields (~70 rows each) belong to specific table types and do not belong in a general metric definition. They likely require typed metric variants in the schema:

| Field group | Row count | Table type |
|---|---|---|
| `xfer`, `coherency`, `expr` | ~72 each | Fabric stall breakdown tables |
| `type`, `transaction`, `read req`, `write req`, `atomic req` | 12 each | Fabric write stall tables |
| `std dev` | 6 | Statistical tables |

---

## Step 4 — Architecture coverage

| Coverage pattern | Count |
|---|---|
| Present in all 7 archs | 18 |
| Present in all CDNA (6 archs) | 259 |
| RDNA only (gfx115x) | 152 |
| Unique to exactly one arch | 497 |
| — of which gfx950-only | 345 |
| — of which gfx115x-only | 152 |
| Partial CDNA subset (2–5 CDNA archs) | 50 |

**Partial CDNA subset examples** (latency metrics not present on gfx940–gfx942):
- `VL1 Lat`: gfx908, gfx90a, gfx950
- `L2 Rd Lat`: gfx908, gfx90a, gfx950
- `L2 Wr Lat`: gfx908, gfx90a, gfx950
- `L1 Access Latency`: gfx908, gfx90a, gfx950
- `L1-L2 Read Latency`: gfx908, gfx90a, gfx950

### Panel-level coverage

20 panel files are not present in all architectures:

| Panel file | Present in | Missing from |
|---|---|---|
| `0500_command_processor_cpc.yaml` | gfx115x | all CDNA |
| `0500_command_processor_cpc_cpf.yaml` | all CDNA | gfx115x |
| `0700_wavefront.yaml` | all CDNA | gfx115x |
| `0700_workgroup_processor.yaml` | gfx115x | all CDNA |
| `0900_l0_cache.yaml` | gfx115x | all CDNA |
| `1100_compute_units_compute_pipeline.yaml` | all CDNA | gfx115x |
| `1300_l2_cache.yaml` | gfx115x | all CDNA |
| `1700_l2_cache.yaml` | all CDNA | gfx115x |
| `3000_mem_bw.yaml` | gfx950 only | all others |
| *(11 more — full list in analysis script)* | | |

---

## Step 5 — Sets cross-reference

- Total metric references across all set files: **141**
- Sets today use **positional numeric ids** (e.g., `2.1.3`), not metric names. Direct cross-referencing by name is not possible without resolving ids to names via the positional index scheme.
- This confirms that migrating sets to name-based references is a required step in the Layer 2 migration, and reinforces the open question about sets layer placement.

---

## Schema implications for FR2.1

| Field | Required / Optional | Rationale |
|---|---|---|
| `id` | Required | 95 metric names have special characters — ids cannot be derived from names |
| `name` | Required | User-facing display name, free to change independently of `id` |
| `description` | Optional today; recommended required | 10% missing — documentation debt |
| `unit` | Required | Drifted in 32 metrics — single declaration eliminates drift |
| `archs` or `implementations` | Required | 497 unique-to-one-arch; 59 Case A multi-impl |
| `peak` | Optional | Absent for some archs in 12 metrics |
| `pct_of_peak` | Optional | 50 metrics |
| `coll_level` | Optional | 12 metrics |
| Aggregation type (`avg/min/max` vs `value`) | Required | 21 metrics drift between types — must be explicit |
| Table-type-specific fields (`xfer`, `coherency`, etc.) | Typed variant | ~70 rows each — schema may need metric type variants |

## Cases requiring manual review before schema finalization

1. **39 metrics with both description and formula drift** — need to be classified as Case A (same metric, multiple impls) or Case B (separate metrics) by domain knowledge, not by automated analysis.
2. **43 "same description, different name" cases** — need canonical name selection.
3. **`units` field (42 rows)** — typo of `unit`; needs cleanup, not schema support.
