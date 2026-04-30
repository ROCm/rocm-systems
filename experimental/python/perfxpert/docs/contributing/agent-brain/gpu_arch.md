# Contributing: new GPU architecture

## What you're adding

Support for a new GPU architecture (e.g., a new CDNA variant, RDNA revision,
or vendor GPU). Requires entries in two knowledge YAMLs and corresponding
test updates.

## File locations

- GPU specs: `perfxpert/knowledge/gpu_specs.yaml`
- VGPR occupancy tables: `perfxpert/knowledge/vgpr_occupancy_tables.yaml`
- Tests: `tests/test_knowledge/test_gpu_specs.py`

## Template

### Entry in `gpu_specs.yaml`

```yaml
gfx950:
  name: "MI350X"
  codename: "CDNA4"
  peak_fp64_tflops: 72.1
  peak_fp32_tflops: 144.2
  peak_fp16_tflops: 2300.0
  peak_bf16_tflops: 2300.0
  peak_fp8_tflops: 4600.0
  peak_int8_tops: 4600.0
  memory_bandwidth_tbs: 8.0
  cu_count: 256
  lds_kb: 160
  lds_per_cu_kb: 160
  wave_size: 64
  max_vgprs_per_thread: 256
  vgprs_per_simd: 512
  simds_per_cu: 4
  max_waves_per_simd: 8
  ridge_point: 18.0
```

### Entry in `vgpr_occupancy_tables.yaml`

```yaml
gfx950:
  - {max_vgprs: 64,  waves_per_eu: 8}
  - {max_vgprs: 80,  waves_per_eu: 6}
  - {max_vgprs: 96,  waves_per_eu: 5}
  - {max_vgprs: 128, waves_per_eu: 4}
  - {max_vgprs: 160, waves_per_eu: 3}
  - {max_vgprs: 256, waves_per_eu: 2}
```

## Schema constraints (CI-enforced)

- `gpu_specs.yaml` validated against JSON schema
- `vgpr_occupancy_tables.yaml` validated
- VGPR table entries ordered by VGPR count (ascending)
- All peak specs (TFLOPS, bandwidth) must cite an AMD data sheet

## Key specs to get right

- **MI300X FP64:** 81.7 TFLOPS (NOT 163.4 — that's FP32)
- **MI300X occupancy:** 32 max waves/CU and 4 SIMDs/CU, so 8 waves/SIMD
- **CDNA4 LDS:** 160 KB/CU (not 64)
- **MI350X public peaks:** 72.1 FP64 TFLOPS, 144.2 FP32 TFLOPS, 8.0 TB/s HBM

## Tests you must add

Update `tests/test_knowledge/test_gpu_specs.py`:

- `test_new_gfx_id_loads()` — YAML loads
- `test_new_gfx_id_has_required_fields()` — all mandatory fields
- `test_new_gfx_id_tflops_reasonable()` — sanity check on compute
- `test_new_gfx_occupancy_table_ordered()` — vgpr table sorted

## Review requirements

- 1 core reviewer
- Data sheet citation must be verifiable (AMD public data sheet or internal memo)
- CI green (schema + tests)

## Common pitfalls

- Don't confuse FP32 and FP64 TFLOPS (MI300X is 163.4 FP32, 81.7 FP64)
- Don't use MI355X higher-bin peaks for the `gfx950` MI350X default unless you
  explicitly split the SKU key first
- VGPR table must be complete (cover typical code patterns: 64, 96, 128, 192, 256 at minimum)
- LDS must match official spec (check against latest AMD app notes)

## Related docs

- GPU schema (spec §6.2 Task 2)
- AMD GPU data sheets: https://www.amd.com/en/products/specifications/processors
- CDNA architecture docs: https://rocmdocs.amd.com
