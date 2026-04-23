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
- gfx_id: "gfx950"
  name: "MI350X"
  family: "CDNA4"
  compute_capability: 9.0
  peak_fp32_tflops: 163.4
  peak_fp64_tflops: 81.7
  memory_bandwidth_gbs: 432
  l2_cache_kb: 8192
  lds_kb_per_cu: 160
  num_cus: 152
  wavefront_size: 64
  vgpr_per_cu: 10240
  sgpr_per_cu: 10240
  lds_banks: 32
  source: "AMD MI350X data sheet v1.0"
```

### Entry in `vgpr_occupancy_tables.yaml`

```yaml
- gfx_id: "gfx950"
  tables:
    - vgprs: 64
      wave_occupancy: 0.5
      waves_per_cu: 8
    - vgprs: 128
      wave_occupancy: 0.25
      waves_per_cu: 4
    # ... more entries
```

## Schema constraints (CI-enforced)

- `gpu_specs.yaml` validated against JSON schema
- `vgpr_occupancy_tables.yaml` validated
- VGPR table entries ordered by VGPR count (ascending)
- All peak specs (TFLOPS, bandwidth) must cite an AMD data sheet

## Key specs to get right

- **MI300X FP64:** 81.7 TFLOPS (NOT 163.4 — that's FP32)
- **CDNA4 LDS:** 160 KB/CU (not 64)
- **Compute capability:** must match Khronos / AMD numbering

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
- VGPR table must be complete (cover typical code patterns: 64, 96, 128, 192, 256 at minimum)
- Compute capability follows Khronos spec (e.g., CDNA1=9.0, CDNA2=9.0, CDNA3=9.0, CDNA4=9.0)
- LDS must match official spec (check against latest AMD app notes)

## Related docs

- GPU schema (spec §6.2 Task 2)
- AMD GPU data sheets: https://www.amd.com/en/products/specifications/processors
- CDNA architecture docs: https://rocmdocs.amd.com
