# RCCL Kernel-Count Guards

CPU-only [pytest](https://pytest.org) guards that catch **kernel leaks** — an
unintended growth in the set of device kernels emitted by RCCL's code
generators. Every extra kernel costs binary size and device-linker build time,
so an unreviewed increase should fail CI and force the author to update a
baseline and justify the growth.

The guards run the generators directly (no GPU, no built RCCL library — just
`python3`) and compare the generated kernel set against committed baselines in
three complementary layers:

1. **per-dimension value-sets** — names the culprit axis when a new
   type/protocol/unroll/algorithm appears (root cause);
2. **per-collective counts** — where the growth landed; catches rule
   relaxations that reuse existing axis values;
3. **grand total** — the net effect a reviewer reads at a glance.

## Coverage

| File | Generator guarded |
|------|-------------------|
| `tests/test_kernel_counts.py` | `src/device/generate.py` (main combinatorial generator), both `ENABLE_ROCSHMEM` OFF and ON |
| `tests/test_symmetric_kernels.py` | `src/device/symmetric/generate.py` |

`src/device/ce_reduce/generate.py` is out of scope: it already has an equivalent
guard in `src/device/ce_reduce/test_generate_ce_reduce.py`.

## Running locally

```bash
cd projects/rccl/test/kernel-count
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest -v
```

## How it runs in CI

Two independent lanes execute exactly these tests:

1. **test_runner** — registered as an `is_pytest` suite in
   `tools/scripts/test_runner/configs/mi300x_mellanox_ib.json` (`test_dir:
   test/kernel-count`, `setup_venv: true`).
2. **Host unit-test workflow** — the `guards` phase of
   `test/host/run_host_tests.sh`, folded into its `run` phase, so every PR
   touching `projects/rccl/**` runs them (no GPU required).

## When a guard fails

The failure prints the total delta, the per-collective delta, and any changed
dimension values, followed by an ACTION line. If the change is intentional,
update the `EXPECTED*` constants in the relevant test file **and explain in the
PR description why the kernel count changed**. Do not blind-update the numbers.

Baselines are seeded from the multi-arch superset
(`BUILD_LOCAL_GPU_TARGET_ONLY=OFF`, arch-guarded variants included), so they are
deterministic and machine-independent; a local-arch-only build legitimately
emits fewer kernels.
