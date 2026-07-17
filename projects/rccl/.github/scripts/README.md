# RCCL multi-node workload runner

Stdlib-only Python orchestration for running GPU workloads across N nodes on a SLURM cluster via mnctl-launched ROCm containers. The workflow (and `run_coverage.sh`) invoke it.

## Layout

```
.github/scripts/
├── ci_targets.yml            # DATA: clusters + targets (edit here; needs PyYAML)
├── ci_configure.py           # loads/validates ci_targets.yml -> build matrix
├── run_coverage.sh           # entry -> python run_workload.py --workload coverage
├── run_workload.py           # entrypoint: pick payload by --workload/$WORKLOAD
├── orchestrator.py           # RunConfig (dataclass) + Orchestrator (generic flow)
└── payloads/
    ├── base.py               # Payload ABC (plugin contract)
    ├── coverage.py           # CoveragePayload
    └── __init__.py           # REGISTRY: workload name -> class
```

## Run

From `projects/rccl` (or anywhere — paths resolve from the script location):

```bash
WORKLOAD=coverage ALLOC_MODE=existing GPU_ARCH=gfx942 TEST_SUITE=ubr_multi_node \
  python3 .github/scripts/run_workload.py
# or the stable entry the workflow uses:
bash .github/scripts/run_coverage.sh
```

Environment variables: `ROCM_IMAGE`, `NODES`, `PARTITION`, `ACCOUNT`, `GPUS_PER_NODE`, `TIME_LIMIT`, `RESERVATION`, `ALLOC_MODE`, `RCCL_DIR`, `MNCTL_DIR`, `RCCL_TESTS_DIR`, `RCCL_TEST_MPI_HOSTFILE`, `REGISTRY_USER`/`REGISTRY_TOKEN`, `MNCTL_REBUILD`, `SHARED_FS_ROOT`. Output lands in `rccl_test_artifacts_<RUN_ID>_<timestamp>/`.

## Adding a workload

Add `payloads/<name>.py` with a `Payload` subclass and register it in `payloads/__init__.py:REGISTRY`. The contract is documented in `payloads/base.py`.
