# mini-nbody (health workload)

Vendored subset of [ROCm/HIP-Examples `mini-nbody`](https://github.com/ROCm/HIP-Examples/tree/master/mini-nbody)
(Apache-2.0; see `LICENSE`).

Used by SPP health scripts (`scripts/run_spp_health_gfx942.sh`):

```bash
hipcc -O2 -std=c++17 -I../ -DSHMOO --offload-arch=gfx942 \
  nbody-block.cpp -o nbody-block
./nbody-block 131072 500
```

## Local patches (vs upstream)

| Change | Why |
|--------|-----|
| `argv[2]` = iteration count | Health scripts pass `131072 ${ITERS}`; upstream hardcoded `nIters = 10` |
| Host buffer via `hipHostMalloc` | Pinned staging for H2D/D2H each iteration |
| Device `hipMalloc` once outside the loop | Stable device VA across iterations (no alloc/free among iters) |

Kernel traffic stays on device memory (`hipMalloc`); host pin is staging only
(does not switch the workload to IO/zero-copy paths).
