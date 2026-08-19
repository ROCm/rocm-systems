# Multiprocess Conjugate-Gradient Sample

This profiling workload assigns two conjugate-gradient-inspired kernels to
independent GPU child processes:

- `kernel_spmv_csr` performs sparse matrix-vector multiplication over a CSR
  matrix with power-law row lengths.
- `kernel_cg_update_reduce` updates `x` and `r`, then performs a block-local
  residual reduction.

Each child uses private data and performs all rounds in one kernel launch. This
is not a convergent CG solver or correctness benchmark: children do not exchange
results, and kernel outputs are not copied back or verified.

## Build

From the repository root, run:

```bash
cmake -S . -B build -DENABLE_TESTS=ON
cmake --build build --target conjugate_gradient --parallel
```

The target creates three colocated files in `tests/`:

```text
tests/conjugate_gradient
tests/cg_module_a.hsaco
tests/cg_module_b.hsaco
```

The HSACO modules must target the runtime GPU. CMake detects the local
architecture when possible; when cross-compiling or building without a GPU,
set it explicitly, for example with `-DCMAKE_HIP_ARCHITECTURES=gfx942`.

## Run

Run the intended three-process workload:

```bash
./tests/conjugate_gradient \
  --processes 3 \
  --kernels spmv,spmv,update \
  --rotate-code-objects
```

This runs SpMV in children 0 and 1 and update/reduction in child 2. Children 0
and 2 load module A before module B, while child 1 reverses that order. Every
child launches its selected kernel from module A. The reversed order gives
module A a different process-local code-object ID in the two SpMV processes,
exercising PID-aware sample correlation.

List all options with:

```bash
./tests/conjugate_gradient --help
```
