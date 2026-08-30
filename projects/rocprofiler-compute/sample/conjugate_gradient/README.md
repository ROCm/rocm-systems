# Multiprocess Conjugate-Gradient Sample

A PC sampling workload in which the same kernel carries a different
process-local code-object ID in each of two processes.

Two conjugate-gradient-inspired kernels are built into separate shared
libraries:

- `kernel_spmv_csr` performs sparse matrix-vector multiplication over a CSR
  matrix with power-law row lengths.
- `kernel_cg_update_reduce` updates `x` and `r`, then performs a block-local
  residual reduction.

The driver forks once, and both processes run both kernels in opposite order.
HIP defers loading a code object until the first launch out of it, so the
process that runs SpMV first gives it the lower ID and the other process gives
it the higher one. That divergence is what makes the pair useful for checking
that sample attribution keys on the process as well as the code object.

The kernels live in separate libraries rather than in the driver because
kernels compiled into one binary share a single fat binary, which the runtime
loads as one code object holding both symbols.

This is not a convergent CG solver or a correctness benchmark: the processes do
not exchange results, and kernel outputs are neither copied back nor verified.

## Build

From the repository root, run:

```bash
cmake -S . -B build -DENABLE_TESTS=ON
cmake --build build --target conjugate_gradient --parallel
```

The build populates `tests/conjugate_gradient/` with the driver and the two
kernel libraries. CMake selects the GPU architecture; when cross-compiling or
building without a GPU, set it explicitly with
`-DCMAKE_HIP_ARCHITECTURES=gfx942` or similar.

## Run

The workload takes no arguments:

```bash
./tests/conjugate_gradient/conjugate_gradient
```

Each process prints its PID, its role, and the order it launched the kernels in.
