# Multiprocess Conjugate-Gradient Sample

This sample runs two GPU kernels taken from a conjugate-gradient (CG) iteration:

- `kernel_spmv_csr` performs sparse matrix-vector multiplication over a CSR
  matrix with power-law row lengths.
- `kernel_cg_update_reduce` updates `x` and `r`, then performs a block-local
  residual reduction.

The sample assigns these phases to independent GPU child processes and can
vary module load order between children. It is an iteration fragment, not a
convergent or cooperative CG solver. The child processes do not exchange data,
and there is no inter-process communication, CPU reference calculation, result
copy-back, or numerical verification. A successful exit establishes only that
setup and GPU execution completed, so this is not a correctness benchmark.

## Build

### CMake

From the repository root, enable and install the test collateral:

```shell
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DENABLE_TESTS=ON \
  -DINSTALL_TESTS=ON
cmake --build build --target conjugate_gradient --parallel
```

The target creates these three colocated files in `tests/`:

```text
tests/conjugate_gradient
tests/cg_module_a.hsaco
tests/cg_module_b.hsaco
```

Both HSACO files contain both kernels and use the same
`CMAKE_HIP_ARCHITECTURES` setting as the executable. CMake detects the local
architecture when possible. On a GPU-less build host or when cross-compiling,
set an explicit target such as `-DCMAKE_HIP_ARCHITECTURES=gfx942`; the
resulting executable and modules must match the GPU on which the sample runs.

To install the sample after configuring with `INSTALL_TESTS=ON`, run:

```shell
cmake --build build --target install --parallel
```

The executable and both modules are installed together under
`${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-compute/tests`
(normally `install/libexec/rocprofiler-compute/tests` for the command above),
so default module discovery continues to work. The `.cpp`, `.hip`, and shared
`cg_args.hpp` sources are installed under
`install/share/rocprofiler-compute/sample/conjugate_gradient`.

### Manual build

The same layout can be built without CMake. Run these commands from the
repository root and replace `native` with a GFX target when necessary:

```shell
CG_BUILD_DIR="$PWD/build-cg"
CG_ARCH=native
mkdir -p "$CG_BUILD_DIR"

amdclang++ -std=c++17 -g -O2 -x hip \
  sample/conjugate_gradient/conjugate_gradient.cpp \
  -o "$CG_BUILD_DIR/conjugate_gradient"

amdclang++ -std=c++17 -g -O2 -x hip --offload-device-only \
  --offload-arch="$CG_ARCH" \
  sample/conjugate_gradient/cg_kernels.hip \
  -o "$CG_BUILD_DIR/cg_module_a.hsaco"

amdclang++ -std=c++17 -g -O2 -x hip --offload-device-only \
  --offload-arch="$CG_ARCH" \
  sample/conjugate_gradient/cg_kernels.hip \
  -o "$CG_BUILD_DIR/cg_module_b.hsaco"
```

## Run

The public options are:

| Option | Default | Meaning |
|---|---:|---|
| `-p`, `--processes N` | `2` | Number of independent child processes. |
| `-k`, `--kernels LIST` | `spmv,update` | Comma-separated child assignments; the list repeats modulo its length. |
| `-r`, `--rounds N` | `400` | Iterations performed inside the selected kernel launch. |
| `-n`, `--rows N` | `65536` | Number of generated CSR matrix rows. |
| `-s`, `--seed N` | `12345` | Seed for deterministic matrix generation. |
| `-d`, `--device N` | `0` | Base device ordinal; child `i` uses `(N + i) % device_count`. |
| `--rotate-code-objects` | off | Reverse A/B module load order for odd-numbered children. |
| `--module-dir PATH` | executable directory | Directory containing both `cg_module_a.hsaco` and `cg_module_b.hsaco`. |
| `-h`, `--help` | off | Print command-line help. |

For the intended three-process mix, run:

```shell
./tests/conjugate_gradient \
  --processes 3 \
  --kernels spmv,spmv,update \
  --rotate-code-objects
```

This deliberately omits `--rounds`, exercising the default of 400. Children 0
and 2 load modules in A,B order; child 1 loads them in B,A order. Every child
still resolves and launches its selected kernel from module A. Module B is
loaded only to perturb process-local code-object ID assignment. Consequently,
the two SpMV processes can execute the same module-A symbol with different
numeric code-object IDs; the literal ID values are not stable interfaces.

Module lookup is independent of the current working directory. By default the
program locates both modules beside the resolved executable; use
`--module-dir` after separating the executable from its modules:

```shell
/path/to/conjugate_gradient --module-dir /path/to/cg-modules
```

The parent makes no HIP calls, then creates each child with `fork()` followed
by `execv("/proc/self/exe", ...)`. The sample therefore requires Linux with a
mounted `/proc` filesystem, including when it runs inside a container.

## Workload behavior

Each child constructs the same deterministic CSR matrix and vector inputs from
the configured row count and seed. The selected kernel performs all configured
rounds inside one launch:

- SpMV uses power-law row lengths and indirect `p[column]` reads, producing an
  irregular global-memory access pattern and different loop lengths across
  lanes.
- Update/reduction streams through `p`, `q`, `x`, and `r`, then reduces one
  partial residual per block through shared memory and barriers.

The block partials are intentionally not combined on the host. Processes run
their assigned phase independently, and their arrays are not shared.
