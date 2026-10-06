<!-- SPDX-License-Identifier: MIT -->

# GFX1201 HSA hardware checks

After the [CMake build](../../README.md#build-and-test), run
`./run_gpu_smokes.sh /tmp/rocddi-cmake/lib` on a machine with a GFX1201
GPU, accessible `/dev/kfd` and DRM render node, a C compiler, and GNU
`timeout`. The script verifies with `ldd` that each C probe resolves the
staged image. The callback probe performs ten cycles where one async
signal handler shuts down HSA while another registration remains active,
then waits for successful reinitialization.

The PC sampling probe checks that system and agent extension queries do not
advertise sampling and that direct entry points reject configuration and
creation without changing the output handle. Each probe has a 30-second
subprocess watchdog.

The logging probe verifies that a non-null caller-owned C `FILE*` is
reported as unsupported without being retained. It then enables null-stream
logging, creates and destroys a GPU queue, and checks that stderr contains the
queue creation record.

Run `./run_kernel_comparison.sh /tmp/rocddi-cmake/lib/libhsa_runtime64.so`
with `hipcc`, `clang-offload-bundler`, and installed ROCr to compare a real
GFX1201 HSA workload. The script compiles the checked-in HIP kernel to one
code object, then runs separate 40-second guarded processes against rocddi and
ROCr. Each
process selects an allocatable fine-grained CPU kernarg pool, grants GPU
access to output and arguments, loads and freezes the code object, and
submits 16 kernel dispatch packets. Every completion must increment the
host-visible output exactly once. The script compares the results and prints
library, code-object, and probe hashes. This checks one dispatch and memory
path on the current GPU; it does not cover other kernels or hardware.
