# Kernel corrections from ROCr gfx1201 qualification

## Wait for initial VM clears before CPU updates

`amdgpu-wait-initial-vm-clear.patch` targets the installed AMD DKMS package
`6.19.14-2364437.24.04`, file `amd/amdgpu/amdgpu_vm.c`, with base SHA-256
`dc2cfcf5c11b86dc16247dce712fe826c7de672cd27f8d3759cb7ca0a1a9d1e5`.

The initial SDMA root clear uses a BOOKKEEP fence that the existing implicit
wait excludes. The patch waits for it before switching to CPU updates. It
limits the new wait to that transition under the root reservation: a later
retry or competing acquisition can carry an unsignaled KFD eviction fence.
The original implicit wait remains on every CPU-mapping path.

The [trap investigation](../rocr-gfx1201-trap-recovery.md) records invalid live
trap page tables with libhsa, valid reference mappings, and a timing diagnostic
that restores both the mapping and passing Abort/Generic callbacks. That
supports the kernel race diagnosis. The hardware checks below now qualify the
patch for these tests on this machine.
No delay workaround was committed to the runtime.

Validation completed:

- Patch applies to the fingerprinted source without offsets or fuzz.
- Independent owner and curator reviews checked initial conversion,
  interruption, CPU-from-open, failed-initialization retry, and competing
  acquire paths. The first unconditional-wait draft was corrected before
  commit because it could wait on an eviction fence.
- The full DKMS module set built in an isolated source copy against
  Linux `6.17.0-23-generic`. The final revision recompiled `amdgpu_vm.o`,
  completed MODPOST, and linked `amdgpu.ko` successfully. BTF generation was
  skipped because `vmlinux` was unavailable. Build warnings were outside the
  changed translation unit; the compiler-name warning compared the same
  GCC 13.3.0 release under two executable names.
- The staged module and installed module have identical vermagic,
  dependencies, and all 1,155 imported symbol/version pairs.

Local artifacts, excluded from Git:

- `target/rocr-test-fixes/kernel/amdgpu.ko`: final module with debug info
  stripped; SHA-256
  `3307828a3f125908c436afdde1fc87be07b64570124623c7db3db459ebc1affb`;
  srcversion `B40219968217AB467B79120`.
- `target/rocr-test-fixes/kernel/amdgpu-original.ko.zst`: copy of the installed
  module for rollback, srcversion `88983A47E73AF492726AD7C`.
- `target/rocr-test-fixes/kernel/manifest.json` and
  `symbol-version-comparison.json`: identities and compatibility comparison.
- `target/rocr-test-fixes/recovery/kernel-fence-build.log` and
  `kernel-fence-build-final.log`: build results.
- `/tmp/rocr-amdgpu-vm-fence-build/`: configured source copy and full objects.

## Temporary-load hardware qualification

After user approval and an empty GPU handle audit, the original module was
unloaded without force and the staged module loaded. The device rebound,
`/dev/kfd` returned, and `/sys/module/amdgpu/srcversion` matched
`B40219968217AB467B79120`. The installed boot module was preserved. Secure Boot
is disabled on this machine; the staged module is unsigned.

The unchanged Rust library (SHA-256
`4cd94bf26ae6b64922adb47dc24fa1794a626b915a4c25ab64127a2269ee0245`)
passed NoTrap (0.415 s), Abort (0.415 s), and Generic (0.465 s), with no preload
or diagnostic delay. `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` isolates trap
handling from the separate GPU core-dump gap. A separate read-only diagnostic,
paused only after queue creation, found a stable valid trap leaf
`0x80000003daae40b1` with read and execute permission.

The 41 queue, lifecycle, memory access/atomic, and virtual-memory regression
cases completed with 39 passes and two expected fabric capability skips.
All four negative queue cases passed. The post-reload kernel delta contains
256 illegal-opcode messages from those malformed-queue cases and 16 workqueue
latency warnings. There are no new VM faults, MES timeouts, or resets. The
final GPU handle audit reports no users.

Evidence is in `kernel/reload-result.json`, `kernel/after-*-kernel.log`,
`kernel/regressions-kernel-delta.log`, `runs/patched-kmd-{notrap,abort,generic}/`,
`runs/patched-kmd-regressions/`, and
`recovery/patched-kmd-pte-notrap-trap-pte.json`. Plain run manifests record the
loaded driver identity and absence of a preload; the PTE diagnostic is separate.

The VM-clear-only module was later superseded by the combined module below.
The installed/on-disk module remains original and available for rollback;
a reboot restores that original driver.
This validation does not claim a persistent installation, stress coverage of
all VM-acquire interleavings, or GPU core-dump support.

## Checked SVM prefetch

`amdgpu-checked-svm-prefetch.patch` targets the same installed DKMS source.
It adds the explicitly identified local checked-prefetch protocol while
preserving legacy ioctl behavior. It is independent of the VM-clear patch;
the previously staged `amdgpu-svm.ko` includes both corrections.

The initial per-page-fill revision applied with zero fuzz and built
successfully. Its module imports, dependencies, and vermagic matched the
previously qualified module. Owner and independent curator reviews closed the
identified findings. After approval for that exact staged module, temporary
loading succeeded with an empty GPU-handle audit.

All eight bounded native/public-HSA probes pass, including real partial
migration and failed split preparation, with empty kernel deltas and GPU-handle
audits. The corrected ROCr SVM test exits zero and exercises both prefetch
subtests; its XNACK-dependent batch-discard section remains skipped.

See [the SVM correction](../rocr-gfx1201-svm-prefetch.md) for the contract,
module and source identities, checks, and remaining coverage limits. This
patch has no persistent installation or upstream UAPI allocation. The running
module is now `DE114CD2BB59945654162BC`; the VM-clear-only module and original
installed boot module remain available for rollback.

The current patch batches fresh destination pages by physical adjacency and
checks each fill of at most 256 MiB before committing migration. Initialized
pages, missing destinations, and nonmigrating pages split runs; physical VRAM
offset zero remains valid. The host regression compiles these two production
helpers from the patched source, using mocked submissions and fences:

```sh
sh runtimes/docs/testing/probes/test-checked-prefetch-batching.sh \
  /path/to/patched/amdgpu-source
```

It verifies exact coverage, four jobs for a contiguous 1 GiB range, bounded
sizes, mixed/reused-page selection, arithmetic checks, and drained ownership
on first/later submission and completion errors. The revised patch applies
with zero fuzz and builds against `6.17.0-23-generic`; its module retains the
same imports, dependencies, and vermagic. Its srcversion is
`58E64DFDCF5F47066CDC872`. This revision has not been loaded or tested on the
GPU. The earlier native results above qualify only `DE114CD2BB59945654162BC`;
they do not qualify the new batching implementation.
