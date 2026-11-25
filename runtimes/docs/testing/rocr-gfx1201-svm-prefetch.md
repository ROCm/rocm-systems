# ROCr SVM prefetch: checked migration completion

Before these corrections, `rocrtstFunc.SvmMemory_Basic_Test` failed on gfx1201 at
`projects/rocr-runtime/rocrtst/suites/functional/svm_memory.cc:558` because
`hsa_amd_svm_prefetch_async` returns 47 (`HSA_STATUS_ERROR_NOT_SUPPORTED`).
This was reproduced with libhsa revision
`b71be8828809c2dcd56f2e5f6e86d0007ace61cf` on November 19, 2026. The preceding
SVM host-memory read/write checks pass. Logs and the library hash are in
`target/rocr-test-fixes/runs/after-modern-profile/` in the test workspace.

This was a backend capability gap, previously documented in the libhsa README.
The [HSA entry point](../../ddi/libhsa/src/api/memory/copy_svm.rs) requests
`PrefetchCompletion::Migration` before accepting asynchronous work. The original
[Linux KFD provider](../../ddi/rocddi/src/driver/linux_kfd/advice.rs) advertised
only `PrefetchCompletion::Attempt`. The
[shared validation](../../ddi/rocddi/src/memory/advice.rs) rejects the stronger
request, so this failing call submits neither a prefetch ioctl nor a worker.
The asynchronous worker independently requests the same guarantee.

The [public HSA extension](../../api-headers/include/hsa/hsa_ext_amd.h), under
`hsa_amd_svm_prefetch_async`, promises that completion is decremented when
migration completes. Same-source C++ ROCr's `Runtime::SvmPrefetch` calls
`hsaKmtSVMSetAttr(PREFETCH_LOC)` and decrements after the call returns. This
establishes a compatibility difference, but not confirmed migration.

The exact driver source used for the earlier VM-clear-only baseline,
`/usr/src/amdgpu-6.19.14-2364437.24.04`, confirms the original provider
limitation. At that baseline, the KFD SVM sources were identical in the
installed tree and isolated build: the trap correction changed only
`amdgpu_vm.c`. The combined module qualified below additionally changes KFD
migration and SVM handling.

A concrete single-GPU failure path is:

1. `amd/amdkfd/kfd_migrate.c:541–547` returns `-ENOSPC` when VRAM reservation
   fails for a requested migration.
2. `kfd_svm.c:3625–3629` records the failed `svm_migrate_to_vram` result only in
   its internal `migrated` output, then returns zero.
3. For an existing range whose prefetch request changes no access mapping,
   `kfd_svm.c:3806–3808` takes the `!migrated && !update_mapping` path. The
   SVM ioctl can therefore return success without migrating the range.

The migration code also collapses nonnegative partial-page counts to zero at
`kfd_migrate.c:591` for GPU placement and `:907` for CPU placement. Waiting for
submitted copy fences at `:199–209` does not establish that every requested
page migrated. Querying `PREFETCH_LOC` cannot close that gap: `kfd_svm.c:3959–3967`
returns the preference stored before migration at `:779–781`. Other source
paths can substitute CPU placement, but their particular topology conditions
are not established on this single-GPU machine.

Source SHA-256 identities:

- `kfd_svm.c`: `fa4fbc726d92b4a40d131fe35bc8861ca2aa8b1c6807dfd5d6e7eada2f8a1879`
- `kfd_migrate.c`: `f811a0b22788814d29537de632c8535d74ca8c0c197bed10f7c613aa1ba8bb15`

This supersedes the earlier corroboration from a separately pinned kernel
checkout. It is source evidence tied to the built driver, not a physical page
placement measurement or an injected VRAM-exhaustion experiment.

Changing the HSA call to request only `Attempt` would weaken its modeled
completion guarantee. The original driver therefore retains the explicit unsupported
result. The checked provider operation described below implements the stronger
contract. Its initial per-page-fill revision was qualified on the gfx1201
test machine; the batching revision below has build and host coverage only.
This is a provider limitation, not an instruction-set exclusion for gfx1201.

The unchanged ROCr test is insufficient to qualify that operation: its
post-prefetch attribute query includes the set-only `AGENT_ACCESSIBLE`
attribute, ignores the query status, and checks a preferred-location value
already initialized to the GPU. Validation needs an independent migration
completion check and failure/partial-migration cases in addition to API
acceptance and completion signaling.

## Implemented correction

The [checked-prefetch kernel patch](patches/amdgpu-checked-svm-prefetch.patch)
adds an opt-in synchronous operation, while preserving the legacy best-effort
operation. An output-only extension of `GET_VERSION` identifies the local
protocol and feature bit. This is an experimental local protocol allocation,
not a claim of upstream KFD support. Old eight-byte version clients keep their
prefix; an extended query to the old driver returns a zero extension.

The strict operation accepts exactly one prefetch-location attribute. It:

- Propagates migration errors and rejects a destination fallback.
- Checks every submitted GART and copy fence before committing destination
  pages, including draining a GART update when copy submission fails.
- Explicitly zeroes untouched destination pages with checked fill completion,
  covering both fresh backing and holes in reused backing. The current patch
  coalesces physically adjacent eligible destinations and limits each job to
  256 MiB, checking completion before submitting the next job.
- Restores source mappings on failed copies, using recorded destination pages
  for cleanup rather than treating DMA addresses as physical page identities.
- Recounts every cloned split piece through a stable HMM snapshot, preserves
  earlier successful VMA migrations on later failure, and safely releases empty
  backing. Unpublished split pieces are unlinked before transaction cleanup.
- Verifies every requested page with an independent HMM query that does not
  fault or migrate pages. Device identity and backing must match the requested
  destination. CPU requests populate absent pages only after confirming no
  device pages remain, then undergo the same independent verification.

A partial failure can leave successfully migrated pages in their new location;
there is no rollback promise. Success establishes placement at completion,
not permanent residency after subsequent eviction or application access.

rocddi negotiates this protocol on the existing native endpoint and advertises
`Migration` only when it matches. Inactive contexts and old drivers retain
`Attempt`; negotiation errors propagate. Strict failures never fall back to
the legacy ioctl. Host-only commit/discard validation remains independent of
KFD activation. libhsa retains its original admission requirement and negative
asynchronous completion behavior.

The [separate harness fix](rocr-gfx1201-svm-harness.md), commit `c1f7626a455`,
rejects negative completion, bounds polling, and checks the prefetch query.

Validation of the initial per-page-fill revision:

- Owner and independent curator reviews closed the identified lifetime,
  accounting, cleanup, process-ID, ABI, and host-only capability regressions.
- The patch applies with zero fuzz to the exact installed source package.
- The isolated driver build succeeds against `6.17.0-23-generic`. The staged
  module has identical vermagic, dependencies, and imported symbol versions
  to the previously qualified module. Existing display-code warnings and the
  compiler executable-name warning are outside changed code.
- libhsa, rocddi, and rocloader host tests: 660 pass, seven existing ignored.
  Native encoding tests cover exact protocol matching, endpoint caching,
  strict error propagation, and unchanged legacy encoding. Real host mappings
  cover commit/discard without activating a GPU under both context lifetimes.
- CTest: ten pass and two expected sandbox capability skips. The CPU-only
  prefetch test still rejects admission without changing either signal.
- Warnings-denied Clippy and formatting checks pass.

Staged module: `target/rocr-test-fixes/kernel/amdgpu-svm.ko`, srcversion
`DE114CD2BB59945654162BC`, SHA256
`108d7ed7d2d4111eb6c9680a631b6e1092168d1edf64b376ab2a92f2b5f60664`.
Source fingerprints and ABI compatibility checks are in `kernel/svm-manifest.json`.
The earlier VM-clear correction is included in this module.

## Hardware qualification of the initial per-page-fill revision

With approval for the exact staged module, the VM-clear-only module was
unloaded normally after an empty GPU-handle audit. The combined module
`DE114CD2BB59945654162BC` loaded and the GPU rebound. The installed boot module
was preserved; reboot restores the original driver. The previous qualified
VM-clear-only module is retained as a rollback artifact.

All eight [native/public-API probes](probes/kfd-checked-prefetch.c) pass:

| Probe | Result |
|---|---|
| Untouched GPU pages | Two GPU/CPU round trips preserve zero contents. |
| Untouched CPU pages | Explicit CPU prefetch populates readable zero pages. |
| Initialized data | Two GPU/CPU round trips preserve every byte. |
| Subrange without access mappings | Interior CPU/GPU migration and whole-range return preserve data, exercising split counts without DMA arrays. |
| Pinned partial migration | An unread `vmsplice` pipe pins one page. GPU prefetch returns `EAGAIN` (11); CPU recovery preserves data; releasing the pin permits a successful retry. |
| Multiple VMAs | A protection boundary splits the range; repeated round trips preserve data across both VMAs. |
| Failed split recount | A protected excluded head causes `EFAULT` (14) for interior prefetch. Restoring protection permits whole-range migration and correct data, exercising transaction cleanup. |
| Public HSA completion | A dependency holds completion at one. After release, real pinned-page partial migration produces negative completion (-1). Unpinning permits successful GPU and CPU prefetch; the destination query and data checks pass. |

Every probe ran in its own bounded process, serially. Kernel deltas were empty
and all GPU-handle audits were empty. Logs and checks are in
`target/rocr-test-fixes/svm-native-*.log`, `kernel/svm-native-health.json`,
`kernel/svm-frontend-health.json`, and `kernel/svm-reload-result.json`.

The corrected `rocrtstFunc.SvmMemory_Basic_Test` now exits zero in 0.515 seconds
without a preload. Both GPU prefetch subtests complete and pass their query
assertions. It retains partial coverage because the separate batch-discard
section requires XNACK, which is disabled. Its kernel delta and GPU-handle
audit are empty (`runs/svm-checked`, `kernel/svm-rocr-health.json`). A subsequent
fresh 133-case sweep repeats this result in 0.465 seconds and completes with
118 passes, 15 skips/partial outcomes, and zero failures. All trap, core-dump,
and malformed-queue cases remain passing; see the [full report](rocr-gfx1201-report.md).

These checks qualify gfx1201 on this exact driver/kernel tuple. They do not
establish multi-GPU or CRIU-restore behavior, pressure/eviction stress, or
execution-error injection into an SDMA fence. Those cleanup paths have source
review coverage; the native negative tests exercise partial placement and
failed range preparation. No persistent installation is claimed.

Build the probe from `runtimes/` with the qualified runtime build directory:

```sh
cc -std=c11 -Wall -Wextra -Wno-unused-parameter -I api-headers/include \
  docs/testing/probes/kfd-checked-prefetch.c \
  -L target/rocr-test-fixes/runtime/lib -lhsa-runtime64 \
  -o target/rocr-test-fixes/svm-native-probe
```

Its cases are `untouched`, `cpu-untouched`, `data`, `split`, `partial`,
`multi-vma`, `split-error`, and `frontend-error`. Each runs in a separate process with a
20-second alarm; the prepared serial runner adds an external watchdog and
kernel/handle audits. The probe contains no driver-loading code.

## PR review: batch checked zero fills

The initial patch submitted and waited for one fill per untouched page.
For a contiguous 1 GiB migration with 4 KiB pages, this means 262,144 jobs and
waits. This is a source-derived operation count, not measured elapsed time.
The current patch scans prepared destination PFNs after successful copies,
coalesces only physically adjacent destinations without source contents, and
fills each run in checked jobs of at most 256 MiB. A contiguous 1 GiB run now
requires four jobs. A hole, initialized source, nonmigrating source, or physical
discontinuity breaks the run. Offset zero is not a missing-page sentinel.

Each job retires outside the entity lock before the next submission. Native
submission and fence errors propagate to the existing strict abort/finalize
path before `migrate_vma_pages` can commit destinations. Earlier successful
chunks are already drained if a later submission fails.

The [host batching regression](probes/test-checked-prefetch-batching.sh)
extracts and compiles the actual two helpers from a patched source tree. It
passes coverage/count, partial-chunk, mixed/reused-page selection, valid-zero,
alignment/overflow, first/later submission, and completion-error checks.
Mock fence assertions require an unlocked wait and no outstanding work at
cleanup. These checks establish host control flow, not SDMA execution.

The regenerated patch applies with zero fuzz to the installed source package
and reproduces all seven patched files byte for byte. A separate full Kbuild
against `6.17.0-23-generic` passes. The new module has identical vermagic,
dependencies, and imported symbol versions, srcversion
`58E64DFDCF5F47066CDC872`, and SHA-256
`fc4f7fe09ca4474a1e9d9199b9a5d6ad6276a4b91adb1722c79f251ec59d9fe4`.
The build and source manifest are in `/tmp/rocddi-pr16293-kernel/` as
`build-batched.log` and `batching-manifest.json`.

This new module has not been loaded or hardware-qualified. The running module
remains `DE114CD2BB59945654162BC`, and the installed boot module is unchanged.
The eight native probes and the 133-case sweep above refer to the earlier
per-page-fill module. Reused-backing, large-range migration, and native fence
failure behavior on the batched module remain validation gaps.
