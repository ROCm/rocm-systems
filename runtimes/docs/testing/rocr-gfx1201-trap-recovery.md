# ROCr shader traps: initial kernel VM-clear race on gfx1201

The November 19, 2026 investigation identified a missing synchronization in
this machine's installed amdgpu driver. Its initial SDMA page-table clear can
race CPU page-table updates during KFD VM acquisition. The libhsa process's
reserved first-level trap address has an invalid root entry before its first
queue executes. Reference ROCr reaches a valid executable mapping. A diagnostic
100 ms delay before `ACQUIRE_VM` restores the mapping with the unchanged Rust
runtime binary. Both Abort and Generic then pass with that same diagnostic
delay and produce no new kernel messages.

The reviewed kernel correction is now temporarily loaded and **all three
shader trap tests pass without the diagnostic delay**. NoTrap and Abort each
pass in 0.415 s; Generic passes in 0.465 s. All use the unchanged runtime
binary, no preload, and `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`. The installed
boot driver remains original. No sleep or speculative runtime workaround was
retained.

## Controlled runtime comparison

The earlier controls were contaminated by preceding failed recovery and four
orphaned HIP test children. Those children were identified and terminated.
After explicit authorization for controlled recovery, each runtime comparison
below started after a successful device-specific MODE1 reset, with no GPU
users reported by `fuser`.

All cases used `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` to isolate trap delivery.
Reference ROCr came from the same source tree. The Rust binary was built at
`784500683cc`; each run retains its own library snapshot and hash.

| Independently recovered state | NoTrap | Abort |
|---|---|---|
| Reference ROCr, default debug setting | Pass, 0.415 s | Pass, 0.415 s |
| Rust libhsa | Pass, 0.415 s | Fail, 5.923 s; no callback |
| Reference ROCr, `HSA_ENABLE_DEBUG=1` | Pass, 0.415 s | Pass, 0.415 s |

The Rust failure reports an SQC instruction-fetch fault at
`0x0000ffffffbf0000`, followed by MES `REMOVE_QUEUE` and `SUSPEND` timeouts.
This is KFD's reserved first-level CWSR trap region:

```
2^48 - 2 MiB CSA - 2 MiB SEQ64 - 64 KiB TRAP = 0x0000ffffffbf0000
```

A monotonic native-call trace places the fault at `179684.502003`, 18 ms after
the final temporary publication queue's destroy request, and before the
five-second test callback timeout. It is not a fault first introduced by
that test's final cleanup.

MES ADD_QUEUE snapshots contain the same trap address, TMA address
`0xffffffbf1800`, VM control `0x03fffc07`, and flags `0x1e00` for both runtimes.
The second-level gfx12 trap binary also matches the reference binary.

## Live page-table evidence

The read-only diagnostic pauses the test after queue creation. It verifies the
process executable and its KFD PASID, finds that PASID's MES page-directory
address, and follows only the entries for the reserved trap VA while the
process remains alive. It reads eight bytes per table level and rejects
unsupported/system/large-page descriptors. The later Rust and delayed controls
also reread the entries to check that they stayed stable during the snapshot.

The running device reports four levels with nine-bit table indices. The trap
VA selects indices 511, 511, 509, and 496. MMHUB 4.1.0's discovered register
base is `0x1a000`; `MMMC_VM_FB_OFFSET` at dword `0x1a4c7` reads zero. Thus the
non-system table addresses translate directly to offsets in `amdgpu_vram`.
No historical page-directory address is read after its test exits.

| Control | Reserved trap mapping before dispatch |
|---|---|
| Rust, normal initialization | Root entry `0x8000000000000000`: invalid |
| Rust, before the first temporary queue executes | Same invalid root entry; stable reread |
| Reference ROCr | All levels valid; leaf `0x80000003f7cfc0b1`, readable and executable |
| Rust, 100 ms delay before ACQUIRE_VM | All levels valid; same leaf flags and value; stable reread |
| Rust, patched kernel, no pre-acquire delay | All levels valid; leaf `0x80000003daae40b1`, readable and executable; stable reread |

The repeated physical page-directory address is allocator reuse across
separate processes, not shared VM ownership. Stored page-table contents do not
establish the instantaneous state of GPU translation caches. Here the invalid
root entry, dispatch-time fault, and source-level race agree.

## Kernel mechanism

The original amdgpu module and installed artifact report version
`6.19.14.31400000`, srcversion `88983A47E73AF492726AD7C`, on Linux
`6.17.0-23-generic`; KFD reports ABI 1.23. The source is
`/usr/src/amdgpu-6.19.14-2364437.24.04`. This machine has fully visible VRAM and
`vm_update_mode=-1`, selecting CPU updates for compute under that source's
default policy.

1. `amdgpu_vm_pt_clear` submits the initial root clear through SDMA. On gfx12
   its invalid non-leaf value is exactly `0x8000000000000000`.
2. `amdgpu_vm_sdma_commit` records that fence on the root reservation with
   `DMA_RESV_USAGE_BOOKKEEP`. The initial clear passes no separate output-fence
   pointer, so that reservation is its relevant synchronization record.
3. `amdgpu_vm_make_compute` calls `amdgpu_bo_sync_wait` before changing to CPU
   updates. Despite its comment, that helper reaches `amdgpu_sync_resv`, which
   iterates only `DMA_RESV_USAGE_READ` and lower. It excludes BOOKKEEP fences
   before applying owner filtering.
4. The conversion replaces `vm->last_update` with a signaled stub. The later
   KFD root wait uses the same filtering, and CPU updates wait only for KERNEL
   fences. Those paths do not unconditionally wait for the initial clear.
5. KFD writes the CWSR mapping with CPU updates. The still-pending SDMA clear
   can overwrite those entries after acquisition. Delaying acquisition lets
   the clear complete before those CPU writes.

Relevant installed source locations:

- `amd/amdgpu/amdgpu_vm_pt.c:361`: clear value and commit without output fence.
- `amd/amdgpu/amdgpu_vm_sdma.c:105`: BOOKKEEP fence publication.
- `amd/amdgpu/amdgpu_vm.c:2697`: conversion and incomplete wait.
- `amd/amdgpu/amdgpu_sync.c:240`: reservation usage filtering.
- `amd/amdgpu/amdgpu_vm_cpu.c:92`: CPU update wait.
- `amd/amdgpu/amdgpu_amdkfd_gpuvm.c:1545`: later KFD wait and eviction fence.
- `amd/amdkfd/kfd_process.c:819` and `:1607`: CWSR allocation/map/sync.

The correction belongs at the conversion: wait for BOOKKEEP-inclusive root
reservation fences before CPU writes begin. The new wait must be limited to the first SDMA-to-CPU transition under the
root reservation. A retry or competing acquisition may already carry an
unsignaled KFD eviction fence after that transition. Retaining the existing
implicit wait preserves synchronization on those later paths. Broadening the
common helper would change unrelated submission and eviction behavior.

The [prepared kernel patch](patches/amdgpu-wait-initial-vm-clear.patch) has
passed independent review and an isolated full driver build. Its
[validation record](patches/README.md) identifies the staged module and rollback
copy. The module is now loaded with srcversion `B40219968217AB467B79120`.
NoTrap, Abort, and Generic pass without a preload. A separate PTE check pauses
only after queue creation and confirms a valid mapping. The subsequent
41-case queue/lifecycle/memory regression has 39 passes and two expected
fabric skips. Its kernel delta has expected malformed-queue errors and
workqueue latency warnings, with no VM fault, MES timeout, or reset; no GPU
handles remain open. The installed boot module is unchanged.

A GEM_VA CLEAR plus output timeline can provide a diagnostic barrier on the
normal installed-driver path. It is not a robust replacement for the kernel
fix: the ioctl can export a stub/prior fence on VM readiness/update failures,
and older DRM versions can ignore its extended fields. A fixed sleep only
changes timing and is not a production synchronization contract.

## Rejected hypotheses and qualification limits

- Reference ROCr passes with TTMP setup enabled; that setting alone does not
  explain the runtime difference.
- Injecting a successful SET_MEMORY_POLICY does not fix Abort. The installed
  gfx12.0 MES packet builder also omits the proposed shader-memory field.
- Binding the trap before publication does not fix Abort.
- Retaining the publication queues does not fix Abort. Its NoTrap dispatch
  completes, but deliberate diagnostic retention makes shutdown busy; this
  run is not a passing lifecycle test.
- Early render probes only query capabilities. Source inspection found no
  queue-destruction path that releases the process VM or CWSR allocation.

All experimental source changes were removed. The diagnostic delay restored
page tables and passed NoTrap. Abort subsequently passed in 0.566 s and Generic
in 0.516 s, including queue callbacks and cleanup, with no new kernel messages
and no remaining GPU users. Two earlier Abort launch attempts failed because
the approval service disconnected; those attempts executed no GPU work. The
successful retry occurred after approval review resumed responding.

The earlier contaminated reference failures remain historical observations,
not evidence of an independent firmware defect. GPU core-dump support and SVM
migration-completion support remain separate gaps.

## Artifacts

Local evidence is under `target/rocr-test-fixes/`:

- `runs/recovered-{reference,rust,debug-reference}-{notrap,abort}/`.
- `runs/policy-injection-*`, `runs/bind-first-*`, `runs/retained-queue-*`.
- `runs/delayed-rust-{abort,generic}/` and their kernel snapshots.
- `runs/patched-kmd-{notrap,abort,generic}/`, `runs/patched-kmd-regressions/`,
  and `kernel/after-*-kernel.log`: qualification without a diagnostic preload.
- `recovery/patched-kmd-pte-notrap-trap-pte.json`: patched-driver live mapping.
- `runs/timed-rust-abort/` and `recovery/timed-rust-abort-kernel.log`.
- `recovery/pte-rust-notrap-v2-trap-pte.json`,
  `pte-first-rust-notrap-trap-pte.json`, `pte-reference-notrap-trap-pte.json`,
  and `pte-delayed-rust-notrap-trap-pte.json`.
- `recovery/read-trap-pte.py`, `snapshot-*.py`, and the diagnostic C injectors.
- `recovery/*reset*`: command results and kernel confirmation of seven
  successful MODE1 recoveries. The first attempted pre-reset raw-format
  dmesg capture was unsupported and empty; later plain captures contain the
  recovery evidence.

The installed `amdgpu_vm.c` base SHA-256 is
`dc2cfcf5c11b86dc16247dce712fe826c7de672cd27f8d3759cb7ca0a1a9d1e5`.
