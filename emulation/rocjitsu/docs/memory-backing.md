# KFD and GEM memory backing

GPU allocations, doorbells, event pages, and GEM objects use the backing adapters
in `memory_backing.h`.
The private `rocjitsu-memory` helper creates fixed-size memfds within its own
inherited hard file limit, raising only its own soft limit. If that limit
cannot accommodate an allocation, it uses
System V shared RAM instead. This also supports a finite inherited hard
`RLIMIT_FSIZE` without changing the application's limits or signal disposition.
Internal scratch that has no exportable handle uses shared anonymous memory
with `MAP_NORESERVE`, preserving sharing across fork without file sizing.

KFD buffer-object backing is fixed at the page-rounded allocation size. The local
KFD mapping path rejects views extending beyond that size with `ENOMEM` instead of
growing the backing with `ftruncate`, as earlier versions did. This applies to
both native memfds and tokens. Valid views, including a final partial page,
remain supported; callers that need more storage must allocate a larger BO.

Shared RAM is retained by opaque Unix socket descriptors. Kernel descriptor
references preserve it across duplication, fork, and `SCM_RIGHTS` transfer while
its creating keeper remains alive.
The helper holds an attachment to each segment, marks it for removal, and drops
that attachment when the final token closes. Mapped views can outlive all
handles. Keeper failure invalidates its tokens: subsequent mapping and reopen
requests fail with `ESTALE`, even if a pre-existing view still retains a segment.
Existing views remain usable, but a token with no remaining view loses its data
when the keeper dies. A replacement keeper serves new allocations; it cannot
recover data from invalidated tokens. Mapping adapters use `shmat` and zero-old-size `mremap` aliases, retain
read-only permissions, and support partial and fixed mappings without copying
live buffer contents. These fallback descriptors require rocjitsu's mapping and
metadata adapters; importing them into a different IPC namespace fails with
`EXDEV`. Size queries through `lseek`/`lseek64` follow dma-buf semantics:
zero-offset `SEEK_END` reports the size, zero-offset `SEEK_SET` reports zero,
and other seeks fail with `EINVAL`. Ordinary memfds retain native Linux behavior.

This fallback assumes trusted same-UID processes in the IPC namespace. A live
segment remains discoverable in `/proc/sysvipc/shm` and can be attached directly
by another process with the same UID, without receiving a token. Socket names
carry descriptive, unauthenticated metadata; they do not establish authority.
Read-only adapter mappings enforce mapping permissions, not isolation against
such processes. Each fallback region also consumes a namespace-wide segment
ID and shared-memory budget (`kernel.shmmni` and `kernel.shmall`) until its last
attachment disappears, even after `IPC_RMID`. The per-segment size limit
`kernel.shmmax` can also reject an allocation even when IDs and aggregate
capacity remain available.

All token mappings temporarily attach the entire segment with `shmat`
before creating the requested view with a zero-old-size `mremap` alias and
detaching the temporary attachment. They therefore need contiguous virtual
address space for the full segment, plus space for the new view. A whole-buffer
view temporarily needs roughly twice the segment's address space; a partial
view still needs the full segment plus the requested range. Either can fail
with `ENOMEM` under `RLIMIT_AS`, even when the requested view is only one page.
This is virtual address space, not eager physical allocation.
A fence-only BO retains one fence page after mapping; only native memfd backing
can create that page view without the temporary full-segment requirement.
A failed fence mapping rejects submission and can be retried after sufficient
address space becomes available.

Socket tokens support these mapping/metadata adapters, not file-offset I/O.
The PCI model's `VramStore` still uses its own memfd with `pread`/`pwrite` and
is outside this KFD backing support for file limits.

## GPU core dump classification

Device allocations retain GPU dump classification across descriptor transfers.
Native memfds carry their render-node name. Device tokens carry the same
classification in their socket metadata and use a reserved creation-key prefix
for System V segments. The keeper attaches and immediately marks each segment
for removal, releasing its key; the kernel retains the original `/SYSV` name
on surviving views. This avoids process-local address tracking across `mremap`,
fork, or closing the final descriptor. Non-device token backing still uses
`IPC_PRIVATE`. If all device creation keys are occupied, device allocations
also fall back to `IPC_PRIVATE`, retaining device classification in their token
metadata and `MADV_DONTDUMP` host-core policy. Such mappings have no device marker
in their System V names, so ROCr cannot select them for GPU core dumps through
proc maps/smaps. Other shared-memory resource limits still apply. A keeper killed
between segment creation and `IPC_RMID` can leave a named segment behind; normal
cleanup releases the key immediately after attachment.

The interposer renders marked System V paths in proc maps/smaps snapshots as
`/dev/dri/renderD-shared`. This is a GPU classification marker, not an actual
render-node path or a device-minor query. ROCr selects GPU core pages by the
`/dev/dri/renderD` prefix. Native mappings retain their specific render minor.
Both device backing forms receive `MADV_DONTDUMP` for host cores; ordinary
host allocations remain eligible. These names are descriptive markers, not
an authentication boundary.

## Public VM handles

Embedders that load `librocjitsu.so` with `dlopen(RTLD_LOCAL)` must map handles
from `rj_vm_get_shared_mem{,_as}` and `rj_vm_cmd_t::shared_handle` with the
public `rj_vm_map_shared_mem` C API. These handles are borrowed from the VM;
they may be memfds or opaque tokens, and ordinary libc `mmap` cannot map tokens.
The same API accepts descriptors duplicated or received through `SCM_RIGHTS`.
Keep the library loaded and the descriptor valid during the call.
Callers retain the allocation size separately and send it alongside the
descriptor when transferring ownership or access with `SCM_RIGHTS`. Use that
size to select the view's offset and length. Non-preloaded host `fstat` sees a
socket with size zero for a token, and host `lseek` cannot query its backing
size; host file metadata is not a portable size source for these handles.

Pass a `rj_vm_map_t` with `MAP_SHARED`, the desired protection, and a
nonnegative page-aligned byte offset within the backing (not the KFD mmap
offset used for handle lookup). The supported mapping flags are `MAP_SHARED`
with optional `MAP_FIXED`, `MAP_FIXED_NOREPLACE`, and `MAP_32BIT` where available.
Both backing forms reject all other flags, including `MAP_PRIVATE`,
`MAP_ANONYMOUS`, `MAP_LOCKED`, `MAP_HUGETLB`, and `MAP_GROWSDOWN`, with `EINVAL`.
Empty ranges and ranges extending beyond the backing also fail with `EINVAL`.
A successful call returns `ROCJITSU_STATUS_SUCCESS` and fills
`mapped_addr`; a failed mapping returns `ROCJITSU_STATUS_ERROR`, sets
`mapped_addr` to `MAP_FAILED`, and captures the error in `map_errno`.
The keeper-lifetime, IPC-namespace, and credential constraints above also apply.
The caller owns the resulting view and releases it with host `munmap`.
Mapped views outlive handles, but the caller must stop simulator access before
unmapping or changing their protections.

## Keeper lifecycle

The helper execs with an independent address space, detaches from application
job control, and watches its owner's pidfd and an open-file-description lock
on an empty memfd. A private `PROT_NONE` mapping retains the owner's locked
file description after its descriptor closes; the empty file is never sized or
accessed. `MADV_DONTFORK` keeps that mapping out of forked children, and exec
removes it. The helper waits for the lock through an independently opened file
description. Retirement unmaps the lease without closing a long-lived application
descriptor, so concurrent `close_range` or descriptor replacement cannot make it
close an unrelated file. A helper thread waits for release of the lock and
notifies the event loop. The helper retires
on owner exit or exec, preserving any exported tokens until they close, without
retaining application mappings. The interposer starts it before application
code can lower file limits. A forked child starts its own helper under the
child's then-current hard limits. Generated topology files and proc-maps
snapshots are sized there and populated through mappings; these small metadata
files must fit that helper's hard file limit. Topology population failures emit
a warning identifying the file. Application writes, including runtime compiler
temporary files, continue to obey the application's current limit.

Changing effective user or group credentials retires the previous helper's
lease before using a new helper; previously exported tokens retain their keeper
and storage. New token mappings still check the caller's current System V IPC
permissions. Segments use mode 0600; a different effective UID without
`CAP_IPC_OWNER` receives `EACCES`, even when it retained a token. Already mapped
views remain usable. Group-only changes do not remove the owning UID's access.
Detached helpers must be reaped by PID 1 or an adopting child subreaper,
including after application children exit. Run containers with a reaping init
such as tini. An application acting as PID 1 or enabling
`PR_SET_CHILD_SUBREAPER` must handle keeper SIGCHLD notifications and drain
exited children with nonblocking waits. Waiting only for known application
children leaves adopted keepers as zombies; a blocking wait for all children
can hang while a keeper is still serving live tokens. rocjitsu diagnoses this
requirement for PID-1 owners, their direct children, and child subreapers.

Native memfds preserve the previous allocation policy: daemon and driver-selected
VA allocations attempt preallocation in the helper; caller-VA allocations remain
sparse. Token allocations remain sparse. `RJ_TEST_FORCE_TOKEN_BACKING=1` forces
only GPU backing to use tokens for integration tests, leaving metadata-file
sizing and application limits unchanged.

Installed objects locate the helper in their installation. Only objects inside
the configured build tree may fall back to its helper executable; an incomplete
installation fails rather than using another build's private protocol.

File-limit handling does not require `HSA_ENABLE_IPC_MODE_LEGACY=1`. Ordinary HIP
allocation, copies, dispatch, and local VMM work with the runtime's default IPC
selection. The System V RAM fallback is independent of HSA's legacy IPC API and requires
Linux System V IPC support; native memfd backing does not.
HIP IPC sharing between processes is a separate compatibility boundary: the
current dma-buf IPC path is unsupported, so daemon clients using
`hipIpcGetMemHandle`/`hipIpcOpenMemHandle` need the legacy mode described in the
[CLI guide](rocjitsu-cli.md#overview). This change does not add dma-buf IPC support.
