# KFD and GEM memory backing

GPU allocations, doorbells, event pages, and GEM objects use `memory_backing`.
The private `rocjitsu-memory` helper creates fixed-size memfds within its own
inherited hard file limit, raising only its own soft limit. If that limit
cannot accommodate an allocation, it uses
System V shared RAM instead. This also supports a finite inherited hard
`RLIMIT_FSIZE` without changing the application's limits or signal disposition.
Internal scratch that has no exportable handle uses shared anonymous memory
with `MAP_NORESERVE`, preserving sharing across fork without file sizing.

Shared RAM is retained by opaque Unix socket descriptors. Kernel descriptor
references preserve it across duplication, fork, and `SCM_RIGHTS` transfer.
The helper holds an attachment to each segment, marks it for removal, and drops
that attachment when the final token closes. Mapped views can outlive all
handles. Mapping adapters use `shmat` and zero-old-size `mremap` aliases, retain
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

Socket tokens support these mapping/metadata adapters, not file-offset I/O.
The PCI model's `VramStore` still uses its own memfd with `pread`/`pwrite` and
is outside this KFD backing support for file limits.

The helper execs with an independent address space, detaches from application
job control, and watches its owner's pidfd and a close-on-exec file lease.
The lease is a process-owned record lock on an empty memfd, so an idle forked
child cannot keep its parent's lease alive after exec. A helper thread waits
for release of that lock and notifies the event loop. The helper retires
on owner exit or exec, preserving any exported tokens until they close, without
retaining application mappings. Closing or overwriting the lease causes the
next request to start a new helper. The interposer starts it before application
code can lower file limits. A forked child starts its own helper under the
child's then-current hard limits. Generated topology files and proc-maps
snapshots are sized there and populated through mappings; these small metadata
files must fit that helper's hard file limit. Topology population failures emit
a warning identifying the file. Application writes, including runtime compiler
temporary files, continue to obey the application's current limit.

Changing effective user or group credentials retires the previous helper's
lease before using a new helper; previously exported tokens remain valid.
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
