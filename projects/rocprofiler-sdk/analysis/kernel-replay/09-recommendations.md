# 9. Recommendations, ordered

Each item states the change, why it matters, and where it lands in the code. The ordering is by
(risk removed or capability unlocked) ÷ (implementation cost), not by how interesting it is.

## P0 — correctness and containment (small changes, large risk reduction)

**R0. Detect untracked device-visible allocations and decline.**
This is now the first recommendation rather than a refinement of R5, because §4.1 shows the exposure
is far wider than the carveout list suggests: Kokkos enables `hipMallocAsync` by default on every
ROCm before 7.0.0, and ROCm's stream-ordered pool is VM-heap backed by default
(`HIP_MEM_POOL_USE_VM = true`), so an ordinary LAMMPS-KOKKOS or LULESH run on ROCm 6.x has
*essentially its entire dataset* outside the snapshot. `snap()` returns `ok = true`, replay proceeds,
and every pass after the first computes on mutated inputs with no diagnostic. PyTorch's
`expandable_segments:True` — the setting its own OOM message recommends — is the same failure.

Wrap the untracked entry points (`hsa_amd_vmem_handle_create`/`vmem_map`, the managed and
stream-ordered pool paths) purely to *count* bytes, and gate replay on the result: if any untracked
device-visible mapping is live on the agent, decline with a diagnostic naming the allocator. Cheap,
because it needs no restore capability for those regions — only knowledge that they exist. Without
it, the feature's soundness depends on a build-time macro in a third-party library that neither the
user nor the tool inspects.

**R1. Add a generation counter to the allocation inventory.**
`with_inventory_check` validates `(base address, size ≥ recorded)`, which cannot distinguish "the
same allocation" from "a different allocation that reused the address". The allocation wrappers are
deliberately outside the replay lock, so free-then-alloc during the window is reachable, and caching
allocators make same-address reuse likely rather than exotic. Add a monotonic `generation` to
`alloc_info_t`, bump it in `record_alloc`, capture it in `mem_block_t`, and require equality in both
`snap`'s and `restore`'s liveness check. ~20 lines in `memory_tracker.*` and `memory_snapshot.cpp`;
removes a silent-corruption class (§6.3 O2).

**R2. Decline, do not abort, on drain timeout.**
`replay_wait_or_fatal` and `replay_drain_agent_or_fatal` end in `ROCP_FATAL`. Every legitimate
"agent will not go idle" case — persistent kernels, spin-waiting cooperative kernels, a collective
waiting on a peer, a blocked completion handler — therefore converts a profiling request into a
process kill, and in an MPI job into a job kill. Convert to: emit an error record naming the
dispatch, release the window, run the dispatch once, and continue with replay disabled for that
kernel. Keep the abort only for states that are genuinely unrecoverable (a partially applied
restore, which is already handled separately and correctly).

**R3. Guard against reentrancy instead of hanging.**
`std::shared_mutex` is not recursive, and there is no timeout on the reader side, so any GPU work
submitted from inside a tool callback or a drained completion handler on the replaying agent
deadlocks the process permanently (§6.4 L2). Set a thread-local "inside replay window" marker when
the writer lock is taken; on the reader path, if the marker is set for this thread, log a fatal
diagnostic naming the API misuse rather than blocking forever. Document the prohibition in the
public header next to the callback declarations.

**R4. Detect and decline collective and peer-touching dispatches.**
Two independent detectors, both cheap:
* a default-on kernel-name denylist for RCCL/NCCL/MSCCL device kernels, user-overridable;
* a kernarg scan that declines when any pointer-like word resolves via `hsa_amd_pointer_info` to an
  allocation whose `agentOwner` is not the replaying agent.
The second also catches hand-written peer-to-peer kernels and IPC-attached buffers, which a name
list cannot. Without this, replaying a collective corrupts peer state and then aborts the job
(§3.2).

**R5. Track peer exposure.**
Wrap `hsa_amd_agents_allow_access` and mark affected inventory entries peer-exposed. Decline replay
while any peer-exposed allocation is live on the replaying agent (or, later, escalate to a
peer-group window per §3.1). This is the only way to make the agent-scoped isolation claim true on a
multi-GPU node with XGMI access enabled, which is the default configuration.

**R6. Admission control on footprint, with a visible diagnostic.**
Compute $P \cdot F$ before snapping. If it exceeds a budget (default derived from measured link
bandwidth and a wall-time ceiling, overridable), decline with a message that names the footprint, the
pass count, the projected added time, and the remedy ("restrict with kernel filtering"). Also add a
process-wide byte budget for concurrent in-flight snapshots so eight agents cannot independently
decide to hold 150 GB each (§1.3.3). Today a footprint that does not fit produces a `bad_alloc`
warning and silently missing counter rows.

**R7. Refuse partitioned agents whose siblings are active.**
In MI300X CPX-style compute partitioning one physical device presents as several HSA agents that
share HBM and last-level cache, so a per-agent writer lock neither isolates the memory nor cleans the
counter blocks (§3.1c). Detect the partition mode, and decline (or widen the lock to all agents on
the same physical device) when sibling partitions are in use by this process. Document that sibling
partitions in *other* processes cannot be detected at all.

## P1 — measurement validity (make the assumptions measurable)

**R8. Canary counter in every group.**
Include one cheap, pass-invariant counter (`SQ_WAVES`, or a wave count derived from grid size) in
every counter group so it is collected in every pass. If it varies across passes for the same
dispatch id, either the passes did not see identical inputs (O1/O3 violated) or the kernel is
nondeterministic — and every derived metric that combines groups is then suspect. Report the
variance per dispatch. Linux `perf` has reported `time_enabled/time_running` for exactly this reason
for twenty years; kernel replay currently reports nothing, so a user has no way to know that the
mechanism silently failed to give them identical inputs.

**R9. `--kernel-replay-verify`: repeated-group self-validation.**
Run $P+1$ passes, with the extra pass repeating pass 0's counter group, and additionally hash the
tracked footprint on the device after each pass. Three outcomes, all actionable:
* identical hashes, matching counters → the replay behaved as claimed;
* identical hashes, differing counters → hardware/measurement nondeterminism (report the spread);
* differing hashes → state leaked across passes (untracked memory, a concurrent writer, or kernel
  nondeterminism) and the results should not be trusted.
This converts the two unprovable obligations of §6.3 into a per-dispatch measurement, and the device
side hash costs HBM bandwidth (tens of milliseconds), not host-link bandwidth. It is the single most
valuable addition in this document because it makes the feature self-diagnosing.

**R10. Cache and clock control policy.**
Passes are not comparable while caches are neither restored nor flushed and the snapshot traffic
itself perturbs L2 and TLB state (§1.4). Provide an explicit policy: invalidate/flush caches before
each pass (the AQL acquire/release scopes make this expressible), optionally discard a warm-up pass,
and record which policy was used. Nsight Compute defaults to flushing between passes and exposes
clock control for the same reason; matching that behaviour is necessary before comparing AMD
kernel-replay counters to anything.

**R11. Record replay conditions in the output.**
Per replayed dispatch: tracked footprint bytes, region count, snap and restore milliseconds, pass
count, canary variance, cache policy, and — for declined dispatches — the decline reason. Two
audiences need this: users deciding whether to trust a number, and the team deciding which of §9's
performance items to build first. It also makes §7's experiments self-reporting rather than
requiring log scraping.

## P2 — performance (turn the cost model from prohibitive to routine)

These three compound, and together they are the difference between "unusable above a few GB" and
"usable on production footprints".

**R12. Snapshot the reachable set, not the agent.**
`snap()` captures every tracked allocation owned by the agent. The kernel can only touch what it can
reach: the pointers in its kernarg segment, module-scope variables, and whatever is reachable
transitively from those. Scan the kernarg segment for words that resolve to tracked allocations, and
snapshot only those (plus module variables); optionally iterate to a transitive closure for
structures that hold device pointers (Kokkos `View`s inside a struct, device-side linked
structures), with a conservative fallback to the full agent footprint when the closure is not
provably complete. Kerncap already demonstrates this pointer-scavenging approach on AMD hardware.
Expected effect: for a GEMM or an MD force kernel inside a large process, $F$ drops from the whole
resident footprint to the kernel's few operands — commonly two to four orders of magnitude.

**R13. Restore only the write set.**
After R12, restore is still a full copy of the reachable set. Nsight Compute determines the subset
the kernel *wrote* on the first pass and restores only that. On AMD the practical route is a
device-side page hash: hash the captured regions on the GPU before pass 0 and after pass 0, and
restore only the pages whose hash changed. Hashing reads at HBM bandwidth (tens of ms for tens of
GB) instead of moving bytes over the host link (seconds to minutes). The design document already
lists hashing as expected future work; this is the concrete form it should take.

**R14. Stop staging through unpinned host memory.**
`mem_block_t::host_copy` is a `std::vector<char>`, so `hsa_memory_copy` cannot DMA into it directly
and must stage or fall back to a CPU copy — which is why the in-tree perf test's floor is 4 GB/s
rather than a pinned-transfer rate (§1.3.1). Three fixes, in order of preference:
1. keep the snapshot **in device memory** when free HBM allows (device-to-device copy at HBM
   bandwidth, ~2 orders of magnitude faster than the host link) and spill to host only when it does
   not — this is what Nsight Compute does, and after R12 the reachable set will usually fit;
2. when host-backed, allocate pinned host memory (`hsa_amd_memory_pool_allocate` from the host pool,
   or `hsa_amd_memory_lock` over the vector's pages) so the copy is a real DMA;
3. issue the per-region copies asynchronously and in flight together rather than as serial blocking
   `hsa_memory_copy` calls, and coalesce small regions, so many-region inventories stop paying
   per-region latency (the perf test already observes a fixed per-inventory offset).

## P3 — scope (new capability)

**R15. Graph replay.** Replace the graph exclusion with graph *support*: a graph launch is already a
captured, relaunchable block of work, which makes it the best range-replay unit on the platform and
the only way this feature applies to graph-captured inference and `torch.compile` steady state
(§5.1A). Decline graphs containing allocation nodes or collective kernels.

**R16. User-driven range replay.** Let the application bracket a range and run the pass loop itself
(§5.1B). It is the only design that works with MPI and collectives, it preserves realistic execution
conditions, and it makes snapshot/restore optional — removing the dominant cost term entirely for
applications whose ranges are naturally repeatable.

**R17. Fence SDMA against the replay window.** Intercept `hsa_amd_memory_async_copy` (and the
`_rect`/`on_engine` variants) and have them take the per-agent reader lock, so an async copy either
completes before the window opens or waits until it closes. This closes the largest remaining
isolation hole that is actually inside the process's control (§6.3 O3); RDMA and other-process
writers remain out of reach and should be documented as such.

**R18. Add metric distribution across ranks as an alternative to replay.**
Nsight Compute's Metric Distributor collects different counter groups on different ranks of one MPI
run: zero extra passes, no snapshot, no drain, at the cost of assuming ranks are statistically
comparable — which for SPMD bulk-synchronous codes they generally are (§10.3, §10.6). For the
multi-rank HPC workloads that motivate much of the demand for cheaper counter collection, this
dominates replay on every axis, and it is far less invasive than the cross-rank rendezvous of §3.3.
It is P3 only because it is a separate feature rather than a fix to this one.

## What to do first, if only three things happen

1. **R9 (verify mode) + R8 (canary).** Everything else in this study is an argument about whether the
   assumptions hold in a given application; these two let the tool answer that question per run, on
   real workloads, without anyone having to reason about allocators.
2. **R12 + R14 (reachable set + device-resident snapshot).** Together they move the cost model from
   "$P$ × whole footprint over a slow host link" to "$P$ × operands at HBM bandwidth", which is what
   makes the feature applicable to anything with a large resident footprint — i.e. to AI at all.
3. **R0 + R2 (detect untracked allocators; decline instead of abort).** These are the two failure
   modes that damage users rather than merely disappointing them: R0 prevents a wrong scientific
   result from a correct-looking run, and R2 prevents a profiling request from killing a multi-hour
   distributed job. A profiler that can do either will not be turned on twice.

If the list is one item long, it is **R0**. Every other recommendation improves a feature that works;
R0 is the difference between a feature that declines on Kokkos-on-ROCm-6.x and one that silently
returns wrong numbers on a large fraction of the HPC portfolio it was justified by.
