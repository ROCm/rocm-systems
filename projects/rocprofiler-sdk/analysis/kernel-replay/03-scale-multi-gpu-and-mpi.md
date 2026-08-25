# 3. Scale: multi-GPU, partitions, collectives, and MPI

The per-agent reader/writer lock is the entire scale-out story in the current implementation. This
section works out what it does and does not buy, and what a coordinated design would have to look
like.

## 3.1 What the per-agent lock actually guarantees

```110:126:projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp
std::shared_mutex&
agent_replay_mutex(rocprofiler_agent_id_t agent_id)
```

One `std::shared_mutex` per `rocprofiler_agent_id_t`. A replay window takes it uniquely; every
non-replayed dispatch takes it shared *for the duration of its submission only*. The GPU tail is
handled separately by `replay_drain_agent_or_fatal`, which polls `active_async_packets()` across
every queue belonging to that HSA agent.

This is sound for the intra-agent, intra-process, AQL-only case, and it is a real improvement over a
global lock: two GPUs can replay simultaneously. Three properties follow that matter at scale.

**(a) Concurrency across agents multiplies host RAM.** Each concurrent window holds its agent's full
tracked footprint in host RAM (§1.3.3). Eight concurrent windows on an 8-GPU node hold eight
footprints. There is no global admission control, so nothing prevents eight agents from
simultaneously deciding to snapshot 150 GB each. The natural fix is a process-wide byte budget for
in-flight snapshots, with windows queueing (or declining) when the budget is exhausted.

**(b) Agent-scoped isolation is not isolation once P2P is enabled.** `hsa_amd_agents_allow_access`
makes a buffer allocated on agent A writable by kernels running on agent B. Agent B's dispatches
take *B's* mutex, which does not exclude A's window, and B's queues are not polled by A's
agent-wide drain (it filters on `sibling->get_agent().get_hsa_agent().handle == agent.handle`). So
on any node with XGMI peer access enabled — which is the default configuration for MI250/MI300
multi-GPU systems — the isolation obligation O3 is violated by construction whenever the profiled
process has cross-GPU buffers in play. The mechanism cannot even detect this today, because nothing
records which tracked allocations have been made peer-accessible.

Two implementable responses, in increasing order of ambition:

* *Detect and decline.* Wrap `hsa_amd_agents_allow_access` and mark the affected inventory entries
  as peer-exposed. If any peer-exposed allocation is live on the replaying agent, decline replay
  (with a diagnostic naming the buffer). Conservative, cheap, and honest.
* *Peer-group windows.* Treat the P2P clique as the unit of isolation: acquire the writer locks of
  every agent in the clique in a fixed global order (agent id order, to make deadlock impossible),
  drain all of them, and snapshot the union of their tracked footprints. Correct, but it serializes
  the whole node and multiplies both snapshot cost and host RAM by the clique size — for an 8-GPU
  fully-connected MI300X node that is 8 × 150 GB. Viable only in combination with the write-set and
  device-resident-snapshot optimizations of §9.

**(c) Partitioned GPUs may break the agent abstraction.** On MI300X, compute partitioning (SPX
through CPX) exposes a single physical GPU as multiple HSA agents that nonetheless share HBM stacks,
the memory fabric, and last-level cache. A per-*agent* writer lock does not exclude work on the
*sibling partitions of the same die*, so in CPX-style modes a replay window on one partition is not
isolated from concurrent work on the other partitions of the same physical device — and the counters
it collects are contaminated by that work in any block whose counters are physically shared. The
correct lock granularity is the physical device (or, more precisely, the smallest unit that owns the
counter blocks being read), not the HSA agent. At minimum the feature should refuse to replay when
the agent is a partition of a device with other active partitions in the same process, and should
document that it cannot detect other *processes* using sibling partitions at all.

## 3.2 Collectives: the hard carveout, and why

RCCL/NCCL collectives are the clearest case where replay cannot work, and the failure mode is worth
spelling out because it is not "wrong numbers", it is "the job dies".

A ring or tree allreduce kernel makes progress by spinning on flags in peer-visible memory: each
rank's kernel writes data and a sequence counter, and waits for its neighbour's counter to advance.
The kernels of all participating ranks must be resident *simultaneously*. Now replay one of them:

1. Pass 0 runs. The peers participate, the collective completes, every rank's kernel exits.
2. `restore()` reverts the replaying agent's tracked memory — including the flags and the
   communication buffers the *peers* observed. On the peer side those buffers may be that rank's
   own memory (untouched) or the replaying rank's memory reached over XGMI (silently rewound).
3. Pass 1 submits the same collective kernel. The peers have already finished and moved on to the
   next operation. Nobody advances the flags pass 1 is waiting for.
4. Pass 1's kernel never completes. `queue.sync()` returns false for twelve 5-second slices, and
   `replay_wait_or_fatal` calls `ROCP_FATAL`. **The process aborts after ~60 s**, and in an MPI job
   that means the whole job.

So the collective case is not merely unsupported; the current failure path converts it into a job
kill. Even before the hang, step 2 has already corrupted peer state.

Detection is straightforward enough that there is no excuse for leaving this to chance:

* *Name-based denylist.* Collective kernels are recognizable (`ncclDevKernel*`, `nccl*`, RCCL's
  equivalents, and MSCCL/MSCCL++ kernels). A default denylist, overridable by the user, costs
  nothing and prevents the abort.
* *Cooperative/multi-agent launch detection.* Any dispatch whose kernarg segment references memory
  whose `hsa_amd_pointer_info().agentOwner` is a different agent, or whose allocation is
  peer-exposed (§3.1), should be declined. This also catches hand-rolled peer-to-peer kernels that a
  name list would miss.
* *Decline, do not abort.* Independently of detection, `replay_wait_or_fatal` should decline the
  replay (emit an error record, run the dispatch once, continue) rather than terminate. A profiler
  that kills long jobs on an unsupported kernel will not be enabled in production.

## 3.3 MPI: what is possible today, and what a coordinated design needs

The tool is loaded into every rank, each rank profiles its own process, and output is per-rank. The
replay window is per-process and completely uncoordinated with other ranks. Consequences:

**Compute-only kernels on one rank: workable, with care.** A kernel with no communication dependency
(a force computation, a stencil interior update, a GEMM) can be replayed in one rank while other
ranks proceed. The other ranks will eventually block at the next synchronization point waiting for
the replaying rank — which is fine, MPI point-to-point and collective waits have no intrinsic
timeout — provided:

* the replay window is short relative to any *external* timeout: job-scheduler watchdogs, a
  progress-thread timeout, an interconnect retry limit, and framework-level watchdogs (PyTorch's
  RCCL/NCCL watchdog aborts after its configured timeout, on the order of minutes, and that abort is
  a hard kill);
* nothing in the window depends on peer progress (§3.2);
* GPU-aware MPI is not concurrently writing the tracked buffers (below).

**GPU-aware MPI is an unmodelled writer.** When MPI is built GPU-aware, the transport writes device
memory directly: RDMA from the NIC into a registered device buffer, or an SDMA copy between device
and staging memory. Neither path goes through an AQL queue, so neither the writer lock nor the
agent-wide drain sees it. Two distinct failures:

* *Restore clobbers an in-flight receive.* A neighbour's halo data lands in a tracked buffer during
  the window, `restore()` reverts it, and the application proceeds with stale halos. Silent, and it
  will look like a physics bug, not a profiler bug.
* *The replayed kernel is the sender.* Replaying a pack kernel that fills a send buffer, then
  reverting it, then re-running it, is harmless if the send has not started; if the send is already
  in flight (nonblocking `MPI_Isend` issued before the window, transport DMA-ing out of that buffer)
  the peer receives a mixture of pass outputs.

**A coordinated design.** Making replay MPI-safe requires an out-of-band rendezvous: all ranks must
agree that a replay window is open and that no rank issues communication inside it. That means
(i) a barrier at window entry and exit, reachable from inside the profiler without deadlocking the
application's own MPI usage — realistically over a side channel (PMIx, a shared-memory rendezvous
per node plus a small TCP mesh across nodes) rather than `MPI_Barrier`, which is unsafe to call from
an interception context; and (ii) agreement on *which* dispatch is being replayed, which requires a
deterministic, cross-rank-stable dispatch identity. For strict SPMD codes a per-rank dispatch
counter plus kernel identity is stable enough in practice; for anything with load balancing,
adaptive refinement, or data-dependent control flow it is not. This is a significant subsystem, and
it is worth comparing honestly against the alternative in §5: if the *application* drives the pass
loop, the application's own `MPI_Barrier` provides the rendezvous for free, and the deterministic
dispatch identity problem disappears entirely. **User-driven range replay is a far cheaper route to
multi-rank profiling than coordinated kernel replay.**

**Practical guidance for today.** Until any of the above exists, the only defensible multi-rank
recipe is: pick one rank, restrict replay to a named compute kernel with no communication
dependency, keep $P$ small, verify that the un-profiled application output is reproduced, and
confirm that no framework watchdog fires. §7 turns this into concrete commands, including the
negative controls that should be run first so the failure modes are observed deliberately rather
than discovered in a user's job.
