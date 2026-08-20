# QP sharing (net_ib_cast) — review tracker

Working tracker for the IB CAST QP-sharing feature. Not product documentation: it lives
next to the code so it stays findable while the feature is in review, and it should be
deleted (or folded into the final PR description) once the feature lands.

Scope reviewed: `ROCm/rocm-systems` PR
[#10251](https://github.com/ROCm/rocm-systems/pull/10251) plus the follow-up fix commits
on `VadimKutovoi/rocm-systems` branch `users/vkutovoi/qpsharing-rocm-ib-tests`
(tip `f2884af` at time of writing).

Target hardware for this feature is **AINIC only** — no mlx5, no Broadcom for now.
Several items below exist only because that scope isn't enforced in code.

---

## Branch inventory

| Branch | Tip | Contents |
| --- | --- | --- |
| `ROCm` `users/karthikarum/qpsharing-rocm-ib` | `1a32566` | PR #10251, 3 commits |
| `ROCm` `users/vkutovoi/qp-sharing` | `2a338b7` | independent earlier variant; forces prepost under sharing, no dead `#if` blocks, but no flush QP for secondaries and no commId recycling |
| `ROCm` `users/vkutovoi/qp-sharing-tests` | `c859625` | above + baseline MPI test suite |
| `ROCm` `users/vkutovoi/qpsharing-rocm-ib-flush-fix` | `93a537f` | PR commit 1 + per-comm flush QP |
| `ROCm` `users/vkutovoi/qpsharing-rocm-ib-tests` | `48333ec` | PR commit 1 + MPI test suite |
| `ROCm` `users/vkutovoi/qpsharing-rocm-ib-tests-flush-fix` | `8beab68` | PR commit 1 + tests + flush QP fix + teardown ordering fix |
| `VadimKutovoi` `users/vkutovoi/qpsharing-rocm-ib-tests` | `f2884af` | **most complete**: full PR + tests + 6 fix commits (see "Closed") |

---

## Environment variables

Both parameters use `RCCL_PARAM`, which expands to `ncclLoadParam("RCCL_" env, ...)` with
**no** NCCL_ fallback (that only exists in `RCCL_PARAM_NCCL_ALIAS`). The working names are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RCCL_IB_COMM_NGROUPS` | 0 | 0 = sharing disabled, N = N sharing groups per peer |
| `RCCL_IB_QP_DEPTH_MULTIPLIER` | 1 | CQ/WR depth scaling; on the fork branch also the per-group admission capacity |

`NCCL_IB_COMM_NGROUPS` and `NCCL_IB_QP_DEPTH_MULTIPLIER` are silently ignored. See QPS-9.

---

## Status summary

| ID | Severity | Item | Reachable from rccl-tests |
| --- | --- | --- | --- |
| QPS-1 | blocking | commId in `wr_id[63:48]` collides with the packed slot bytes | yes, needs >= 7 grouped recvs |
| QPS-2 | blocking | pool key is node IP with the port stripped | yes, needs > 1 rank per (node, NIC) |
| QPS-3 | blocking | shared CQ polled by several proxy threads; cross-comm request mutation | yes, needs >= 2 comms per process |
| QPS-4 | blocking | primary/secondary decision not negotiated; connect-side pool lookups unlocked | partly |
| QPS-5 | high | `RCCL_IB_COMM_NGROUPS` not gated on AINIC | yes, on a non-AINIC node |
| QPS-6 | high | sharing paths assume `vProps.ndevs == 1`, merging is on by default | yes |
| QPS-7 | medium | admission capacity ~32x too conservative; sharing is a no-op at defaults | yes |
| QPS-8 | medium | Ionic UDMA mask chosen by group parity; unbalanced for odd group counts | yes (perf A/B) |
| QPS-9 | medium | documented env var names don't work (false-negative risk in validation) | n/a |
| QPS-10 | medium | `IbCastIflush` INFO logging on the data path, NULL deref on flush-slot miss | yes |
| QPS-11 | medium | `IbCastFindSharedQpByQpn` keys on `qp_num` alone | no |
| QPS-12 | low | `IbCastCountGroupQpSlots` counts flush entries | no (inspection) |
| QPS-13 | low | `IbCastCountPeerTotalRefcount` ignores its `ibDevN` argument | observable only |
| QPS-14 | low | `destroyedCqs[NCCL_IB_MAX_DEVS_PER_NIC]` written without a bound check | no |
| QPS-15 | low | commId free stack has no double-push guard | no |
| QPS-16 | low | `fail:` paths free the comm while its commId is still registered | fault injection only |
| QPS-17 | cleanup | dead code, dead stores, tab indentation | n/a |
| QPS-18 | process | no functional tests in PR #10251 | n/a |

---

## Testing preconditions

Two things make otherwise-green runs meaningless.

**1. Raise the depth multiplier on the fork branch.** `capacityUnits = RCCL_IB_QP_DEPTH_MULTIPLIER`
(default 1), so the second comm in a group fails admission and silently falls back to
unshared. Set it to at least the expected number of comms per group.

**2. Prove sharing was actually active** before believing a pass:

```bash
RCCL_IB_COMM_NGROUPS=2 RCCL_IB_QP_DEPTH_MULTIPLIER=8 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET \
mpirun -np 2 --map-by ppr:8:node \
  all_reduce_perf -b 8M -e 8M -f 2 -g 1 -n 5 2>&1 \
  | grep -cE "QP sharing (PRIMARY|SECONDARY)"
```

### Why the baseline run passes

`all_reduce_perf -b 1K -e 16G -f 2 -g 1 -n 20 -c 1 -w 5 -N 10` passes even without the
follow-up fixes, and that is consistent with every open item:

- `-g 1` with one rank per NIC gives one proxy sub per connection, so `irecv` is always
  called with `n = 1`, `nreqs == 1`, and `wr_id` only ever uses byte 0 (closes QPS-1).
- One communicator means one `ncclProxyState` (`comm->proxyState = comm->sharedRes->proxyState`),
  so a single proxy thread performs every connect and every `IbCastTest` poll (closes QPS-3
  and the racy half of QPS-4).
- In-flight depth is `min(NCCL_STEPS, NCCL_SHARED_STEPS / nsubs)` <= 8 per sub against a
  256-deep RQ, so no queue limit is approached (this is also why the withdrawn `rxPosts`
  item is harmless, and why QPS-7 matters).
- Create-once / destroy-once: `-N` repeats the size sweep without re-initialising, so no
  commId or pool-slot churn (the exhaustion and leak bugs need churn).
- One rank per (node, NIC) keeps the pool key unique per peer (closes QPS-2).

---

## Open issues

### QPS-1 — commId in `wr_id[63:48]` collides with the packed slot bytes

- **Status:** open on all branches.
- **Where:** `p2p.cc`, `IbCastMultiSend`; `qp_sharing.h`, `IbCastStripCommId` / `IbCastRouteCommFromWrId`.
- **What:** `wr_id += (uint64_t)(slot & 0xff) << (r * 8)` runs for `r` in `[0, nreqs)` with
  `nreqs` up to `NCCL_NET_IB_MAX_RECVS` (8), so bytes 6 and 7 — bits 48..63 — are already
  occupied when 7 or 8 receives are aggregated. `wr_id | (commId << 48)` then merges the two
  fields: routing reads a corrupted commId and `IbCastStripCommId` erases slot bytes.
- **Trigger:** >= 7 proxy subs grouped into one `irecv`, i.e. several local ranks sharing one
  `netRecvComm` (the `maxRecvs > 1 && NCCL_NET_SHARED_COMMS` path, PXN). Note sharing opens
  its own gate here: `IbCastGetProperties` returns `maxRecvs = 1` only while CTS offload is
  enabled, and sharing force-disables CTS offload, so `maxRecvs` becomes 8.
- **Repro:**

```bash
RCCL_IB_COMM_NGROUPS=2 RCCL_IB_QP_DEPTH_MULTIPLIER=16 \
NCCL_NET_SHARED_COMMS=1 NCCL_PXN_DISABLE=0 \
mpirun -np 2 --map-by ppr:1:node \
  alltoall_perf -b 512K -e 64M -f 2 -g 8 -n 50 -c 1
```

  A pass proves nothing unless `nreqs` reached 7. Add a permanent tripwire in
  `IbCastMultiSend`: `if (nreqs > 6 && comm->base.commId != 0) WARN(...)`.
- **Fix direction:** pick an encoding that cannot overlap the existing layout, or carry the
  commId out of band.

### QPS-2 — pool key is the node IP with the port stripped

- **Status:** open on all branches.
- **Where:** `qp_sharing.cc`, `IbCastStripPort` / `IbCastSharedQpKey`; sender key built from
  `handle->connectAddr`, receiver key from the accepted socket's peer address.
- **What:** stripping the port (necessary, since each `listenComm` has a fresh port) reduces
  the key to (peer node IP, `remIbDevIdx`). The only per-process discriminator left is the
  remote device index, so two remote ranks behind the same NIC collapse into one key and a
  comm can adopt RC QPs connected to a different rank.
- **Trigger:** > 1 rank per (node, IB device) — `NCCL_IB_HCA` pinned to one device, partially
  populated nodes, PXN. Safe by construction in a rail-local 1:1 GPU:NIC layout.
- **Repro:** one comm per process, so no intra-process concurrency confound; alltoall gives
  each rank 8 remote peers sharing one IP, and the HCA pin makes `remIbDevIdx` identical too:

```bash
NCCL_IB_HCA=<one AINIC dev> \
RCCL_IB_COMM_NGROUPS=1 RCCL_IB_QP_DEPTH_MULTIPLIER=8 \
mpirun -np 16 --map-by ppr:8:node \
  alltoall_perf -b 1M -e 64M -f 2 -g 1 -n 20 -c 1
```

  Expect `-c 1` mismatches or a hang. If it passes, downgrade to "assert the assumption".
- **Fix direction:** carry a process-unique peer id in `ncclIbConnectionMetadata` and put it
  in the key; or refuse to share when a slot with the same key belongs to a different peer.

### QPS-3 — shared CQ polled by several proxy threads, cross-comm request mutation

- **Status:** open on all branches.
- **Where:** `connect.cc` secondary paths (CQ redirect); `p2p.cc`, `IbCastTest` dispatching to
  `targetBase`; `qp_sharing.cc`, `IbCastRouteCommFromWrId` / `IbCastRouteCommFromImmData`.
- **What:** secondaries point `devs[i].base.cq` at the primary's CQ, so several comms call
  `wrap_ibv_poll_cq` on one CQ concurrently (libibverbs requires external serialisation), and
  one comm's thread mutates another comm's `req->events[]`, `rxPosts[]` and `sendReqs[]`.
  `g_IbCastCommTable` is read lock-free in the routing helpers while teardown writes it under
  `g_IbCastSharedQpMutex` — a commId can be freed mid-poll, so this is a use-after-free on the
  comm pointer, not just a torn read.
- **Trigger:** >= 2 independently created comms in one process whose pool keys collide.
  rccl-tests reaches this: `initComms` wraps per-device `ncclCommInitRankConfig` in
  `ncclGroupStart/End`, so N comms initialise on N async-job threads and run N proxy threads.
- **Repro:**

```bash
NCCL_IB_HCA=<one AINIC dev> \
RCCL_IB_COMM_NGROUPS=1 RCCL_IB_QP_DEPTH_MULTIPLIER=16 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET \
mpirun -np 2 --map-by ppr:1:node \
  all_reduce_perf -b 8M -e 512M -f 2 -g 8 -n 50 -c 1
```

  Symptoms: `could not retreive a request`, `sendReq(...)->events=... <= 0`,
  `ncclInternalError`, `-c 1` mismatches, hangs. The same rank count with `-g 1 -np 16`
  (one comm per process) should be clean — that delta is the evidence.
- **Fix direction:** one designated poller per shared CQ group, or a per-group lock around
  poll-and-dispatch; make the comm table lookup safe against concurrent free.

### QPS-4 — primary/secondary decision not negotiated; connect-side lookups unlocked

- **Status:** partly addressed on the fork (register/join/leave/unregister self-lock, closing
  the refcount TOCTOU). `IbCastFindSharedQp`, `IbCastCountGroupQpSlots` and
  `IbCastCountPeerTotalRefcount` are still called bare from connect, so find-then-join and the
  primary-vs-secondary decision still race.
- **What:** each side decides primary vs secondary from its own pool; only `sharedGroupIdx`
  crosses the wire. Sender-side fallback is safe because `meta.sharedGroupIdx = -1` propagates
  and the receiver requires `>= 0`. The reverse is not signalled: if the sender joined as a
  secondary and the receiver then falls back, the sender is wired to the *primary* receiver's
  QPs while this receiver holds fresh unconnected QPs — silent misdelivery or a hang.
- **Trigger:** any fallback firing. Rare on PR #10251 (pool/commId exhaustion only), routine on
  the fork, where capacity fallback is an expected path the tests assert.
- **Repro:** fork branch, `RCCL_IB_COMM_NGROUPS=2` with the multiplier left at 1 and more than
  two comms per group; then check both ends' INFO logs for a receiver-side
  "at capacity, falling back" without a matching sender-side fallback.
- **Fix direction:** put the admission outcome in the handshake; lock the whole
  find-then-join / find-then-register sequence.

### QPS-5 — `RCCL_IB_COMM_NGROUPS` not gated on AINIC

- **Status:** open on all branches.
- **What:** nothing ties the parameter to `IbCastAinicRoce`. Only `ncclIbCreateQpIonic` applies
  `cqDepthMultiplier` to `max_send_wr` / `max_recv_wr`; the `mlx5dv` branch and the generic
  fallback in `IbCastQpCreate` do not. On non-AINIC hardware you therefore get a scaled CQ over
  unscaled queues, and on the fork an admission check that admits `depthMultiplier` comms onto
  queues that were never widened.
- **Repro:** run the precondition check on an mlx5 node; sharing should be refused and isn't.
- **Fix direction:** one line beside the existing CTS-offload disable in `IbCastInitDevices` —
  if sharing is requested without AINIC, warn and force it to 0.

### QPS-6 — sharing paths assume `vProps.ndevs == 1`

- **Status:** open on all branches; TODOs in `connect.cc` acknowledge it.
- **What:** the secondary path assigns `existingSlot->primaryCq` (device 0's CQ) to every local
  device, and the key mapping assumes a single device. `NCCL_IB_MERGE_NICS` defaults to 1 and
  `IbCastMakeVDevice` permits vDevice creation on AINIC at that default, so a merged device is
  reachable in a stock configuration.
- **Repro:**

```bash
NCCL_IB_MERGE_NICS=1 \
RCCL_IB_COMM_NGROUPS=2 RCCL_IB_QP_DEPTH_MULTIPLIER=8 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET \
mpirun -np 2 --map-by ppr:1:node \
  all_reduce_perf -b 8M -e 512M -f 2 -g 8 -n 20 -c 1
```

  Confirm from the log that a merged device was created (`NCCL MergedDev`, `ndevs=2`).
- **Fix direction:** refuse to enable sharing when `vProps.ndevs > 1` until the multi-device
  design exists.

### QPS-7 — admission capacity too conservative; sharing is a no-op at defaults

- **Status:** open on the fork (introduced with the ISSUE-1 admission check).
- **What:** `capacityUnits = RCCL_IB_QP_DEPTH_MULTIPLIER`, default 1, so the first comm takes
  the only slot in its group and everything after falls back to unshared:
  `RCCL_IB_COMM_NGROUPS=2` alone yields two shared QPs per peer and nothing else. The bound is
  also far tighter than the queues need — the RQ is `NET_IB_MAX_REQUESTS` (256) deep while
  per-comm outstanding receives are capped at `min(NCCL_STEPS, NCCL_SHARED_STEPS / nsubs)` <= 8,
  so a shared QP can safely host on the order of `32 x multiplier` comms.
- **Repro:**

```bash
RCCL_IB_COMM_NGROUPS=2 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET \
mpirun -np 2 --map-by ppr:8:node \
  all_reduce_perf -b 8M -e 8M -f 2 -g 1 -n 5 2>&1 \
  | grep -cE "falling back to unshared|QP sharing SECONDARY"
```

- **Fix direction:** derive capacity from per-comm WQE demand (about `NCCL_STEPS x nqps`)
  rather than from the depth multiplier, and document the coupling either way.

### QPS-8 — Ionic UDMA mask chosen by group parity

- **Status:** open on all branches.
- **Where:** `connect.cc`, `ncclIbCreateQpIonic`.
- **What:** shared QPs select `IONIC_UDMA_MASK_LOW/HIGH` by `qpSharingGroupIdx % 2`, replacing
  the per-channel alternation that balanced regardless of configuration. `ngroups=1` puts every
  shared QP on LOW; `ngroups=3` splits 2:1. Separately, `wrap_ionicdv_pd_set_udma_mask` mutates
  the *shared* per-device PD immediately before `ibv_create_qp`, so concurrent connects on one
  device can interleave mask-set and create and land a QP on the wrong engine.
- **Repro:**

```bash
for G in 0 1 2 3 4; do
  RCCL_IB_COMM_NGROUPS=$G RCCL_IB_QP_DEPTH_MULTIPLIER=8 \
  mpirun -np 2 --map-by ppr:8:node \
    all_reduce_perf -b 256M -e 2G -f 2 -g 1 -n 20 -w 5
done
```

  Expect 1 and 3 to trail 2 and 4 by more than the QP-count difference explains.
- **Fix direction:** balance by a running count of registered primaries per device; cover
  mask-set plus create with the same lock used for the pool.

### QPS-9 — documented env var names do not work

- **Status:** open on all branches.
- **What:** `qp_sharing.h`'s file header says "Controlled via NCCL_IB_COMM_NGROUPS", the comment
  above the extern says "Accessible as RCCL_IB_COMM_NGROUPS / NCCL_IB_COMM_NGROUPS", and the PR
  description documents `NCCL_IB_COMM_NGROUPS` / `NCCL_IB_QP_DEPTH_MULTIPLIER`. Plain
  `RCCL_PARAM` reads only `RCCL_<env>`. Anyone following the docs runs with sharing silently
  off and reports "no regression" — a false negative in exactly the validation runs this
  feature needs.
- **Fix direction:** switch both to `RCCL_PARAM_NCCL_ALIAS` (which checks `RCCL_` then `NCCL_`),
  or correct the comments and the PR text.

### QPS-10 — `IbCastIflush` logging on the data path

- **Status:** open on PR #10251 and the fork (arrived with the flush-sharing commit); absent on
  the branches that use a per-comm flush QP.
- **What:** two unconditional `INFO` calls per flush, i.e. per receive. The second dereferences
  `comm->devs[i].gpuFlush.qp.qp->qp_num`, which is NULL exactly when the secondary's flush-slot
  lookup missed — that path only `WARN`s at connect time and continues.
- **Repro:**

```bash
RCCL_IB_COMM_NGROUPS=2 RCCL_IB_QP_DEPTH_MULTIPLIER=8 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET \
mpirun -np 2 --map-by ppr:8:node \
  all_reduce_perf -b 1M -e 16M -f 2 -g 1 -n 20 2>&1 \
  | grep -cE "posting flush dev|could not find shared flush QP"
```

- **Fix direction:** demote to `TRACE`; make the flush-slot miss a hard failure or a real
  fallback. Prefer the per-comm flush QP approach from
  `users/vkutovoi/qpsharing-rocm-ib-flush-fix`, which also removes QPS-12.

### QPS-11 — `IbCastFindSharedQpByQpn` keys on `qp_num` alone

- **Status:** open on all branches.
- **What:** QP numbers are unique per device context, not per process, so with more than one
  local NIC teardown can decrement the refcount on an entry belonging to another device.
- **Trigger:** colliding `qp_num` across two local devices; not steerable from a command line.
- **Fix direction:** include `ibDevN` in the lookup, or assert the found entry's `key.ibDevN`.

### QPS-12 — `IbCastCountGroupQpSlots` counts flush entries

- **Status:** open on PR #10251 and the fork (only they register flush QPs in the pool).
- **What:** the filter checks group, peer, `remIbDevIdx` and direction but not `qpIdx`, so flush
  entries (`qpIdx == IBCAST_FLUSH_QP_IDX`) inflate the receiver's `primaryNqps` by `ndevs`.
  Currently masked because `nqps <= primaryNqps`, but it defeats the `q % primaryNqps` mapping.
- **Fix direction:** add a `qpIdx >= 0` filter (or adopt the per-comm flush QP).

### QPS-13 — `IbCastCountPeerTotalRefcount` ignores its `ibDevN` argument

- **Status:** open on all branches.
- **What:** the parameter is unused, so group assignment sums refcounts across all local
  devices and skews on multi-NIC nodes.
- **Observe:** count PRIMARY/SECONDARY per group in an INFO log on a multi-NIC run.

### QPS-14 — `destroyedCqs` written without a bound check

- **Status:** open on all branches.
- **Where:** `qp_sharing.cc`, `IbCastCleanupGroupCqs`.
- **What:** `struct ibv_cq* destroyedCqs[NCCL_IB_MAX_DEVS_PER_NIC]` is filled while iterating the
  whole pool with no bound on `nDestroyed`. Any path that puts more distinct CQs in one group
  (merged NICs, duplicate primaries) overflows the stack array.

### QPS-15 — commId free stack has no double-push guard

- **Status:** open on all branches; the QP free stack added on the fork has the same shape.
- **What:** `IbCastFreeCommIdLocked` pushes unconditionally without checking `used` or bounding
  `g_IbCastCommIdFreeTop`, so a double free overflows the stack and aliases commIds. The traced
  paths look single-push today; the guard is cheap insurance.

### QPS-16 — `fail:` paths free the comm while its commId is still registered

- **Status:** open on all branches.
- **What:** `IbCastConnect` and `IbCastAccept` end in `fail: free(comm)`. Any failure after
  commId allocation leaves `g_IbCastCommTable` holding a pointer to freed memory, which the
  routing helpers will dereference; pool entries also keep the freed comm's CQ.
- **Trigger:** injected failure (`ENABLE_FAULT_INJECTION` build).
- **Fix direction:** unwind commId and pool registrations on the failure path.

### QPS-17 — cleanups

- `#if 0` block in `common.cc` (the prepost override) and `#if 1 / #else` in `p2p.cc`.
- `remDevInfo->mtu = rtrAttr->mtu;` in `IbCastSenderQpsToRts` — unrelated to sharing and a
  dead store (`remDevs` is copied from `remMeta` before the call).
- `peerAddr` computed via `ncclSocketGetAddr` and never used in the sender registration block
  (the key uses `handle->connectAddr`).
- `meta.sharedGroupIdx` assigned twice around the label; `sharedPrimaryNqps` and
  `base.remIbDevIdx` written but never read.
- Unused `base` parameter in `IbCastRouteCommFromImmData`.
- Tab indentation in `connect.cc` (6 lines); `qp_sharing.{h,cc}` use 4-space indent against the
  2-space style of the rest of `net_ib_cast`.
- Magic `wc->wr_id & 0xFFFF` / `& 0xff` should go through `IbCastStripCommId` or the `WR_ID_*`
  defines.
- Extract the `rcclParamIbCastCommNGroups() > 0` branches into helpers — `IbCastConnect` and
  `IbCastAccept` each carry ~200 lines of near-duplicate logic, and the `goto`-based flow
  through `qp_sharing_skip_*` / `qp_sharing_done_*` is what made the refcount-leak-on-fallback
  bug easy to miss. Requested by @VadimKutovoi in review.

### QPS-18 — no functional tests in PR #10251

The MPI suite (9 tests, an introspection hook, a reference model and CI presets) lives on the
`*-tests*` branches only. rccl-tests cannot cover comm-lifecycle churn — it creates comms once
and `-N` only repeats the size sweep — so `QpShareStressConnectionChurn` and
`QpShareStressBatchCreateDestroy` are the only coverage for the pool/commId/refcount lifecycle.
The suite also can't currently reach QPS-1: every `PostRecv` in `NetIbMPITestBase` passes
`n = 1`.

---

## Closed on `VadimKutovoi` `users/vkutovoi/qpsharing-rocm-ib-tests`

| Commit | Item |
| --- | --- |
| `4b38e38` | use-after-free: `primaryDevBase` pointed into a freed comm's inline `devs[]`; replaced with `primaryIbDevN` by value |
| `ffcb249` | shared CQ/PD teardown ordering — `IbCastCleanupGroupCqs` moved after the flush-QP and MR loop (was `EBUSY` + orphaned CQ/PD) |
| `ec978a8` | PD refcount leak for secondary comms |
| `71b4d5a` | pool exhaustion via slot reclaim, plus registration-failure rollback instead of a silently unregistered "primary" |
| `90c1aab` | CQ size clamped to device `max_cqe` |
| `f2884af` | admission check on group joins (`IbCastTryJoinSharedQp` / `IbCastLeaveSharedQp`), closing the bare-`refcount++` TOCTOU |

Also fixed within PR #10251 itself: commId exhaustion (free stack, commit `1a32566`) and the
NULL flush QP for secondary receivers (commit `d1d3536`, by sharing the flush QP — the
`*-flush-fix` branches instead keep it per-comm, which is the cleaner option).

## Withdrawn

**`rxPosts` drift on the shared RQ.** Originally raised as a correctness issue and as the
justification for the `#if 0` prepost override in `common.cc`. The per-comm counters sum to the
group's actual RQ occupancy, so a comm saturating at `NET_IB_MAX_REQUESTS` is always offset by
others that have gone negative and keep posting — the RQ cannot starve from drift, and with
<= 8 outstanding receives per comm nothing approaches the cap anyway. The comment in the
disabled block overstates the mechanism. Leaving it compiled out is fine (but see QPS-17: the
dead block should still go).
