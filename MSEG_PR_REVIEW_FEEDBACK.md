# Multi-segment stack PR review feedback

Triage of Copilot, Argus, and human (Bertan) comments on the multi-segment DMA-BUF stack, plus the uncommitted fixes that landed on each layer.

Fixes are **not committed**. They live as working-tree changes on:

| Layer | Branch | Worktree | GitHub PR |
|---|---|---|---|
| 01 | `users/atulkulk/mseg/01-ce-recv-window-offset` | `_wt/mseg-pr01` | [#10887](https://github.com/ROCm/rocm-systems/pull/10887) |
| 02 | `users/atulkulk/mseg/02-amd-host-vmm-compat` | `_wt/mseg-pr02` | [#10891](https://github.com/ROCm/rocm-systems/pull/10891) |
| 03 | `users/atulkulk/mseg/03-registration-range` | `_wt/mseg-pr03` | [#10885](https://github.com/ROCm/rocm-systems/pull/10885) |
| 05 | `users/atulkulk/mseg/05-gin-rma-multisegment` | `_wt/mseg-tests-stack05` | [#10884](https://github.com/ROCm/rocm-systems/pull/10884) |
| 06 | `users/atulkulk/mseg/06-classic-netib-multisegment` | `_wt/mseg-tests-stack06` | [#10886](https://github.com/ROCm/rocm-systems/pull/10886) |
| 07 | `users/atulkulk/mseg/07-hybrid-stress-ci` | `_wt/mseg-tests-stack07` | [#10890](https://github.com/ROCm/rocm-systems/pull/10890) |
| 08 | `users/atulkulk/mseg/08-cast-netib-multisegment` | `_wt/mseg-tests-stack08` | [#10889](https://github.com/ROCm/rocm-systems/pull/10889) |
| final | `users/atulkulk/mseg/multi-segment-dmabuf-cast-gin` | `_wt/mseg-wire` | [#10888](https://github.com/ROCm/rocm-systems/pull/10888) |

No PR04. Global bots (`therock-pr-bot` unit-test filename warning, RCCL perf-gate “NO VERDICT”) are noise and are omitted below.

**Verdicts**

- **Fixed** — real bug or real test hole; code/config changed on the introducing layer.
- **Partial** — the comment is right, but the patch covers the practical hole rather than the strongest theoretical bound.
- **Skipped** — not a bug, out of scope, or a design change that would not pay for itself on this stack.

---

## PR01 [#10887](https://github.com/ROCm/rocm-systems/pull/10887) — CE receive-window offset

### Copilot — `Symmetric_Lsa` runner has no CE env (`RegistrationMPITests.cpp:1224`)

**Comment.** The new AFTER/BEFORE tests skip unless `RCCL_CE_ALLREDUCE=1` and `NCCL_CTA_POLICY=2`. The runner entry Copilot cited did not set those, so CI would skip both regressions.

**Verdict: Fixed on PR07** (that is where the suite JSON lives). `UBR_MultiSegment_Symmetric_Lsa_AFTER` now sets `NCCL_CTA_POLICY=2` as well as `RCCL_CE_ALLREDUCE=1` in all six platform configs. BEFORE already had both. `ubr_multisegment_ce_before` already forces CE + CTA ZERO.

**Why it works.** Windowed CE AllReduce is gated on CTA policy ZERO (integer `2` or the documented `ZERO` spelling). Without that env, `Symmetric_Lsa` hits `GTEST_SKIP` and the offset fix is never executed. Putting the knobs on the suite that *names* the AFTER test makes the advertised regression actually run.

Bertan asked to ignore adding `NCCL_CTA_POLICY`. That is wrong for *these* tests: they explicitly skip unless the policy is ZERO. The production CE path can be selected other ways; this regression cannot.

### Copilot — comment said `rank * chunkBytes` (`RegistrationMPITests.cpp:1286`)

**Comment.** Production used `rank * shardBytes`. `chunkBytes` can differ on multi-chunk reductions.

**Verdict: Fixed.** `ce_fault_inject.h` and the BEFORE test comment now say `shardBytes` and “no window offset”.

**Why it works.** The fault only zeros `recvWindowOffset`. Phase 3 still adds `rank * shardBytes`. Documenting `chunkBytes` would send the next reader to the wrong term in `ce_coll.cc`.

### Argus — `CE_FAULT_LEGACY_RECV_OFFSET` is silent (`ce_coll.cc:1941`)

**Comment.** The other CE faults go through `ceFaultCheck`, which WARNs and returns `ncclSystemError`. This bit corrupts the result with no log, and `FAULT_INJECTION` defaults ON.

**Verdict: Fixed.** WARN at the inject site. Do **not** fail the call.

**Why it works.** The BEFORE test needs AllReduce to *complete* with a corrupted recv window. Returning `ncclSystemError` would make the test fail on the API status instead of observing the legacy bug. A WARN still makes an accidentally armed bit visible in `NCCL_DEBUG` logs, matching the spirit of `ceFaultCheck` without breaking the control.

### Argus — BEFORE/AFTER skip only env, not CE eligibility (`RegistrationMPITests.cpp:1301`)

**Comment.** `count` is \(2^{25}\) bfloat16 elements. At 3/5/6/7 ranks, `count % nRanks != 0` takes the kernel path; the result is correct and `EXPECT_FALSE` fails. `ncclCeImplemented` / `symmetricSupport` can also decline CE.

**Verdict: Partial.** Skip when `count % nRanks != 0`. Do not try to inspect the internal CE selector.

**Why it works.** The CE shard path requires an even split. If the selector falls through to RING, AllReduce is correct, so a BEFORE that demands corruption is a false failure. Skipping on divisibility removes the ranks Argus named. Full “was CE selected?” would need a comm-internal query that these tests do not have; the suite already forces `RCCL_CE_ALLREDUCE=1` and CTA ZERO.

### Argus — `EXPECT_FALSE(verifyAllReduceResult)` is a false pass (`RegistrationMPITests.cpp:1358`)

**Comment.** Any mismatch passes, including a failed `hipMemcpy` or a no-op. Argus wanted the corruption *signature*: own shard holds the AllReduce sum, other recv shards stay zero.

**Verdict: Partial.** After `hipMemset(recv, 0)`, assert recv is still the zero pattern **and** `verifyAllReduceResult` is false.

**Why it works.** The legacy offset only affects *peer* LSA stores (`ncclDevrGetLsaRankPtr` with `recvSlotOffset`). Remote shards land in the send half; recv is not the correct AllReduce. A memcpy/null failure that leaves garbage fails the zero check. A correctly dispatched CE AllReduce fails the zero check. Residual: the local self-copy into `myRecvSlot` does not use the window offset, so shard `rank` in recv *may* be filled; if that shows up on a given runtime, Argus’s per-shard check is the tighter signature. The zero-plus-not-correct pair is enough to reject “any mismatch”.

### Argus — `atoi(ctaPolicy) != 2` misses documented `ZERO` (`RegistrationMPITests.cpp:1223`)

**Comment.** `init.cc` accepts `ZERO` case-insensitively (and `|`-combined forms). `atoi("ZERO")` is 0, so a correctly configured hand run skips.

**Verdict: Fixed.** `envCtaPolicyIsZero()` accepts integer `2` or case-insensitive `ZERO`.

**Why it works.** It matches the documented spelling and the in-tree `2` used by CI. Combined `|` forms are not used by these suites.

### Argus — `fastPath` is rank-local (`ce_coll.cc:1819`)

**Comment.** Fast path vs fallback uses different `ncclMemOpSync` counts. If ranks disagree on the predicate, they hang on `ceSeqNum` rather than returning an error. Argus asked to AllReduce the boolean or fail.

**Verdict: Skipped.** Symmetric windows already imply a shared recv layout; hang only if ranks disagree on recv offsets, which is a broken caller. Cross-rank AND of `fastPath` is a real hardening idea, not a bug in this offset fix.

### Argus — containment fallback untested (`ce_fault_inject.h:49`)

**Comment.** The fault never forces `fastPath` false. Both new tests sit on the in-window boundary.

**Verdict: Skipped.** Out of scope for the offset regression. A “recvbuff past window end” case would be a new test, not a fix to the offset bug.

### Argus — BEFORE/AFTER DRY (~45 duplicated lines)

**Verdict: Skipped.** Style-only.

### Argus unanchored (not inlined)

| Finding | Verdict |
|---|---|
| Fallback stages into ~512 MiB `ceARTmpBuf` with no size cap | Skipped — pre-existing CE staging, not introduced by the offset term |
| Uniform AFTER data is permutation-invariant | Skipped — the AFTER still checks `hasNumSegments` and a correct AllReduce; a slot permutation that preserves the uniform fill is a different test |
| AFTER does not memset recv, BEFORE does | Skipped — AFTER verifies the reduced value; uninitialized recv would fail that |
| PR description links internal Atlassian items | Skipped — process, not code |

---

## PR02 [#10891](https://github.com/ROCm/rocm-systems/pull/10891) — AMD host VMM

### Bertan — `rocmwrap.h` only included if `ROCM_VERSION >= 71200` (`alloc.h:257`)

**Comment.** `NCCL_CUMEM_HOST_VERSION_SUPPORTED(HIP_VERSION)` at the host-alloc gate needs that header on 7.0.2.

**Verdict: Fixed.** Include `rocmwrap.h` on HIP without the 7.12 gate (`enqueue.cc` already does this).

**Why it works.** On HIP 7.0.2, `ROCM_VERSION < 71200`, so the old include was skipped. An undefined function-like macro in `#if` evaluates as 0, so `ncclCuMemHostAlloc` was compiled out on the backport that is supposed to support host VMM. Including the header makes the predicate real: 7.0.2.x and 7.12+ are true, other HIP versions false.

### Bertan — change other `ROCM_VERSION` host-VMM checks to the macro (`alloc.h:794`)

**Verdict: Partial.** The compile-breaking include was the live bug. Remaining `ROCM_VERSION >= 71200` tests that *intentionally* skip pre-7.12 host VMM were left; AllocTests already use the macro.

### Bertan + Copilot — rewrite every rank to host if any rank is host (`dev_runtime.cc:420/445`)

**Comment.** HIP reports imported host VMM as device. The workaround stamped **every** rank’s type to host. A genuine mixed host+device layout at the same index then maps the wrong handle (`reuseLocal`). Bertan: what if layouts differ?

**Verdict: Fixed.** Count host vs device owners. HIP lie is `nHost == 1` plus imported “device” ranks → still rewrite. Reject `nHost > 1 && nDevice > 0` (true mixed owners).

**Why it works.** The HIP bug is: the allocating rank exports host, importers look like device. That is exactly one host owner. Two host owners plus a device rank cannot be that bug; treating the device owner as host would `cuMemMap` the wrong handle. 2-rank genuine host+device is still indistinguishable from the HIP lie and still rewritten — same as before, called out in the comment.

### Copilot — host VMM still reaches `windowRegisterNonSym` → `cudaIpcGetMemHandle` (`dev_runtime.cc:1336`)

**Comment.** `ncclDevrCheckRegistrationSupport` only checks `NCCL_ELASTIC_BUFFER_REGISTER`. Default is 1, so host VMM on LSA>1 still takes the non-sym IPC path that cannot export host VMM.

**Verdict: Fixed.** After a successful probe on the non-sym path, if `hasSysmemSegment && comm->localRanks > 1`, return `ncclInvalidArgument`.

**Why it works.** The reject is **only** on the non-sym path. Symmetric host VMM (the point of PR02) still uses handle sharing, not IPC. Ordinary (non-VMM) allocations are not probed on 7.0.2.2 (HIP faults in `hipMemRetainAllocationHandle`). One-rank / no-IPC windows keep working.

### Copilot — GIN stores AMD `hipMemLocationTypeHost`; device fence only checks `HOST_NUMA` (`dev_runtime.cc:698`)

**Comment.** Device put path in `gin__funcs.h:556` only escalates to system-scope release for `CU_MEM_LOCATION_TYPE_HOST_NUMA`. AMD host sources miss the sysmem fence.

**Verdict: Fixed** by normalizing at store: host segments are recorded as `CU_MEM_LOCATION_TYPE_HOST_NUMA`. Device code is unchanged (it already compiles against the CUDA/HIP-compat `HOST_NUMA` enumerator).

**Why it works.** The fence predicate is a device-side compare. Teaching it `hipMemLocationTypeHost` is fragile on NVIDIA/device bitcode. Storing the enum the device already tests makes AMD host puts take the system-scope path.

---

## PR03 [#10885](https://github.com/ROCm/rocm-systems/pull/10885) — NET registration range

### Copilot — `Generic` is unreachable (`RegistrationMPITests.cpp:1057`)

**Comment.** `UBR_MultiSegment::SetUp` skips unless **exactly one node**. `Generic` requires `min_nodes=2`. The assertion never runs.

**Verdict: Fixed.** Removed the fixture’s blanket single-node skip. Per-test `validateTestPrerequisites` owns node limits (`Symmetric_Lsa` stays `max_nodes=1`, `Generic` stays `min_nodes=2`).

**Why it works.** The skip was a leftover “until dmabuf lands” gate. Dmabuf is this stack. Leaving it in place made the NET `numSegments 8` regression dead code. Tests that must stay single-node already say so.

### Bertan — why `min_nodes=2`? Want single-node IPC too

**Verdict: Skipped as a product change.** `Generic` is specifically “IPC-only cannot satisfy NET `numSegments 8`”. A single-node IPC variant is `Generic_Reuse`, which already allows 1 node. Dropping `min_nodes=2` would make Copilot’s “assert NET” check pass on IPC logs.

### Copilot — `hasNumSegments` matches any `numSegments N` (`RegistrationMPITests.cpp:1115`)

**Comment.** NET failure logs also print `numSegments 8`.

**Verdict: Fixed.** `hasNETRegistrationWithSegments(n)` requires `NET register userbuff` and `numSegments n` on the same line (the success INFO in `net.cc`).

**Why it works.** The failure path is `failed to NET register userbuff` / “not registered as DMABuf”, which does not match `NET register userbuff … numSegments N`.

---

## PR05 [#10884](https://github.com/ROCm/rocm-systems/pull/10884) — classic GIN/RMA

### Copilot + Bertan — send QP depth (`connect.cc:666`)

**Comment.** `max(2*NET_IB_MAX_REQUESTS, NCCL_RMA_MAX_SIGNAL_WRS)` is 512. One `iputSignal` can post 33 WRs. Inflight is 256. The 16th such request can overflow the SQ. Size for inflight×WRs **or** add WR-credit backpressure. Bertan: looks valid.

**Verdict: Partial.** Additive depth: `2 * NET_IB_MAX_REQUESTS + NCCL_RMA_MAX_SIGNAL_WRS` (545). Same idea as CAST’s `NCCL_IB_RMA_MAX_SEND_WRS`, without porting CAST’s credit loop onto classic GIN.

**Why it works.** `max(512, 33)` never grew the SQ for GIN; 512 was already the classic NET budget. Adding 33 gives room for one full GIN signal chain on top of that budget. Residual: 256 concurrent 33-WR iputs would still overflow; that needs CAST-style `rmaWrsOutstanding` credits, which classic GIN does not have. 545 is the same class of fix CAST used for the RMA QP.

### Copilot — flush QP depth (`connect.cc:1300`)

**Comment.** One flush posts up to 16 reads; pool allows 256 inflight flushes.

**Verdict: Partial.** `NET_IB_MAX_REQUESTS + NCCL_RMA_MAX_FLUSH_WRS`. Same rationale as send: one extra max-width flush on top of the classic budget.

### Copilot — `EXPECT_EQ` on `RegMr` then `iput` (`RmaMultiSegmentMPITests.cpp:279`)

**Verdict: Fixed.** `ASSERT_EQ` so a failed register stops before `iput` on a null handle.

**Why it works.** `EXPECT_*` is non-fatal. The next statement used `sendMh`/`recvMh`. A registration bug then SIGSEGVs instead of a gtest failure.

### Copilot — `SyncSkip(r == ncclSuccess)` hides unilateral success (`RmaMultiSegmentMPITests.cpp:1064`)

**Comment.** `SyncSkip` is ANY-rank. If rank 0 succeeds (same enumeration) and rank 1 is rejected, everyone skips. The bug this test exists to catch never fires.

**Verdict: Fixed.** Skip only if `allRanksTrue(r == ncclSuccess)`.

**Why it works.** Identical enumeration → both succeed → skip (no asymmetry). One rank succeeds and the other is rejected → do not skip → `EXPECT_EQ(r, ncclInvalidUsage)` fails on the succeeding rank. That is the collective-guard hole.

---

## PR06 [#10886](https://github.com/ROCm/rocm-systems/pull/10886) — classic NET/IB P2P

### Copilot + Bertan — multi-seg `netIbRegMrMultiSeg` failure falls through to HSA (`net.cc:2535` / `:2614`)

**Comment.** `needReg` stays true, then `ncclHsaRegMrDmaBuf` registers the whole VA as one DMA-BUF (first physical segment only). Bertan agreed.

**Verdict: Fixed.** HSA path is `needReg && numSegments <= 1`. Multi-seg failure then hits the existing `numSegments > 1 → goto fail` (null handle / staging).

**Why it works.** The comment above `netIbRegMrMultiSeg` already claimed “on failure leave unregistered”. HSA still ran. Whole-range HSA is exactly the first-segment-only MR this stack exists to avoid. Gating it on `numSegments <= 1` makes the comment true.

### Copilot — segmented send skips profiler init (`p2p.cc:132`)

**Comment.** Completion still stops event handles. Stale `nEventHandles` / `qpIndex` on profiling builds.

**Verdict: Fixed.** Reset `reqs[r]->pInfo[0].nEventHandles = 0` before the QP loop, matching `ncclIbMultiSend`.

**Why it works.** The completion path always iterates `nEventHandles`. Without a reset, a previous request’s count is reused and `ncclProfilerNetEventStop` runs on leftover handles.

### Copilot — SQ vs `NCCL_IB_MAX_WRS_PER_SEND` / CTS two-WR (`common.h:281`, `p2p.cc:235`, `:724`)

**Comment.** Bertan: covered in #10884.

**Verdict: Partial / skipped here.** Send/flush QP growth is on PR05 (shared `connect.cc`). CAST-style CTS two-WR under resiliency is not reproduced on classic: resilient CTS is one signaled WR per request. Residual SQ pressure from two concurrent 256-WR segmented sends remains the same class as PR05’s partial depth bump.

---

## PR07 [#10890](https://github.com/ROCm/rocm-systems/pull/10890) — hybrid / stress / CI

### Copilot + Bertan — `HybridVmmHelpers.hpp` included before the `hipMemLocationTypeHost` shim (`RegistrationMPITests.cpp:28`)

**Comment.** On ROCm < 7.0.1 the helper does not compile. Bertan: looks valid.

**Verdict: Fixed.** Shim moved into `HybridVmmHelpers.hpp` immediately after `<hip/hip_runtime.h>`. Duplicate shim removed from the .cpp.

**Why it works.** The helper uses `hipMemLocationTypeHost` at parse time. A `#define` after the include is too late. The header is also used from RMA tests, so one shim covers every TU.

### Copilot + Bertan — DeepEP alloc skip is rank-local (`RegistrationMPITests.cpp:1893`)

**Comment.** One rank `GTEST_SKIP`s, peers enter `ncclCommWindowRegister` and hang. Register path already uses `allRanksTrue`. Alloc path needs the same.

**Verdict: Fixed.** `allRanksTrue(buf.totalSize != 0)` before skip, for DeepEP elastic and mixed host-VMM tests.

**Why it works.** `GTEST_SKIP` returns from that rank only. MPI collectives after it have no matching call. Reducing allocation success first makes every rank skip or every rank proceed.

### Copilot — AFTER suites missing `NCCL_CTA_POLICY=2` (six JSON files)

Covered under PR01. **Fixed** despite Bertan’s “ignore”.

---

## PR08 [#10889](https://github.com/ROCm/rocm-systems/pull/10889) — CAST P2P multi-segment

### Copilot + Bertan — remap leak on failed post (`p2p.cc:341`)

**Comment.** `IbCastQpSchedGetRemap` then `wrap_ibv_post_send`. Remap is freed only on CQE (`IbCastQpSchedFreeRemap`). Failed post / fault inject leaks an entry and can exhaust the pool. Bertan: same leak in `IbCastMultiSend`.

**Verdict: Fixed.** Hoist `remapWrId`. On fault inject and failed `ibv_post_send`, `IbCastQpSchedFreeRemap` if non-null. Both segmented send and `IbCastMultiSend`.

**Why it works.** The remap object is a pool entry keyed by WR id. A CQE never arrives if the WR was not posted, so the free-on-CQE path never runs. Freeing on the error path returns the slot. Successful posts still free on CQE only.

### Copilot — CTS two-WR vs QP sized `NET_IB_MAX_REQUESTS` under resiliency (`p2p.cc:912`)

**Comment.** Bertan: covered in #10884.

**Verdict: Skipped.** CAST connect already documents: resiliency signals every CTS, so the SQ is one WR per request (`NET_IB_MAX_REQUESTS * (resiliency ? 1 : 2)`). The two-WR (unsignaled + signaled) budget is the non-resiliency case, which already uses `* 2`.

### Copilot — flush QP still `NET_IB_MAX_REQUESTS` (`p2p.cc:1185` / `connect.cc:1499`)

**Comment.** One flush can post `NCCL_NET_IB_MAX_RECVS * NCCL_IB_MAX_SEGMENTS` = 8×16 = 128 WQEs. `NET_IB_MAX_REQUESTS` is 256.

**Verdict: Skipped as not currently overflowing.** 128 < 256 for a single flush. Multiple outstanding max-width flushes could still pile up; CAST RMA flush depth is handled on the final PR.

### Copilot — segmented send telemetry (`p2p.cc:345`)

**Verdict: Skipped.** Metrics-only; does not change data path.

### Copilot + Bertan — CAST multi-seg suite never sets `NCCL_GDR_FLUSH_DISABLE=0` (`net_ib_transport.json:478`)

**Comment.** CAST defaults `IbCastGdrFlushDisable=1`. `IbCastIflush` is a no-op. Bertan agreed.

**Verdict: Fixed.** `cast_multisegment` sets `NCCL_GDR_FLUSH_DISABLE=0`.

**Why it works.** `connect.cc` enables the flush QP only when that param is 0 (plus GDR). Without it, `MultiSegmentFlush*` tests skip or pass without posting per-segment flush WRs.

---

## Final PR [#10888](https://github.com/ROCm/rocm-systems/pull/10888) — CAST GIN/RMA

### Copilot — AINIC recovery recreates flush QP at `NET_IB_MAX_REQUESTS` (`connect.cc:1500` / `p2p_resiliency_recovery_ainic.cc:78`)

**Comment.** Initial RMA flush QP is `NCCL_IB_RMA_MAX_FLUSH_WRS` (271). Credits still allow 271 after recovery, but the new QP is 256 deep.

**Verdict: Fixed.** `IbCastBuildDataQpCreateAttr` uses `isRma ? NCCL_IB_RMA_MAX_FLUSH_WRS : NET_IB_MAX_REQUESTS` on the recv/flush side. Recovery no longer overwrites `maxSendWorkRequest` after that helper.

**Why it works.** Recovery already called `IbCastBuildDataQpCreateAttr` then clobbered the depth. Removing the clobber keeps RMA depth across AINIC destroy/recreate. Non-RMA flush QPs stay at `NET_IB_MAX_REQUESTS`.

### Copilot — `ginMrHandle` calloc before first AllGather (`gin.cc:607`)

**Comment.** One-rank `NCCLCHECK` return while peers block at the transcript AllGather.

**Verdict: Partial.** Allocate `registrations` first (still fail-out if that calloc fails — no recv buffer). Allocate `ginMrHandle` with `goto reconcile` on failure so the existing status AllGather still runs. Guard `memcpy` of `segOff` if the handle is null.

**Why it works.** The hang is “I returned, you AllGather”. Putting the handle failure *into* `localRegistration.status` and still gathering lets peers see `ncclSystemError` and take `fail` together. A failed `registrations` calloc still cannot gather; that path needs a bootstrap scratch buffer that this function does not have.

### Copilot — `IbCastRmaReleaseWrs` on any `post_send` error (`gin.cc:857`)

**Comment.** `ibv_post_send` can accept a prefix and return `bad_wr` at the first unposted WR. Releasing all credits and freeing the request leaves unsignaled WRs in flight against a recycled slot.

**Verdict: Fixed.** `IbCastRmaPostWrs` walks the WR list to `bad_wr`. Posted==0 → release credits, caller frees the request. Posted>0 → charge only the accepted prefix, WARN, leave the request for CQ drain, return the error **and** `*request`.

**Why it works.** Only the last WR is signaled. A prefix in flight has no CQE until the signaled tail — which never posted. Keeping `rmaNwrs` and the request prevents credit underflow and use-after-free. Returning the request lets `test()` poll if the caller still holds it; `NCCLCHECK` abort still leaks that slot until comm destroy, which is safer than recycling it.

### Copilot — CAST RMA JSON `extends` merges tests by name (`mi300_mellanox_ib_func.json:2910`, `mi300x_mellanox_ib.json:4524`)

**Comment.** Child adds a new test name, so parent `NetIbMultiSegmentMPITest.*` and UBR proxy tests are inherited. CAST RMA runs twice plus unrelated NET/UBR under `NCCL_NET=IB-CAST`.

**Verdict: Fixed.** `multisegment_cast_rma_multinode` now extends `default` and copies ranks/timeout/env itself, with only the CAST RMA filter.

**Why it works.** `test_config.py` merges tests by `name`. A unique child name cannot replace the parent list. A test-free base (or not extending a test-bearing parent) is the only way to get a single filter.

---

## Residual / follow-ups (not patched)

1. Classic GIN still has no WR-credit loop; SQ is “classic + one max GIN chain”, not `256 × 33`.
2. BEFORE recv-zero vs Argus per-shard signature if the local self-copy fills shard `rank`.
3. CAST segmented-send telemetry counters.
4. `fastPath` cross-rank AND.
5. CE containment-fallback (recv past window) test.
6. Host-VMM 2-rank genuine mix vs HIP import lie (unavoidable without a better runtime type).
7. `CE_FAULT_LEGACY_RECV_OFFSET` is `0x08`; a future develop CE-fault PR must not reuse that bit.

---

## How to land

Commit on the **introducing** branch, then `git rebase --onto <new-parent> <old-parent>` down the stack. Stash `mseg-wire` IB/`test_executor.py` dirt around the final rebase. Do not force-push unless asked. Do not mix the HIP 7.0.2.2 `CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE` stash into this stack.
