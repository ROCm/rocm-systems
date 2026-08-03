# KFD Dispatch-Log: Signal-less Kernel-Dispatch Completion (design plan v3)

Status: PLAN (not yet implemented). Signal-less mode must ship feature-disabled
until the full Phase 2+3 change-set lands and passes the test matrix below.

## Motivation

On the inline queue-interposition path, rocprofiler-sdk currently allocates and
arms an HSA completion signal for **every** kernel dispatch, enables HSA queue
profiling, and runs an async waiter thread that fires `get_dispatch_time()` +
`dispatch_complete()` + correlation-id teardown when the signal completes. When
the KFD dispatch-log is available, the firmware record already carries the true
hardware dispatch timestamps *and* (per guarantee G3 below) is an authoritative
completion event. The signal machinery is therefore pure per-dispatch overhead
for KFD-served dispatches.

Primary goal: for a KFD-eligible inline dispatch, STOP allocating/arming the HSA
completion signal, STOP enabling HSA queue profiling, and STOP the async signal
wait. The firmware EOP record becomes the completion authority.

Secondary goal: seal the current per-dispatch KFD/HSA mixing race — a KFD-eligible
dispatch must not silently use HSA timestamps just because its firmware record was
late; it waits for the record. Only a detected loss resolves otherwise.

## Guarantees / constraints (maintainer-confirmed ground truth)

- G1. Ring size is fully controllable (env-var, default a few MB). `buffer_size`
  ioctl field is `uint32`. Reduces but does not eliminate overrun.
- G2. No backpressure. The ring can overrun; a record can be permanently lost.
- G3. EOP-record-visible == the kernel launch has completed. An observed EOP is an
  authoritative completion event.
- G4. Inline is the only KFD path. Counters/ATT/PC-sampling/SPM/HIP-graph/serializer
  force the legacy path (registration.cpp gate), so they never coexist with an
  inline KFD dispatch. The completion signal's only SDK consumer on the inline path
  is kernel-dispatch tracing itself.
- G5. 4 KiB doorbell page is guaranteed on all supported platforms. The reader's
  page-relative slot reduction (`kDoorbellSlotsPerPage=1024`) and the capture-side
  `sysconf(_SC_PAGESIZE)` reduction therefore always agree; no page-geometry gate
  is required.

## Policy decisions

- P1. Loss policy = "leak and shout." On overrun (the only expected loss source),
  do NOT force any terminal completion. Affected pending entries are stranded: no
  record emitted (absent record), correlation ids deliberately NOT retired
  (accepted leak, including at global finalization). A loud warning names that an
  overrun occurred, how many dispatches / unique correlation ids were stranded, and
  tells the user to raise the ring via the env var. No watchdog-forced completion,
  no wrong-dispatch completion, ever.
- P2. Absent record on loss is acceptable. There is intentionally no public
  "timestamp unavailable" record/status today. TODO: add such an API in the future
  (tracked separately); out of scope now.
- P3. App-provided completion signals: in KFD (signal-less) mode the SDK does NOT
  touch the packet's completion signal at all — no copy-modify, no +1 at enqueue,
  no -1 at completion, no pooled-signal borrow. The packet the GPU sees is
  byte-identical to what the app submitted (app signal present or null). The SDK
  learns completion solely from the EOP record; it never waits on, bumps, or reads
  the app signal. App-signal presence is irrelevant to KFD eligibility. Documented
  behavior change: SDK record emission is decoupled from / no longer ordered before
  the app observing its own signal's terminal value.

## Overrun detection: `wptr` is definitive

`wptr[region]` is a monotonic atomic integer the kernel advances one-per-record,
AFTER writing that record's slot. The reader reads it `__ATOMIC_ACQUIRE`. The key
ordering property is directional: the producer's write frontier is at `wptr` and
advances AWAY from the reader's `rptr`. So every slot the reader scans in
`[rptr, w0)` is a slot the producer has already moved past — it is not being
concurrently written — UNLESS the producer has wrapped the whole ring and its
frontier has caught back up to `rptr`. That wrap-collision is exactly the overrun
condition, so `wptr` is the single, ordering-sound authority for overrun. The
`KFD_DLOG_STREAM_OP_STATUS` `source_overruns` counter is diagnostics-only and the
design does not depend on its ordering.

Accepted-risk edge (maintainer-approved): the ONLY window where the reader could
read a slot the producer is actively writing is when the producer's frontier has
lapped and collided with `rptr` (`w - rptr >= region_slots`). That is already an
overrun — data around that position is already presumed lost/corrupt — so possible
corruption of the record(s) at exactly that collision point is acceptable and is
subsumed by the leak-and-shout policy (P1). We do not add a per-slot
sequence/commit marker; the collision is rare (requires the reader to fall a full
ring behind) and its records are discarded as overrun anyway.

Boundary predicate (IMPORTANT): the safe-scan region is `w - rptr < region_slots`;
the overrun latch fires at `w - rptr >= region_slots` (NOT strict `>`). At exactly
`w - rptr == region_slots` the producer's next write target IS the slot at `rptr`
(the collision), so a full ring is treated as overrun, not a safe drain. (The
current non-signal-less drain uses strict `>` with silent
`scan = w - region_slots + 1` recovery at `dlog_drain.hpp:161`; signal-less mode
replaces that with the `>=` latch + no silent recovery.)

Transactional per-region drain:

1. `w0 = load_acquire(wptr[region])`.
2. If `w0 - rptr[region] >= region_slots` -> overrun: publish nothing, latch
   session `LOSS_POISONED`, transition all still-matchable pending entries ->
   `LEAKED`, loud warning, disable signal-less for the process. (No silent
   `w - region_slots + 1` recovery in signal-less mode — that recovery is what
   silently drops lapped records.)
3. Else scan `[rptr, w0)` and stage candidate pairs (do not publish yet).
4. `w1 = load_acquire(wptr[region])`. If `w1 - rptr[region] >= region_slots` the
   producer lapped us mid-scan -> discard staged pairs, latch overrun as in (2).
5. Else commit staged pairs to the hub and advance `rptr[region] = w0` (release).

Both reads are of the same acquire-ordered atomic, so there is no cross-channel
ordering hazard. The before/after bracket closes the mid-scan lap window; the
`>=` boundary ensures the only unstable slot (the frontier==rptr collision) is
classified as overrun rather than scanned.

## Completion terminal states (three, not two)

| Terminal | Cause | Record emitted | Corr-id retired |
|---|---|---|---|
| `RESULT_READY` | EOP arrived, START matched (start_ticks known), tick-convert + sanity OK | yes (KFD timestamps) | yes |
| `COMPLETED_NO_TIMING` | EOP arrived (kernel done per G3) but EITHER start_ticks unknown (START lost, EOP matched a unique current PENDING under a non-overrun drain) OR tick-convert/sanity failed | no (absent) | yes (normal) |
| `LEAKED` | EOP lost to `wptr` overrun, reader dead, slot quarantine, or teardown-while-PENDING | no (absent) | no (deliberate leak) |

(`EOP_PROVEN` is the non-terminal handoff state between the reader proving
completion and the worker producing one of `RESULT_READY`/`COMPLETED_NO_TIMING`;
see the state machine.)

Tick-conversion failure detail: `hsa_amd_profiling_convert_tick_to_system_domain`
rebases raw firmware GPU ticks onto the CLOCK_BOOTTIME system domain. Per ROCr
source it only fails on invalid/non-GPU agent (a should-never-happen-for-a-live-
dispatch condition) — it is pure arithmetic otherwise. It is distinct from the
`kfd_time_is_sane` correlation guard. Because the EOP proves completion (G3),
conversion failure is NOT a leak: emit no record but retire the correlation id
normally (`COMPLETED_NO_TIMING`).

## Eligibility (decided once per BATCH at enqueue, before writer()/doorbell publish)

A batch is signal-less iff ALL hold:
- inline path (always true here per G4);
- session live for THIS gpu_id AND reader alive (not `LOSS_POISONED`/`READER_DEAD`);
- every packet's doorbell resolves to a slot with an INJECTIVE live owner (no live
  page-relative-slot collision across ALL live compute queues on the GPU);
- the owned completion payload can be constructed safely (no raw `Queue&` / raw
  context pointers escaping — value data + stable ids/tokens only).

If ANY packet fails, the WHOLE batch keeps today's signal path (no mixed-mode
batches). App-signal presence is NOT a gate (P3).

## Architecture

Invert control: the reader matches each drained pair to a pending-completion
registry entry (the "hub") and hands finalization to the existing inline async
task-group. The reader NEVER runs client callbacks / tick conversion / teardown
and holds no hub/queue lock across the handoff.

- Enqueue (signal-less batch): register a pending entry per dispatch BEFORE
  publishing the packet/doorbell, keyed by `correlation_key`, holding an owned
  minimal payload (callback_record; owned/validated tracing context handles;
  corr-id refcount handle; tid; external corr ids; queue token + full submit index
  for invariant checks). Do NOT touch the packet signal, do NOT enable HW profiling.
- Reader: on a committed pair (a matched START+EOP, OR an EOP matching a unique
  current PENDING entry whose START was lost to a non-overrun drain — see the state
  machine), under the hub lock transition `PENDING -> EOP_PROVEN` exactly once, take
  ownership, release lock, hand payload to the task-group finalizer. The reader does
  NOT convert or emit; it only proves completion and hands off. (It never sets
  `RESULT_READY`/`COMPLETED_NO_TIMING` — those are the worker's terminal outcomes.)
- No-signal finalizer: accepts the owned KFD tick pair directly; converts +
  sanity-checks; on success emits the KFD record and retires the corr-id; on
  convert/sanity failure emits nothing but retires normally (`COMPLETED_NO_TIMING`).
  Contains NO HSA signal fallback (the app may have destroyed its signal). Uses an
  RAII cleanup guard so a callback failure cannot skip corr-id / queue-counter
  cleanup.
- Lazy HW profiling: `hsa_amd_profiling_set_profiler_enabled(queue,true)` only on a
  queue's first SIGNAL-path batch. Pure signal-less queues never enable it.
  (ROCr tick conversion does not require queue profiling; agent-level clock sync
  handles it. Confirm on hardware.)

## Correctness requirements that must land WITH signal-less enablement

These bite even with perfect delivery because removing the signal removes the safe
HSA fallback that currently masks them:

1. Overrun = permanent loss latch (above). Not an observability-only epoch.
2. Remove the 5-second start eviction for live signal-less pending entries — else a
   legitimate >5s kernel is silently stranded without an overrun. Store
   `start_ticks` in the pending entry when START arrives; clear only on EOP,
   `LEAKED`, quarantine, reader death, or global loss poisoning.
3. Reverse doorbell-owner map for ALL live compute queues on the session GPU (not
   just signal-less ones). On a second live owner for a slot: quarantine the slot,
   stop new reservations for every owner, leak existing pending entries or complete
   a proven drain barrier, then let the colliding queue publish via the signal path.
   Quarantine the slot for the rest of the process.
4. Quiesce-or-quarantine on EVERY generation transition / queue destroy: before
   bumping the doorbell generation (queue_controller.cpp destroy path), mark the
   queue/slot closing under the hub lock (release it), fence in-progress
   `process_doorbell_impl()` via `QueueState::gate_lock` (release it), transition
   remaining pending -> `LEAKED`, permanently quarantine the slot, clear reader-side
   unmatched starts / stale results for the slot. Never hold a hub/queue lock across
   a reader wait or a `gate_lock` acquire.
5. No-signal finalizer never reaches the generic HSA `get_dispatch_time` fallback.
6. `correlation_id_finalize()` must SKIP force-retirement for IDs in the P1 loss
   ledger (explicit code change). `LEAKED` entries release their payload but omit
   `sub_kern_count`/`sub_ref_count`; move them to non-matchable tombstones so low-32
   key reuse cannot reactivate them.
7. Hub-aware synchronization (lands WITH Phase 2, not Phase 3): code-object unload,
   queue-controller sync, client detach/finalize, and global finalize must fence
   (a) interceptor registration/publication, (b) reader drain/result production,
   (c) ready-task handoff, (d) task execution. `interposition_sync()` today joins
   only already-enqueued tasks, not records still in the firmware ring.
   TOTAL TEARDOWN ORDER (strict, no reordering) — this is the ONLY ordering that
   guarantees no `EOP_PROVEN` entry is stranded and no task is enqueued after join:
   1. Stop new inline reservations (eligibility returns false; no new PENDING).
   2. Quiesce in-flight interceptor registration/publication (fence via the queue
      gate_lock path) so no new PENDING/EOP_PROVEN can be created.
   3. Stop + join the reader (final status query + final drain first) so no new
      `PENDING -> EOP_PROVEN` and no new task/retry-owner insertions can originate
      from the reader.
   4. Flush the retry owner: any `EOP_PROVEN` still held by the bounded retry owner
      is finalized in place now (steps 1-3 guarantee NO producer can add to it after
      this point — this closes the "late retry insertion after flush" window).
   5. Deliver all ready/`EOP_PROVEN` tasks; transition any still-`PENDING` -> `LEAKED`.
   6. Join the task-group (safe now: steps 1-4 mean no producer remains).
   7. Then run the existing queue_controller_fini / kfd::finalize /
      correlation_id_finalize (with the loss-ledger exclusion).
   Execution context: the retry owner's synchronous finalize-in-place (permanent
   task-submission rejection) runs the SAME no-signal finalizer as a worker, but on
   the thread that performs the flush (step 4, i.e. the teardown/interposition-sync
   thread) — NOT the reader thread and NOT while holding any hub/queue lock. Client
   callbacks it invokes are therefore on a normal SDK thread, identical to the
   worker case, satisfying invariant 11.
8. Fork: disable ALL inline interposition in the child (the inline task-group is a
   separate static object not rebuilt by the general fork handler). Owner-PID/epoch
   gates all hub use; the atfork child handler writes only lock-free scalar state
   (no map clear, no mutex, no allocation, no logging). Inherited hub/task-pool
   objects are abandoned without running destructors; child must survive a normal
   `exit()`.

## Rendezvous invariants (hub)

Session modes: `RUNNING -> {LOSS_POISONED | READER_DEAD | STOPPING | CHILD_STALE}`;
no terminal mode returns to `RUNNING` without recreating the stream.

Entry state machine (note: EOP arrival and result finalization are SEPARATE
steps on separate threads, so there is a non-terminal handoff state between them —
the reader proves completion, the task-group worker later converts + finalizes):
```
ABSENT --whole-batch atomic register--> PENDING(start_ticks: none|present)

// reader thread, under hub lock: an EOP proves completion (G3), take ownership,
// hand payload to the task group. This is the single "result path wins" point.
// Two EOP shapes both prove completion and both go to EOP_PROVEN:
//   (i)  matched START+EOP           -> start_ticks known
//   (ii) EOP matching a unique current PENDING entry whose START was lost, drained
//        under a NON-overrun (loss-free) drain -> start_ticks unknown
PENDING --loss-free EOP (matched or unique-current), drain committed--> EOP_PROVEN

// task-group worker, off all hub/queue locks: convert ticks + sanity-check.
// A missing start_ticks (shape ii) goes straight to COMPLETED_NO_TIMING.
EOP_PROVEN --start_ticks known, convert + sanity OK--> RESULT_READY (emit KFD ts, retire)
EOP_PROVEN --start_ticks unknown OR convert/sanity fail--> COMPLETED_NO_TIMING (absent, retire)

// loss path (competes with the PENDING->EOP_PROVEN transition under the hub lock):
PENDING --overrun/reader-dead/quarantine/teardown--> LEAKED (absent record, NO retire)

// task-submission disposition (does NOT change the completion outcome; kernel is
// done per G3, corr-id WILL be retired):
//   temporary reject  -> stays EOP_PROVEN, held by the bounded retry owner, retried
//   permanent reject   -> the retry owner finalizes SYNCHRONOUSLY in place
//                         (convert+emit-or-COMPLETED_NO_TIMING, retire), never LEAKED
// The retry owner is drained/flushed before task-group join at teardown, so no
// EOP_PROVEN entry is ever left un-retired.
```
`EOP_PROVEN`, `RESULT_READY`, and `COMPLETED_NO_TIMING` are all COMPLETION
outcomes (kernel is done per G3): once an entry leaves `PENDING` via the EOP path
it can NEVER become `LEAKED`, and it retires the corr-id exactly once regardless of
which of the two terminal conversion outcomes it reaches. The result-vs-loss race
is decided by a single winner at the `PENDING` exit under the hub lock: either the
reader claims `PENDING->EOP_PROVEN` or loss handling claims `PENDING->LEAKED`.
After that claim, conversion failure only chooses between `RESULT_READY` and
`COMPLETED_NO_TIMING`; it cannot cross to `LEAKED`. No watchdog transition. No HSA
fallback after any of these. `RESULT_READY`/`COMPLETED_NO_TIMING`/`EOP_PROVEN`
never cross to `LEAKED` and vice versa. Task-submission rejection (U17) keeps the
entry in `EOP_PROVEN` owned by a bounded retry owner that is drained before
task-group join at teardown — it never reverts to `PENDING` or `LEAKED`.

1. Whole-batch atomicity: all entries inserted before the first packet publishes,
   or none inserted and the whole batch uses the signal path.
2. One matchable owner per key; its slot has exactly one live queue owner.
3. No overwrite semantics: duplicate register / duplicate START / ambiguous
   duplicate result is a quarantine/error, not first-writer-wins.
4. Single race winner under the hub lock: result delivery XOR loss handling.
5. No early publication: candidate pairs commit only after the after-scan `wptr`
   bracket proves no concurrent lap.
6. Exact cleanup: `RESULT_READY`/`COMPLETED_NO_TIMING` retire the corr-id at most
   once; `LEAKED` never retires.
7. Per-queue outstanding count: ++ on register, -- on terminal; queue destroy waits
   for it or explicitly leaks it.
8. Loss ledger of unique leaked correlation ids excluded from force-retirement.
9. A slot cannot change owner/generation while it has a matchable entry or retained
   START.
10. Payloads hold no app-signal ownership, no raw `Queue&`, no unprotected
    context/code-object lifetime.
11. No hub/queue lock held across task enqueue, conversion, buffer emplace, logging,
    client callbacks, or waits.
12. Producer quiescence before task-group join at teardown.

## Phasing

- Phase 0 (small, independently shippable): env-var ring size + validated sizing
  math; `wptr`-based overrun detection with the loud P1 warning; `OP_STATUS`
  diagnostics; reader liveness (`reader_dead` + clear `session_ready`).
- Phase 1 (small, signals RETAINED): replace the one-shot `take()` with an
  event-driven rendezvous (CV notified by `deposit()`, one absolute batch deadline,
  wait on the packet's own signal before HSA fallback, fallback made visible); fix
  the async-handler teardown hazard. IMPORTANT: making KFD hits more reliable makes
  a wrong-queue collision more consequential, and Phase 1 still selects a KFD result
  whenever it passes the broad `kfd_time_is_sane` CPU-window guard (which two
  colliding queues with overlapping windows can both pass). Therefore Phase 1 must
  satisfy EXACTLY ONE of: (a) BOTH the full all-live-queue injective-owner guarantee
  (requirement 3 below, including queues that existed before session establishment)
  AND the generation-transition/reuse closure (requirement 4 below) are implemented
  and active, so a KFD result is only ever selected for a slot with a unique live
  owner AND cannot be an old-generation/reused-slot record; or (b) KFD result
  selection is GLOBALLY DISABLED in Phase 1 (the rendezvous only makes an
  already-uniquely-owned result wait; if the requirement 3+4 tracking is not in
  Phase 1, Phase 1 ships as pure race-seal with KFD selection off and delivers no
  KFD timestamps, its value being the deterministic race-seal + rendezvous/teardown
  groundwork). "Minimal/where-known" gating is NOT acceptable. NOTE: current-owner
  uniqueness alone (requirement 3 without 4) is INSUFFICIENT for option (a) — it
  does not prove a record belongs to the current owner vs a previous generation of
  the slot.
- Phase 2+3 (the primary goal, must land TOGETHER, feature-disabled until complete):
  signal-less eligibility + hub + reader->task-group handoff + no-signal finalizer
  + lazy profiling + P3 no-touch-packet + all correctness requirements 1-8 above.

## Test matrix (write alongside development)

Unit / deterministic (gtest, `kfd-dlog-*` targets, no GPU; TSan/ASan/UBSan/LSan):

- U1. Hub state machine: every legal transition; illegal transitions rejected.
- U2. Whole-batch atomic register; rollback on any per-packet failure leaves NO
  partial registration and NO signal mutation.
- U3a. (Phase 1 rendezvous) result-deposited-before-waiter and waiter-before-deposit
  both resolve exactly once. U3b. (Phase 2 hub) result-before-register is REJECTED
  (a Phase 2 result with no matching PENDING entry is stale/ambiguous -> not cached,
  not applied to a later same-key dispatch); register-before-result resolves once.
  (Split: the two phases have OPPOSITE correct behavior here.)
- U4. Result-vs-loss race: under the hub lock exactly one of `PENDING->EOP_PROVEN`
  (reader) or `PENDING->LEAKED` (loss) wins; the loser is a no-op
  (linearizability test under TSan). After `EOP_PROVEN`, convert failure yields
  `COMPLETED_NO_TIMING`, never `LEAKED`.
- U5. `wptr` overrun latch uses `w - rptr >= region_slots` (NOT strict `>`) before
  scan -> publish nothing, all still-matchable entries `LEAKED`, session
  `LOSS_POISONED`, one warning.
- U6. Mid-scan lap: after-scan `wptr` bracket (`w1 - rptr >= region_slots`) catches a
  producer lap during the scan -> staged pairs discarded, overrun latched.
- U7. Boundary: `w - rptr == region_slots - 1` drains fully (safe); `w - rptr ==
  region_slots` (exactly full) is treated as OVERRUN (the frontier==rptr collision),
  NOT a safe drain. (Corrects the earlier "exactly full is safe" assumption.)
- U7b. Accepted-risk collision: with `w - rptr >= region_slots`, the record(s) at
  the frontier==rptr position may be torn; assert this is classified as overrun
  (LEAKED + warning), i.e. corruption is never committed as a valid pair — it is
  discarded as loss. (No per-slot sequence marker; this is the maintainer-approved
  edge.)
- U8. START retained beyond 5s for a live pending entry (no eviction); its later EOP
  still pairs.
- U9. EOP-without-START matching a unique current PENDING entry under a loss-free
  (non-overrun) drain -> `EOP_PROVEN` (start_ticks unknown) -> `COMPLETED_NO_TIMING`
  (retire). An unmatched EOP with NO PENDING entry, or any EOP under an overrun
  drain, completes nothing (must not fabricate a completion for an arbitrary EOP).
- U10. Tick-convert failure -> `COMPLETED_NO_TIMING` (absent record, normal retire),
  never HSA fallback, never `LEAKED`.
- U11. Low-32 dispatch-id recurrence: a leaked entry is non-matchable (tombstoned) so
  a later same-low-32 record cannot reactivate it.
- U12. Reverse-owner collision detection: two traced queues; traced + signal-path
  queue; traced + untraced live queue — all quarantine the slot before the second
  owner publishes.
- U13. Generation reuse: an old signal-path/late record cannot be tagged with a new
  queue's generation and matched to a new dispatch.
- U14. Loss-ledger excluded from `correlation_id_finalize()` force-retire; leaked
  entry omits `sub_kern_count`/`sub_ref_count`.
- U15. Warning counts dispatches AND unique correlation ids separately (a batch
  shares one corr-id with one ref per dispatch).
- U16. Env-var ring size parsing: empty/zero/negative/default/max-accepted/over-
  `uint32`/overflow-adjacent; sizing math (arr_bytes, signal_off, alloc_size,
  stride) validated before use.
- U17. Task-group handoff ownership matrix: submission success; temporary rejection
  (entry stays `EOP_PROVEN`, owned by the bounded retry owner, drained before join);
  permanent-closing rejection; conversion failure; callback reentrancy; buffer-emplace
  failure; teardown while retry-owned. Exactly one owner at all times; reader never
  runs the callback; EOP-proven work never crosses to `LEAKED`; cleanup runs once.
- U18. Teardown: producer cannot enqueue a task after task-group join begins;
  ready/`EOP_PROVEN` entries delivered, still-`PENDING` -> `LEAKED`, before corr-id
  finalize.
- U19. Fork child: owner-PID/epoch short-circuits all hub ops without touching an
  inherited mutex/map/pool; survives normal `exit()` (must exercise real process
  exit + static-object destruction, not just a direct handler call).
- U20 (split into deterministic cases with barriers): U20a stop-context during a
  callback; U20b destroy-queue during a callback; U20c finalize called from a
  callback worker (no self-join); U20d reader-thread exit mid-handoff. Each asserts
  no double-cleanup and no cleanup while a kernel is live.

Real-GPU (rocprofiler-ian, report actual gfx; cover a CDNA dlog class + gfx1201):

- G-1. P3 app-signal: launch with an app signal init to 1; it reaches 0 with NO
  SDK +1/-1, and the dispatch is traced exactly once.
- G-2. Null-signal dispatch still traces via KFD.
- G-3. Per-dispatch API audit: eligible dispatches issue NO signal
  create/add/sub/load/wait and NO queue profiling-enable (measure per dispatch, not
  per queue create/destroy which still makes internal queue signals).
- G-4. Lazy profiling: pure signal-less queue never enables profiling; first
  signal-path batch enables it exactly once before publication.
- G-5. Tick conversion without queue-profiling-enable yields sane system-domain
  timestamps on each gfx class.
- G-6. Long kernel (> current 5s eviction interval) receives a valid record.
- G-7. Induced overrun (tiny test ring and/or pause-reader hook): one loud warning;
  affected dispatches -> absent records; their corr-ids NOT retired incl. at
  finalize; no late record completes them; subsequent dispatches take the signal
  path.
- G-8. Queue destroy after app completion, then force doorbell reuse: slot stays
  signal-path-only; old late records complete nothing.
- G-9. Clean destroy race: queue destruction vs interceptor registration/publication
  under sanitizer/fault hooks.
- G-10. >512 live queues to force a real page-relative slot collision; verify
  collision closure before the second owner publishes (or deterministic slot-alias
  hook if HW limits prevent it).
- G-11. Code-object unload immediately after app-signal completion: no callback runs
  against invalidated symbol metadata (hub-aware sync).
- G-12. Fork while reader + completion tasks active: child passes through / traces
  nothing inline, finalizes via `exit()` without hang/crash; parent keeps tracing.
- G-13. Multi-GPU with the one-session restriction: non-session GPU dispatches use
  the signal path.

## Test seams (introduce these so most of the above are UNIT tests, not GPU)

Keep each seam narrow; do not build a framework around production code.

- S1. Scriptable ring view: `load_wptr(region)`, `read_record(region, idx)`,
  `store_rptr(region)`, with injectable barriers at pre-scan / per-record-read /
  post-scan so producer laps and torn writes are deterministic (no real data race).
  Enables U5-U7b, P0-overwrite, tiny-ring exhaustion.
- S2. Non-singleton hub + controllable executor (accept / reject / pause / close /
  join) for U3b, U4, U17, U18, model test.
- S3. Injectable monotonic clock + tick converter (force each convert/sanity branch
  and each deadline outcome) for U10, Phase 1 rendezvous, sanity boundaries.
- S4. Injectable `(gpu_id, queue_token) -> doorbell_slot` resolver + full submit
  index, for U12/U13 collision/generation without a GPU.
- S5. Capturing packet writer + fail-on-call HSA API table (packet byte compare;
  fail the test if any signal create/add/sub/load/wait/destroy or profiling-enable
  is called) for P3 byte-identity and the "no signal ops" audit as a UNIT test.
- S6. Reader OS backend hooks (poll/eventfd/ioctl/mmap) for liveness + partial
  setup-failure unwind.
- S7. Retirement observer: records corr-id decrement / retirement / ledger
  membership / tombstone state without parsing logs.

## Model / property test (highest coverage-per-effort)

Stateful model test over events {RegisterBatch, Publish, START, EOP, OverwriteSlot,
PublishWptr, Poison, ReaderDead, Collision, DestroyQueue, SubmitTask, RejectTask,
RunTask, ConvertFail, Stop, ForkEpoch} against a small reference model, with 1-2
GPUs, two colliding slots, tiny rings, repeated low-32 ids. After EVERY event
assert: no emitted record without an EOP proven for that exact pending owner; at
most one emission + cleanup per dispatch; retire iff completion proven and not
leaked; no leaked entry becomes matchable; no signal API call for a signal-less
batch; no registration after poison; no task submit after join begins; no slot
owner/generation change with retained state. Plus: exhaustive tiny-ring schedule
exploration (capacity 1/2/4, partial-write-before-wptr); geometry/parser fuzz under
ASan/UBSan; TSan lifecycle stress (producer + reader + destroy + context-stop +
finalize threads, randomized barriers).

## Tests-first ordering (write with each phase to fail fast)

- Before/with Phase 0: U5, U6, U7, U7b, U16 (+ P0 overwrite/torn/schedule-table and
  setup-failure-unwind), reader-liveness classification. Seams S1, S6.
- Before/with Phase 1: U3a, U4 (rendezvous form), Phase-1 own-signal-before-fallback,
  multi-packet absolute deadline, Phase-1 overrun-uses-signal-fallback (NOT LEAKED),
  the feature-gate regression test (signal-less stays OFF). GATING Phase 1 shipping,
  EXACTLY ONE of: (option a) BOTH the FULL all-live owner-alias test (U12, incl.
  pre-session queues) AND the generation/reuse-closure test (U13) must pass — U13 is
  MANDATORY for option (a), not deferrable to Phase 2, because current-owner
  uniqueness without reuse closure is insufficient; OR (option b) the
  global-KFD-disable test (KFD selection off, rendezvous still seals the race).
  Seams S2, S3, S4, S5.
- Before/with Phase 2: U1 (final state machine incl. EOP_PROVEN), U2 (+ positive
  byte-identity), U3b, U8-U15, U17, eligibility decision table, poison-vs-
  register/publish, result-before-register rejection, unmatched/duplicate firmware
  events, owner-registry lifecycle, generation-transition races, loss-ledger matrix,
  tombstone bounds, hub-aware-sync stage matrix. Seams S2-S7 + the model test.
- Before/with Phase 3: U18, U19, U20a-d, fork-at-held-lock, code-object-unload sync.

## Open items / future work

- Public "timestamp unavailable / lost" record status API (P2) — future.
- `source_overruns` remains diagnostics-only; revisit if a lossless/backpressured
  ring mode is ever added.
