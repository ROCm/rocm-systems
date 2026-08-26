(kernel-replay-multi-service)=
# Kernel Replay with Services Other Than Counters

Kernel replay re-executes a dispatch once per pass and restores device memory between passes. Nothing
about that is specific to hardware counters. The SDK deliberately exposes replay as a *callback
tracing* domain rather than a counter-collection mode, and the localized context control API exists so
a tool can decide, per pass, which of its services are active.

Despite that, `rocprofv3 --kernel-replay-beta-enabled` collects **counters only**, and now rejects
`--att`, PC sampling and `--spm` instead of accepting them and producing misleading output. This page
explains what is actually missing, which parts are ordinary integration work, and which part is a
genuine hardware problem rather than a wiring problem.

## Why the CLI refuses today

The restriction is not that the other services are incompatible with replaying a kernel. It is that
the tool has no *pass schedule*.

`kernel_replay_callback` in `rocprofiler-sdk-tool/tool.cpp` does exactly two things:

1. On `KERNEL_REPLAY_CONFIG` enter, it installs `pass_count_cb`, which returns the number of counter
   groups collectable on the dispatch's agent.
2. On `KERNEL_REPLAY_PASS` enter/exit, it publishes the SDK's pass index into a thread-local
   (`tl_current_replay_pass`) and clears it, so `counter_dispatch_callback` can select the matching
   counter group.

It never calls `rocprofiler_replay_local_start_context` or `..._stop_context`. No context is toggled
at any point. The counter context is enabled for every pass; what varies is which *group* the counter
callback hands back.

That design is fine for counters and wrong for everything else. A service that is enabled when replay
begins simply stays enabled for all `N` passes:

| Service | Consults the local override? | Behavior under replay today |
|---|---|---|
| Dispatch counters | yes, `counters/dispatch_handlers.cpp:87` | correct — group varies per pass |
| SPM | yes, `spm/dispatch_handlers.cpp:150` | would trace every pass; no toggle is ever recorded |
| ATT / thread trace | partially, `thread_trace/core.cpp:452` — honors a forced *off*, but does not AND with an enabled flag | would emit `N` traces per kernel |
| PC sampling | **no** — production code never calls `local_context_override` | samples the whole window regardless |

All of those records carry the same `dispatch_id`, because replay deliberately reuses one reserved
dispatch ID across the passes of a dispatch. So the duplication is not even distinguishable after the
fact without a pass field the records do not have. That is why the CLI refuses rather than warns:
the output looks plausible and counts each kernel `N` times.

## What the SDK already provides

The replay window itself is service-agnostic, and this is worth stating plainly because it determines
how much of the remaining work is SDK work versus tool work.

- The per-agent writer lock, the queue drain, the agent-wide drain, the snapshot, the pass loop and
  the restore in `hsa/queue.cpp` know nothing about counters.
- `scoped_local_context_control` installs a thread-local override map for the duration of the loop,
  and `set_toggles_armed` restricts toggling to the `PASS` enter window. Global context state is never
  mutated, so a locally stopped context is still globally started and its exit-side routing is
  unaffected.
- Kernel dispatch *tracing* records are already filtered by the override in `queue.cpp`, so a locally
  disabled context does not receive dispatch callbacks for that pass.
- SPM and ATT already consult the override at their dispatch decision points, and both have unit
  tests covering it (`spm/tests/local_context.cpp`, `thread_trace/tests/local_context.cpp`).
- Working sample clients already demonstrate the pattern end to end: `samples/kernel_replay/`
  contains `att_client.cpp` (counters on pass 0, ATT on pass 1) and `service_sequence_client.cpp`.

So for ATT and SPM the SDK plumbing exists and is exercised. The gap is in the tool and the CLI.

## Per-service assessment

### ATT / thread trace — integration work

Dispatch-mode ATT injects trace start/stop AQL packets around the kernel through the queue callback,
and its dispatch path already returns early when the override forces it off. What is missing:

- A pass schedule in the tool: something that decides "pass 0 collects counter group 0, pass `k`
  collects the trace", and calls the local start/stop functions during `PASS` enter accordingly.
- A pass count that is no longer simply "number of counter groups". With ATT requested, the count
  becomes groups + 1 (or groups + however many trace passes are wanted).
- Pass identification in the output. ATT output is written per dispatch; with replay, one dispatch ID
  now covers several passes and the consumer needs to know which pass a trace came from.
- ATT and SPM must never share a pass — both inject AQL around the same kernel. That is currently the
  tool's responsibility, enforced nowhere.

The `enabled` check is also weaker than the counter one: ATT honors a forced *off* but does not AND
with a service-enabled flag the way `counters/dispatch_handlers.cpp` does. That asymmetry is harmless
while nothing toggles it, and should be tightened before anything starts toggling it.

### SPM — integration work, plus a configure-time conflict to route around

SPM is per-dispatch (it injects barrier plus start/stop AQL in `pre_kernel_call`) and already consults
the override. The obstacles are above it:

- `spm/core.cpp` and `spm/service.cpp` reject SPM in the same context as PMC counter collection. Since
  replay is driven by counter groups, SPM has to live in a *separate context* from the counters. The
  samples already do this; the tool creates its contexts in one place and would need the same split.
- `start_context` calls `enable_serialization()`. The replay loop does not use global serialization,
  and the interaction between an always-serialized SPM context and replay's per-agent writer lock is
  not characterized. This needs to be measured, not reasoned about.
- Whether aqlprofile's own device buffers survive `memory_snapshot::restore` between passes is
  untested. The snapshot deliberately excludes executable-flag pools, which is where those buffers
  should land, but no test covers SPM specifically across a restore.

### PC sampling — not integration work

This is the one that is genuinely hard, for two independent reasons.

**It is agent-wide, not per-dispatch.** The hardware sampler is started by
`pc_sampling_service_start()` and runs continuously. The per-dispatch marker packet injected in
`queue.cpp` only supplies correlation; it does not gate collection. Its injection is guarded solely by
`pc_sampling::is_pc_sample_service_configured(agent_id)` and never consults the override, so during a
replay loop the marker fires once per pass with the same dispatch ID, and samples continue to be
collected during passes the tool believes PC sampling is off for.

Making a local stop mean anything therefore requires either gating the marker and accepting that
samples still accrue, or pausing and resuming the sampler hardware around each pass. The SDK's own
test states the position explicitly:

> PC sampling is agent-wide and does not consult `local_context_override()`. A recorded local stop
> must succeed (the TLS map is service-agnostic) but must not flip the sampler's enabled flag.
> Reprogramming PCS hardware per pass is exactly what the sticky override is meant to avoid; until a
> consumer is wired, collection is a no-op.

That is a design decision, not an oversight. Reprogramming `hsa_ven_amd_pcs_start`/`stop` at every
pass boundary of every replayed dispatch is a large amount of hardware traffic inside a window that
already holds an exclusive per-agent lock.

**It conflicts with counters in hardware.** `pc_sampling/service.cpp` returns
`ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT` when the context already has dispatch or device counter
collection, and the comment explains why: counter collection can enable clock gating while PC sampling
is active, and the result is a hang. Separate contexts avoid the configure-time error but not the
hardware hazard — the two must not be active on the same pass.

Since replay's pass count is derived from counter groups, and PC sampling cannot be co-resident with
counters on a pass, a correct PC-sampling-plus-replay run needs the sampler genuinely quiescent during
every counter pass. Given that the sampler is agent-wide and the replay lock is per-agent, "quiescent"
means actually stopping the hardware, which is the thing the current design avoids.

## What the work would look like

Staged, with the cheap and verifiable parts first.

**Stage 1 — pass schedule in the tool (enables ATT and SPM).**
Replace "pass count = counter groups" with an explicit schedule: an ordered list of passes, each
naming which contexts are active. Derive it from the requested services. During `PASS` enter, toggle
the contexts for that pass through the local start/stop API; during exit, restore. Counter-only runs
must produce the identical schedule they produce today, which is directly testable.

Touches `tool.cpp` (`kernel_replay_pass_count_callback`, `kernel_replay_callback`, context creation)
and `rocprofv3.py` (relax the rejection this branch added, per service). Contained, but it changes the
shape of the tool's replay integration rather than adding to it.

**Stage 2 — pass identity in output.**
Counter records already carry `replay_pass` in JSON. ATT and SPM records do not, and their output
paths key on dispatch ID. Either add a pass field to those records, or make the SDK deliver the pass
index to service callbacks directly instead of via the tool's thread-local — which is already listed
as an open item in the callback API design notes. Without this, stage 1 produces correct collection
with ambiguous output.

**Stage 3 — validate SPM and ATT across a restore.**
Confirm on hardware that trace and SPM buffer state is clean from pass to pass, and characterize the
serialization interaction. Needs a GPU; cannot be settled by reading code.

**Stage 4 — PC sampling.**
Decide the policy question first: does a local stop pause the sampler, or does PC sampling simply
occupy dedicated passes with counters absent? Only then wire a consumer. This stage carries the
clock-gating hang risk and is the reason PC sampling should not be bundled with the ATT and SPM work.

## Effort and risk

| Stage | Where | Risk |
|---|---|---|
| 1. Pass schedule | tool + CLI | Moderate. Must not perturb counter-only replay; that is testable without a GPU. |
| 2. Pass identity in records | SDK + tool + output schema | Moderate. Touches a public-facing output schema. |
| 3. SPM/ATT across restore | validation only | Unknown until measured. May surface real defects. |
| 4. PC sampling | SDK, hardware programming | High. Hang risk, agent-wide sampler versus per-pass intent, contradicts a deliberate design choice. |

Stages 1 through 3 are ordinary integration work on infrastructure that already exists and is tested
at the unit level. Stage 4 is a design question about hardware behavior that should be answered before
any code is written.

## Open questions

These are stated as questions because the code does not answer them:

1. Does aqlprofile's SPM state survive `memory_snapshot::restore` between passes? No test covers it.
2. Is ATT trace control state fully reset per pass when locally toggled? The samples check record
   counts, not buffer contents.
3. How does an SPM or ATT context's `enable_serialization()` interact with replay's per-agent writer
   lock over a whole replay window?
4. For PC sampling, should a local stop pause the sampler, or should the tool guarantee separation by
   scheduling instead? This determines whether stage 4 is an SDK change or a tool change.

## See also

- {ref}`using-kernel-replay` — tool-author guide, including localized context control
- {ref}`using-kernel-replay-rocprofv3` — the CLI limitations this page explains
- [Callback API and tool configuration](kernel_replay_callback_api.md) — service combination limits
- [Concurrency and isolation](kernel_replay_concurrency_and_isolation.md) — the replay window
- `samples/kernel_replay/att_client.cpp`, `service_sequence_client.cpp` — working multi-service replay
