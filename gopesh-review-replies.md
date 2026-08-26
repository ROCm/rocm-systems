# Replies to bgopesh's review — paste-ready as amd-vkale

These were posted as `cursor[bot]`, not as amd-vkale. A covering attribution note is now
posted on each PR (7960 and 8891) naming every affected comment, so no action is strictly
required — the bot replies can stay as they are.

If you'd rather they appear under your own name, delete the bot ones and paste the text
below. Each reply here is prefixed with the attribution line; drop that line if you're
posting as yourself and would rather word it differently.

Note: the agent cannot delete or edit the bot's comments (403 on `DELETE` and `PATCH`),
so deletion has to be done by hand.

Bot comments, if you want them gone:

| PR | comment id | thread |
|---|---|---|
| 7960 | 3856619554 | queue.cpp:927 null corr_id (reply 1) |
| 7960 | 3856630985 | queue.cpp:927 null corr_id (reply 2) |
| 7960 | 3856620841 | queue.cpp:960 create_signal (reply 1) |
| 7960 | 3856632045 | queue.cpp:960 create_signal (reply 2) |
| 7960 | 3856622336 | memory_tracker.cpp:135 free ordering |
| 7960 | 3856623548 | memory_snapshot.cpp:105 size cap |
| 7960 | 3856624845 | memory_snapshot.cpp:179 ROCP_FATAL |
| 7960 | 3856626025 | callback_tracing.cpp:185 check-then-set |
| 8891 | 3856628439 | counters/queue_hooks.cpp:110 unsynchronized walk |
| 8891 | 3856629974 | counters/core.cpp:199 mutex across drain |

Direct link form: `https://github.com/ROCm/rocm-systems/pull/<pr>/files#r<comment id>`

---

## PR 7960 — thread 1: `hsa/queue.cpp:927`, null `corr_id`

### Reply 1

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


You're right, and thanks for pointing at line 546 — that's the precedent I should have followed and didn't.

`corr_id` is null whenever a dispatch arrives outside a tracked correlation scope, and the `has_kernel_replay` check is process-wide, so we walk into this block for those dispatches regardless of whether they were ever going to be replayed. Taking your suggestion as written.

### Reply 2

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


Fixed.

`projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp:925-931`

```cpp
// corr_id is null for dispatches submitted outside a tracked correlation scope (direct-HSA
// submissions, for example). has_kernel_replay is process-global, so this block is reached
// for those too -- guard the same way process_packet_batch does above.
const auto thr_id           = (corr_id) ? corr_id->thread_idx : common::get_tid();
const auto internal_corr_id = (corr_id) ? corr_id->internal : 0;
const auto ancestor_corr_id = (corr_id) ? corr_id->ancestor : 0;
```

Same three guards as `queue.cpp:546-548`, plus a note saying why `corr_id` can be null here so it doesn't get dropped again on the next edit.

---

## PR 7960 — thread 2: `hsa/queue.cpp:960`, unchecked signal creation

### Reply 1

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


I went and checked this one, and I don't think the failure path you're describing is reachable — but the code gives you no way to know that, which is a fair complaint on its own.

`Queue::create_signal` ends in a `ROCP_FATAL_IF` when `hsa_amd_signal_create` returns anything other than success or `INFO_BREAK` (`queue.cpp:1348-1352`). We call it with `use_pool=false` and attribute 0, so we take that branch, and a creation failure aborts the process inside `create_signal` rather than returning to us with an unset handle. So there is no path that submits the barrier with a null completion signal and then waits on it.

I'd rather not add a status check for a condition that can't occur, because it reads as if the abort isn't there and the next person has to re-derive this. Two options I'd consider instead:

- Write the invariant down at the call site. Cheap, and it answers the question you just had to ask.
- Make `create_signal` return the status instead of aborting, and have callers decline gracefully. That's the better long-term shape, but every `create_signal` caller in this file has the same pattern, so it's a change to all of them and doesn't belong in this PR.

I've done the first for now. If you'd prefer the second, I'm happy to open it separately — it's a real improvement, just a wider one.

### Reply 2

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


Invariant written down at the call site.

`projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp:961-969`

```cpp
// Drain barrier: fence the CPU against all prior in-flight GPU work on this queue so
// device memory is stable before snapshotting.
//
// The return value is intentionally unused and there is no status to check: with
// use_pool=false, create_signal aborts via ROCP_FATAL_IF if hsa_amd_signal_create
// fails, and returns nullptr on the path that reaches here. It therefore cannot hand
// back an unset drain_signal, so the wait below always gets a valid handle.
hsa_signal_t drain_signal = null_hsa_signal;
Queue::create_signal(0, &drain_signal, /*use_pool=*/false);
```

No behavior change. If you'd rather have `create_signal` return the status instead of aborting, that's a separate change across all its callers and I'll open it on its own.

---

## PR 7960 — thread 3: `kernel_replay/memory_tracker.cpp:135`, free ordering

### Reply 1 (only reply — not fixed in this round)

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


Agreed, this is a real bug and your reading of it is correct.

`next_pool_free` hands the memory back to the runtime, and the inventory still lists the pointer until `record_free` runs on the next line. A `snap()` landing in that window reads memory that's already been released, and a `restore()` landing in it can write into whatever the runtime handed out next. The read lock only proves the entry is still in the map, and the map is exactly the thing that's stale here — so it doesn't help. You're also right that the per-agent replay lock doesn't cover it, since the alloc/free wrappers aren't routed through it.

I want to fix it the way you sketched: take the entry out first, put it back if the free fails, so the inventory under-reports instead of over-reports. Three details the sketch doesn't cover, which is why I'm not pushing it in this round:

- `record_free` erases from two maps, `inventory()` and `unsupported_executable_inventory()` (`memory_tracker.cpp:188-189`). The take/restore pair has to remember which one the pointer came from and put it back in the right one.
- The restore can't go through `record_alloc`, because that calls `query_alloc(ptr)` to decide trackability and classify the agent. Querying a pointer whose free just failed is not something I want to rely on. It needs a direct insert of the saved `alloc_info_t`.
- `memory_free_wrapper` just below has the same shape and needs the same treatment, otherwise we fix half of it.

None of that is hard, but this sits on every free in the process, so I'd rather add it with a test that actually exercises a free during the snapshot window than put it in untested. That's the next change I'll make on this file.

If you'd rather see it here before this PR goes any further, say so and I'll move it up.

---

## PR 7960 — thread 4: `kernel_replay/memory_snapshot.cpp:105`, silent 1 GiB cap

### Reply 1

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


Fair, and the silent part is the real problem. A variable over the cap never gets saved or restored, so pass 2 sees pass 1's writes to it, the counters come out wrong, and there's nothing in the log to explain why.

I've taken the first of your two options: the literal now has a name, and going over it warns once, naming the variable.

I didn't take the decline-replay option, and I want to be clear that's a judgement call rather than a disagreement. Declining means any application with one oversized global stops being replayable at all, which is a bigger behavior change than the missing diagnostic, and I'd rather not fold it in alongside a logging fix. If you'd prefer the strict version I'm fine switching — it's a small change now that the cap has a name and a single place that decides.

### Reply 2

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


Fixed, in `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp`.

`:82-86` — the literal now has a name and a reason:

```cpp
// Per-variable sanity cap on module-scope variables we are willing to snapshot. A variable above
// this is skipped, which means a kernel's mutations to it persist into the next pass instead of
// being rolled back. That is a correctness gap, not just a memory-budget choice, so the skip is
// reported rather than silent.
constexpr uint64_t module_variable_size_cap = 1ULL << 30;
```

`:136-151` — going over it now reports, naming the variable:

```cpp
if(size > module_variable_size_cap)
{
    static std::atomic<bool> _warned_oversized_variable{false};
    if(!_warned_oversized_variable.exchange(true, std::memory_order_relaxed))
        ROCP_WARNING << fmt::format(
            "kernel replay: module-scope variable '{}' is {} bytes, above the {} byte snapshot "
            "cap, so it is not saved or restored between replay passes. Kernel writes to it "
            "will carry over into later passes and may skew their counters",
            get_symbol_name(symbol),
            size,
            module_variable_size_cap);
    return HSA_STATUS_SUCCESS;
}
```

`:88-107` — new `get_symbol_name` helper for the message. Best effort, returns `<unknown>` rather than failing, since it only ever runs on a path that's already declining to snapshot something. It uses the `HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH` then `_NAME` sequence.

The warn-once uses the same `static std::atomic<bool>` and `exchange` pattern as the graph and multi-packet decline warnings at `queue.cpp:461-473`, so a hot dispatch loop can't flood the log — `discover_module_variables` runs per replayed dispatch, so an unguarded warning here would have been very loud.

---

## PR 7960 — thread 5: `kernel_replay/memory_snapshot.cpp:179`, "Shouldn't it be just ROCP_FATAL?"

### Reply 1 (only reply — no change)

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


I'd like to keep it a warning here, and I think the distinction is worth spelling out.

That branch is a host-side allocation failure while copying a region into the snapshot. It means the machine is short on host memory — it doesn't mean anything is corrupt or that we've lost track of state. Returning `false` sets `out.ok = false` (`memory_snapshot.cpp:277` and `:289`), and the caller then declines replay for that dispatch and runs it once the ordinary way.

So the application still gets correct results. It just doesn't get replayed counters, and it gets a warning saying why. `ROCP_FATAL` would kill an application that was only under memory pressure, and it would kill it from inside a profiling tool the application didn't ask to be fatal.

Declining rather than aborting is the behavior I've tried to apply everywhere replay can't safely proceed — the graph case and the multi-packet case in `queue.cpp` do the same thing. Somewhere we genuinely can't recover, `ROCP_FATAL` is right; this isn't one of those.

---

## PR 7960 — thread 6: `callback_tracing.cpp:185`, check-then-set on the replay owner

### Reply 1

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


Configuration is single-threaded, so the two threads in your scenario can't both get through — but you're right that nothing here says so, and I had to go and check rather than just read it.

The guard is at the top of the same function: `if(rocprofiler::registration::get_init_status() > -1) return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;` (`callback_tracing.cpp:149-150`). Every configure call is rejected once initialization has started, so all of them run in the tool configuration phase before `rocprofiler_initialize` returns, and that phase is single-threaded.

Taking your first suggestion and writing that down at the check.

One note on the alternative: a compare-exchange wouldn't actually close this even if configuration were concurrent. The check is a walk of the context registry, not a read of a flag, so making `replay_service_configured_flag` atomic wouldn't make the check-and-set atomic. If configuration ever does become concurrent, this needs a real owner handle that the check and the set both go through. I've said that in the comment too, so whoever hits it next doesn't reach for the CAS and think they're done.

### Reply 2

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


Added at `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/callback_tracing.cpp:169-176`, immediately above the check:

```cpp
// This check and the set_replay_service_configured() below are deliberately not atomic with
// respect to each other. They do not need to be: the get_init_status() > -1 guard at the top of
// this function restricts every configure call to the single-threaded tool configuration phase,
// before rocprofiler_initialize() returns. Two threads cannot both pass this check, because
// there is only one thread here. If configuration ever becomes callable concurrently, this pair
// has to become a compare-exchange on a dedicated owner handle -- the registry walk below is
// not something a CAS can cover on its own.
```

Comment only, no behavior change.

---

## PR 8891 — thread 7: `counters/queue_hooks.cpp:110`, unsynchronized registry walk

### Reply 1 (only reply — not fixed)

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


This one is real and I don't have a one-line answer for it. Let me take your points in order, including the one where your inference was off, because I think the conclusion survives it anyway.

On the `has_value()` guards being evidence: they aren't, quite. They're there because `get_registered_contexts` already had callers that could observe a slot before it was engaged, independent of this PR. So they're not a fossil of a crash report. But your actual point stands — they make the common case survive without making the read atomic, and I can't point at anything that makes the read safe.

I also can't give you the reassurance you offered as the alternative. `stable_vector` not moving existing elements when it grows means a pointer to an already-constructed context stays valid, and that's the property the container was chosen for. It says nothing about one thread reading a slot while another is running `optional::emplace` on it. Those are different guarantees and I'd be talking myself into it if I claimed the first covers the second. So the race is there.

On the fix, I agree with you that taking `get_contexts_mutex()` in the exit hook is the wrong direction, for the reason you gave — this runs from the HSA async signal handler and blocking there isn't allowed. The snapshot is the right shape. What I want to build is a published list of contexts that have counter collection configured, updated under the contexts mutex at register and deregister, with the exit hook reading it through a `shared_ptr` load. Contexts are registered during configuration and torn down at finalize, so the list changes rarely and the read side stays cheap on the completion path.

Two things I want to settle before writing it:

- Whether one published list can serve thread trace and SPM as well. Their exit hooks walk the registry exactly the same way, so if I solve this only for counters I'll be solving it three more times, slightly differently each time.
- What the enter hook should do. It walks `get_active_contexts()` rather than the registered set, and that one does change on every start and stop, so the same snapshot trick doesn't transfer directly.

I'd rather do this once across the services than four times. Give me a bit and I'll come back with a concrete shape.

Flagging plainly: this is the part of the PR I'm least comfortable with as it stands, so if you want to hold on this specific point, that's a reasonable place to hold.

---

## PR 8891 — thread 8: `counters/core.cpp:199`, contexts mutex held across GPU drain

### Reply 1 (only reply — not fixed)

> Cursor helped draft this comment. I (@amd-vkale) own this comment.


Agreed, and I think you've framed it exactly right: the mutex hold is old, the drain is new, and putting the two together is what this PR did. That's the part I own.

To be precise about the exposure so we're describing the same thing: `stop_context` takes `get_contexts_mutex()` at `context/context.cpp:396`, and with the reordering `counters::stop_context()` now runs inside that hold, starting with `hsa::queue_controller_sync()`, which waits on every queue. So any start or stop anywhere in the process waits behind one context's GPU work, including on agents that context never touched. And `set_dispatch_agents` at `core.cpp:228` takes the same mutex, so it inherits the stall, as you say.

I don't believe the drain has to be inside the lock. What it has to be inside is the window where the service is already marked disabled — that's the whole reason the ordering moved, so that no new packets get instrumented while we're draining. That property comes from the disable, not from holding the mutex. So dropping the lock across the drain and reacquiring for the CAS should be safe on that count.

The part I want to think through before moving it is that `stop_context` isn't the only writer. Releasing the lock in the middle means a concurrent `start_context` can interleave with another context's drain, and I need to decide what a start arriving in that window should do — proceed, wait, or fail. That's a real question and I don't want to guess at it and find out from a flaky test later.

So: I agree with the direction and I'll move the drain outside the lock. I'd like to settle the interleaving question first rather than move the call and discover the answer afterwards. When I do, the comment will say what's held and why, not just what order things happen in — you're right that it currently explains the ordering thoroughly and is silent on the lock scope, which is the thing that actually bites.
