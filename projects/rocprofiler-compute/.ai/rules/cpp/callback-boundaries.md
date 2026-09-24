# C++ Callback Boundaries

Every C++ library we ship runs inside somebody else's callback. The
rocprofiler-sdk tool callbacks call us. The torch dispatcher calls us through a
`RecordFunction` observer. Python calls the extension module.

We do not own the thread, we do not own the stack, and we do not own the
lifetime. That shapes three things at once, which is why they are in one file.

## What thread are we on

Assume any callback can run on a thread we have never seen, at any time,
concurrently with itself on another thread. Nothing about our own call order
holds.

On the sampling and dispatch path:

- No allocation.
- No blocking. No mutex you might wait on, no I/O, no logging to a file.
- No call back into the host API that invoked us.

The tools for shared state on that path, in order of preference:

1. `std::atomic` for counters. Use `memory_order_relaxed` when the value is only
   ever read as a statistic. `Stats` in
   `src/lib/torch_trace_collector/stats.h` is used this way.
2. `thread_local` for per-thread state, so there is no sharing to coordinate.
   `thread_state()` in `src/lib/torch_trace_collector/torch_trace_collector.cpp`
   holds the marker stack this way.
3. `synchronized_t<T>` from `src/lib/utils/synchronized/synchronized.hpp` when
   state really is shared and a lock is unavoidable. Keep the critical section
   to the shortest possible span, and shard the data if contention is real.
   `SnapshotStore` shards its map for that reason.

## What must not cross the boundary

An exception must never escape into rocprofiler-sdk, torch, or the Python
interpreter. Unwinding through a foreign frame is undefined behaviour, and in
practice it terminates the profiled application, which is the one thing a
profiler must not do.

So every function that the host can call directly is a firewall:

- Wrap the body in `try` and catch `...` at the outermost level.
- Convert the failure into whatever the host expects. A status code for
  rocprofiler-sdk, a null or a no-op for a torch observer, a Python exception
  set through pybind for the module.
- Report it. Bump an error counter and write one line to `stderr`. Do not throw,
  do not allocate a message on the hot path, and do not retry.
- Leave our own state consistent on the way out. `handle_start_error` in
  `src/lib/torch_trace_collector/torch_trace_collector.cpp` unwinds the marker
  stack before it returns, and its cleanup is itself inside a `try`, because
  cleanup running during error handling must not throw either.

Recovery code that can itself fail gets its own `catch (...)` with an empty
body. That is the one place where swallowing an exception silently is correct.

## Process-lifetime state

We need state that outlives every object we own. The reason is the callback API,
not how the library gets loaded: the host calls us, and there is no object of
ours that owns the state and no parameter we can thread it through.

`LD_PRELOAD` is the wrong test for this. `torch_trace_collector` is an ordinary
Python extension module and it still needs a process-wide `ProcessState`.

### Allowed

Only for state that a host callback API leaves us no place to own:
rocprofiler-sdk tool callbacks, torch `RecordFunction` observers, and module
init.

It must take one of two forms. Which one depends on a single question: **does
the host call us after our static destructors would have run?**

| | Use | Because |
|---|---|---|
| No, or you are not sure | Function-local static behind an accessor | Lazy, thread-safe, and the object cannot be used before it exists |
| Yes | Never-destroyed heap object behind a namespace-scope reference | A destructor at all is the bug, so there must not be one |

The first is the default. Reach for the second only when you can name the host
teardown path that forces it, and say so in the comment.

**A function-local static behind an accessor.**

```cpp
ProcessState& process_state() {
    static ProcessState state;
    return state;
}
```

Initialization is lazy and thread-safe, and the object cannot be used before it
exists.

**A never-destroyed heap object behind a namespace-scope reference.** Use this
only when the host tears down after our static destructors would have run, so a
destructor at all is the bug. `rocprofiler_compute_tool.cpp` does this because
rocprofiler-sdk calls `tool_fini` from its own `_dl_fini`, and the comment above
the declarations says so.

Either way, never a namespace-scope object whose destructor runs at shutdown.

### Required in both cases

- It stays reachable from tests. Either the accessor hands the object out by
  reference so consumers can be given a different one, or there is an explicit
  reset and injection seam. `test_knobs::set_sdk_wrapper` and
  `test_knobs::reset_cfg` in `rocprofiler_compute_tool.cpp` are that seam.
- A comment says which host API forces it. Name the API and the callback.

### Banned everywhere else

Process-lifetime state is not a shortcut for passing state between our own
functions. If the call chain is ours end to end, pass a parameter.

[`testability.md`](testability.md) adds only one thing on top of this: whatever
form the state takes, the accessor has to stay injectable.
