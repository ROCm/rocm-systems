# Minimal

Small, single-file programs that each exercise one specific component or code path of the rocprofiler-systems instrumentation in isolation. They are intentionally trivial so any discrepancy between expected and recorded behavior can be attributed to the profiler rather than the workload. These are targeted reproducers and regression checks, not benchmarks.

## Source Files

- `recursion.cpp` - Calls a single `__attribute__((noinline))` `recurse` function to a configurable depth (default `100`, override via `argv[1]`). Used to verify that recursive frames are not collapsed by the per-thread region cache.
- `pthreads.cpp` - Spawns a single worker thread with `pthread_create` and joins it. Used to verify that the `pthread_create` gotcha records the call's argument annotations on success.
- `sleep_interrupts.cpp` - Sleeps in 50 ms increments (default `60` iterations, override via `argv[1]`) and reports how many were cut short by a signal. Used to verify that pausing sampling stops the underlying timers, rather than only discarding the samples they generate.

## Building

```bash
cmake -B <build_dir> -S <project_root>/examples/minimal
cmake --build <build_dir>
```

| Target | Description |
| -------- | ------------- |
| `minimal-recursion` | Recursion test for the per-thread region cache |
| `minimal-pthreads` | `pthread_create` gotcha trace-args (argument annotations + `return`) |
| `minimal-sleep-interrupts` | Sampler timers are stopped, not just muted, while sampling is paused |
