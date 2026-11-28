# Minimal

Small, single-file programs that each exercise one specific component or code path of the rocprofiler-systems instrumentation in isolation. They are intentionally trivial so any discrepancy between expected and recorded behavior can be attributed to the profiler rather than the workload. These are targeted reproducers and regression checks, not benchmarks.

## Source Files

- `recursion.cpp` - Calls a single `__attribute__((noinline))` `recurse` function to a configurable depth (default `100`, override via `argv[1]`). Used to verify that recursive frames are not collapsed by the per-thread region cache.
- `pthreads.cpp` - Spawns a single worker thread with `pthread_create` and joins it. Used to verify that the `pthread_create` gotcha records the call's argument annotations on success.
- `sleep_interrupts.cpp` - Sleeps in 50 ms increments (default `60` iterations, override via `argv[1]`) and reports how many were cut short by a signal. Used to verify that pausing sampling stops the underlying timers, rather than only discarding the samples they generate.
- `tail_call_main.cpp` - `main()` ends with `return run_workload(argc);`, which the compiler turns into a jump instead of a call followed by a return. Used to verify that the profiler does not finalize at that jump, before the workload has run.
- `exit_in_main.cpp` - `main()` ends with a call to `exit()` instead of a return. Used to verify that the profiler still finalizes at that end of `main()` and does not mistake it for a tail call.
- `mixed_main_end.cpp` - `main()` has a tail call, a return and a call to `exit()`, and the arguments select which one runs. Used to verify that only the tail call is skipped and the session is finalized after the workload whichever exit is taken.
- `tail_call_workload_lib.cpp` and `tail_call_main_shared.cpp` - `main()` ends with a tail call into a function of a shared library, which is a jump through the PLT. Used to verify the tail call is skipped when the callee is in another object.

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
| `minimal-tail-call-main` | `main()` ends with a tail call, so finalization must wait for the real end of the program |
| `minimal-exit-in-main` | `main()` ends with `exit()`, which is not a tail call |
| `minimal-mixed-main-end` | `main()` has a tail call, a `ret` and an `exit()` call, selected by the arguments |
| `minimal-tail-call-main-shared` | `main()` ends with a tail call into a shared library (`libminimal-tail-call-workload.so`) |
