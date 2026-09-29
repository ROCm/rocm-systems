# Clock-counter sampling regression test

Builds the actual Linux `src/time.c` with a deterministic ioctl transport and
monotonic clock. No GPU or kernel module is required.

```sh
cmake -S projects/rocr-runtime/libhsakmt/tests/time -B build/hsakmt-time
cmake --build build/hsakmt-time
ctest --test-dir build/hsakmt-time --output-on-failure
```

Checks delayed GPU/CPU correlations, selection of an intact counter tuple,
second-boundary arithmetic, CPU-only nodes, ioctl/clock failures without partial
output, and invalid runtime state or node IDs. Hardware validation must also
check unadjusted profiling timestamps against CPU enqueue/completion bounds.
