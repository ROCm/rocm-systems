# Queue event lifetime regression

This test repeatedly creates and destroys AQL queues without dispatching kernels. It interposes on KFD event ioctls and delays background SET_EVENT calls by one millisecond, then fails if DESTROY_EVENT overlaps any outstanding write to that event. This enlarges the queue callback completion window that previously allowed the final queue to destroy an event still used by its retained signal.

The executable runs on a KFD GPU or under rocJITsu CPU emulation. The normal rocrtst build includes and installs it. It can also be built alone against an installed runtime:

```sh
cmake -S projects/rocr-runtime/rocrtst/suites/functional/queue_event_lifetime \
  -B <build-dir> -G Ninja -DCMAKE_PREFIX_PATH=<rocm-prefix>
cmake --build <build-dir> --parallel 4
ctest --test-dir <build-dir> --output-on-failure
```

For CPU emulation, run the executable through rocJITsu with a supported virtual GPU configuration:

```sh
rocjitsu --config <gfx942-config.json> -- <build-dir>/queue_event_lifetime_test
```

Set the library search path to the runtime being tested. A missing GPU agent or a configuration that exercises no background KFD event writes returns CTest skip code 77. Other runtime failures and overlapping event destruction return failure. The test reports how many delayed writes executed, so a skipped path cannot be mistaken for validation. The delay is a stress mechanism; it does not prescribe a deterministic thread schedule or rely on private runtime symbol addresses.
