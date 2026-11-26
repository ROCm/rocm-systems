<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Testing rocr-debug-agent

rocr-debug-agent is the HSA tool library that dumps registers, LDS, and
disassembly for faulting AMD GPU wavefronts.  It is loaded via `HSA_TOOLS_LIB`
and built on top of **amd-dbgapi** and the ROCr runtime (`hsa-runtime64`).

## Test architecture

There are two tiers:

- **Integration tests** are GPU-bound: each test builds and runs a real HIP
  kernel with the debug agent attached, then inspects its output.
- **Unit tests** are CPU-only GTest tests under `test/unit/`, built into the
  `rocm-debug-agent-unit-tests` binary when `ENABLE_UNIT_TESTS=ON`.  They
  need no GPU.

Tests live under `test/`.  The `rocm-debug-agent-test` binary contains the
HIP kernel scenarios; `test/run-test.py` drives it, sweeping
`HIP_ENABLE_DEFERRED_LOADING` across multiple settings per run and producing
a `PASS` / `FAIL` / `UNSUPPORTED` verdict per check.  Read the comment header
in `run-test.py` before adding a test. It is the authoritative guide for
test structure and harness conventions.

The unit tier is still small; coverage of the ELF/DWARF and option-parsing
logic in `src/` is yet to be added.

Tier that does not exist yet: performance tests for wave-dump latency.

## Adding a test

Every fix or feature should include a test.  Logic that can run without a
GPU should get a unit test in `test/unit/`; anything that needs a real
wavefront must be exercised through a full GPU scenario.

A new GPU integration test consists of three parts:

1. A HIP workload in `test/` that triggers the behavior under test.
2. A new scenario wired into `rocm-debug-agent-test` so the harness can
   invoke it.
3. An entry in `test/run-test.py`'s `TEST_DEFINITIONS` with regex patterns
   to match against the agent's output, or a custom handler for cases that
   need filesystem checks or signal coordination.

New scenarios automatically run under all `HIP_ENABLE_DEFERRED_LOADING`
settings.  See the existing tests and `run-test.py`'s comment header for
the conventions to follow.

## Running the suite

**Local build tree:**

```bash
make test
```

**Against a full TheRock/ROCm stack** (matches CI):

```bash
OUTPUT_ARTIFACTS_DIR=<rocm-tree> <rocm-tree>/tests/rocm-debug-agent/test_rocr-debug-agent.py
# or, if ROCM_PATH points at that tree:
tests/rocm-debug-agent/test_rocr-debug-agent.py --try-rocm-path
```

Which tests run, and which binaries the wrapper requires, depend on the flags:

| Flags                      | Runs                | Required binaries                          |
|----------------------------|---------------------|--------------------------------------------|
| none                       | integration + unit  | integration; unit is optional (skipped with a warning if missing, for compatibility) |
| `--gpu-tests`              | integration only    | integration                                |
| `--cpu-tests`              | unit only (no GPU)  | unit                                       |
| `--gpu-tests --cpu-tests`  | both                | integration and unit                       |

The wrapper fails before running anything if a required binary is missing.
The unit test binary is only built with `ENABLE_UNIT_TESTS=ON`.

The wrapper prepends the ROCm tree's `bin` and `lib` directories to `PATH` and
`LD_LIBRARY_PATH`, drops `ROCM_PATH` from the environment unless
`--try-rocm-path` is given, and retries the integration tests (up to 3x with
backoff).  Both test tiers run in this modified environment.  `HSA_TOOLS_LIB`
is set by `run-test.py` for the integration tests, not by the wrapper; the
unit tests do not load the agent.  Unit tests are deterministic and run once,
without retries.  Prefer the wrapper over invoking the test binaries directly.

rocr-debug-agent requires amd-dbgapi, the ROCr runtime, and a HIP compiler.
Build it as part of a full ROCm stack via TheRock; an isolated build will not
produce a runnable test binary.

## CI and hardware coverage

CI triggers and hardware coverage are defined in `.github/`.  Tests run on
every PR touching rocr-debug-agent or its runtime dependencies; a periodic
run sweeps additional GPU architectures.

The suite iterates all devices `hipGetDeviceCount()` reports.  Unsupported
architectures produce a hard `FAIL`, not a skip.  There is no per-check
expected-failure mechanism. A failing check is always a hard failure after
CI's 3x retry is exhausted.
