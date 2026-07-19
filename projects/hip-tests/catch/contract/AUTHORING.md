<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Authoring a HIP contract test

This guide is for anyone — a contributor or an AI agent — adding a public-API
semantic contract test to `projects/hip-tests/catch/contract/`. It captures the
conventions the suite already follows so a new test is correct the first time.

If you added or changed a public HIP API in `projects/hip/include/hip/`, the
`HIP contract-test coverage` CI check (`tools/check_contract_coverage.py`) will
fail the PR until the new API is either covered by a contract test or added to
`uncovered_apis.txt` with a reason. This guide tells you how to do the former.

## What a contract test is (and is not)

A contract test pins a **small, portable, device-only semantic guarantee** of a
public HIP API: an invariant that must hold on any conforming backend. It is
**not** behavioral or performance coverage. One API has many modes; the contract
layer intentionally asserts only the few guarantees that are portable and cheap
to verify without special hardware or external producers.

Typical contract shapes (pick the strongest one the API supports device-only):

- **Round-trip**: set a value, read it back, assert equality (e.g. a
  stream/attribute/param setter+getter pair, or a malloc → memcpy → free flow).
- **Accepted-or-unsupported**: a well-formed call must return `hipSuccess` or a
  documented `hipErrorNotSupported` (treat the latter as a capability skip), not
  crash or return a wrong code.
- **Invalid-input rejection**: a null/zero/reserved-flag argument must return a
  defined error rather than crashing or silently succeeding. Use this when the
  success path needs a resource a device-only harness can't build (e.g. an
  external semaphore or a graphics resource).

See `projects/hip-tests/CONTRACT_COVERAGE.md` for the coverage philosophy and the
per-API notes on why each gap exists.

## Step-by-step: add a test

1. **Pick or create a domain directory** under `contract/`. Group by API family
   (e.g. `stream_props`, `graph_update`, `texture_reference_symbol`). One domain
   = one test executable.

2. **Write `test_hip_<domain>_contract.cc`.** Use the Catch2 `HIP_TEST_CASE`
   macro and name each case `Contract_<Area>_<Behavior>` so the intent is legible
   in `ctest` output. Put an `// @asserts:` intent tag on the line directly above
   each case (see the checklist below — it feeds the generated test plan). Include
   `hip_test_common.hh` (pulls in Catch2, `HT_AMD`, `CHECK_IMAGE_SUPPORT`,
   `HIP_CHECK`, `HIP_SKIP_TEST`) and, if you allocate anything,
   `contract_cleanup.hh`. See the skeleton below or copy `TEMPLATE.cc.txt`.

3. **Add a `CMakeLists.txt`** in the domain dir:

   ```cmake
   set(TEST_SRC test_hip_<domain>_contract.cc)
   hip_add_exe_to_target(NAME Contract<Domain>Test
                         TEST_SRC ${TEST_SRC}
                         TEST_TARGET_NAME contract_tests)
   ```

4. **Register the domain** in `contract/CMakeLists.txt` with
   `add_subdirectory(<domain>)`.

5. **Add a YAML entry per `HIP_TEST_CASE` name** to
   `catch/config/configs/contract.yaml`:

   ```yaml
     Contract_<Area>_<Behavior>:
       <<: *level_0
       tags: [<domain>]
   ```

   This is **required, not optional**. `HIP_TEST_CASE(name)` expands through
   `GET_TAGS(name)` → `SECOND_ARG(name)`, which needs a `name` macro that the
   build **generates from this YAML** into `hip_test_config.hh`. A test-case name
   with no YAML entry fails to compile with
   *"too few arguments to function-like macro SECOND_ARG"*. After editing the
   YAML you must re-run CMake configure so the header regenerates.

6. **Regenerate the test plan and update the docs**: run
   `tools/gen_test_plan.py` to refresh `TEST_PLAN.md` (the CI check fails if it is
   stale), bump the counts and the domain line in
   `projects/hip-tests/CONTRACT_COVERAGE.md`, and add/adjust the domain
   description in `catch/contract/README.md`. Run
   `tools/check_contract_coverage.py` — it prints the numbers to copy and confirms
   the new API is now counted as covered.

## Conventions checklist

- **Tag each case with `// @asserts:`.** On the line directly above every
  `HIP_TEST_CASE`, write `// @asserts: <API> - <one-line portable invariant this
  case pins>` (keyboard-friendly spaced-hyphen separator; the API before it, the
  invariant after). This is the machine-readable intent that
  `tools/gen_test_plan.py` compiles into `TEST_PLAN.md`, the cross-tier inventory
  used to spot redundant coverage between the contract, unit, integration, and
  other tiers. Write the invariant from what the case *asserts*, not what it
  allocates. If the case is `#if`-guarded, put the tag directly above the
  `HIP_TEST_CASE` line (inside the guard), not above the `#if`. Exemplar:
  `mem_batch_copy_3d/test_hip_mem_batch_copy_3d_contract.cc`.

- **Clean up every resource with `ContractCleanup`.** Declare a
  `hip::contract::ContractCleanup cleanup;` first, then `cleanup.Add(...)` right
  after each successful allocation. Capture handles **by value** (`[ptr]{...}`),
  **not** by reference (`[&]`) — the guard is declared before the resources, so a
  by-reference capture reads a dangling variable at scope exit. Exemplar:
  `mem_batch_copy_3d/test_hip_mem_batch_copy_3d_contract.cc`.

- **Image-gate texture/array/image APIs** with `CHECK_IMAGE_SUPPORT;` at the top
  of the case (after any backend-neutral null-argument checks). It skips cleanly
  where image support is absent (e.g. a WSL2 iGPU) and runs on image-capable
  datacenter GPUs. Exemplar: `texture_reference_symbol`.

- **Gate backend-divergent or CUDA-removed APIs** with `#if HT_AMD` (from
  `hip_test_context.hh`: 1 on AMD, 0 on NVIDIA). `#if HT_AMD` around the whole
  translation unit compiles it to an empty binary on NVIDIA instead of failing to
  build. For a per-assertion difference use `#if HT_AMD ... #else ... #endif`.
  For CUDA-version-specific availability, match the `nvidia_detail` header's own
  `#if CUDA_VERSION >= ...` guard rather than a blanket `HT_AMD`.

- **Mark real divergences with a `BACKEND-DIFF:` comment.** Any place a test
  accommodates a genuine behavioral or availability difference between AMD and
  NVIDIA gets a `BACKEND-DIFF:` marker explaining the difference and what parity
  would require. `grep -rn BACKEND-DIFF` is the reconciliation worklist. Purely
  mechanical portability shims (const-cast helpers, `hipFree(0)` context primes)
  get an ordinary comment, not this marker.

- **Clear the sticky error after every intentional rejection.** After a call you
  expect to fail, add `(void)hipGetLastError();`. A leaked thread-global sticky
  error poisons the next test's `HIP_CHECK(hipGetLastError())` — this manifests
  as a cross-test failure that only shows up in full-suite order (and often only
  on NVIDIA). If a test passes alone but fails in-suite, suspect this.

- **Probe the runtime; never assume a return value.** Before writing a divergent
  assertion or declaring an API uncoverable, build a tiny standalone probe and
  print the *actual* code the runtime returns. The installed runtime can differ
  from the source you read, and a shallow-copy constructor is not proof that a
  handle is dereferenced only later. The GL-interop and external-semaphore gaps
  were both settled this way — and both times the naive source-based assumption
  was wrong.

- **Treat `hipErrorNotSupported` as a capability skip**, not a failure, for
  accepted-or-unsupported contracts: `if (status == hipErrorNotSupported)
  HIP_SKIP_TEST("...");`.

## When an API genuinely cannot be covered

Some APIs cannot be exercised by a device-only harness (they need an external
producer, crash by design, or have a Windows-only positive path). For those:

1. Confirm the gap is real by **probing the actual runtime** (see above), ideally
   cross-arch (local iGPU behavior is not always representative — check a
   datacenter GPU too).
2. Add the API to `uncovered_apis.txt` with a specific reason.
3. Note it in the relevant "Largest remaining gaps" entry in
   `CONTRACT_COVERAGE.md`.

That satisfies the CI gate. When a future runtime makes the API coverable, remove
its allowlist line and add the test — the checker flags redundant entries.

## Skeleton

A minimal starting point (also in `TEMPLATE.cc.txt`):

```cpp
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

// One or two sentences: which public API this pins and what portable guarantee
// the case asserts (round-trip / accepted-or-unsupported / rejection).
// @asserts: hipSomeApi - one-line portable invariant this case pins
HIP_TEST_CASE(Contract_<Area>_<Behavior>) {
  hip::contract::ContractCleanup cleanup;

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 256));
  cleanup.Add([ptr] { (void)hipFree(ptr); });  // capture BY VALUE

  // ... exercise the API and REQUIRE(...) the portable invariant.
  // For an intentional rejection, capture the status, REQUIRE it is an error,
  // then clear the sticky error:
  //   const hipError_t status = hipSomeApi(/* invalid input */);
  //   REQUIRE(status != hipSuccess);
  //   (void)hipGetLastError();
}
```

Remember to register the case in `contract.yaml`, add the `CMakeLists.txt`, and
`add_subdirectory` the domain — then re-run CMake configure.
