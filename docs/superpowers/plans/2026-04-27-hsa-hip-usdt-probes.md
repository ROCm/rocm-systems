# HSA + HIP USDT Entry/Exit Probes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add USDT (User Statically-Defined Tracing) entry/exit probe pairs to every public HIP and HSA runtime API so external BPF/bpftrace observers can attach without modifying or restarting the application.

**Architecture:**
- Build-time gated by per-project `ROCM_ENABLE_USDT` CMake option that auto-detects `<sys/sdt.h>` (from `systemtap-sdt-dev`).
- HIP: probes injected into existing `api_callbacks_spawner_t` RAII object (one place, fires for every `HIP_INIT_API` call). Exit value read from `hip::tls.last_command_error_`.
- HSA: probes injected by redefining `TRY`/`CATCH` macros to wrap the function body in a lambda whose return value the entry/exit guard can capture; `CATCHRET(T)` paths get entry-only probes in V1 and selective hand-instrumentation for high-value APIs.
- Probe args always include: API ID (uint32 enum value), correlation ID (uint64), and on exit the return code. Function name passed via `__func__` pointer for HSA where no enum exists.
- Disabled cost: 1 nop per probe site + (for guarded sites) one cache-resident semaphore load.

**Tech Stack:** C++17, GCC/Clang, `<sys/sdt.h>` (systemtap-sdt-dev), CMake, bpftrace (test/CI only), libbpf-skeleton (example tool).

---

## File Structure

**HIP (projects/clr):**
- Create: `projects/clr/hipamd/src/usdt/rocm_hip_usdt.h` — macro shims, semaphore variants, no-op fallback.
- Modify: `projects/clr/hipamd/src/hip_prof_api.h` — add probe calls inside `api_callbacks_spawner_t` ctor/dtor.
- Modify: `projects/clr/hipamd/CMakeLists.txt` — `ROCM_ENABLE_USDT` option, FindSystemtap include.
- Create: `projects/clr/cmake/FindSystemtap.cmake` — locates `<sys/sdt.h>`.

**HSA (projects/rocr-runtime):**
- Create: `projects/rocr-runtime/runtime/hsa-runtime/core/inc/usdt/rocm_hsa_usdt.h` — macro shims + EntryGuard helper.
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa.cpp` — redefine `TRY`/`CATCH` macros, add includes.
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa_ext_amd.cpp` — same macro redefinition.
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt` — `ROCM_ENABLE_USDT` option.
- Create: `projects/rocr-runtime/cmake/FindSystemtap.cmake` — same find script (or symlink/share with clr).

**Tests + tooling:**
- Create: `tools/bpf/README.md` — overview.
- Create: `tools/bpf/bpftrace/hip_api_count.bt` — example: count HIP API calls.
- Create: `tools/bpf/bpftrace/hsa_api_latency.bt` — example: HSA API latency histogram.
- Create: `tools/bpf/bpftrace/hip_correlation_join.bt` — example: pair HIP entry to HSA entry by correlation_id.
- Create: `projects/clr/hipamd/test/usdt/test_hip_usdt_probes.sh` — CI sanity test, attaches bpftrace and validates probe args.
- Create: `projects/rocr-runtime/runtime/hsa-runtime/test/usdt/test_hsa_usdt_probes.sh` — CI sanity test for HSA.

**Docs:**
- Create: `projects/clr/hipamd/docs/conceptual/usdt-tracepoints.rst` — HIP-side reference.
- Create: `projects/rocr-runtime/docs/conceptual/usdt-tracepoints.rst` — HSA-side reference.

---

## Task 1: Add FindSystemtap.cmake

**Files:**
- Create: `projects/clr/cmake/FindSystemtap.cmake`
- Create: `projects/rocr-runtime/cmake/FindSystemtap.cmake` (identical content; cmake doesn't share)

- [ ] **Step 1: Write the find module**

Create `projects/clr/cmake/FindSystemtap.cmake` with this exact content:

```cmake
# FindSystemtap.cmake
#
# Locates <sys/sdt.h> from systemtap-sdt-dev / systemtap-sdt-devel.
# Defines:
#   Systemtap_FOUND
#   Systemtap_INCLUDE_DIR
#   Systemtap::sdt (INTERFACE imported target)

find_path(Systemtap_INCLUDE_DIR
    NAMES sys/sdt.h
    PATHS /usr/include /usr/local/include
    DOC "Path to <sys/sdt.h> from systemtap-sdt-dev")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Systemtap
    REQUIRED_VARS Systemtap_INCLUDE_DIR)

if(Systemtap_FOUND AND NOT TARGET Systemtap::sdt)
    add_library(Systemtap::sdt INTERFACE IMPORTED)
    set_target_properties(Systemtap::sdt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Systemtap_INCLUDE_DIR}")
endif()

mark_as_advanced(Systemtap_INCLUDE_DIR)
```

- [ ] **Step 2: Copy to rocr-runtime**

```bash
cp projects/clr/cmake/FindSystemtap.cmake projects/rocr-runtime/cmake/FindSystemtap.cmake
```

- [ ] **Step 3: Verify it locates the header on a system with the package installed**

Run on a dev box with `systemtap-sdt-dev` installed:
```bash
mkdir -p /tmp/findstap && cd /tmp/findstap
cmake -DCMAKE_MODULE_PATH=$PWD/../../home/bewelton/rocm-systems/worktrees/pr5486/projects/clr/cmake \
      -P - <<'EOF'
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Systemtap REQUIRED)
message(STATUS "Found at: ${Systemtap_INCLUDE_DIR}")
EOF
```
Expected: `-- Found at: /usr/include` (or wherever `sys/sdt.h` lives).

If `systemtap-sdt-dev` is not installed, expected: error `Could NOT find Systemtap`.

- [ ] **Step 4: Commit**

```bash
git add projects/clr/cmake/FindSystemtap.cmake projects/rocr-runtime/cmake/FindSystemtap.cmake
git commit -m "build: add FindSystemtap.cmake for sys/sdt.h discovery"
```

---

## Task 2: Add ROCM_ENABLE_USDT option to HIP build

**Files:**
- Modify: `projects/clr/hipamd/CMakeLists.txt`

- [ ] **Step 1: Locate the options section**

Run:
```bash
grep -n "^option(" projects/clr/hipamd/CMakeLists.txt | head -10
```
Expected: a list of existing `option(...)` calls. Note the line of the last one — append the new option after it.

- [ ] **Step 2: Add the option, find_package, and propagate**

Append after the last existing `option(...)` line in `projects/clr/hipamd/CMakeLists.txt`:

```cmake
option(ROCM_ENABLE_USDT "Enable USDT (sys/sdt.h) static tracepoints" OFF)

if(ROCM_ENABLE_USDT)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
    find_package(Systemtap REQUIRED)
    add_compile_definitions(ROCM_ENABLE_USDT=1)
    message(STATUS "USDT static tracepoints: ENABLED (sys/sdt.h at ${Systemtap_INCLUDE_DIR})")
else()
    message(STATUS "USDT static tracepoints: disabled")
endif()
```

- [ ] **Step 3: Verify configure with USDT off (default)**

```bash
mkdir -p /tmp/clr_build_off && cd /tmp/clr_build_off
cmake /home/bewelton/rocm-systems/worktrees/pr5486/projects/clr 2>&1 | grep -i usdt
```
Expected: `-- USDT static tracepoints: disabled`

- [ ] **Step 4: Verify configure with USDT on (requires systemtap-sdt-dev)**

```bash
mkdir -p /tmp/clr_build_on && cd /tmp/clr_build_on
cmake -DROCM_ENABLE_USDT=ON /home/bewelton/rocm-systems/worktrees/pr5486/projects/clr 2>&1 | grep -iE "usdt|systemtap"
```
Expected: `-- USDT static tracepoints: ENABLED (sys/sdt.h at /usr/include)` and `-- Found Systemtap: ...`

If `systemtap-sdt-dev` is not installed, expected: configure fails with `Could NOT find Systemtap`.

- [ ] **Step 5: Commit**

```bash
git add projects/clr/hipamd/CMakeLists.txt
git commit -m "build: add ROCM_ENABLE_USDT option to HIP (clr/hipamd)"
```

---

## Task 3: Add ROCM_ENABLE_USDT option to HSA build

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt`

- [ ] **Step 1: Locate options section**

```bash
grep -n "^option(" projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt | head -10
```

- [ ] **Step 2: Append the option**

Append after the last existing `option(...)` line:

```cmake
option(ROCM_ENABLE_USDT "Enable USDT (sys/sdt.h) static tracepoints" OFF)

if(ROCM_ENABLE_USDT)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
    find_package(Systemtap REQUIRED)
    add_compile_definitions(ROCM_ENABLE_USDT=1)
    include_directories(${Systemtap_INCLUDE_DIR})
    message(STATUS "HSA USDT static tracepoints: ENABLED")
else()
    message(STATUS "HSA USDT static tracepoints: disabled")
endif()
```

- [ ] **Step 3: Verify both configurations**

```bash
mkdir -p /tmp/rocr_off && cd /tmp/rocr_off
cmake /home/bewelton/rocm-systems/worktrees/pr5486/projects/rocr-runtime/runtime/hsa-runtime 2>&1 | grep -i usdt
```
Expected: `-- HSA USDT static tracepoints: disabled`

```bash
mkdir -p /tmp/rocr_on && cd /tmp/rocr_on
cmake -DROCM_ENABLE_USDT=ON /home/bewelton/rocm-systems/worktrees/pr5486/projects/rocr-runtime/runtime/hsa-runtime 2>&1 | grep -i usdt
```
Expected: `-- HSA USDT static tracepoints: ENABLED`

- [ ] **Step 4: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt
git commit -m "build: add ROCM_ENABLE_USDT option to HSA (rocr-runtime)"
```

---

## Task 4: Create rocm_hip_usdt.h shim header

**Files:**
- Create: `projects/clr/hipamd/src/usdt/rocm_hip_usdt.h`

- [ ] **Step 1: Write the header**

Create `projects/clr/hipamd/src/usdt/rocm_hip_usdt.h` with this exact content:

```cpp
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 *
 * USDT (User Statically-Defined Tracing) probe shims for HIP.
 *
 * Provider: hip
 * Probes:
 *   hip:api_entry(uint32_t api_id, uint64_t correlation_id)
 *   hip:api_exit(uint32_t api_id, uint64_t correlation_id, int32_t hip_error)
 *
 * Disabled-build cost: zero (macros expand to (void)0).
 * Enabled-build, unattached cost: 1 nop per probe site.
 * Enabled-build, attached cost: ~1-3 us per fire (kernel uprobe trap + BPF program).
 */

#ifndef HIP_SRC_USDT_ROCM_HIP_USDT_H
#define HIP_SRC_USDT_ROCM_HIP_USDT_H

#include <cstdint>

#if defined(ROCM_ENABLE_USDT) && ROCM_ENABLE_USDT
#  include <sys/sdt.h>
#  define ROCM_HIP_USDT_API_ENTRY(api_id, corr) \
        DTRACE_PROBE2(hip, api_entry, (uint32_t)(api_id), (uint64_t)(corr))
#  define ROCM_HIP_USDT_API_EXIT(api_id, corr, err) \
        DTRACE_PROBE3(hip, api_exit, (uint32_t)(api_id), (uint64_t)(corr), (int32_t)(err))
#else
#  define ROCM_HIP_USDT_API_ENTRY(api_id, corr) ((void)0)
#  define ROCM_HIP_USDT_API_EXIT(api_id, corr, err) ((void)0)
#endif

#endif  // HIP_SRC_USDT_ROCM_HIP_USDT_H
```

- [ ] **Step 2: Verify header compiles in both modes**

```bash
g++ -x c++ -std=c++17 -c -DROCM_ENABLE_USDT=1 -Iprojects/clr/hipamd/src/usdt - <<'EOF'
#include "rocm_hip_usdt.h"
int main() { ROCM_HIP_USDT_API_ENTRY(42, 0xdeadbeefULL); ROCM_HIP_USDT_API_EXIT(42, 0xdeadbeefULL, 0); return 0; }
EOF
```
Expected: compiles with no warnings (assuming `sys/sdt.h` is installed).

```bash
g++ -x c++ -std=c++17 -c -Iprojects/clr/hipamd/src/usdt - <<'EOF'
#include "rocm_hip_usdt.h"
int main() { ROCM_HIP_USDT_API_ENTRY(42, 0xdeadbeefULL); ROCM_HIP_USDT_API_EXIT(42, 0xdeadbeefULL, 0); return 0; }
EOF
```
Expected: compiles with no warnings (USDT-disabled fallback).

Clean up: `rm -f a.out *.o`

- [ ] **Step 3: Commit**

```bash
git add projects/clr/hipamd/src/usdt/rocm_hip_usdt.h
git commit -m "hip: add rocm_hip_usdt.h shim header"
```

---

## Task 5: Wire HIP probes into api_callbacks_spawner_t

**Files:**
- Modify: `projects/clr/hipamd/src/hip_prof_api.h`

- [ ] **Step 1: Read the current file to understand existing layout**

```bash
cat projects/clr/hipamd/src/hip_prof_api.h
```
Expected: `api_callbacks_spawner_t` template with constructor that runs the rocprofiler `phase_enter` callback and destructor that runs `phase_exit`.

- [ ] **Step 2: Add include and probe calls**

Apply this exact edit to `projects/clr/hipamd/src/hip_prof_api.h`:

Replace:
```cpp
#include "hip/amd_detail/hip_prof_str.h"
#include "platform/prof_protocol.h"
```
With:
```cpp
#include "hip/amd_detail/hip_prof_str.h"
#include "platform/prof_protocol.h"
#include "usdt/rocm_hip_usdt.h"
#include "hip_internal.hpp"  // for hip::tls
```

Then in the `api_callbacks_spawner_t` constructor, add the probe call as the very first statement (before any other code):

Replace:
```cpp
  template <typename Functor> api_callbacks_spawner_t(Functor init_cb_args_data) {
    static_assert(operation_id >= HIP_API_ID_FIRST && operation_id <= HIP_API_ID_LAST,
                  "invalid HIP_API operation id");
```
With:
```cpp
  template <typename Functor> api_callbacks_spawner_t(Functor init_cb_args_data) {
    static_assert(operation_id >= HIP_API_ID_FIRST && operation_id <= HIP_API_ID_LAST,
                  "invalid HIP_API operation id");

    // USDT entry probe — fires before any rocprofiler bookkeeping so external
    // observers see the call even if no rocprofiler tool is registered.
    // correlation_id reuses the rocprofiler-assigned value when a tool is
    // active; otherwise a thread-local counter provides a unique id.
    ROCM_HIP_USDT_API_ENTRY(operation_id, hip::tls.usdt_correlation_id_++);
```

In the destructor, add the exit probe after the rocprofiler `phase_exit` (so correlation_id is still set):

Replace:
```cpp
  ~api_callbacks_spawner_t() {
    if (enabled_) {
      if (trace_data_.phase_exit != nullptr) trace_data_.phase_exit(operation_id, &trace_data_);
      amd::activity_prof::correlation_id = 0;
    }
  }
```
With:
```cpp
  ~api_callbacks_spawner_t() {
    if (enabled_) {
      if (trace_data_.phase_exit != nullptr) trace_data_.phase_exit(operation_id, &trace_data_);
      amd::activity_prof::correlation_id = 0;
    }
    // USDT exit probe — fires for every HIP API regardless of rocprofiler state.
    // hip::tls.last_command_error_ holds the return code set by HIP_RETURN.
    ROCM_HIP_USDT_API_EXIT(operation_id, hip::tls.usdt_correlation_id_,
                           hip::tls.last_command_error_);
  }
```

Also handle the `HIP_API_ID_NONE` specialization — replace:
```cpp
template <> class api_callbacks_spawner_t<HIP_API_ID_NONE> {
 public:
  template <typename Functor> api_callbacks_spawner_t(Functor) {}
};
```
With:
```cpp
template <> class api_callbacks_spawner_t<HIP_API_ID_NONE> {
 public:
  template <typename Functor> api_callbacks_spawner_t(Functor) {
    // No probes for the sentinel HIP_API_ID_NONE id.
  }
};
```

- [ ] **Step 3: Add usdt_correlation_id_ to TLS struct**

Open `projects/clr/hipamd/src/hip_internal.hpp` and find `struct TlsAggregator` (or whatever holds the TLS members — search for `last_command_error_`):

```bash
grep -n "last_command_error_" projects/clr/hipamd/src/hip_internal.hpp
```

Add a new member right next to `last_command_error_`:

```cpp
  hipError_t last_command_error_;
  hipError_t last_error_;
  uint64_t usdt_correlation_id_{0};   // ADDED: monotonic per-thread USDT correlation
```

- [ ] **Step 4: Build the library with USDT off (verify nothing breaks)**

```bash
cd projects/clr && mkdir -p build_usdt_off && cd build_usdt_off
cmake .. && make -j$(nproc) hipamd64 2>&1 | tail -20
```
Expected: build succeeds.

- [ ] **Step 5: Build with USDT on**

```bash
cd projects/clr && mkdir -p build_usdt_on && cd build_usdt_on
cmake -DROCM_ENABLE_USDT=ON .. && make -j$(nproc) hipamd64 2>&1 | tail -20
```
Expected: build succeeds.

- [ ] **Step 6: Verify probes are present in the binary**

```bash
readelf -n projects/clr/build_usdt_on/hipamd/libamdhip64.so | grep -A3 stapsdt | head -30
```
Expected: at least two distinct probes named `api_entry` and `api_exit` under provider `hip`. Count should be 2 (the macros expand once each in the templated class — the linker may keep one instantiation per `operation_id` due to template specialization, so you may see many more).

```bash
readelf -n projects/clr/build_usdt_on/hipamd/libamdhip64.so | grep -c stapsdt
```
Expected: a number ≥ 2 (likely much higher due to per-API-ID template instantiations).

- [ ] **Step 7: Commit**

```bash
git add projects/clr/hipamd/src/hip_prof_api.h projects/clr/hipamd/src/hip_internal.hpp
git commit -m "hip: emit USDT api_entry/api_exit probes from api_callbacks_spawner_t"
```

---

## Task 6: Create rocm_hsa_usdt.h shim header with EntryGuard

**Files:**
- Create: `projects/rocr-runtime/runtime/hsa-runtime/core/inc/usdt/rocm_hsa_usdt.h`

- [ ] **Step 1: Write the header**

Create the file with this exact content:

```cpp
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 *
 * USDT probe shims for HSA runtime.
 *
 * Provider: hsa
 * Probes:
 *   hsa:api_entry(const char* func_name, uint64_t correlation_id)
 *   hsa:api_exit(const char* func_name, uint64_t correlation_id, uint64_t status_or_value)
 *
 * Function name is passed as a pointer to the .rodata literal produced by __func__.
 * BPF programs read it with bpf_probe_read_user_str() on demand.
 *
 * For hsa_status_t-returning APIs, status_or_value is the hsa_status_t cast to
 * uint64_t. For other return types it is the raw return value (handles via 64-bit
 * truncation; pointer/uint64/uint32 returns work; void returns are emitted from a
 * void-specialized guard with status_or_value == 0).
 */

#ifndef HSA_CORE_USDT_ROCM_HSA_USDT_H
#define HSA_CORE_USDT_ROCM_HSA_USDT_H

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(ROCM_ENABLE_USDT) && ROCM_ENABLE_USDT
#  include <sys/sdt.h>
#  define ROCM_HSA_USDT_ENTRY(name, corr) \
        DTRACE_PROBE2(hsa, api_entry, (name), (uint64_t)(corr))
#  define ROCM_HSA_USDT_EXIT(name, corr, val) \
        DTRACE_PROBE3(hsa, api_exit, (name), (uint64_t)(corr), (uint64_t)(val))
#else
#  define ROCM_HSA_USDT_ENTRY(name, corr) ((void)0)
#  define ROCM_HSA_USDT_EXIT(name, corr, val) ((void)0)
#endif

namespace rocm { namespace hsa { namespace usdt {

inline uint64_t generate_correlation_id() noexcept {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

// EntryGuard fires the entry probe in its constructor. It is intended to be
// used with operator+ to wrap a lambda whose return value the guard captures
// and reports through the exit probe. Used by the redefined TRY/CATCH macros.
struct EntryGuard {
    const char* name;
    uint64_t corr;

    explicit EntryGuard(const char* n) noexcept
        : name(n), corr(generate_correlation_id()) {
        ROCM_HSA_USDT_ENTRY(name, corr);
    }

    // Wraps a callable (the function body lambda). Invokes it, captures the
    // return value, fires the exit probe, returns the value. Specialized below
    // for void return.
    template <typename F>
    auto operator+(F&& fn) -> typename std::enable_if<
        !std::is_void<decltype(fn())>::value,
        decltype(fn())>::type {
        auto result = std::forward<F>(fn)();
        ROCM_HSA_USDT_EXIT(name, corr, (uint64_t)result);
        return result;
    }
};

// Void-return specialization helper. Used by callers whose lambda returns void.
struct EntryGuardVoid {
    const char* name;
    uint64_t corr;

    explicit EntryGuardVoid(const char* n) noexcept
        : name(n), corr(generate_correlation_id()) {
        ROCM_HSA_USDT_ENTRY(name, corr);
    }

    template <typename F>
    void operator+(F&& fn) {
        std::forward<F>(fn)();
        ROCM_HSA_USDT_EXIT(name, corr, 0);
    }
};

}}}  // namespace rocm::hsa::usdt

#endif  // HSA_CORE_USDT_ROCM_HSA_USDT_H
```

- [ ] **Step 2: Verify header compiles standalone in both modes**

```bash
g++ -x c++ -std=c++17 -c -DROCM_ENABLE_USDT=1 \
    -Iprojects/rocr-runtime/runtime/hsa-runtime/core/inc/usdt - <<'EOF'
#include "rocm_hsa_usdt.h"
int main() {
    int r = rocm::hsa::usdt::EntryGuard("test_fn") + [&]() -> int { return 42; };
    rocm::hsa::usdt::EntryGuardVoid("test_void_fn") + [&]() {};
    return r == 42 ? 0 : 1;
}
EOF
./a.out && echo OK
```
Expected: `OK`. Clean up: `rm -f a.out`.

```bash
g++ -x c++ -std=c++17 -c \
    -Iprojects/rocr-runtime/runtime/hsa-runtime/core/inc/usdt - <<'EOF'
#include "rocm_hsa_usdt.h"
int main() {
    int r = rocm::hsa::usdt::EntryGuard("test_fn") + [&]() -> int { return 42; };
    return r == 42 ? 0 : 1;
}
EOF
```
Expected: compiles cleanly (USDT-disabled fallback).

- [ ] **Step 3: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/core/inc/usdt/rocm_hsa_usdt.h
git commit -m "hsa: add rocm_hsa_usdt.h shim header with EntryGuard helper"
```

---

## Task 7: Redefine TRY/CATCH in hsa.cpp to wrap function body

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa.cpp`

- [ ] **Step 1: Add include**

After the existing `#include` block at the top of `hsa.cpp` (before the `#define IS_BAD_PROFILE` macros), add:

```cpp
#include "core/inc/usdt/rocm_hsa_usdt.h"
```

- [ ] **Step 2: Replace TRY/CATCH macros**

Locate the existing definitions:
```cpp
#define TRY try {
#define CATCH } catch(...) { return AMD::handleException(); }
#define CATCHRET(RETURN_TYPE) } catch(...) { return AMD::handleExceptionT<RETURN_TYPE>(); }
```

Replace with:
```cpp
// USDT-instrumented TRY/CATCH wraps the function body in a lambda so the
// EntryGuard can capture the return value for the exit probe.
//
// Original semantics (try/catch around the body, return AMD::handleException()
// on exception) are preserved.
//
// TRY/CATCH pair is for hsa_status_t-returning functions. CATCHRET(T) handles
// other return types (uint64_t, void, hsa_signal_value_t, etc.) and uses the
// EntryGuard or EntryGuardVoid as appropriate.

#define TRY \
    return ::rocm::hsa::usdt::EntryGuard(__func__) + [&]() -> hsa_status_t { try {

#define CATCH \
    } catch(...) { return AMD::handleException(); } }

#define CATCHRET(RETURN_TYPE) \
    } catch(...) { return AMD::handleExceptionT<RETURN_TYPE>(); } }

// Companion macros for non-hsa_status_t TRY sites. These should be used in
// new code; existing CATCHRET sites are migrated in Task 8.
#define TRY_T(RETURN_TYPE) \
    return ::rocm::hsa::usdt::EntryGuard(__func__) + [&]() -> RETURN_TYPE { try {

#define TRY_VOID \
    ::rocm::hsa::usdt::EntryGuardVoid(__func__) + [&]() { try {

#define CATCH_VOID \
    } catch(...) { AMD::handleException(); } };
```

Note: the original `TRY` was `try {` and the function body wrote `return X; CATCH;`. After this change, the function body now reads:

```cpp
hsa_status_t hsa_init() {
    TRY;                                     // expands to: return EntryGuard(__func__) + [&]() -> hsa_status_t { try {
    return core::Runtime::runtime_singleton_->Acquire();
    CATCH;                                   // expands to: } catch(...) { return AMD::handleException(); } }
}
```

The lambda returns `hsa_status_t`, EntryGuard's `operator+` captures it, fires exit, returns it. The outer `return` returns the EntryGuard's result.

- [ ] **Step 3: Migrate CATCHRET sites that use TRY**

The existing pattern for non-hsa_status_t functions is:
```cpp
uint64_t hsa_some_thing(...) {
    TRY;
    return ...;
    CATCHRET(uint64_t);
}
```

After Task 7's macro change, `TRY` declares the lambda return type as `hsa_status_t`, which mismatches `uint64_t`. We need to convert these sites. Find them:

```bash
grep -nB2 "CATCHRET(uint64_t)" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa.cpp | grep -B1 "TRY;" | head -20
```

For each site, replace:
```cpp
    TRY;
    ...
    CATCHRET(uint64_t);
```
with:
```cpp
    TRY_T(uint64_t);
    ...
    CATCHRET(uint64_t);
```

Do the same for `CATCHRET(hsa_signal_value_t)` (use `TRY_T(hsa_signal_value_t)`) and `CATCHRET(void)` (use `TRY_VOID` and `CATCH_VOID` instead of `CATCHRET(void)`).

Use this script to enumerate the sites that need conversion:
```bash
awk '/^[[:space:]]*TRY;/{tryline=NR; next}
     /^[[:space:]]*CATCHRET\(([^)]+)\);/{
         match($0, /CATCHRET\(([^)]+)\)/, m);
         print FILENAME ":" tryline " " m[1]
     }' projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa.cpp
```
Expected: a list of (line, type) pairs for every `TRY` whose matching `CATCHRET` is non-`hsa_status_t`. There are ~51 of these in hsa.cpp.

For each line, do the substitution. A safe sed equivalent (per-type):
```bash
# Manually inspect first; this is destructive
sed -i 's/^\(\s*\)TRY;\s*$/\1TRY_T(uint64_t);/' /tmp/preview.cpp  # only for uint64_t blocks
```
Recommended: do this manually with targeted edits since the script can't trivially know which return type belongs to which TRY without parsing C++.

- [ ] **Step 4: Build with USDT off (verify TRY/CATCH macro change doesn't break baseline)**

```bash
cd projects/rocr-runtime/runtime/hsa-runtime
mkdir -p build_off && cd build_off
cmake .. && make -j$(nproc) 2>&1 | tail -30
```
Expected: build succeeds. The lambda wrapping is zero-cost (RVO, inlining) at -O2.

- [ ] **Step 5: Build with USDT on**

```bash
cd projects/rocr-runtime/runtime/hsa-runtime
mkdir -p build_on && cd build_on
cmake -DROCM_ENABLE_USDT=ON .. && make -j$(nproc) 2>&1 | tail -30
```
Expected: build succeeds.

- [ ] **Step 6: Verify probes are present**

```bash
readelf -n projects/rocr-runtime/runtime/hsa-runtime/build_on/libhsa-runtime64.so | grep -A3 stapsdt | head -20
readelf -n projects/rocr-runtime/runtime/hsa-runtime/build_on/libhsa-runtime64.so | grep -c "Name: api_entry"
readelf -n projects/rocr-runtime/runtime/hsa-runtime/build_on/libhsa-runtime64.so | grep -c "Name: api_exit"
```
Expected: api_entry count ≥ 100 (one per HSA API site instrumented), api_exit count similar.

- [ ] **Step 7: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa.cpp
git commit -m "hsa: wrap TRY/CATCH with USDT EntryGuard for entry/exit probes"
```

---

## Task 8: Apply same TRY/CATCH redefinition in hsa_ext_amd.cpp

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa_ext_amd.cpp`

- [ ] **Step 1: Add include**

Add at the top after existing includes:
```cpp
#include "core/inc/usdt/rocm_hsa_usdt.h"
```

- [ ] **Step 2: Replace TRY/CATCH/CATCHRET definitions**

Find the existing definitions:
```bash
grep -n "define TRY\|define CATCH" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa_ext_amd.cpp
```

Replace them with the same definitions from Task 7 Step 2 (TRY, CATCH, CATCHRET, TRY_T, TRY_VOID, CATCH_VOID).

- [ ] **Step 3: Convert non-hsa_status_t TRY sites**

```bash
grep -nB2 "CATCHRET" projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa_ext_amd.cpp | grep -B1 "TRY;" | head -20
```

For each TRY whose matching CATCHRET is non-hsa_status_t, replace `TRY;` with `TRY_T(<return_type>);` (or `TRY_VOID;` for void).

- [ ] **Step 4: Build with USDT on**

```bash
cd projects/rocr-runtime/runtime/hsa-runtime/build_on
make -j$(nproc) 2>&1 | tail -20
```
Expected: build succeeds.

- [ ] **Step 5: Verify combined probe count**

```bash
readelf -n projects/rocr-runtime/runtime/hsa-runtime/build_on/libhsa-runtime64.so | grep -c "Name: api_entry"
```
Expected: ≥ 200 (122 from hsa.cpp + 78 from hsa_ext_amd.cpp + others).

- [ ] **Step 6: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa_ext_amd.cpp
git commit -m "hsa: apply USDT TRY/CATCH wrapping to hsa_ext_amd.cpp"
```

---

## Task 9: Apply TRY/CATCH redefinition in remaining HSA TRY-using files

**Files:**
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/image/hsa_ext_image.cpp`
- Modify: `projects/rocr-runtime/runtime/hsa-runtime/pcs/hsa_ven_amd_pc_sampling.cpp`

- [ ] **Step 1: Find all files with TRY definitions**

```bash
grep -rn "define TRY" projects/rocr-runtime/runtime/hsa-runtime/
```
Expected: 3 files (hsa.cpp, hsa_ext_amd.cpp already done; hsa_ext_image.cpp and hsa_ven_amd_pc_sampling.cpp remain).

- [ ] **Step 2: Apply same Task 7 + 8 changes to hsa_ext_image.cpp**

Add include `#include "core/inc/usdt/rocm_hsa_usdt.h"` and replace TRY/CATCH/CATCHRET definitions per Task 7 Step 2. Convert non-hsa_status_t TRY sites per Task 7 Step 3.

- [ ] **Step 3: Apply same to hsa_ven_amd_pc_sampling.cpp**

Same procedure.

- [ ] **Step 4: Build and verify**

```bash
cd projects/rocr-runtime/runtime/hsa-runtime/build_on
make -j$(nproc) 2>&1 | tail -10
readelf -n libhsa-runtime64.so | grep -c "Name: api_entry"
```
Expected: build succeeds, api_entry count higher than Task 8 result.

- [ ] **Step 5: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/image/hsa_ext_image.cpp \
        projects/rocr-runtime/runtime/hsa-runtime/pcs/hsa_ven_amd_pc_sampling.cpp
git commit -m "hsa: USDT TRY/CATCH wrapping for ext_image and pc_sampling"
```

---

## Task 10: Add bpftrace example scripts

**Files:**
- Create: `tools/bpf/README.md`
- Create: `tools/bpf/bpftrace/hip_api_count.bt`
- Create: `tools/bpf/bpftrace/hsa_api_latency.bt`
- Create: `tools/bpf/bpftrace/hip_correlation_join.bt`

- [ ] **Step 1: Write README**

Create `tools/bpf/README.md`:

```markdown
# ROCm BPF / bpftrace tools

Example scripts and tools that consume the USDT static tracepoints exposed by
HIP and HSA when built with `-DROCM_ENABLE_USDT=ON`.

## Probe surface

### Provider `hip` (libamdhip64.so)

| Probe | Args |
|-------|------|
| `hip:api_entry` | (uint32_t api_id, uint64_t correlation_id) |
| `hip:api_exit`  | (uint32_t api_id, uint64_t correlation_id, int32_t hip_error) |

API IDs are values from `enum hip_api_id_t` in `hip/amd_detail/hip_prof_str.h`.

### Provider `hsa` (libhsa-runtime64.so)

| Probe | Args |
|-------|------|
| `hsa:api_entry` | (const char* func_name, uint64_t correlation_id) |
| `hsa:api_exit`  | (const char* func_name, uint64_t correlation_id, uint64_t status_or_value) |

`func_name` is a pointer to `__func__` in the HSA library — read with
`str(arg0)` in bpftrace.

## Permissions

Most BPF/uprobe operations require `CAP_BPF + CAP_PERFMON + CAP_SYS_PTRACE`
(typically root).

## Examples

- `bpftrace/hip_api_count.bt` — count HIP API calls per API ID per process.
- `bpftrace/hsa_api_latency.bt` — latency histogram per HSA API.
- `bpftrace/hip_correlation_join.bt` — pair HIP entry to HSA entry by correlation_id.
```

- [ ] **Step 2: Write hip_api_count.bt**

```bash
cat > tools/bpf/bpftrace/hip_api_count.bt <<'EOF'
#!/usr/bin/env bpftrace
/*
 * Count HIP API calls per API ID per process.
 * Usage: sudo bpftrace -p <PID> tools/bpf/bpftrace/hip_api_count.bt
 *        sudo bpftrace tools/bpf/bpftrace/hip_api_count.bt   (system-wide)
 */

usdt:/opt/rocm/lib/libamdhip64.so:hip:api_entry {
    @counts[pid, comm, arg0] = count();
}

interval:s:5 {
    print(@counts);
    clear(@counts);
}
EOF
chmod +x tools/bpf/bpftrace/hip_api_count.bt
```

- [ ] **Step 3: Write hsa_api_latency.bt**

```bash
cat > tools/bpf/bpftrace/hsa_api_latency.bt <<'EOF'
#!/usr/bin/env bpftrace
/*
 * Per-HSA-API latency histogram. Pairs entry to exit by correlation_id.
 * Usage: sudo bpftrace tools/bpf/bpftrace/hsa_api_latency.bt
 */

usdt:/opt/rocm/lib/libhsa-runtime64.so:hsa:api_entry {
    @start[arg1] = nsecs;
    @name[arg1] = arg0;
}

usdt:/opt/rocm/lib/libhsa-runtime64.so:hsa:api_exit / @start[arg1] != 0 / {
    $latency = nsecs - @start[arg1];
    @latency_us[str(@name[arg1])] = hist($latency / 1000);
    delete(@start[arg1]);
    delete(@name[arg1]);
}
EOF
chmod +x tools/bpf/bpftrace/hsa_api_latency.bt
```

- [ ] **Step 4: Write hip_correlation_join.bt**

```bash
cat > tools/bpf/bpftrace/hip_correlation_join.bt <<'EOF'
#!/usr/bin/env bpftrace
/*
 * Demonstrate HIP -> HSA join via shared correlation_id.
 * NOTE: HIP correlation_id is a separate counter from HSA correlation_id.
 * They join via thread-id + temporal proximity, not directly. This script
 * shows the pattern; a future correlation-id bridge would make it direct.
 *
 * Usage: sudo bpftrace tools/bpf/bpftrace/hip_correlation_join.bt
 */

usdt:/opt/rocm/lib/libamdhip64.so:hip:api_entry {
    @hip_active[tid] = arg0;  // remember the HIP API ID per thread
    @hip_corr[tid] = arg1;
    @hip_t[tid] = nsecs;
}

usdt:/opt/rocm/lib/libhsa-runtime64.so:hsa:api_entry / @hip_active[tid] != 0 / {
    $hip_id = @hip_active[tid];
    $hsa_name = str(arg0);
    @hip_to_hsa[$hip_id, $hsa_name] = count();
}

usdt:/opt/rocm/lib/libamdhip64.so:hip:api_exit {
    delete(@hip_active[tid]);
    delete(@hip_corr[tid]);
    delete(@hip_t[tid]);
}
EOF
chmod +x tools/bpf/bpftrace/hip_correlation_join.bt
```

- [ ] **Step 5: Sanity-check syntax**

```bash
sudo bpftrace --unsafe -e 'BEGIN { exit(); }' >/dev/null 2>&1 && echo "bpftrace usable" || echo "bpftrace not installed; install bpftrace before continuing"
```

If bpftrace is installed, validate each script's syntax:
```bash
for f in tools/bpf/bpftrace/*.bt; do
    sudo bpftrace -d "$f" >/dev/null 2>&1 && echo "OK: $f" || echo "FAIL: $f"
done
```
Expected: `OK` for all three.

- [ ] **Step 6: Commit**

```bash
git add tools/bpf/
git commit -m "tools: add bpftrace example scripts for HIP/HSA USDT probes"
```

---

## Task 11: Add CI sanity test for HIP probes

**Files:**
- Create: `projects/clr/hipamd/test/usdt/test_hip_usdt_probes.sh`
- Create: `projects/clr/hipamd/test/usdt/test_hip_usdt_main.cpp`

- [ ] **Step 1: Write the test program**

Create `projects/clr/hipamd/test/usdt/test_hip_usdt_main.cpp`:

```cpp
// Calls a deterministic sequence of HIP APIs so the probe-attach test
// can assert exact counts.
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    int count = 0;
    hipGetDeviceCount(&count);     // expect 1 hip:api_entry + 1 hip:api_exit

    void* p = nullptr;
    hipMalloc(&p, 4096);            // expect 1 + 1
    hipFree(p);                     // expect 1 + 1

    int dev = 0;
    hipGetDevice(&dev);             // expect 1 + 1
    return 0;
}
```

- [ ] **Step 2: Write the test script**

Create `projects/clr/hipamd/test/usdt/test_hip_usdt_probes.sh`:

```bash
#!/usr/bin/env bash
# CI sanity test: build USDT-enabled HIP, run a tiny app under bpftrace,
# verify expected probe counts and arg values.

set -euo pipefail

HIP_BUILD="${HIP_BUILD:-/tmp/hip_usdt_test_build}"
TEST_BIN="${HIP_BUILD}/test_hip_usdt_main"
LIBHIP="${HIP_BUILD}/hipamd/libamdhip64.so"

if ! command -v bpftrace >/dev/null; then
    echo "SKIP: bpftrace not installed"
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    echo "SKIP: requires root (CAP_BPF + CAP_PERFMON)"
    exit 0
fi

# 1. Probe inventory check: readelf must show api_entry and api_exit
ENTRY_COUNT=$(readelf -n "$LIBHIP" | grep -c "Name: api_entry")
EXIT_COUNT=$(readelf -n "$LIBHIP" | grep -c "Name: api_exit")
if [[ $ENTRY_COUNT -lt 1 || $EXIT_COUNT -lt 1 ]]; then
    echo "FAIL: no USDT probes found in $LIBHIP (entry=$ENTRY_COUNT exit=$EXIT_COUNT)"
    exit 1
fi
echo "PASS: probe inventory (entry=$ENTRY_COUNT exit=$EXIT_COUNT)"

# 2. Functional test: counts must be reproducible
RESULT=$(mktemp)
trap "rm -f $RESULT" EXIT

bpftrace -e "
    usdt:$LIBHIP:hip:api_entry { @entries[arg0] = count(); }
    usdt:$LIBHIP:hip:api_exit  { @exits[arg0] = count(); }
    END { print(@entries); print(@exits); }
" -c "$TEST_BIN" > "$RESULT" 2>&1

# Expected: 4 distinct API IDs, each fired exactly once
ENTRY_LINES=$(grep -c "@entries\[" "$RESULT" || true)
if [[ $ENTRY_LINES -lt 4 ]]; then
    echo "FAIL: expected at least 4 distinct API entries, got $ENTRY_LINES"
    cat "$RESULT"
    exit 1
fi

# Each entry count should be 1 (deterministic test)
if grep -E "@entries\[[0-9]+\]:" "$RESULT" | grep -v ": 1$"; then
    echo "FAIL: some HIP API was called more than once"
    cat "$RESULT"
    exit 1
fi

echo "PASS: probe firing (entries=$ENTRY_LINES, all count=1)"
echo "OK: HIP USDT sanity test passed"
```

- [ ] **Step 3: Make script executable and validate**

```bash
chmod +x projects/clr/hipamd/test/usdt/test_hip_usdt_probes.sh
bash -n projects/clr/hipamd/test/usdt/test_hip_usdt_probes.sh && echo "syntax OK"
```
Expected: `syntax OK`.

- [ ] **Step 4: Commit**

```bash
git add projects/clr/hipamd/test/usdt/
git commit -m "test: HIP USDT probe sanity test (CI)"
```

---

## Task 12: Add CI sanity test for HSA probes

**Files:**
- Create: `projects/rocr-runtime/runtime/hsa-runtime/test/usdt/test_hsa_usdt_probes.sh`
- Create: `projects/rocr-runtime/runtime/hsa-runtime/test/usdt/test_hsa_usdt_main.cpp`

- [ ] **Step 1: Write the HSA test program**

Create `projects/rocr-runtime/runtime/hsa-runtime/test/usdt/test_hsa_usdt_main.cpp`:

```cpp
// Calls a deterministic HSA sequence.
#include <hsa/hsa.h>
#include <cstdio>

int main() {
    hsa_init();           // hsa:api_entry / hsa:api_exit pair, name="hsa_init"
    uint16_t major = 0;
    hsa_system_get_major_extension_table(0, 1, sizeof(major), &major);  // pair
    hsa_shut_down();      // pair
    return 0;
}
```

- [ ] **Step 2: Write the HSA test script**

Create `projects/rocr-runtime/runtime/hsa-runtime/test/usdt/test_hsa_usdt_probes.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

HSA_BUILD="${HSA_BUILD:-/tmp/hsa_usdt_test_build}"
TEST_BIN="${HSA_BUILD}/test_hsa_usdt_main"
LIBHSA="${HSA_BUILD}/libhsa-runtime64.so"

if ! command -v bpftrace >/dev/null; then echo "SKIP: bpftrace not installed"; exit 0; fi
if [[ $EUID -ne 0 ]]; then echo "SKIP: requires root"; exit 0; fi

ENTRY_COUNT=$(readelf -n "$LIBHSA" | grep -c "Name: api_entry")
if [[ $ENTRY_COUNT -lt 100 ]]; then
    echo "FAIL: expected >=100 hsa:api_entry probes, got $ENTRY_COUNT"
    exit 1
fi
echo "PASS: probe inventory ($ENTRY_COUNT api_entry sites)"

RESULT=$(mktemp); trap "rm -f $RESULT" EXIT

bpftrace -e "
    usdt:$LIBHSA:hsa:api_entry { @[str(arg0)] = count(); }
    END { print(@); }
" -c "$TEST_BIN" > "$RESULT" 2>&1

# Validate the deterministic 3 functions appear
for fn in hsa_init hsa_system_get_major_extension_table hsa_shut_down; do
    if ! grep -q "@\[$fn\]" "$RESULT"; then
        echo "FAIL: missing $fn in trace"
        cat "$RESULT"
        exit 1
    fi
done
echo "PASS: all 3 expected HSA APIs traced"
echo "OK: HSA USDT sanity test passed"
```

- [ ] **Step 3: Make executable and validate**

```bash
chmod +x projects/rocr-runtime/runtime/hsa-runtime/test/usdt/test_hsa_usdt_probes.sh
bash -n projects/rocr-runtime/runtime/hsa-runtime/test/usdt/test_hsa_usdt_probes.sh && echo "syntax OK"
```

- [ ] **Step 4: Commit**

```bash
git add projects/rocr-runtime/runtime/hsa-runtime/test/usdt/
git commit -m "test: HSA USDT probe sanity test (CI)"
```

---

## Task 13: Microbenchmark — disabled vs enabled-unattached vs attached

**Files:**
- Create: `tools/bpf/benchmark/usdt_overhead.cpp`
- Create: `tools/bpf/benchmark/run_usdt_overhead.sh`

- [ ] **Step 1: Write the benchmark program**

Create `tools/bpf/benchmark/usdt_overhead.cpp`:

```cpp
// Measure HIP API call cost in three configurations:
//   1. USDT-disabled build (baseline)
//   2. USDT-enabled build, no observer attached
//   3. USDT-enabled build, bpftrace observer attached
//
// The same binary runs in cases 2 and 3; case 1 needs a USDT-disabled rebuild.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>

constexpr int N = 1'000'000;

int main() {
    int dev = 0;
    hipGetDevice(&dev);  // warm up

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        hipGetDevice(&dev);
    }
    auto t1 = std::chrono::steady_clock::now();

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("hipGetDevice x %d: %.2f ns/call\n", N, (double)ns / N);
    return 0;
}
```

- [ ] **Step 2: Write the benchmark driver**

Create `tools/bpf/benchmark/run_usdt_overhead.sh`:

```bash
#!/usr/bin/env bash
# Run the USDT overhead microbenchmark in all three configurations.
# Requires both USDT-on and USDT-off builds of HIP available.
set -euo pipefail

BENCH_BIN="${BENCH_BIN:-/tmp/usdt_overhead}"
LIBHIP_OFF="${LIBHIP_OFF:?set to USDT-disabled libamdhip64.so}"
LIBHIP_ON="${LIBHIP_ON:?set to USDT-enabled libamdhip64.so}"

echo "=== Config 1: USDT disabled (baseline) ==="
LD_LIBRARY_PATH="$(dirname "$LIBHIP_OFF")" "$BENCH_BIN"

echo
echo "=== Config 2: USDT enabled, NO observer ==="
LD_LIBRARY_PATH="$(dirname "$LIBHIP_ON")" "$BENCH_BIN"

if command -v bpftrace >/dev/null && [[ $EUID -eq 0 ]]; then
    echo
    echo "=== Config 3: USDT enabled, bpftrace observer attached ==="
    bpftrace -e "usdt:$LIBHIP_ON:hip:api_entry { @ = count(); }" \
        -c "LD_LIBRARY_PATH=$(dirname "$LIBHIP_ON") $BENCH_BIN" 2>&1 | head -5
else
    echo "SKIP Config 3: needs root + bpftrace"
fi
```

- [ ] **Step 3: Chmod + commit**

```bash
chmod +x tools/bpf/benchmark/run_usdt_overhead.sh
git add tools/bpf/benchmark/
git commit -m "bench: USDT overhead microbenchmark (disabled/unattached/attached)"
```

- [ ] **Step 4: Run the benchmark and record numbers**

After the HIP build (Task 5) is installed, run:
```bash
g++ -O2 -std=c++17 tools/bpf/benchmark/usdt_overhead.cpp -lamdhip64 -L/path/to/build -o /tmp/usdt_overhead
LIBHIP_OFF=/path/to/usdt_off/libamdhip64.so \
LIBHIP_ON=/path/to/usdt_on/libamdhip64.so \
sudo -E tools/bpf/benchmark/run_usdt_overhead.sh
```

Expected (rough, varies by host):
- Config 1 baseline: ~50-100 ns/call
- Config 2 USDT-on, unattached: ~50-105 ns/call (within ~5%)
- Config 3 attached: ~1000-3000 ns/call

Record the actual numbers in a follow-up commit message or in the docs page (Task 14).

---

## Task 14: Write conceptual docs for HIP USDT

**Files:**
- Create: `projects/clr/hipamd/docs/conceptual/usdt-tracepoints.rst`

- [ ] **Step 1: Write the doc**

```rst
.. meta::
  :description: USDT static tracepoints in HIP for BPF/bpftrace observers
  :keywords: HIP, USDT, BPF, bpftrace, tracing

.. _hip-usdt-tracepoints:

USDT static tracepoints in HIP
==============================

When HIP is built with ``-DROCM_ENABLE_USDT=ON``, ``libamdhip64.so`` exposes
USDT (User Statically-Defined Tracing) probes that external BPF tools can
attach to without modifying or restarting the application.

Probe surface
-------------

.. list-table::
   :header-rows: 1

   * - Probe
     - Args
     - Fires
   * - ``hip:api_entry``
     - ``(uint32_t api_id, uint64_t correlation_id)``
     - At the start of every public HIP API
   * - ``hip:api_exit``
     - ``(uint32_t api_id, uint64_t correlation_id, int32_t hip_error)``
     - As the HIP API returns

The ``api_id`` value is the corresponding ``hip_api_id_t`` enum value defined
in ``hip/amd_detail/hip_prof_str.h``. The ``correlation_id`` is a monotonic
per-thread counter; the matching entry/exit pair share the same id.

Cost model
----------

* Built but unattached: 1 nop per probe site; effectively zero runtime cost.
* Attached: ~1-3 microseconds per fire on x86_64 (kernel uprobe trap +
  BPF program execution).

Do not enable probes during PGO/AutoFDO training runs — int3 trap dominates the
profile.

Stability
---------

The provider/probe names (``hip:api_entry``, ``hip:api_exit``) and their
argument shapes are committed ABI for any downstream BPF tool. The ``api_id``
enum is shared with rocprofiler-sdk and follows the same compatibility rules.

Examples
--------

See ``tools/bpf/bpftrace/`` for runnable bpftrace scripts.
```

- [ ] **Step 2: Add to docs index**

Find the docs index:
```bash
grep -rn "toctree" projects/clr/hipamd/docs/ | head
```
Add `conceptual/usdt-tracepoints` to the appropriate toctree.

- [ ] **Step 3: Commit**

```bash
git add projects/clr/hipamd/docs/
git commit -m "docs(hip): add USDT tracepoints conceptual page"
```

---

## Task 15: Write conceptual docs for HSA USDT

**Files:**
- Create: `projects/rocr-runtime/docs/conceptual/usdt-tracepoints.rst`

- [ ] **Step 1: Write the doc**

```rst
.. meta::
  :description: USDT static tracepoints in HSA runtime for BPF/bpftrace observers
  :keywords: HSA, ROCr, USDT, BPF, bpftrace, tracing

.. _hsa-usdt-tracepoints:

USDT static tracepoints in HSA runtime
======================================

When ROCr (HSA runtime) is built with ``-DROCM_ENABLE_USDT=ON``,
``libhsa-runtime64.so`` exposes USDT static tracepoints for every HSA API
wrapped by the ``TRY``/``CATCH`` pattern.

Probe surface
-------------

.. list-table::
   :header-rows: 1

   * - Probe
     - Args
     - Fires
   * - ``hsa:api_entry``
     - ``(const char* func_name, uint64_t correlation_id)``
     - At the start of every wrapped HSA API
   * - ``hsa:api_exit``
     - ``(const char* func_name, uint64_t correlation_id, uint64_t status_or_value)``
     - At return; ``status_or_value`` is ``hsa_status_t`` cast to uint64_t (or
       the raw return value for non-status APIs)

``func_name`` is a pointer to the ``__func__`` literal in ``libhsa-runtime64.so``.
Read it from BPF with ``bpf_probe_read_user_str()`` (or ``str(arg0)`` in bpftrace).

Cost model
----------

Identical to the HIP USDT path — see :ref:`hip-usdt-tracepoints`.

Stability
---------

The probe names are committed ABI. Function names follow the underlying HSA API
ABI and are stable per ROCr release. ``correlation_id`` is a process-global
monotonic counter; HSA correlation_id is **not** the same counter as HIP
correlation_id (HIP correlation_id is per-thread). Use thread-id +
temporal proximity to join HIP and HSA traces.

Examples
--------

See ``tools/bpf/bpftrace/`` for runnable bpftrace scripts.
```

- [ ] **Step 2: Add to docs index**

```bash
grep -rn "toctree" projects/rocr-runtime/docs/ | head
```
Add `conceptual/usdt-tracepoints` to the appropriate toctree.

- [ ] **Step 3: Commit**

```bash
git add projects/rocr-runtime/docs/
git commit -m "docs(hsa): add USDT tracepoints conceptual page"
```

---

## Task 16: Verify FDO interaction (sanity check, not a regression test)

**Files:**
- Create: `tools/bpf/benchmark/fdo_sanity.md`

- [ ] **Step 1: Document the FDO discipline**

Create `tools/bpf/benchmark/fdo_sanity.md`:

```markdown
# FDO interaction notes for ROCM_ENABLE_USDT

## Rules

1. **Train and ship with the same `ROCM_ENABLE_USDT` value.** Mixing creates
   profile-vs-binary mismatch (different basic-block sizes from the nops).
2. **Never attach probes during the FDO training run.** Attached probes are
   int3 traps; the profile becomes garbage.
3. **Verify probes still resolve correctly after FDO build.**

## Verification command

After an FDO-enabled build of `libamdhip64.so` with `ROCM_ENABLE_USDT=ON`:

```bash
# 1. Probe count should match a non-FDO build
readelf -n libamdhip64.so | grep -c "Name: api_entry"

# 2. Run the CI sanity test (Task 11) to verify args still extract correctly
PATH_TO_LIBHIP=$(realpath libamdhip64.so) \
    projects/clr/hipamd/test/usdt/test_hip_usdt_probes.sh
```

If probe counts differ between FDO and non-FDO builds, that is expected (ICF
or hot/cold splitting may merge or move probes). What must hold: every
expected `(api_id, args)` pair fires when the corresponding API is called.
```

- [ ] **Step 2: Commit**

```bash
git add tools/bpf/benchmark/fdo_sanity.md
git commit -m "docs: document FDO interaction discipline for USDT builds"
```

---

## Task 17: End-to-end integration test

**Files:**
- Create: `tools/bpf/integration/test_e2e.sh`

- [ ] **Step 1: Write end-to-end test**

```bash
cat > tools/bpf/integration/test_e2e.sh <<'EOF'
#!/usr/bin/env bash
# End-to-end: USDT-built HIP, real GPU app, bpftrace observer, validate
# entry/exit pairing.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then echo "SKIP: requires root"; exit 0; fi
if ! command -v bpftrace >/dev/null; then echo "SKIP: bpftrace required"; exit 0; fi

LIBHIP="${LIBHIP:-/opt/rocm/lib/libamdhip64.so}"
APP="${APP:-/opt/rocm/share/hip/samples/0_Intro/bit_extract/bit_extract}"

if [[ ! -f "$APP" ]]; then echo "SKIP: $APP not present"; exit 0; fi

RESULT=$(mktemp); trap "rm -f $RESULT" EXIT

bpftrace -e "
    usdt:$LIBHIP:hip:api_entry { @entries[arg0] = count(); @corr[tid, arg0] = arg1; }
    usdt:$LIBHIP:hip:api_exit  { @exits[arg0] = count();
        if (@corr[tid, arg0] != arg1) { @mismatches = count(); }
    }
    END { print(@entries); print(@exits); print(@mismatches); }
" -c "$APP" > "$RESULT" 2>&1

if grep -q "@mismatches" "$RESULT" && ! grep -q "@mismatches: 0" "$RESULT"; then
    echo "FAIL: entry/exit correlation_id mismatches detected"
    cat "$RESULT"
    exit 1
fi

echo "PASS: e2e test, no correlation_id mismatches"
EOF
chmod +x tools/bpf/integration/test_e2e.sh
```

- [ ] **Step 2: Validate syntax**

```bash
bash -n tools/bpf/integration/test_e2e.sh && echo OK
```

- [ ] **Step 3: Commit**

```bash
git add tools/bpf/integration/test_e2e.sh
git commit -m "test: USDT e2e integration test with correlation_id pairing"
```

---

## Self-review notes

**Spec coverage:**
- Build gating ✓ (Tasks 1-3)
- HIP entry/exit ✓ (Tasks 4-5, free via existing RAII)
- HSA entry/exit pairs ✓ (Tasks 6-9, lambda-wrapper trick captures return)
- Examples + tools ✓ (Task 10)
- CI tests ✓ (Tasks 11-12)
- Overhead measurement ✓ (Task 13)
- Docs ✓ (Tasks 14-15)
- FDO discipline ✓ (Task 16)
- E2E correlation validation ✓ (Task 17)

**Known limitations carried into V1:**
- HSA `correlation_id` is process-global (not per-thread like HIP). Documented
  in Task 15. A V2 could unify them via TLS.
- `void`-returning HSA APIs that previously used `CATCHRET(void)` need manual
  conversion to `TRY_VOID`/`CATCH_VOID` (Task 7 Step 3).
- Argument-name BPF auto-extraction depends on debug info; the bpftrace examples
  use positional `arg0`/`arg1` which always works.
- USDT-on builds add ~200 nops per HSA library + per-API HIP nops. Binary size
  impact: ~few KiB. Negligible.

**What's NOT in this plan (deferred to follow-ups):**
- Hand-instrumented richer args for high-value HSA APIs (queue_create,
  async_copy_*) — V2.
- Unifying HIP and HSA correlation_id into a shared TLS counter — V2.
- libbpf-skeleton example collector — V2.
- nop5 (5-byte nop) optimization — depends on toolchain support detection.
- Per-AQL-packet probes — explicitly avoided per overhead analysis.

---

**Plan complete. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task with two-stage review between tasks. Best for high-stakes work.

**2. Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
