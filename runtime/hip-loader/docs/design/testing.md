# HIP Loader Testing

This document specifies the test backend and test strategy for the HIP loader
implementation. The main goal is to test loader behavior without requiring a
GPU, HSA runtime initialization, or the current CLR implementation.

## Test Backend Purpose

The loader needs a deterministic backend that behaves like a HIP backend at the
ABI boundary but does not perform GPU work. This backend should be the first
backend implemented for the loader because it makes these properties cheap to
test in CI:

* Backend discovery and selection.
* Versioned loader-backend identity handshake.
* Backend function-table dispatch.
* Public wrapper behavior.
* Compiler-private ABI wrapper behavior.
* `hipGetProcAddress` public-name behavior.
* Public-to-backend API name translation.
* Compatibility shim plumbing.
* Symbol hygiene and dynamic-linker isolation.
* Loader logging and error reporting.

The test backend is not a HIP emulator. It only needs enough behavior to verify
the loader contract.

## Backend Targets

The implementation should add these test-only backend fixtures:

* Good backend: `hip-loader-test-backend`
* Wrong-major backend: `hip-loader-test-backend-wrong-major`
* Missing-handshake backend: `hip-loader-test-backend-missing-handshake`
* Bad-public-export backend: `hip-loader-test-backend-bad-public-export`

Suggested output names:

* Linux good backend: `libamdhip-test.so`
* Windows good backend: `amdhip-test.dll`
* Linux wrong-major backend: `libamdhip-test-wrong-major.so`
* Windows wrong-major backend: `amdhip-test-wrong-major.dll`

The good backend is compiled with:

```text
HIP_ABI_MODE=HIP_ABI_MODE_BACKEND
HIP_API_VERSION=<newest supported HIP API version>
```

It exports the minimal backend identity handshake for that API major, for
example:

```text
hipBackendV8GetInterface
```

It also exports generated backend symbols such as `hipBackendV8Init` and
compiler-private backend symbols such as
`hipBackendV8CompilerRegisterFatBinary`. The handshake returns backend metadata
only; the loader resolves backend symbols separately and builds its own dispatch
tables.

It must not export public HIP API symbols such as `hipInit`,
`hipRuntimeGetVersion`, or `hipMemcpyAsync`, and it must not export public
compiler-private ABI symbols such as `__hipRegisterFatBinary`.

## Backend Behavior

The good test backend should implement every generated private symbol required by
the test manifest. API entries that are not explicitly modeled should point at
generated unsupported stubs rather than be omitted. This lets tests distinguish
"loader table is malformed" from "API is not modeled by this backend".

Default behavior:

* `hipInit(0)` succeeds.
* `hipInit(nonzero)` returns `hipErrorInvalidValue`.
* `hipRuntimeGetVersion` returns a deterministic sentinel version derived from
  the backend API version, for example `80000000` for backend API 8.
* `hipDriverGetVersion` returns the same sentinel version.
* `hipGetDeviceCount` returns the configured fake device count.
* `hipGetDevice` and `hipSetDevice` operate on process-local backend state.
* `hipDeviceGetName` returns `HIP Loader Test Backend Device <ordinal>`.
* `hipGetLastError` returns and clears the backend's last error.
* `hipPeekAtLastError` returns the backend's last error without clearing it.
* `hipGetErrorName` and `hipGetErrorString` return stable strings for modeled
  errors.
* Modeled unsupported APIs return `hipErrorNotSupported` where the return type
  allows it.

Minimum modeled API families:

* Initialization and version APIs.
* Device count, current device, device name, and simple attributes.
* Error query APIs.
* Stream create, destroy, query, and synchronize.
* Event create, record, query, synchronize, and destroy.
* Host-backed fake `hipMalloc`, `hipFree`, `hipMemcpy`, and `hipMemset`.
* `hipGetProcAddress` coverage through the loader, not as a backend public
  symbol.
* Compiler-private fat binary registration and unregister callbacks.
* Compiler-private launch configuration push and pop callbacks.

Fake allocation handles should use host memory plus a small header containing a
magic value and size. `hipMemcpy` between fake device allocations and host memory
can be implemented as `memcpy` with validation. Async APIs can perform the work
synchronously and return success.

The backend must not depend on HSA, ROCR, COMGR, rocprofiler, or GPU devices.

For compiler-private ABI tests, the backend can store fake fat binary handles in
process-local state. `RegisterFatBinary` returns an opaque handle for a non-null
input pointer, registration calls attach metadata to that handle, and
`UnregisterFatBinary` invalidates the handle.

## Test Control ABI

The good test backend should export a small test-only control ABI in addition to
the backend identity and generated private symbols. These symbols are not part of
the HIP loader backend ABI and must not be exported by production backends.

The control ABI should be plain C so tests can drive it from Python `ctypes`:

```c
typedef struct hip_test_backend_call_v1 {
  uint32_t struct_size;
  uint32_t hip_api_version;
  const char* canonical_name;
  const char* backend_symbol;
  uint64_t sequence;
  uint32_t phase;
  uint32_t arg_count;
  void* const* args;
  void* return_value;
} hip_test_backend_call_v1;

typedef int (*hip_test_backend_api_callback_v1)(
    const hip_test_backend_call_v1* call, void* user_data);

int __testBackendSetAPICallback(
    hip_test_backend_api_callback_v1 callback, void* user_data);
int __testBackendClearAPICallback(void);
int __testBackendResetState(void);
uint64_t __testBackendGetCallCount(void);
```

`phase` should distinguish at least function entry and function exit. The entry
callback may return a HIP error code to force the modeled API to return that
error where the API's return type allows it. Returning `hipSuccess` or zero lets
the default backend behavior run. For APIs with non-`hipError_t` return types,
the backend should document which callback return values are meaningful and
otherwise ignore forced errors.

The callback receives pointers to the live arguments. This intentionally allows
tests to inspect input arguments, write through output pointers for selected
APIs, and observe output pointers before and after the backend writes them. Tests
must treat the pointers as valid only for the duration of the callback. The
backend must not retain callback-provided pointers after returning.

For Python `ctypes`, the test driver can:

* Load the public loader library with `ctypes.CDLL`.
* Select the test backend by environment or package configuration.
* Load the test backend DSO/DLL with `ctypes.CDLL`.
* Install `__testBackendSetAPICallback`.
* Call public HIP APIs through the loader.
* Observe canonical API names, backend symbol names, argument pointers, call
  order, and forced return behavior through the callback.

Structured out-of-process logs may still be useful for crash diagnostics, but
the callback is the primary test inspection mechanism. It avoids requiring every
test to launch a subprocess just to collect backend call traces.

## Backend Configuration

The test backend may read backend-specific environment variables. These are for
tests only and should not become public loader configuration:

* `HIP_TEST_BACKEND_DEVICE_COUNT`: fake device count, default `2`.
* `HIP_TEST_BACKEND_FAIL_INIT`: when set to `1`, `Init` returns a deterministic
  failure.
* `HIP_TEST_BACKEND_FAIL_AFTER_CALL`: when set to a positive integer, the backend
  starts failing modeled APIs after that many backend calls.
* `HIP_TEST_BACKEND_LOG_FILE`: optional structured call log for tests that need
  out-of-process inspection or crash diagnostics.

Tests should prefer loader environment variables for selecting the backend, for
example:

```text
HIP_LOADER_BACKEND_PATH=<path to libamdhip-test.so>
HIP_LOADER_LOG=error,warn,info
```

## Negative Fixtures

The wrong-major backend exports a valid identity handshake and private symbols
for a different backend API major, for example `hipBackendV7GetInterface` and
`hipBackendV7*` symbols when the release's loaders require
`hipBackendV8GetInterface` and `hipBackendV8*` symbols. The expected result is a
controlled loader failure before dispatching any HIP API call to the backend.
This must fail the same way whether the application entered through an older
public ABI library or the current public ABI library.

The missing-handshake backend is a loadable DSO/DLL with no backend identity
handshake.
The expected result is a controlled loader failure that names the missing
handshake symbol when diagnostics are enabled.

The bad-public-export backend intentionally exports one public HIP symbol such
as `hipInit`. It is not used for normal loader execution. It exists to prove the
symbol-hygiene CI check fails when a backend leaks public ABI symbols.

A second bad export variant should intentionally export a compiler-private ABI
symbol such as `__hipRegisterFatBinary`. It exists to prove the symbol-hygiene
CI check fails when a backend leaks public compiler ABI symbols.

## Test Categories

### Unit Tests

Unit tests should cover pure loader logic:

* Environment parsing and precedence.
* Safe handling of disabled environment overrides in elevated contexts.
* Backend path validation.
* Backend selection diagnostics.
* Manifest lookup by canonical public name.
* Public-name to backend-table mapping.
* `hipGetProcAddress` alias and `_spt` lookup rules.
* Error conversion when backend load or negotiation fails.

These tests do not need to load a shared library.

### Dynamic Load Tests

Dynamic load tests use the good backend by explicit path:

* Loader opens the backend with `RTLD_NOW | RTLD_LOCAL` on ELF.
* Loader finds the versioned identity handshake symbol.
* Loader validates the returned metadata size and API version.
* Loader resolves generated private symbols into loader-owned dispatch tables.
* Loader dispatches modeled APIs through those tables.
* Test driver can load the same backend DSO/DLL and install
  `__testBackendSetAPICallback`.
* Loader rejects wrong-major and missing-handshake fixtures.
* Loader leaves backend loaded for process lifetime.

On Windows, equivalent tests use `LoadLibrary` and `GetProcAddress` with the
versioned backend identity handshake and generated private symbols.

### Public ABI Tests

Public ABI tests link or load the public loader library and verify:

* `hipInit(0)` succeeds through the test backend.
* `hipRuntimeGetVersion` returns the backend sentinel version through the public
  wrapper.
* `hipGetDeviceCount` and `hipSetDevice` preserve process-local state.
* Fake memory operations round-trip bytes correctly.
* Unsupported modeled APIs return the expected error.
* `hipGetProcAddress("hipInit", ...)` returns a callable public ABI-compatible
  pointer.
* `hipGetProcAddress` handles aliases such as versioned device-property names
  according to manifest rules.
* `__testBackendSetAPICallback` observes each modeled backend call with
  canonical public API name and backend symbol name.
* A callback can force a modeled `hipError_t` API to return a selected error
  code.
* A callback can write deterministic values through output pointers for modeled
  APIs that return structs or scalar outputs.
* A simulated HIP user shared library with a static constructor can resolve and
  call `__hipRegisterFatBinary`, `__hipRegisterFunction`, `__hipRegisterVar`,
  and `__hipUnregisterFatBinary`.
* Launch lowering helpers `__hipPushCallConfiguration` and
  `__hipPopCallConfiguration` dispatch through the loader private table.

The tests should run without a GPU.

### Compatibility Shim Tests

When two HIP public API majors are available, tests should load the older public
library and dispatch to the newest test backend:

* Older HIP public wrappers load the current release's latest backend API.
* The compatibility metadata marks every modeled ABI-breaking symbol for that
  older public major and names the manual compat function that handles it.
* Unchanged symbols use generated direct wrappers into the newest backend table.
* Changed symbols use the named manual compat wrapper, such as a
  `loader_v6_compat.cpp` function for HIP 6.
* Structure translation is applied for modeled compatibility structs.
* Callback-observed output pointers show that the backend wrote the newest
  backend struct layout before the loader translated it to the older public
  layout.
* Callback-forced error codes propagate through compatibility shims without
  corrupting output structures.
* Callback-observed backend symbol names prove that old public calls dispatched
  to newest backend private symbols, not old backend symbols.
* The good test backend intentionally does not export legacy backend symbols
  such as `hipBackendV7GetDevicePropertiesR0000` or
  `hipBackendV7ChooseDeviceR0000`; HIP 6 legacy public calls must still work.
* Removed or incompatible APIs fail with documented compatibility behavior.
* `hipGetProcAddress` on the older public library returns ABI-compatible older
  public wrappers, not newest backend function pointers.

The HIP 6 proof of concept should include real compatibility tests for the known
device-property ABI split:

* `hipGetDevicePropertiesR0600` from the HIP 6 public loader returns a
  HIP 6-layout `hipDeviceProp_tR0600` result.
* Legacy exports `hipGetDeviceProperties` and `hipGetDevicePropertiesR0000`
  return an old-layout `hipDeviceProp_tR0000` result.
* The test backend callback observes the newest concrete device-property backend
  symbol, for example `hipBackendV7GetDevicePropertiesR0600`, and can write
  deterministic newest-layout fields before the compat layer converts them.
* `hipChooseDeviceR0600` accepts HIP 6-layout input properties, while legacy
  `hipChooseDevice` and `hipChooseDeviceR0000` accept old-layout input
  properties. All dispatch to newest backend behavior after conversion or
  equivalent manual matching.
* `hipGetProcAddress("hipGetDeviceProperties", hipVersion=600, ...)` returns a
  `hipGetDevicePropertiesR0600`-compatible public wrapper address.
* `hipGetProcAddress("hipGetDeviceProperties", hipVersion < 600, ...)` returns
  a legacy-compatible public wrapper address when that behavior is part of the
  old public ABI being carried.
* Calling either returned pointer still records a newest-backend private symbol
  in the test backend callback.
* `hipDrvGraphAddMemsetNode` and `hipDrvGraphExecMemsetNodeSetParams` from the
  HIP 6 public loader accept old `HIP_MEMSET_NODE_PARAMS` layouts, convert to
  newest `hipMemsetParams`, and dispatch to newest backend graph memset symbols.
* `hipPointerGetAttributes`, `hipFreeAsync`, and any selected behavior-only
  compatibility wrappers return HIP 6 behavior before dispatching to newest
  backend behavior where needed.
* If HIP 6 `hiprtc*` runtime exports are in the POC scope, they are exported by
  the HIP 6 loader and either forward to `libhiprtc` or fail with documented
  compatibility behavior. They must not appear in the HIP 7 loader merely
  because stale source version scripts list them.

Until a real future major such as HIP 8 exists, forward-looking compatibility
can be tested with synthetic manifest versions and a small modeled struct whose
layout intentionally differs between test API versions. The HIP 6 proof should
use real archived or checked-in HIP 6 ABI data where available.

### Symbol Hygiene Tests

Linux checks:

* `readelf` or `llvm-readelf` verifies the good backend exports only the allowed
  backend identity handshake and generated private symbols.
* `readelf` or `llvm-readelf` verifies the good backend has no dynamic
  relocations to public HIP API symbols or public compiler-private `__hip*`
  symbols.
* `readelf --version-info` verifies each public loader DSO has exactly the
  expected version-node names for its public major.
* `readelf --dyn-syms` verifies compatibility symbols land on the expected
  version nodes, for example `hipGetDevicePropertiesR0000@@hip_4.2`,
  `hipGetDevicePropertiesR0600@@hip_6.0`,
  `hipDrvGraphAddMemsetNode@@hip_5.6`, and
  `hipDrvGraphExecMemsetNodeSetParams@@hip_6.2` in the HIP 6 loader.
* The HIP 7 loader is checked against its own baseline, including the fact that
  ROCm 7.0 introduced `hip_6.5` rather than `hip_7.0`.
* Actual dynamic exports are checked, not only source version scripts, so stale
  entries such as old `hiprtc*` names do not leak into newer loaders.
* The bad-public-export fixture fails the forbidden-export check.
* The bad compiler-private export fixture fails the forbidden-export check.

Windows checks:

* `dumpbin` or `llvm-readobj` verifies the good backend exports only the allowed
  backend identity handshake and generated private symbols.
* `dumpbin` or `llvm-readobj` verifies public loader DLL exports match the
  per-public-major Windows baseline. Windows has no ELF version nodes, and its
  historical DEF files do not necessarily match Linux symbol sets.
* The bad-public-export fixture fails the forbidden-export check.
* The bad compiler-private export fixture fails the forbidden-export check.

These checks should be buildable and runnable without GPU hardware.

### `LD_PRELOAD` Interaction Tests

Linux should have a small public HIP interposer test library that exports one or
two public APIs, such as `hipInit`, and records that it was called.

Expected behavior:

* An application call to `hipInit` can be intercepted at the public boundary when
  `LD_PRELOAD` is used.
* The loader still dispatches to the backend through the loader-owned private
  function table.
* Backend internals do not call public `hip*` symbols and therefore do not route
  through the interposer.
* If an old public HIP runtime is preloaded, the test records the observed
  behavior and the loader emits a diagnostic when it can detect the conflict.

This test demonstrates that `LD_PRELOAD` remains a diagnostic/interposition
mechanism, not the backend selection strategy.

### Logging Tests

Logging tests should verify:

* Backend selection logs include backend name, path, and selected API version.
* Rejected backend logs include the missing or mismatched handshake name.
* API trace logging records canonical public API names, not backend symbols.
* Sensitive environment overrides are not logged in a way that leaks secrets.
* Test backend structured logs, when enabled, contain enough data to correlate
  process, thread, call sequence, canonical API name, backend symbol, phase, and
  result code.

## CI Integration

The test backend and loader tests should be ordinary CTest tests with labels:

* `hip-loader`
* `hip-loader-backend`
* `hip-loader-no-gpu`

Tests that require `LD_PRELOAD` should be Linux-only and labeled:

* `hip-loader-ld-preload`

The no-GPU test set should be cheap enough to run on every pull request that
touches:

* HIP loader sources.
* HIP API headers.
* API manifest tooling.
* Generated loader/backend wrappers.
* Public export maps or DEF files.
* Backend symbol-hygiene tooling.
* Test backend control ABI.

## Acceptance Criteria

The test backend is sufficient when CI can prove:

* A good backend loads, passes metadata validation, resolves private symbols, and
  dispatches through loader-owned tables.
* Wrong-major and missing-handshake backends fail deterministically.
* Public loader wrappers expose public HIP names while backend symbols remain
  private and versioned.
* Public loader wrappers expose required compiler-private ABI names while backend
  private ABI symbols remain private and versioned.
* The test backend callback can observe calls and force return codes from Python
  `ctypes`.
* Compatibility tests can verify output-struct translation using callback-visible
  backend arguments.
* Historic public loaders, starting with the HIP 6 POC, use manual compat
  functions for changed symbols while unchanged symbols remain generated direct
  dispatch.
* Historic public loaders preserve exact Linux version nodes and Windows export
  baselines for their public major.
* The HIP 6 POC proves legacy `R0000` public symbols are implemented in the
  loader and do not require backend `R0000` symbols.
* Backend libraries do not leak public HIP exports.
* Backend libraries do not leak public compiler-private `__hip*` exports.
* `hipGetProcAddress` returns public ABI-compatible pointers for public names.
* `LD_PRELOAD` interposes only the public boundary in the intended test.
* Tests run without GPU hardware.

## Alternatives Considered

### Start With CLR As The First Backend

This would test the real implementation first, but it couples loader bring-up to
GPU availability, HSA initialization, and CLR conversion work. It also makes
negative tests harder. This is rejected for initial loader CI.

### Mock Only At The Loader Unit-test Layer

Pure unit mocks are useful, but they do not test dynamic loading, export lists,
platform symbol lookup, or ELF/Windows symbol hygiene. This is rejected as the
only strategy.

### Use A Passthrough Prototype As The Test Backend

The passthrough prototype reviewed in [research.md](research.md) is useful prior
art, but it depends on loading another
HIP implementation and does not enforce the private backend ABI model. This is
rejected as the primary test backend.

### Require A GPU For Loader Tests

GPU-backed tests are needed later for end-to-end validation, but they are too
expensive and environment-dependent for the loader's basic ABI and dispatch
contract. This is rejected for the core CI gate.
