# HIP Loader Design

This document proposes the first HIP loader architecture. It is intentionally
conservative: the loader owns the public ABI, backends use private versioned ABI
symbols, and generation is used for repetitive ABI surfaces.

## Chosen Architecture

The process model is:

```text
application
  -> public HIP loader library for ABI major N
     -> optional loader logging/interception
        -> loader-generated function table for newest supported backend symbols
           -> selected backend implementation
```

The loader is the only component that exports the public HIP runtime ABI. Backend
libraries are implementation DSOs/DLLs with unique names and private symbols.
The loader-owned ABI includes both documented user-callable APIs and
compiler-private ABI entry points emitted by HIP-Clang into user objects and
shared libraries.

On Linux:

* Public HIP 7 ABI: `libamdhip64.so.7`
* Public HIP 8 ABI: `libamdhip64.so.8`
* Backend example: `libamdhip-clr.so`
* Backend example: `libamdhip-example_backend.so`

On Windows:

* Public HIP 7 ABI: `amdhip-7.dll`
* Public HIP 8 ABI: `amdhip-8.dll`
* Backend example: `amdhip-clr.dll`
* Backend example: `amdhip-example_backend.dll`

The unversioned Linux development symlink and any Windows transition name are
packaging policy. They should point at a loader library, not a concrete backend.

## Version Model

There are three distinct version concepts:

* HIP public ABI major: the ABI exported to applications, such as HIP 7 or HIP 8.
* HIP backend API version: the newest HIP API implemented by backends.
* Loader-backend ABI version: the private protocol used to identify a backend
  and validate its metadata and required private symbols.

Backends target the newest HIP backend API in the installed release. Older HIP
public ABI libraries are compatibility shims owned by the loader; they do not
select older backend ABI versions.

Example:

* `libamdhip64.so.7` exports HIP 7 signatures and structs.
* `libamdhip64.so.8` exports HIP 8 signatures and structs.
* `libamdhip-clr.so` implements HIP backend API 8.
* HIP 7 and HIP 8 loader libraries both load HIP backend API 8.
* HIP 7 loader wrappers translate HIP 7 calls and structures to HIP backend API
  8 before dispatch.

Breaking public signatures or structure layouts require a new HIP public ABI
major. Compatible additions within a major are allowed only when CI verifies that
existing public symbols, symbol versions, structure layouts, and source
contracts remain compatible.

## Historic Public ABI Compatibility

Every supported public HIP major gets a loader library, but all of those loaders
target the newest backend API in the installed release. A HIP 6 public loader in
a release whose backend API is HIP 8 loads a HIP 8 backend, validates
`hipBackendV8GetInterface`, resolves `hipBackendV8*` private symbols, and
adapts only the ABI-breaking HIP 6 call surfaces before dispatch.

The manifest should carry explicit compatibility metadata for each historic
public major. The metadata is a routing and coverage contract, not an attempt to
automatically rewrite arbitrary ABI changes:

```yaml
compatibility:
  - public_major: 6
    target_backend_api: 7
    source_file: loader_v6_compat.cpp
    changed_symbols:
      - canonical_name: hipGetDeviceProperties
        concrete_public_symbols:
          - symbol: hipGetDevicePropertiesR0600
            version_node: hip_6.0
            public_layout: hipDeviceProp_tR0600
            dispatch: direct_or_latest_layout_compat
          - symbol: hipGetDevicePropertiesR0000
            version_node: hip_4.2
            public_layout: hipDeviceProp_tR0000
            compat_function: hip_loader_v6_compat_GetDevicePropertiesR0000
          - symbol: hipGetDeviceProperties
            version_node: hip_4.2
            compatibility_alias_of: hipGetDevicePropertiesR0000
            compat_function: hip_loader_v6_compat_GetDevicePropertiesR0000
        backend_symbol: hipBackendV7GetDevicePropertiesR0600
        reason: output_struct_layout
      - canonical_name: hipChooseDevice
        concrete_public_symbols:
          - symbol: hipChooseDeviceR0600
            version_node: hip_6.0
            public_layout: hipDeviceProp_tR0600
            dispatch: direct_or_latest_layout_compat
          - symbol: hipChooseDeviceR0000
            version_node: hip_4.2
            public_layout: hipDeviceProp_tR0000
            compat_function: hip_loader_v6_compat_ChooseDeviceR0000
          - symbol: hipChooseDevice
            version_node: hip_4.2
            compatibility_alias_of: hipChooseDeviceR0000
            compat_function: hip_loader_v6_compat_ChooseDeviceR0000
        backend_symbol: hipBackendV7ChooseDeviceR0600
        reason: input_struct_layout
      - canonical_name: hipDrvGraphAddMemsetNode
        concrete_public_symbols:
          - symbol: hipDrvGraphAddMemsetNode
            version_node: hip_5.6
            public_layout: HIP_MEMSET_NODE_PARAMS
            compat_function: hip_loader_v6_compat_DrvGraphAddMemsetNode
        backend_symbol: hipBackendV7DrvGraphAddMemsetNode
        reason: input_struct_layout_same_symbol
```

Generated public wrappers use this metadata in two ways:

* Unchanged symbols call the newest backend table directly.
* Changed symbols call the named human-authored compatibility function from the
  historic loader's compat source file.

For example, `libamdhip64.so.6` may generate a normal direct wrapper for
`hipInit`, but route `hipGetDeviceProperties` and `hipChooseDevice` through
`loader_v6_compat.cpp` if their public HIP 6 signatures or structures differ
from the newest backend API. The compatibility source owns old-layout structure
definitions, field-by-field conversion, special default values, error handling,
and `hipGetProcAddress` alias behavior.

`backend_symbol` is a concrete manifest field, not a name inferred only from the
canonical API name. If the newest backend API keeps a historical suffix such as
`R0600` as its concrete ABI spelling during the POC, the manifest should name
that backend symbol explicitly.

Historic compatibility shims must not require old backend entry points. In the
HIP 6 POC, the backend should export the newest concrete API symbols only, such
as `hipBackendV7GetDevicePropertiesR0600` if the newest API still uses the
`R0600` spelling. It should not export `hipBackendV7GetDevicePropertiesR0000` or
`hipBackendV7ChooseDeviceR0000`. If an old public symbol requires old layout
semantics, the loader performs that conversion before or after calling the
newest backend symbol.

Compatibility metadata should classify each change:

* `binary_layout`: old and newest public ABIs use different pointed-to structure
  layouts or by-value structures.
* `source_signature`: source declarations changed but the C ABI is effectively
  the same, such as `void*` to `const void*` or `int*` to `unsigned int*`.
* `behavior`: the signature and layout are unchanged but the public-major
  semantics differ.
* `removed_export`: the old public major exported a symbol that the newest
  public major no longer exports.
* `addition`: a symbol is new in a later public major and should not appear in
  older public loaders.

This keeps the common path generated and cheap while making ABI-breaking changes
explicit. It also matches current CLR precedent: existing `R0000` and `R0600`
device-property symbols are implemented with manual compatibility code, not a
generic struct translation engine.

When a new HIP public major is created, maintainers must update every carried
historic compatibility loader:

* Diff old and new manifests, headers, version scripts, and DEF files.
* Mark every ABI-breaking signature, structure, enum, typedef, symbol, and
  `hipGetProcAddress` lookup change in compatibility metadata.
* Add or update the matching manual `loader_vN_compat.cpp` functions.
* Add test-backend coverage that proves the old public ABI sees old semantics
  while the backend receives newest-API calls.
* Keep direct wrappers for unchanged symbols generated from the manifest.

The initial HIP 6 proof of concept should use current archaeology around
`hipDeviceProp_tR0000`, `hipDeviceProp_tR0600`,
`hipGetDevicePropertiesR0000`, `hipGetDevicePropertiesR0600`,
`hipChooseDeviceR0000`, `hipChooseDeviceR0600`, `HIP_MEMSET_NODE_PARAMS`,
`hipMemsetParams`, `hiprtc*` runtime exports, and the behavior changes listed in
[research.md](research.md). Additional HIP 6 deltas should be added as they are
found in release headers and symbol maps.

## Public Symbol Version Maps

Linux public loader libraries must preserve the exact symbol-version map for
their public SONAME major. These version nodes are historical export groups, not
the same concept as the HIP public ABI major.

The observed ROCm 6 and 7 history demonstrates this:

* ROCm 6.4.4 exports nodes through `hip_6.4`.
* ROCm 7.0.0 adds a `hip_6.5` node, not a `hip_7.0` node.
* The `hip_6.5` node must be treated as historical ABI truth for HIP 7.0-era
  exports, not normalized to the product major.
* The current installed HIP 7 runtime exports `hip_7.1`, `hip_7.2`,
  `hip_profiler_ext`, and `hip_7.14` in addition to older nodes.
* `hipDrvGraphAddMemsetNode@@hip_5.6` keeps the same symbol and version node
  while the pointed-to struct contract changes between HIP 6 and HIP 7.

The manifest therefore needs per-public-major export baselines:

```yaml
public_abi:
  major: 6
  soname: libamdhip64.so.6
  linux_version_nodes:
    - name: hip_4.2
      symbols:
        - hipGetDeviceProperties
        - hipGetDevicePropertiesR0000
        - hipChooseDevice
        - hipChooseDeviceR0000
    - name: hip_5.6
      symbols:
        - hipDrvGraphAddMemsetNode
    - name: hip_6.0
      symbols:
        - hipGetDevicePropertiesR0600
        - hipChooseDeviceR0600
        - hipExtGetLastError
    - name: hip_6.2
      symbols:
        - hipDrvGraphExecMemsetNodeSetParams
    - name: hip_6.4
      symbols:
        - hipEventRecordWithFlags
```

Windows DEF generation has no version nodes and must use its own per-major
export baseline. The split-repo `rocm-6.4.4` DEF file had a malformed
`hipLinkDestroyhipEventRecordWithFlags` entry that was fixed at `rocm-7.0.0`;
the loader generator should model the intended exported symbols and test the
actual DLL export table.

CI should validate actual built artifacts, not just source map files. A source
version script can contain stale names. For example, current Linux source still
lists some `hiprtc*` names, but the installed current `libamdhip64.so.7` does
not export them because hipRTC was removed from `amdhip64`.

## HIP Header ABI Mode

HIP headers should support an explicit ABI mode for declarations that form the C
runtime ABI:

```c
#define HIP_ABI_MODE_PUBLIC 1
#define HIP_ABI_MODE_BACKEND 2

#ifndef HIP_ABI_MODE
#define HIP_ABI_MODE HIP_ABI_MODE_PUBLIC
#endif

#ifndef HIP_API_VERSION
#define HIP_API_VERSION HIP_VERSION_MAJOR
#endif

#define HIP_API_PUBLIC_SYMBOL_I(suffix) hip##suffix
#define HIP_API_PUBLIC_SYMBOL(suffix) HIP_API_PUBLIC_SYMBOL_I(suffix)

#define HIP_API_BACKEND_SYMBOL_I(version, suffix) hipBackendV##version##suffix
#define HIP_API_BACKEND_SYMBOL(version, suffix) HIP_API_BACKEND_SYMBOL_I(version, suffix)

#if HIP_ABI_MODE == HIP_ABI_MODE_PUBLIC
#define HIP_API_SYMBOL(suffix) HIP_API_PUBLIC_SYMBOL(suffix)
#elif HIP_ABI_MODE == HIP_ABI_MODE_BACKEND
#define HIP_API_SYMBOL(suffix) HIP_API_BACKEND_SYMBOL(HIP_API_VERSION, suffix)
#else
#error "Unsupported HIP_ABI_MODE"
#endif
```

Normal public declarations then become:

```c
hipError_t HIP_API_SYMBOL(Init)(unsigned int flags);
hipError_t HIP_API_SYMBOL(DriverGetVersion)(int* driverVersion);
hipError_t HIP_API_SYMBOL(RuntimeGetVersion)(int* runtimeVersion);
```

Public mode emits:

```c
hipError_t hipInit(unsigned int flags);
```

Backend mode with `HIP_API_VERSION=8` emits:

```c
hipError_t hipBackendV8Init(unsigned int flags);
```

This keeps the function declaration readable while making ABI naming explicit.
It also gives source tools a stable marker: `HIP_API_SYMBOL(Init)` identifies the
canonical public API `hipInit` regardless of ABI mode.

The same convention should be used on function definitions:

```c
extern "C" hipError_t HIP_API_SYMBOL(Init)(unsigned int flags) {
  ...
}
```

Do not define global public-name replacement macros such as:

```c
#define hipInit hipBackendV8Init
```

That would affect user source, stringification, documentation examples,
profiler IDs, `hipGetProcAddress`, overload wrappers, and compatibility aliases
in ways that are difficult to audit.

### Nonstandard Public Names

Most runtime APIs can use `HIP_API_SYMBOL(Suffix)`, where public name is
`hip` plus `Suffix`. Other exported names need manifest metadata and, where
needed, a small macro family:

* `__hip*` support entry points.
* Profiler extension entry points.
* Historical aliases such as `hipGetDevicePropertiesR0600`.
* Per-thread default stream variants such as `hipMemcpyAsync_spt`.
* Any C++-mangled compatibility exports that must remain public.

The implementation should not rely on the prefix rule alone. The generated API
manifest records the canonical public name and concrete public/backend symbols.

### Compiler-private ABI Symbols

HIP implementations must export a small set of compiler-private ABI symbols
because HIP-Clang emits calls to them in static constructors, destructors, and
launch lowering paths. The loader owns these exports just as it owns `hip*`
exports.

The current required set includes:

* `__hipRegisterFatBinary`
* `__hipUnregisterFatBinary`
* `__hipRegisterFunction`
* `__hipRegisterVar`
* `__hipRegisterManagedVar`
* `__hipRegisterSurface`
* `__hipRegisterTexture`
* `__hipPushCallConfiguration`
* `__hipPopCallConfiguration`

The manifest should treat these as compiler-private ABI entries. They are not
ordinary user APIs, but they are public dynamic-linker exports of
`libamdhip64.so.N` and the Windows public loader DLL.

The source naming macro for these should be separate from both
`HIP_API_SYMBOL` and ordinary non-compiler private helpers. Proposed spelling:

```c
void** HIP_COMPILER_API_SYMBOL(RegisterFatBinary)(const void* data);
void HIP_COMPILER_API_SYMBOL(UnregisterFatBinary)(void** modules);
```

In public mode, `HIP_COMPILER_API_SYMBOL(RegisterFatBinary)` expands to
`__hipRegisterFatBinary`. The public-mode spelling always has the `__hip`
prefix. In backend mode with `HIP_API_VERSION=8`, it expands to an appropriately
mangled backend compiler ABI symbol such as
`hipBackendV8CompilerRegisterFatBinary`.

Public loader wrappers for compiler-private ABI entries dispatch through the
loader's generated private table:

```c
typedef struct hip_backend_private_api_v8 {
  uint32_t struct_size;
  void** (*RegisterFatBinary)(const void* data);
  void (*UnregisterFatBinary)(void** modules);
  void (*RegisterFunction)(void** modules, const void* hostFunction,
                           char* deviceFunction, const char* deviceName,
                           unsigned int threadLimit, uint3* tid, uint3* bid,
                           dim3* blockDim, dim3* gridDim, int* wSize);
} hip_backend_private_api_v8;
```

The full table is generated from the manifest and should track the current CLR
compiler dispatch table unless the HIP compiler contract changes.

## Backend ABI

The first POC should keep backend changes mechanical. Backends export a minimal
versioned `GetInterface` symbol plus versioned private HIP symbols produced by
`HIP_API_SYMBOL(Suffix)`, `HIP_PRIVATE_SYMBOL(Suffix)`, and
`HIP_COMPILER_API_SYMBOL(Suffix)`. The loader uses
`GetInterface` to confirm that the library is a HIP loader backend and to read
metadata, then builds its own generated function tables by resolving private
symbols from that backend handle.

For HIP backend API 8, all public loader majors in that release resolve symbols
such as:

```text
hipBackendV8GetInterface
hipBackendV8Init
hipBackendV8DriverGetVersion
hipBackendV8RuntimeGetVersion
hipBackendV8CompilerRegisterFatBinary
hipBackendV8CompilerUnregisterFatBinary
```

The `GetInterface` entry point deliberately does not return API dispatch tables
in the first POC:

```c
typedef struct hip_loader_backend_info_v1 {
  uint32_t struct_size;
  uint32_t loader_backend_abi_version;
  uint32_t hip_api_version;
  const char* backend_name;
  const char* backend_version;
} hip_loader_backend_info_v1;

extern "C" const hip_loader_backend_info_v1* hipBackendV8GetInterface(void);
```

The loader-side generated API table uses canonical suffixes, not public or
backend symbol names:

```c
typedef struct hip_loader_backend_api_v8 {
  uint32_t struct_size;
  hipError_t (*Init)(unsigned int flags);
  hipError_t (*DriverGetVersion)(int* driverVersion);
  hipError_t (*RuntimeGetVersion)(int* runtimeVersion);
} hip_loader_backend_api_v8;
```

The loader-side generated private table follows the same pattern:

```c
typedef struct hip_loader_backend_private_api_v8 {
  uint32_t struct_size;
  void** (*RegisterFatBinary)(const void* data);
  void (*UnregisterFatBinary)(void** modules);
} hip_loader_backend_private_api_v8;
```

The exact structure names can change during implementation, but these rules
should not:

* `GetInterface` is a private, versioned backend identity symbol, not a public
  HIP API.
* `GetInterface` returns metadata only in the first POC, not API tables.
* Backend private symbols are versioned and are not public HIP API symbols.
* Wrong HIP backend API major fails because `GetInterface` is missing or returns
  incompatible metadata, or because required private symbols are missing.
* Public ABI major does not determine the backend private symbol version; release
  policy does.
* The loader dispatches through function pointers resolved from the backend
  handle, not public `hip*` or `__hip*` symbol names.
* Compiler-private ABI exports dispatch through the loader private table, not
  backend public `__hip*` symbols.
* Backends are loaded with an explicit path and `RTLD_NOW | RTLD_LOCAL` on ELF.
* Backends are process-lifetime objects and are not `dlclose`d during normal
  execution.

## Backend Symbol Hygiene

Backend-mode libraries must not export public user API symbols or public
compiler-private ABI symbols. This is required even when the backend has a
unique SONAME.

Required backend checks:

* No global exported public `hip*` user API entry points.
* No global exported public compiler-private `__hip*` entry points.
* Export only the minimal backend identity symbol and explicitly allowed private
  backend symbols generated from the manifest, such as
  `hipBackendV8GetInterface`, `hipBackendV8Init`, and
  `hipBackendV8CompilerRegisterFatBinary`.
* No ELF PLT relocations from backend code to public HIP API symbols.
* No ELF PLT relocations from backend code to public compiler-private `__hip*`
  symbols.
* No Windows exports of public HIP API names in backend DLLs.
* Hidden default visibility for implementation symbols, with generated export
  lists allowing only generated backend private symbols.

`RTLD_LOCAL` is not enough protection. If a backend exports `hipMemcpyAsync` and
has an internal PLT relocation to `hipMemcpyAsync`, ELF lookup can bind that
call to an earlier global symbol from the loader or from `LD_PRELOAD`. The
installed current HIP runtime already has self-PLT relocations to public HIP
symbols, so this risk is practical, not theoretical.

## Loader Configuration

The loader should have a small, documented environment contract. Proposed names:

* `HIP_LOADER_BACKEND`
* `HIP_LOADER_BACKEND_PATH`
* `HIP_LOADER_LOG`
* `HIP_LOADER_LOG_FILE`
* `HIP_LOADER_TRACE_API`

`HIP_LOADER_BACKEND` selects a named backend known to the loader or package.
`HIP_LOADER_BACKEND_PATH` selects a user-provided backend library by path and is
intended for development, testing, and explicit product integration.

Environment handling must be security-sensitive:

* Ignore unsafe user override variables in setuid or otherwise elevated Linux
  contexts.
* Ignore unsafe user override variables in elevated Windows contexts.
* Log the selected backend and the reason for rejecting a requested backend when
  diagnostics are enabled.
* Do not require `LD_PRELOAD` for normal backend selection.

Configuration precedence should be fixed before implementation. A reasonable
default is:

1. Explicit safe environment override.
2. Product/package configuration.
3. Built-in default backend.

## Public API Dispatch

The first implementation should generate ordinary C wrappers:

```c
extern "C" hipError_t hipInit(unsigned int flags) {
  return hip_loader_current_backend()->api.Init(flags);
}
```

This is easy to debug and gives a reliable correctness baseline. Hot-path
optimization can later replace selected wrappers with generated tail-call
trampolines or assembly stubs inspired by Vulkan Loader and Implib.so.

The wrapper generator must support:

* Public ABI symbol version maps on Linux.
* Public DEF files on Windows.
* Loader-side backend table type declarations.
* Loader-side backend private table type declarations.
* Generated routing to manual compatibility wrappers for older public ABI
  majors.
* Compiler-private ABI wrappers such as `__hipRegisterFatBinary`.
* Optional logging at the public boundary.
* Validation of backend metadata from `GetInterface`.
* Validation that all required backend private symbols were resolved.

## `hipGetProcAddress` And String APIs

`hipGetProcAddress`, `hipGetDriverEntryPoint`, and related APIs accept public
HIP string names. They must not expose backend symbol names as the user-facing
contract.

The loader handles these APIs by:

* Parsing the public requested name.
* Applying public compatibility rules such as `_spt` selection and historical
  aliases.
* Returning the address of the public loader wrapper when public ABI semantics
  are required.
* Returning a generated backend-table function pointer only when the API
  explicitly requires backend dispatch semantics and the pointer type is safe
  for the requested public ABI version.

The manifest must distinguish:

* Canonical public name.
* ABI category.
* Public exported symbol name by major.
* Backend symbol name by major.
* String lookup aliases.
* Compatibility shim requirements.
* Per-public-major compatibility records and named manual compat functions.

## API Manifest And Generated Outputs

The HIP API manifest should be generated from HIP headers, with a cheap
Clang-based syntax/AST validation step, plus a small annotation file for
semantic data that C declarations do not carry. During migration, CLR trace
typedefs remain a fallback source for exports that are not yet represented cleanly
in public headers.

The manifest should drive:

* Public loader declarations.
* Public loader wrappers.
* Loader-side backend table declarations.
* Backend metadata handshake declaration.
* Backend private symbol declarations.
* Symbol version scripts.
* Windows DEF files.
* Profiler IDs and argument records.
* API logging names and argument metadata.
* Compatibility metadata and generated routing to manual shim source.
* CI diff baselines.

The manifest must preserve canonical public names even when generated source is
built in backend ABI mode.

## Impact On Existing Code

### HIP Headers

Header impact is moderate to large but mostly mechanical. Public C ABI
declarations need to move from plain names to `HIP_API_SYMBOL(Suffix)`.

Inline C++ helpers need review because many call public APIs directly:

```c++
return ::hipGetSymbolAddress(devPtr, (const void*)&symbol);
```

These calls should either use `HIP_API_SYMBOL(GetSymbolAddress)` or a generated
canonical call helper, depending on whether the helper is intended to call the
public loader ABI or the backend ABI in the current compilation mode.

Existing compatibility macros such as
`hipGetDeviceProperties -> hipGetDevicePropertiesR0600` should move toward
manifest metadata. They should not stack unpredictably with `HIP_API_SYMBOL`.

### CLR

CLR impact is large but feasible as a mechanical migration:

* Add one opt-in CMake flag for backend mode, proposed as
  `HIP_ENABLE_LOADER_BACKEND_MODE`.
* Default production CLR builds keep today's behavior: target/output names,
  public exports, installed CMake packages, and dependency surfaces remain
  unchanged when the flag is off.
* Function definitions need to use `HIP_API_SYMBOL(Suffix)`.
* When the flag is on, the shared library target outputs a unique backend name
  such as `libamdhip-clr.so` and compiles with backend ABI naming.
* Public version scripts and DEF files should become generated from the manifest
  for loader builds.
* Backend builds should export only generated backend private symbols.
* Internal symbol lookup by public strings must move to generated table lookup
  or explicit public-string translation.
* Self-calls should prefer internal helpers or direct functions that cannot be
  interposed through public HIP names.

Threading backend mode through exported CMake package files is a separate
packaging step. The POC should avoid changing the default exported package
surface. After the loader, test backend, header macros, manifest generation, and
CLR backend-mode build are staged, TheRock can make an atomic build-system change
that enables the loader as the public runtime and configures CLR with
`HIP_ENABLE_LOADER_BACKEND_MODE=ON`.

### Profiler And Stub Generators

The current profiler generator should not remain regex-based on preprocessed
public names.

Transition plan:

* Run existing profiler generation in public ABI mode.
* Add raw-header parsing for `HIP_API_SYMBOL(Suffix)` so canonical names remain
  stable.
* Generate profiler IDs from the manifest once available.

Long-term:

* Callback IDs are keyed by canonical public API names.
* Log strings are canonical public API names.
* Backend symbols are never used as profiler API identity.
* The generated profiler dispatch wrappers can be table transforms, similar to
  the loader interception model.

## CI And Validation

Required CI checks for the implementation phase:

* Public Linux export list matches the checked-in manifest and version map.
* Public Linux version nodes match the per-public-major baseline from the built
  DSO, including inherited node names such as `hip_5.6` and `hip_6.0`.
* Public Windows DEF exports match the manifest.
* Public loader libraries export required compiler-private ABI symbols.
* Public ABI manifest diff rejects breaking changes within a major.
* Backend libraries do not export public HIP symbols.
* Backend libraries do not export public compiler-private `__hip*` symbols.
* Backend libraries do not contain PLT relocations to public HIP symbols.
* Backend libraries do not contain PLT relocations to public compiler-private
  `__hip*` symbols.
* Wrong-version backend selection fails at `GetInterface` lookup, metadata
  validation, or required private-symbol resolution.
* `hipGetProcAddress` returns public ABI-compatible pointers for public names.
* Every ABI-breaking delta for a carried historic public major is recorded in
  compatibility metadata and routed to a manual compat function.
* Public-major compatibility loaders export the expected old symbols and call
  newest backend symbols after manual adaptation.
* Historic compatibility loaders do not resolve old backend symbols such as
  backend `R0000` device-property variants.
* Source map files are checked against actual dynamic exports so stale version
  script entries do not become accidental loader exports.
* Profiler IDs and names are stable across public and backend ABI modes.
* Generated files are checked with a `--verify` mode.

The first implementation should use the deterministic no-GPU backend specified
in [testing.md](testing.md). It validates loader mechanics before CLR or any
other real HIP backend is converted. Its `__testBackend*` control symbols are
test-only inspection hooks and are not part of the production backend ABI.

Recommended test scenarios:

* Application linked against `libamdhip64.so.7` loads the default backend.
* Application linked against `libamdhip64.so.7` selects a backend by path.
* HIP 6 and HIP 7 public shims dispatch to the newest backend API carried by the
  release, using manual compatibility functions only for ABI-breaking calls.
* HIP 6 legacy device-property symbols dispatch to the newest backend
  device-property symbol and never to a backend `R0000` symbol.
* HIP 6 graph memset driver symbols convert `HIP_MEMSET_NODE_PARAMS` to the
  newest backend `hipMemsetParams` contract while preserving the old public
  Linux version nodes.
* A HIP-compiled or simulated user shared library can load and run static
  constructors that call `__hipRegisterFatBinary` through the loader.
* `LD_PRELOAD` interposes the public loader boundary without intercepting
  backend internals.
* Preloading an old public HIP runtime is diagnosed as unsupported or produces a
  controlled failure.
* Backend with only `hipBackendV7GetInterface` or `hipBackendV7*` private symbols
  is rejected by any loader from a release that requires `hipBackendV8GetInterface`
  and `hipBackendV8*` private symbols.
* Backend with accidental `hipInit` export fails the symbol-hygiene check.

## Alternatives Considered

### Keep Backends As `libamdhip64.so.N`

This preserves the current runtime shape but prevents multiple implementations
from coexisting cleanly. It also keeps public symbol collisions between loader
and backend implementations. This is rejected.

### Unique Backend SONAME Only

Renaming CLR to `libamdhip-clr.so` avoids package and SONAME conflicts, but it
does not solve ELF interposition if the backend still exports public `hip*` or
compiler-private `__hip*` symbols. This is necessary but insufficient.

### Loader `dlsym` Of Public Backend `hip*` Symbols

This matches the simplest passthrough approach, but it requires backends to
export the same public names as the loader. It is vulnerable to `LD_PRELOAD` and
global-scope interposition. The same problem applies to compiler-private
`__hip*` ABI symbols. It also makes wrong-major binding too easy. This is
rejected for the primary design.

### Table-returning `GetInterface`

A backend `GetInterface` that returns complete API tables is a cleaner runtime
contract and may be the right long-term backend ABI. It is too much API
invention for the first POC because current HIP implementations already export
functions and do not appear to maintain explicit complete API tables. The first
POC should keep backend surgery mechanical: add a small metadata-only
`GetInterface`, rename exported functions with `HIP_API_SYMBOL` and
`HIP_COMPILER_API_SYMBOL` or `HIP_PRIVATE_SYMBOL` as appropriate, and let the
loader build generated dispatch tables with explicit symbol resolution.

### `LD_PRELOAD` As The Loader Strategy

`LD_PRELOAD` can be useful for diagnostics, but it is Linux-specific, fragile,
security-sensitive, and difficult to support in products. It is rejected as the
normal backend selection mechanism.

### `RTLD_DEEPBIND`

`RTLD_DEEPBIND` can reduce some ELF self-interposition cases on glibc, but it is
not portable and can conflict with sanitizers, debuggers, profilers, and
intentional interposition. It is rejected as a default.

### `dlmopen` Isolation

`dlmopen` can create a separate glibc link-map namespace, but HIP depends on
process-global runtime libraries and profiler registration. Duplicating those
dependencies is risky, Linux-specific, and not viable as a cross-platform base
design. It is rejected as a default.

### Per-function `#if HIP_ABI_MODE` Blocks

This is explicit but unmaintainable across hundreds of APIs. It also creates
more chances for declarations and definitions to diverge. This is rejected.

### Full Declaration Macros

A macro such as `HIP_API_DECL(hipError_t, Init, (unsigned int flags))` is easy
for generators but makes the headers less like normal C declarations. It also
becomes awkward for comments, attributes, conditional declarations, and function
pointer typedefs. This is rejected in favor of `HIP_API_SYMBOL(Suffix)`.

### Global Public-name Replacement

Global macros such as `#define hipInit hipBackendV8Init` minimize source edits,
but they rewrite ordinary user-visible identifiers and stringification contexts.
They are especially risky for `hipGetProcAddress`, profiler IDs, C++ inline
wrappers, and existing compatibility aliases. This is rejected.

### Export-only Renaming

ELF asm labels, `#pragma redefine_extname`, Windows DEF aliases, Mach-O linker
aliases, and AIX export files can all influence external symbol names. They are
useful for generated platform glue, but they are not a uniform source-level HIP
header contract. This is rejected as the primary strategy.

## Open Questions

* Whether Windows keeps a current-major compatibility DLL name during transition.
* Exact environment variable names and precedence.
* Exact backend handshake structure layout.
* Whether backend API version should be named after the newest HIP source/API
  major or use an independent monotonically increasing loader-backend number
  after the first implementation.
* How much of the current profiler generator should be adapted before replacing
  it with manifest-driven generation.
