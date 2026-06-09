# HIP Loader Scope

This directory captures the initial design scope for a HIP loader layer. The
loader is intended to become the public HIP runtime ABI boundary while allowing
multiple concrete HIP implementations to coexist during migration.

The first deliverable is design documentation. The implementation can start as a
proof of concept, but the design should be review-grade and should avoid known
dead ends such as relying on install-tree surgery or `LD_PRELOAD` for normal
routing.

## Goals

The loader should provide these capabilities:

* Export the public HIP runtime ABI on Linux and Windows.
* Select a concrete HIP backend at process startup through loader policy and
  documented configuration.
* Allow user-provided HIP implementations without replacing files in an install
  tree.
* Allow multiple in-tree backend prototypes to exist at the same time.
* Preserve semantic-version compatibility within a HIP major version.
* Provide compatibility libraries for older HIP major versions when the newest
  backend API has breaking signature or structure changes.
* Support historic public ABI loaders, including a HIP 6 proof of concept, that
  route into the newest backend API carried by the release.
* Preserve exact public export baselines for each public major, including Linux
  symbol-version nodes and Windows DEF exports.
* Provide a path to generated wrappers, dispatch tables, export lists, ABI
  manifests, and compatibility shims.
* Provide diagnostic logging hooks at the loader boundary.
* Avoid dynamic-linker symbol collisions between the loader, preloaded
  interposers, and backend implementations.
* Provide a deterministic no-GPU test backend for loader bring-up and CI.
* Preserve compiler-emitted HIP private ABI entry points such as fat binary
  registration callbacks.

The default model is one selected HIP backend per process. Routing different
devices or calls to different HIP backends in one process is outside this first
scope.

## Public ABI Scope

On Linux, the loader owns the public HIP SONAME:

* `libamdhip64.so.7` exports the HIP 7 public ABI.
* `libamdhip64.so.8` exports the HIP 8 public ABI when HIP 8 exists.
* The unversioned `libamdhip64.so` development symlink points at the default
  public major selected by the package.

On Windows, the loader should move to a versioned namespace, for example:

* `amdhip-7.dll`
* `amdhip-8.dll`

Keeping an `amdhip64.dll` or `amdhip64_<major>.dll` transition name for the
current major can be considered for compatibility, but the design should not
assume the old Windows naming scheme remains the long-term public ABI boundary.

Backend implementations are not public HIP ABI libraries. They use unique names
such as `libamdhip-clr.so`, `libamdhip-example_backend.so`,
`amdhip-clr.dll`, and `amdhip-example_backend.dll`.

The public loader ABI is broader than documented user-callable `hip*` APIs. HIP
compiled user objects and shared libraries also reference compiler-private entry
points such as `__hipRegisterFatBinary`, `__hipRegisterFunction`,
`__hipRegisterVar`, and `__hipUnregisterFatBinary` from static constructors and
destructors. The loader must export these symbols with their public ABI names so
existing HIP-compiled code links and loads correctly.

## HIP Header ABI Mode Scope

The design includes changes to HIP headers so the same implementation source can
be built for either the public ABI or a private backend ABI.

The proposed public contract is:

```c
#define HIP_ABI_MODE_PUBLIC 1
#define HIP_ABI_MODE_BACKEND 2

#ifndef HIP_ABI_MODE
#define HIP_ABI_MODE HIP_ABI_MODE_PUBLIC
#endif

#ifndef HIP_API_VERSION
#define HIP_API_VERSION HIP_VERSION_MAJOR
#endif

hipError_t HIP_API_SYMBOL(Init)(unsigned int flags);
```

In public mode, `HIP_API_SYMBOL(Init)` expands to `hipInit`. In backend mode, it
expands to a versioned backend symbol such as `hipBackendV8Init`.

This scope deliberately avoids global replacement macros such as
`#define hipInit hipBackendV8Init`. Public HIP identifiers should remain usable
as ordinary source identifiers, string names, profiler names, and documentation
names. Only declarations and definitions that form the exported ABI should use
`HIP_API_SYMBOL(Suffix)`.

The exact helper macro spelling is an implementation detail, but it should be
based on ordinary token pasting and numeric preprocessor constants, not string
comparisons or platform-specific linker aliases.

## Backend ABI Scope

Backends should expose a minimal versioned `GetInterface` identity symbol plus
versioned private HIP API symbols for the newest supported HIP API in the
installed release. The loader uses `GetInterface` for backend identity and
metadata, then builds its own generated function tables by resolving private
symbols from the backend handle.

Older public HIP ABI libraries do not require older backend implementations.
They contain compatibility code that translates older public signatures and
structures to the latest backend API. For example, if a release's newest backend
API is HIP 8, both `libamdhip64.so.7` and `libamdhip64.so.8` loader libraries
may load a backend exporting `hipBackendV8GetInterface` plus private symbols
such as `hipBackendV8Init` and `hipBackendV8CompilerRegisterFatBinary`.

The loader should fail early only when the selected backend does not implement
the backend API required by that loader release. A backend that exports only
`hipBackendV7GetInterface` or `hipBackendV7*` private symbols should be rejected
by a release whose loaders require `hipBackendV8GetInterface` and
`hipBackendV8*` private symbols, regardless of whether the application entered
through the HIP 7 or HIP 8 public ABI.

Backend-mode shared libraries must not export official public `hip*` symbols.
This is a correctness requirement, not just a cleanliness preference. It avoids
ELF interposition hazards, `LD_PRELOAD` collisions, and accidental binding of a
backend's internal calls to the loader's public ABI entry points.

## Metadata And Tooling Scope

The HIP headers remain the source of truth, but the implementation should move
toward a generated API manifest. The manifest should record:

* Canonical public API name, such as `hipInit`.
* Declaration suffix, such as `Init`.
* ABI category, such as user API, compiler-private ABI, profiler extension, or
  compatibility alias.
* Public symbols by HIP major.
* Linux version-node membership by HIP major and symbol.
* Windows DEF membership by HIP major.
* Backend symbols by HIP major.
* Return type, parameters, calling convention, and attributes.
* Struct, union, enum, and typedef layout information.
* API introduction, deprecation, removal, and compatibility metadata.
* Per-public-major compatibility records for ABI-breaking signature, struct, and
  symbol changes that require manual loader shims.
* Special string lookup names used by `hipGetProcAddress` and related APIs.
* Historical aliases such as `hipGetDevicePropertiesR0600`.
* Per-thread default stream variants such as `_spt` entry points.

Generated outputs should include loader wrappers, loader-side backend tables,
export maps, Windows DEF files, compatibility shims, profiler IDs, and CI
comparison data.

## Constraints

The design must account for these constraints:

* Existing HIP applications link to `libamdhip64.so.N` on Linux.
* Current CLR builds the shared library target `amdhip64` and uses Linux version
  scripts and Windows DEF files to export public HIP symbols.
* CLR backend mode must be opt-in behind one CMake flag. The default CLR build
  must remain production-compatible until the loader and backend-mode changes are
  staged and activated together.
* Current HIP headers mostly expose plain C declarations under default
  visibility, rather than a per-function export-name macro.
* Current profiler generation parses HIP headers and implementation sources by
  public API names.
* `hipGetProcAddress` and `hipGetDriverEntryPoint` are public string-name APIs;
  their input names must remain public HIP names regardless of backend ABI mode.
* Dynamic loading must work on Linux and Windows, and design choices should not
  preclude MacOS or AIX where ordinary C symbol names and generated export lists
  are available.
* Normal backend selection must not require `LD_PRELOAD`.
* Environment variables must be handled with security-sensitive behavior modeled
  after established loader practice. Setuid/elevated contexts should not honor
  unsafe user overrides.

## Initial Documentation Boundary

This first docs-only change does not implement the loader, but the follow-on POC
is expected to include:

* Loader code.
* A backend-mode build path for CLR controlled by one CMake flag.
* HIP header changes for public and backend ABI naming.
* API manifest generation.
* Compatibility metadata and manual historic-loader shims, starting with a HIP 6
  proof of concept where the necessary API archaeology is tractable.
* Concrete `HIP_API_VERSION=N` source compatibility syntax for the headers
  touched by the POC.
* An eventual atomic build-system switch that enables the loader and configures
  CLR for backend mode after all pieces are staged.

## Non-goals

The project does not currently require:

* Supporting multiple active HIP backends in one process.
* Preserving `LD_PRELOAD` as a supported backend selection mechanism.
* Supporting an ABI-incompatible backend through public `hip*` symbol lookup.
* Using the test backend as a full HIP emulator.

## Acceptance Criteria

The design documentation is complete enough when it:

* Defines the public loader ABI and private backend ABI boundaries.
* Specifies the `HIP_API_SYMBOL(Suffix)` header direction.
* Explains why backend-mode libraries must not export public HIP entry points or
  public compiler-private `__hip*` entry points.
* Explains why Linux version-node names must be preserved exactly and validated
  against actual built DSOs.
* Identifies impacts on CLR, profiler generation, export maps, DEF files, and
  CI.
* Describes the dynamic-linker and `LD_PRELOAD` risks.
* Specifies the test backend and no-GPU CI strategy in
  [testing.md](testing.md).
* Lists alternatives considered and why they are not the preferred path.
