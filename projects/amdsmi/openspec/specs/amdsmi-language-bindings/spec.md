# amdsmi-language-bindings Specification

## Purpose

Defines what the two non-Python language bindings maintained in this tree
actually guarantee: the Go shim, made of `goamdsmi.go` and `goamdsmi_shim/`,
and the Rust crate in `rust-interface/`. Both are named as part of what AMD SMI
ships, both sit on `include/amd_smi/amdsmi.h`, and neither has had its contract
written down — so a caller has no way to know which subset of the C surface is
reachable, how failure arrives, how the binding tracks the header, or which
delivery channel puts it on disk.

They are one capability because the questions are the same and the answers are
opposite, and the contrast is the useful part. The Rust crate is a mechanically
generated FFI layer under a hand-written safe layer that translates every
`amdsmi_status_t` into `Result`, covers 116 of the 178 functions its bindings
declare, and reaches no user through any delivery channel. The Go binding is
not a binding to the header at all: it is a hand-written C shim exporting a
fixed set of about forty metric getters, a third of which return a constant
without calling the library, wrapped one-to-one by cgo declarations, shipped
only by the channels that install every CMake component.

Both are small and both are partly stale. This specification says so rather
than implying a guarantee neither offers: the obligations below are mostly
about what a consumer must not assume.

Deliberately excluded: the header, the status codes, the sentinel convention
and the initialization lifecycle these bindings translate, which are
[amdsmi-c-api-abi]; the Python binding, whose far larger contract is
[amdsmi-python-api] and whose generated-wrapper rule this capability mirrors
for Rust; what the library enumerates and how discovery degrades, which is
[amdsmi-device-discovery]; the options that gate these subdirectories, which
are inventoried in [amdsmi-build-configuration]; and the install tree and the
channels themselves, which are [amdsmi-install-layout],
[amdsmi-rocm-os-packages], [amdsmi-therock-subproject] and
[amdsmi-therock-artifact].

## Requirements

### Requirement: The Go Binding Is A Fixed-Function Shim, Not A Binding To The Header

The Go surface SHALL be the hand-written C entry points declared in
`goamdsmi_shim/smiwrapper/amdsmi_go_shim.h`, wrapped one-to-one by cgo
declarations in `goamdsmi.go`. Nothing about it is generated from
`amdsmi.h`, and adding an AMD SMI API does not add a Go function: both the C
shim and the Go declaration must be written by hand. A consumer SHALL treat the
shim's entry-point list, not the header, as the definition of what Go can
reach.

The shim SHALL be understood as covering a fixed metric set — device count,
device id, power, temperature, SCLK and MCLK, GPU and memory utilization,
memory usage and total, UMA carveout, the TTM pages limit, and a set of
CPU/HSMP counters — and not as a general interface to the library.

A caller SHALL NOT expect the following entry points to consult the library at
all. Each returns a compiled-in constant regardless of the device, while the
doc comments in `amdsmi_go_shim.h` still describe the `rsmi_*` call each once
made:

| Entry point | Always returns |
| ----------- | -------------- |
| `goamdsmi_gpu_shutdown` | `false` |
| `goamdsmi_gpu_dev_name_get`, `goamdsmi_gpu_dev_vendor_name_get`, `goamdsmi_gpu_dev_vbios_version_get` | a freshly allocated `"NA"` |
| `goamdsmi_gpu_dev_pci_id_get` | `0xFFFFFFFFFFFFFFFF` |
| `goamdsmi_gpu_dev_perf_level_get`, `goamdsmi_gpu_dev_overdrive_level_get`, `goamdsmi_gpu_dev_mem_overdrive_level_get` | `0xFFFFFFFF` |
| `goamdsmi_gpu_od_volt_freq_range_min_get_sclk`, `..._min_get_mclk`, `..._max_get_sclk`, `..._max_get_mclk` | `0xFFFFFFFFFFFFFFFF` |

#### Scenario: A device inventory built through Go has no device names

- **WHEN** a Go exporter enumerates GPUs and calls `GO_gpu_dev_name_get`,
  `GO_gpu_dev_vendor_name_get` and `GO_gpu_dev_vbios_version_get` to label its
  metrics
- **THEN** every device is labelled `"NA"` on every machine, because the shim
  allocates the literal and returns it without a library call — the labels are
  not "unsupported on this ASIC", they are unimplemented for all ASICs, and no
  amount of driver or firmware change alters them

#### Scenario: The string getters leak unless the caller frees them

- **WHEN** a Go caller uses one of the three string getters in a loop
- **THEN** it receives the raw `*C.char` the shim obtained from `malloc`;
  `goamdsmi.go` neither converts it to a Go `string` nor frees it, and
  `amdsmi_go_shim.h` states that the caller must free the buffer, so a caller
  that treats the result as garbage-collected leaks 256 bytes per call

#### Scenario: Two Go wrappers call the wrong shim function

- **WHEN** a caller invokes `GO_gpu_dev_overdrive_level_get` or
  `GO_gpu_dev_mem_overdrive_level_get`
- **THEN** the first reaches `goamdsmi_gpu_dev_perf_level_get` and the second
  reaches `goamdsmi_gpu_dev_overdrive_level_get`, one step off in each case;
  the crossed wiring is invisible today only because all three targets return
  the same constant, so it would become a live defect the moment any one of
  them is implemented

### Requirement: The Go Boundary Reports Failure Only As A Sentinel Value

No Go entry point SHALL return a status. Failure and success SHALL be
distinguished only by the returned value, using the constants in
`goamdsmi_shim/smiwrapper/goamdsmi.h`:

| Return kind | Failure value |
| ----------- | ------------- |
| `bool` initializer | `false` |
| `uint16_t` getter | `0xFFFF` |
| `uint32_t` getter | `0xFFFFFFFF` |
| `uint64_t` getter | `0xFFFFFFFFFFFFFFFF` |
| `char*` getter | `"NA"` |
| `int32_t` UAPI setter or getter | `-1` |

These are the same all-ones values [amdsmi-c-api-abi] defines for a field the
platform cannot report, so a caller SHALL NOT attempt to distinguish "the call
failed", "the ASIC does not support this" and "the driver reported no value" —
at this boundary they are one outcome. Diagnostic detail exists only as
`printf` output compiled in by `-DENABLE_DEBUG_LEVEL=<n>` at CMake time, which
is off by default and is not machine-readable.

The shim SHALL rescale three families for compatibility with the values the
older ROCm SMI Go interface returned, and a caller SHALL NOT assume the C
library's units: GPU power is multiplied by 1000000 to microwatts, GPU
temperature by 1000 to millidegrees, and CPU socket power and power cap by 1000
to milliwatts.

#### Scenario: An unreported power reading arrives as four billion watts

- **WHEN** `amdsmi_get_power_info` succeeds but reports no average socket
  power, which for that `uint32_t` field means `0xFFFFFFFF`
- **THEN** the shim's fallback to `current_socket_power` does not fire, because
  it compares against `0xFFFF` — the 16-bit sentinel, not the one the field's
  declared width licenses — and the value is then multiplied by 1000000 and
  returned as `4294967295000000`, which is neither the shim's own failure
  sentinel nor recognizable as a sentinel at all; the caller records roughly
  four billion watts as a measurement

#### Scenario: A CPU getter after a failed initialization terminates the process

- **WHEN** `GO_cpu_init` returns `false` and the caller proceeds anyway to
  `GO_cpu_core_energy_get` or `GO_cpu_core_boostlimit_get`
- **THEN** the shim computes `thread_index % num_cpu_physicalCore_inAllSocket`
  with a zero divisor and the process dies on the arithmetic exception, so
  checking the initializer's boolean is not defensive style but a hard
  precondition of the CPU surface

#### Scenario: A build without ESMI reports zero threads and succeeds anyway

- **WHEN** the library is configured with `ENABLE_ESMI_LIB` off and a Go caller
  runs the CPU surface
- **THEN** `GO_cpu_init` can still succeed, because it validates only the
  processor counts the enumeration produced, while
  `goamdsmi_cpu_core_energy_get`,
  `goamdsmi_cpu_socket_energy_get`, `goamdsmi_cpu_prochot_status_get`,
  `goamdsmi_cpu_core_boostlimit_get` and `goamdsmi_cpu_threads_per_core_get`
  are compiled out to their sentinels and `GO_cpu_number_of_threads_get`
  reports `0` — socket power and power cap keep working, because those two are
  the only CPU getters the shim leaves outside the conditional

### Requirement: Go Shim State Is Process-Global, Fixed-Capacity, And Never Torn Down

The shim SHALL hold all handles in file-scope arrays with compiled-in maxima —
4 sockets, 24 GPU devices, 4 CPUs per socket, 384 physical cores — and SHALL
index every getter by position in those arrays rather than by a handle the
caller holds. There is therefore one library session per process, shared by
every goroutine, with no mutex of its own; the per-device serialization
described in [amdsmi-c-api-abi] is all the concurrency protection there is.

Initialization SHALL probe for drivers by file existence —
`/sys/module/amdgpu/initstate` for the GPU path and `/dev/hsmp` for the CPU
path — rather than by asking the library. When both exist, a single
`amdsmi_init` with the GPU and CPU flags combined SHALL enumerate both classes
at once; otherwise the requested class SHALL be initialized alone. The outcome
SHALL be latched: once the CPU or GPU path has run, later calls return a
verdict recomputed from the stored counts and never re-probe.

The shim SHALL never call `amdsmi_shut_down`. A process that uses it holds the
library initialized until it exits.

#### Scenario: A host beyond the compiled-in maxima writes past the arrays

- **WHEN** the shim initializes on a system whose socket count exceeds the
  array it is about to fill
- **THEN** it passes the true count the library reported back as the fill
  capacity, without clamping it to the array size, so the library writes past
  the static buffer; the maxima are assumptions about the fleet, not enforced
  limits, and a denser host corrupts adjacent state rather than reporting an
  error

#### Scenario: Four getters accept a device index the others reject

- **WHEN** a caller passes a device index at or beyond the discovered device
  count
- **THEN** `goamdsmi_gpu_dev_id_get`, the power, temperature and clock getters
  and the UMA and TTM entry points return their sentinel after a bounds check,
  while `goamdsmi_gpu_dev_gpu_busy_percent_get`,
  `goamdsmi_gpu_dev_gpu_memory_busy_percent_get`,
  `goamdsmi_gpu_dev_gpu_memory_usage_get` and
  `goamdsmi_gpu_dev_gpu_memory_total_get` index the static array unchecked —
  inside the array they pass a zeroed handle the library rejects, past it they
  read out of bounds

#### Scenario: A driver loaded after the first call is never noticed

- **WHEN** a long-running Go daemon calls `GO_gpu_init` before the `amdgpu`
  driver is up, and calls it again after the driver loads
- **THEN** the second call returns the latched failure without re-probing
  `/sys/module/amdgpu/initstate` or re-entering the library, so recovery
  requires restarting the process

### Requirement: The Rust Crate Exposes Only Its Hand-Written Safe Layer

The crate SHALL be built from three files and expose exactly one of the two
layers:

| Layer | Files | Visibility |
| ----- | ----- | ---------- |
| Generated FFI | `src/amdsmi_wrapper.rs` | `mod amdsmi_wrapper;` — private, no glob re-export |
| Safe wrappers and type re-exports | `src/amdsmi.rs`, `src/utils.rs` | `pub use` from `src/lib.rs` |

The public API SHALL therefore be the 116 functions in `src/amdsmi.rs` plus the
enumerated type, struct, union and constant re-exports in `src/utils.rs`. None
of the crate's public functions is `unsafe`; all `unsafe` is confined to the
`call_unsafe!`, `define_cstr!`, `cstr_to_string!` and `impl_cstr_getters!`
macros. Types SHALL be renamed from the C spelling to upper camel case by the
`bindgen` parse callback in `callbacks.rs`, which also renames every enum
variant, so `amdsmi_status_t` is `AmdsmiStatusT` and
`AMDSMI_INIT_AMD_GPUS` is `AmdsmiInitFlagsT::AmdsmiInitAmdGpus`.

The CPU and HSMP surface SHALL be absent from the crate entirely. This is
structural rather than an omission in the safe layer: `build.rs` invokes
`bindgen` with no `ENABLE_ESMI_LIB` definition, so the three conditional blocks
in `amdsmi.h` never reach the FFI module. The 178 functions the generated
module declares are exactly the header's unconditional functions, and
regenerating the bindings as the tooling stands today will not add the other 68.

#### Scenario: An unwrapped API has no escape hatch

- **WHEN** a Rust caller needs one of the 62 functions that the generated
  module declares but `src/amdsmi.rs` does not wrap
- **THEN** it cannot reach it: the FFI module is private and `src/utils.rs`
  re-exports types only, so the caller must add a safe wrapper to the crate
  rather than dropping into `unsafe` at the call site — which is the reason the
  crate can promise a fully safe surface

#### Scenario: A CPU query does not compile rather than failing at runtime

- **WHEN** a Rust program tries to read CPU socket energy or the HSMP metrics
  table on an x86_64 host whose `libamd_smi.so` exports them
- **THEN** the names do not exist in the crate at any layer, so the failure is
  a compile error naming an unknown function, not the
  `AMDSMI_STATUS_NOT_SUPPORTED` a C or Python caller would see — the
  compile-time conditional described in [amdsmi-c-api-abi] is resolved once, at
  binding-generation time, in the off position

#### Scenario: The convenience macro panics where the functions return errors

- **WHEN** a caller uses the crate's one exported macro,
  `amdsmi_get_processor_handles!()`, to flatten every socket's processors into
  one vector
- **THEN** it aborts the thread on any error, because the macro expands to
  `expect` on both underlying calls; it exists for the doc tests and is not
  interchangeable with the fallible functions it wraps

### Requirement: The Rust Error Model Is The C Status Enumeration Itself

Every fallible function SHALL return `AmdsmiResult<T>`, defined as
`Result<T, AmdsmiStatusT>`. The crate SHALL define no error type of its own:
the `Err` variant is the C status enumerator, whose meanings are
[amdsmi-c-api-abi]'s. `call_unsafe!` SHALL return `Err(status)` for anything
other than `AmdsmiStatusT::AmdsmiStatusSuccess`, so no status code is ever
visible as a return value.

Human-readable text SHALL come from the library. `Display for AmdsmiStatusT`
calls `amdsmi_status_code_to_string` and falls back to
`"An unknown error occurred"` only when that call itself fails, so a status the
crate predates still prints whatever the loaded library calls it.

Strings crossing the boundary SHALL be owned `String` values decoded with
`to_string_lossy`, and fixed-size C character arrays inside returned structures
SHALL be reachable through generated accessor methods rather than as raw byte
arrays. `AmdsmiBdfT` SHALL additionally render as `domain:bus:device.function`
through `Display`.

#### Scenario: Unsupported is an error, not a value

- **WHEN** a Rust caller queries a field the ASIC does not implement and the
  library returns `AMDSMI_STATUS_NOT_SUPPORTED`
- **THEN** the call yields `Err(AmdsmiStatusT::AmdsmiStatusNotSupported)`, so
  portable code must match on the status rather than test the returned value —
  the opposite convention from the Go shim above, where the same condition is
  an in-band sentinel

#### Scenario: The error does not compose with other error types

- **WHEN** a caller wants to use `?` to lift an AMD SMI failure into an
  application error enum or a `Box<dyn Error>`
- **THEN** it must write the conversion itself, because `AmdsmiStatusT`
  implements `Display` but not `std::error::Error`; the crate's error type is a
  plain FFI enum and carries no source, no backtrace and no context about which
  call produced it

#### Scenario: A successful call can still carry unreported fields

- **WHEN** a returned structure such as `AmdsmiGpuMetricsT` contains members the
  platform did not fill
- **THEN** they hold the all-ones sentinel defined in [amdsmi-c-api-abi]
  verbatim; the crate performs no substitution comparable to the `"N/A"`
  convention in [amdsmi-python-api], so every consumer of a metrics structure
  must compare against the maximum of each field's own type itself

### Requirement: The Rust FFI Module Is Generated And Never Hand-Edited

`rust-interface/src/amdsmi_wrapper.rs` SHALL be produced by `bindgen` from
`include/amd_smi/amdsmi.h` through `build.rs`, and SHALL NOT be edited by hand,
for the same reason the Python wrapper carries that rule in
[amdsmi-python-api]: a hand edit disappears on the next regeneration and until
then the Rust-side struct layout disagrees with the header the `.so` was
compiled from, which corrupts data rather than failing loudly. It is committed
to the tree so that a consumer needs no libclang.

Regeneration SHALL be opt-in and SHALL NOT happen during an ordinary build:
`build.rs` regenerates only when the `AMDSMI_GENERATE_RUST_WRAPPER` environment
variable is set, which CMake sets only under `REGENERATE_RUST_WRAPPER=ON`.
Regeneration SHALL write into the source tree, not the build tree.

`tools/update_rust_wrapper.sh` SHALL be the reproducible path, running the
build and `bindgen` inside a container image tagged by its Dockerfile hash so
that two developers produce the same file — the same mechanism
`tools/update_wrapper.sh` uses for the Python wrapper. Reproducibility SHALL be
understood as bounded by the crate's unpinned dependencies: `.gitignore`
excludes `*.lock`, so no `Cargo.lock` is committed and `bindgen = "0.70.1"`
resolves to whatever compatible release exists at generation time.

Which functions the header exposes is decided by `bindgen`'s allowlists — types
matching `amdsmi*`, functions matching `amdsmi*`, variables matching `AMDSMI*`
— so a new API reaches the FFI module by name alone, while a differently
prefixed one does not.

#### Scenario: A header change lands without regeneration and nothing notices

- **WHEN** a struct gains a field, or a function's signature changes, and the
  contributor does not run `tools/update_rust_wrapper.sh`
- **THEN** the committed bindings keep the old layout and the build succeeds,
  because regeneration is opt-in and no build or check compares the two — the
  disagreement surfaces only when someone links the crate against the new
  library and reads a field at the wrong offset

#### Scenario: Auto-fixing hooks may rewrite the generated file

- **WHEN** pre-commit runs on a change that touches the regenerated bindings
- **THEN** `codespell -w`, `trailing-whitespace` and `end-of-file-fixer` apply
  to it, because only `py-interface/amdsmi_wrapper.py` is excluded from those
  hooks; a hook that edits a generated file produces exactly the hand-edited
  state the never-edit rule exists to prevent

### Requirement: Version Coupling Is A Three-Constant Rewrite, Not A Regeneration

The only automatic coupling between either binding and the library version
SHALL be a configure-time rewrite of three constants in the Rust bindings.
Configure SHALL compare the `AMDSMI_LIB_VERSION_MAJOR`, `_MINOR` and `_RELEASE`
values found in `rust-interface/src/amdsmi_wrapper.rs` against the version
[amdsmi-c-api-abi] derives from the header, and SHALL overwrite the file's
three constants when the header's version is strictly greater. The rewrite
SHALL be:

| Property | Consequence |
| -------- | ----------- |
| Unconditional on `BUILD_RUST_WRAPPER` | every configure of AMD SMI touches the file, whether or not Rust is being built |
| Written to the source tree | a plain `cmake -B build` modifies a tracked file and leaves the working tree dirty |
| One-directional | a mirror ahead of the header is left alone |
| Limited to three integer constants | no struct, enum, signature or allowlisted symbol is touched |

The Go binding SHALL be understood as having no version coupling at all: it
declares no version constant, and the SOVERSION of `libgoamdsmi_shim64.so` is
authored independently in `goamdsmi_shim/CMakeLists.txt` rather than derived
from the header, so the shim's own version number carries no information about
which AMD SMI it was built against.

#### Scenario: The repaired constants are unreachable by any consumer

- **WHEN** a Rust program wants to compare the version its bindings were
  generated from against the library it loaded
- **THEN** it cannot: `src/utils.rs` re-exports an explicit list of constants
  that does not include the three version macros, and the FFI module is
  private, so the only version a crate consumer can obtain is the runtime one
  from `amdsmi_get_lib_version()` — the mirror configure maintains is consumed
  by nothing

#### Scenario: Repaired constants can outrun the bindings that carry them

- **WHEN** the header moves ahead of the checked-in bindings, as it stands
  today with the header at minor version 1 and the bindings still declaring 0
- **THEN** the next configure rewrites the constants to match the header while
  leaving every struct layout and signature at the older shape, so the file
  claims a currency its contents do not have; the rewrite is a version-number
  repair and is not a substitute for regeneration

#### Scenario: A version-only configure dirties an unrelated worktree

- **WHEN** a developer configures the project to build the C library alone
- **THEN** `git status` reports `rust-interface/src/amdsmi_wrapper.rs` as
  modified if the tree's constants were stale, because the rewrite runs at the
  top of the top-level `CMakeLists.txt` before any option is consulted

### Requirement: The Go Shim Is Always Built And The Rust Crate Never Is

Build participation SHALL differ between the two, and neither SHALL be assumed
present because the other is. The options named here are inventoried in
[amdsmi-build-configuration]; what this capability fixes is what their defaults
mean for a consumer of each binding:

| Binding | Build | Installed artifacts |
| ------- | ----- | ------------------- |
| Go shim | unconditional; `goamdsmi_shim` is added as a subdirectory with no guarding option | `lib/libgoamdsmi_shim64.so*` under component `libgoamdsmi_shim`, and `include/goamdsmi.h` plus `include/amdsmi_go_shim.h` with no component |
| Rust crate | only under `BUILD_RUST_WRAPPER=ON`, which defaults off and which no delivery channel enables | `share/amd_smi/rust-wrapper/source.tar.gz` under component `dev`, plus example binaries only under `BUILD_RUST_EXAMPLES=ON` |

The Rust interface SHALL be delivered as source rather than as a binary: the
installed tarball carries `Cargo.toml`, `build.rs`, `callbacks.rs`, `src/`,
`examples/`, a copy of `amdsmi.h` and a copy of the built `libamd_smi.so`, and
the consumer compiles it. Because [amdsmi-therock-subproject] passes only
`CMAKE_VERBOSE_MAKEFILE` and `BUILD_TESTS`, and the deb/rpm and wheel builds do
not set the option either, the tarball is produced by no channel and the Rust
binding reaches users only through a git checkout.

Which channels carry the Go shim SHALL follow from how each one installs.
Channels that run `cmake --install` over the whole build, which is what
[amdsmi-therock-subproject] does, place the shim library and its two headers
into the stage tree, where [amdsmi-therock-artifact]'s `lib/**` and `include/**`
patterns route them into the `lib` and `dev` components and onward into every
TheRock-derived package. AMD SMI's own CPack packages SHALL NOT contain them,
because `CPACK_COMPONENTS_ALL` lists only `dev` and `tests` and neither the
shim's own component nor the unspecified component the headers land in is a
member.

`goamdsmi.go` itself SHALL NOT be installed by any channel. It is consumed from
the repository, as the Go how-to documents, and the tree contains no `go.mod`,
so the package cannot be built in place by a module-aware Go toolchain.

#### Scenario: The documented Go workflow does not work on a packaged install

- **WHEN** a user installs the `amd-smi-lib` deb or rpm and follows the Go
  how-to, whose cgo directives hard-code `-I/opt/rocm/include`,
  `-L/opt/rocm/lib` and `-lgoamdsmi_shim64`
- **THEN** the link fails, because the package contains neither
  `libgoamdsmi_shim64.so` nor `amdsmi_go_shim.h`; the shim is reachable from a
  local `make install` and from the TheRock-derived channels, and from nowhere
  else

#### Scenario: The shim travels into channels that never asked for it

- **WHEN** a downstream consumer installs the `core-amdsmi` lib and dev
  components described in [amdsmi-therock-artifact]
- **THEN** it receives `libgoamdsmi_shim64.so` and the two Go headers alongside
  `libamd_smi.so`, because the subdirectory is unconditional and the artifact
  patterns are path-based — nothing in the descriptor mentions Go, and nothing
  documents the shim as part of those packages

#### Scenario: The cgo prefix is fixed regardless of where ROCm is installed

- **WHEN** ROCm is installed anywhere other than `/opt/rocm`, including the
  version-scoped `/opt/rocm/core-X.Y` prefix of the TheRock native packages
- **THEN** the cgo directives still search `/opt/rocm`, because they are
  literals in `goamdsmi.go` and consult neither `ROCM_PATH` nor `pkg-config`,
  so the consumer must patch the file or override the flags in its own build

#### Scenario: A missing shim symbol is deferred to run time

- **WHEN** the installed `libgoamdsmi_shim64.so` is older than `goamdsmi.go`
  and does not export an entry point the Go file declares
- **THEN** the link still succeeds, because the cgo directives pass
  `-Wl,--unresolved-symbols=ignore-in-object-files`, and the failure surfaces
  as a dynamic symbol lookup error when the program runs

### Requirement: Neither Binding Is Verified By Continuous Integration

No workflow SHALL be assumed to compile, test, lint or format either binding.
No job runs `go build`, `go vet`, `gofmt`, `cargo build`, `cargo test` or
`cargo clippy`, and neither `BUILD_RUST_WRAPPER` nor `REGENERATE_RUST_WRAPPER`
is set anywhere in CI. What coverage exists SHALL be understood as incidental:

| Artifact | What actually checks it |
| -------- | ----------------------- |
| `goamdsmi_shim/smiwrapper/*.c`, `*.h` | compiled by every project build, since the subdirectory is unconditional; formatted by the `clang-format` pre-commit hook |
| `goamdsmi.go` | parsed textually by the documentation build's `go-api-ref` directive; never compiled |
| `rust-interface/**` | nothing |

The crate's only test mechanism SHALL be `cargo test --doc`, whose cases are
the examples embedded in the doc comments. Each begins by calling
`amdsmi_init` and `expect`ing success, so the suite requires a host with AMD
hardware and a loaded driver and cannot run on the GPU-less runners the Python
and packaging jobs use.

An unverified binding is a standing risk rather than a guarantee, and this
capability records it as such: a consumer SHALL treat a green pull request as
evidence about the C library, the Python layer and the CLI, and as no evidence
at all about Go or Rust.

#### Scenario: A defect survives indefinitely because nothing executes the code

- **WHEN** a change crosses two Go wrapper functions onto the wrong shim
  entry points, or leaves a Rust safe wrapper referring to a renamed generated
  symbol
- **THEN** every check passes: the Go file is never compiled, the crate is
  never built, and the header-versus-merge-base ABI comparison described in
  [amdsmi-c-api-abi] inspects only `amdsmi.h` — the break is found by the first
  user who builds the binding

#### Scenario: A compile break in the shim is the one failure that is caught

- **WHEN** a change to `amdsmi.h` removes or renames a function
  `amdsmi_go_shim.c` calls
- **THEN** every build of the project fails, because the shim is compiled
  unconditionally as part of the default target — the single automatic guard
  either binding has, and it protects only the C half of the Go path
