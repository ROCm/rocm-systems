# amdsmi-c-api-abi Specification

## Purpose

Defines the contract of `libamd_smi` as a C shared library: what it exports,
what a caller may rely on across a rebuild, and what changing it costs. Every
other channel sits on this. The Python wrapper, the CLI, the Go shim, and the
Rust bindings are all generated from or bound to `include/amd_smi/amdsmi.h` and
call into the same exported symbols, so a change here propagates to all of them
at once.

Three artifacts define the contract and no others: the public header
`include/amd_smi/amdsmi.h`, the linker version script `src/CMakeLists.txt`
generates at configure time, and the SOVERSION derived from the version macros
in that header. The header is the compile-time surface, the version script is
the link-time surface, and the SOVERSION is the promise that the two have not
changed incompatibly since the last major bump.

The library signals absence at two different granularities, and conflating them
is the most common integration error: a whole operation that the platform
cannot perform returns a distinct `amdsmi_status_t`, while an individual field
that the platform cannot fill is returned inside a *successful* call as an
all-ones sentinel. Both are specified below, and this capability is the
authoritative statement of both — the language bindings translate them rather
than redefining them.

Deliberately excluded: how the library decides what hardware exists and what
each device is called ([amdsmi-device-discovery]); how the statuses and
sentinels defined here become Python exceptions and `"N/A"` values
([amdsmi-python-api]) and CLI output ([amdsmi-cli]); what the Go shim and the
Rust crate expose of this surface, and how each translates the same statuses
and sentinels ([amdsmi-language-bindings]); where the built library and header
are placed ([amdsmi-install-layout]); how a Python process finds and opens the
library ([amdsmi-python-loader]); and the wheel's separately-named private copy
([amdsmi-python-wheel]). Those capabilities consume the SONAME, the exported
surface, and the signalling conventions defined here.

## Requirements

### Requirement: The Header Version Macros Are The Single Source Of Truth

`AMDSMI_LIB_VERSION_MAJOR`, `AMDSMI_LIB_VERSION_MINOR`, and
`AMDSMI_LIB_VERSION_RELEASE` in `include/amd_smi/amdsmi.h` SHALL be the only
place the library version is authored. The build SHALL extract them by regex
and derive from them, without a second authored copy:

| Artifact | Derived value |
| -------- | ------------- |
| Shared library `SOVERSION` | `MAJOR` |
| Shared library `VERSION` | `MAJOR.MINOR.RELEASE` |
| CPack package version | `MAJOR.MINOR.RELEASE.<ROCM_LIBPATCH_VERSION>` |
| `amdsmi_get_lib_version()` output | the same three macros, compiled in |
| Rust binding version constants | rewritten by configure when they lag |

`MAJOR` SHALL change for any header change that breaks ABI, and `MINOR` for an
API change that does not alter the header. Configure SHALL rewrite a mirror of
these constants that has fallen behind the header rather than let the two
diverge silently.

How configure extracts the macros and turns them into the CPack package
version, the installed CMake package version file, and the wheel version — and
why a `-D` switch cannot override any of them — is
[amdsmi-build-configuration].

#### Scenario: A major bump obsoletes every linked consumer

- **WHEN** `AMDSMI_LIB_VERSION_MAJOR` moves from 26 to 27
- **THEN** the SONAME becomes `libamd_smi.so.27` and every binary already
  linked against `libamd_smi.so.26` must be relinked, because the dynamic
  linker will not substitute one for the other — which is the entire point of
  tying SOVERSION to the macro rather than to a package version or a git tag

#### Scenario: A drifted binding mirror is repaired at configure time

- **WHEN** the Rust wrapper still declares `AMDSMI_LIB_VERSION_MINOR = 0` while
  the header has moved to `1`
- **THEN** configure rewrites the wrapper's constants to match the header,
  because the header version is greater; a mirror can lag in the tree but
  cannot survive a build

#### Scenario: The version is answerable without a working library state

- **WHEN** `amdsmi_get_lib_version()` is called before `amdsmi_init()`, or
  after the final `amdsmi_shut_down()`
- **THEN** it succeeds and reports the compiled-in macros, because the values
  are constants rather than discovered state — this is what lets a caller check
  compatibility before deciding whether to initialize at all

### Requirement: The Exported Symbol Set Is The ABI

The shared library SHALL be linked with a version script that places every
exported symbol in the version node `AMDSMI_1` and exports exactly:

- everything matching the glob `amdsmi_*`;
- `rsmi_init` and `rsmi_is_P2P_accessible`, retained for external consumers
  that already reach them through `libamd_smi.so`.

Everything else — the internal `amd::smi::*` C++ implementation, the remaining
`rsmi_*` compatibility layer, the `smi_nic_*` NIC interface — SHALL be `local`.
The export criterion is therefore the *symbol name*, not membership in the
public header.

#### Scenario: The library and rocm_smi coexist in one process

- **WHEN** `libamd_smi.so` and `librocm_smi64.so` are both loaded, as happens
  when a framework pulls in one and the application the other
- **THEN** the `rsmi_*` symbols compiled into `libamd_smi.so` do not collide
  with the ones `librocm_smi64.so` exports, because all but the two
  grandfathered names are hidden

#### Scenario: A name-prefixed internal helper becomes ABI by accident

- **WHEN** an internal helper is named `amdsmi_free_name_value_pairs` and given
  C linkage in `include/amd_smi/impl/amd_smi_utils.h`
- **THEN** the glob exports it from the shared library even though `amdsmi.h`
  never declares it, so it is indistinguishable from public API to a linker and
  removing it is an ABI break — internal helpers must not take the `amdsmi_`
  prefix

#### Scenario: Installed impl headers are not a linkable interface

- **WHEN** a consumer of the dev package includes `amd_smi/impl/*.h` and calls
  one of the `smi_amdgpu_*` or `amd::smi::*` entry points it declares
- **THEN** the link fails, because those symbols are `local` in the shared
  library; the impl headers are installed for build convenience, not as a
  supported surface

#### Scenario: The static archive is not filtered

- **WHEN** the build produces `libamd_smi_static.a` alongside the shared
  library
- **THEN** the archive carries every symbol, because the version script is
  applied only to shared targets — a consumer that links the archive can reach
  internals that the `.so` hides, and inherits no ABI promise about them

### Requirement: amdsmi_status_t Is The Universal Return Type

Every public function declared in `amdsmi.h` SHALL return `amdsmi_status_t` and
SHALL report all outcomes through it; none returns a value or a pointer
directly. Enumerator values SHALL be stable once assigned and SHALL avoid
multiples of 256, because a shell truncates an exit status modulo 256 and would
render such a code indistinguishable from success. The distinctions a caller
must act on differently are:

| Status | Meaning for the caller |
| ------ | ---------------------- |
| `AMDSMI_STATUS_NOT_SUPPORTED` | The platform, ASIC, or driver cannot do this. Retrying, elevating privileges, or upgrading the library will not help. |
| `AMDSMI_STATUS_NOT_YET_IMPLEMENTED` | The entry point exists but has no implementation on this path. A later release may add it. |
| `AMDSMI_STATUS_NO_PERM` | The operation is supported but requires admin privileges. |
| `AMDSMI_STATUS_NOT_INIT` | `amdsmi_init()` has not been called, or the last `amdsmi_shut_down()` already ran. |
| `AMDSMI_STATUS_INVAL` | An argument is null or malformed, including an unrecognized socket handle. |
| `AMDSMI_STATUS_NOT_FOUND` | The processor handle is not one this library owns. |
| `AMDSMI_STATUS_INSUFFICIENT_SIZE` | The caller's buffer was too small; only the documented partial result was written. |
| `AMDSMI_STATUS_MORE_DATA` | The call succeeded but more records remain; resume with the returned cursor. |
| `AMDSMI_STATUS_MAP_ERROR` | An internal backend code had no `amdsmi_status_t` mapping. |

`amdsmi_status_code_to_string()` SHALL translate any of these to a human string
and SHALL itself be callable without initialization.

#### Scenario: Capability probing distinguishes three kinds of absence

- **WHEN** a caller classifies whether an API is usable on a given system
- **THEN** only `AMDSMI_STATUS_NOT_SUPPORTED`,
  `AMDSMI_STATUS_NOT_YET_IMPLEMENTED`, and `AMDSMI_STATUS_NO_HSMP_MSG_SUP` mean
  "not supported here", and at least one `AMDSMI_STATUS_SUCCESS` is required
  before declaring the API supported, so a permission failure or a transient
  I/O error is never mistaken for a missing capability

#### Scenario: An unsupported operation is reported as a status, not a sentinel

- **WHEN** `amdsmi_get_gpu_compute_partition()` is called on a consumer-class
  GPU that has no partitioning support
- **THEN** it returns `AMDSMI_STATUS_NOT_SUPPORTED` and leaves the caller's
  buffer alone, because the whole operation is unavailable rather than one
  field of a result

#### Scenario: An unmapped backend code is surfaced rather than coerced

- **WHEN** an internal `rsmi_`, ESMI, or NIC status has no entry in the
  translation maps
- **THEN** the caller receives `AMDSMI_STATUS_MAP_ERROR` rather than
  `AMDSMI_STATUS_SUCCESS` or a plausible-looking neighbour, so a newly added
  backend code cannot silently present as success

### Requirement: Unsupported Field Values Use All-Ones Sentinels

A call that reaches the driver SHALL return `AMDSMI_STATUS_SUCCESS` even when
individual members of the output structure are unavailable, and SHALL signal
each such member by setting it to the maximum value of its own type: `0xFFFF`
for `uint16_t`, `0xFFFFFFFF` for `uint32_t`, and `UINT64_MAX` for `uint64_t`.
Enumerations SHALL carry an explicit invalid enumerator at the same all-ones
value rather than overload zero. A caller SHALL therefore check both the status
and each field it reads.

The sentinel is defined by the width of the *public* field, so where a public
field is wider than the driver or metrics-table field it mirrors, the copy
SHALL widen the sentinel along with the value: an all-ones source SHALL produce
an all-ones destination. A plain cast would zero-extend `0xFFFFFFFF` into
`0x00000000FFFFFFFF`, and a consumer comparing against the 64-bit maximum — the
only comparison the field's declared type licenses — could not then tell "not
reported" from a real reading. Same-width and narrowing copies are unaffected.
Exactly one field departs from this rule, and the header documents the
departure inline at its declaration; no other field may.

#### Scenario: One structure mixes real and unavailable fields

- **WHEN** `amdsmi_get_power_info()` is called on a GPU that reports an average
  socket power but no instantaneous socket power
- **THEN** the call returns success with `average_socket_power` holding a real
  reading and `current_socket_power` holding `0xFFFFFFFF`, so treating a
  successful status as "the whole struct is valid" yields a power figure of
  about four billion watts

#### Scenario: A widened accumulator does not read as four billion

- **WHEN** `amdsmi_get_gpu_metrics_info()` runs on a GPU that does not report
  `gfx_activity_acc`, `mem_activity_acc`, `pcie_nak_sent_count_acc`,
  `pcie_nak_rcvd_count_acc`, or `pcie_lc_perf_other_end_recovery`, each of
  which is `uint64_t` in the public structure while the metrics table still
  reports 32 bits
- **THEN** the field holds `UINT64_MAX`, not `4294967295` — a literal
  4294967295 is both indistinguishable from a real counter and implausible as
  an activity accumulator, and it is what a plain widening cast produces

#### Scenario: The single exception is spelled out at its declaration

- **WHEN** `amdsmi_get_gpu_activity()` cannot obtain GFX activity
- **THEN** `gfx_activity` is `0x0000FFFF` rather than `0xFFFFFFFF`, because that
  `uint32_t` deliberately carries the `uint16_t` sentinel it inherits from
  `average_gfx_activity`; the declaration says so, which is the only reason a
  caller can be expected to compare against the narrower value there

#### Scenario: Versioned metrics blank out fields their revision does not define

- **WHEN** a metrics table populated by one revision is returned in a structure
  shaped for several
- **THEN** the members and array elements that revision does not define hold
  the all-ones sentinel for their type, and the caller identifies the active
  revision from `common_header.format_revision` and `content_revision` rather
  than by probing for non-sentinel values

### Requirement: Out-Parameter And Buffer Conventions

Public functions SHALL validate pointer arguments before dereferencing them and
SHALL return `AMDSMI_STATUS_INVAL` for a null required out-parameter. For
list-returning functions, passing a null array SHALL make the call a pure count
query; passing a non-null array SHALL fill at most the caller's stated capacity
and set the count to the number actually written. Callers SHALL NOT assume that
output storage is left untouched when a call fails, since no such blanket
guarantee is made.

#### Scenario: A null out-pointer is rejected instead of dereferenced

- **WHEN** `amdsmi_status_code_to_string()` or `amdsmi_get_esmi_err_msg()` is
  called with a null `status_string`
- **THEN** it returns `AMDSMI_STATUS_INVAL`; earlier releases dereferenced the
  null and crashed the caller, which is why this is enforced by unit tests that
  need no GPU

#### Scenario: Only the count query reports the true total

- **WHEN** `amdsmi_get_socket_handles()` is called with a capacity of 1 on a
  two-socket system
- **THEN** it returns success, writes one handle, and sets the count to 1 — the
  caller learns nothing about the second socket, so the two-call form with a
  null array first is the only reliable way to size the buffer

#### Scenario: Truncation is silent unless the function documents otherwise

- **WHEN** `amdsmi_get_socket_info()` is given a two-byte buffer for a
  twelve-character socket identifier
- **THEN** it returns success with a truncated, still null-terminated string,
  whereas the functions that document `AMDSMI_STATUS_INSUFFICIENT_SIZE` return
  that status for the same situation — the buffer contract is per-function and
  cannot be generalized from one getter to another

#### Scenario: A large result set is drained by cursor

- **WHEN** the CPER retrieval API is called with buffers smaller than the
  driver cache
- **THEN** it returns `AMDSMI_STATUS_MORE_DATA` with a cursor the caller passes
  to the next call, and returns `AMDSMI_STATUS_OUT_OF_RESOURCES` only when the
  buffer cannot hold even one entry, so "more data" and "buffer unusable"
  remain distinguishable

### Requirement: Initialization Is Reference Counted And First-Call Wins

`amdsmi_init()` SHALL accept a bitmask of `amdsmi_init_flags_t` selecting which
processor classes to discover, and SHALL take effect only on the transition
from uninitialized to initialized. Subsequent calls SHALL increment a reference
count and return `AMDSMI_STATUS_SUCCESS` without applying their flags.
`amdsmi_shut_down()` SHALL decrement the count and SHALL tear down state only
when it reaches zero, using the flags recorded by the *first* successful
initialization. A `shut_down` with no matching `init` SHALL return
`AMDSMI_STATUS_SUCCESS` and do nothing. Public bit positions SHALL remain
within the low bits reserved for `AMDSMI_INIT_*`; a reserved internal flag
SHALL live in a high bit that `AMDSMI_INIT_ALL_PROCESSORS` cannot set. Which
discovery failures are fatal to the call and which are skipped is specified in
[amdsmi-device-discovery].

#### Scenario: A second init with different flags is silently ineffective

- **WHEN** a process initializes with `AMDSMI_INIT_AMD_GPUS`, and a library it
  loaded later initializes with `AMDSMI_INIT_AMD_CPUS`
- **THEN** the second call returns success but discovers no CPUs, and the CPU
  handles the second caller expects never appear — a component that needs a
  processor class must be the one that initializes first, or agree on a
  combined mask

#### Scenario: An intermediate shutdown does not tear anything down

- **WHEN** two components have each called `amdsmi_init()` and one calls
  `amdsmi_shut_down()`
- **THEN** the count drops to one, discovery state survives, and the other
  component's handles keep working — which is the reason the count exists at
  all

#### Scenario: Initializing with no flags succeeds and discovers nothing

- **WHEN** `amdsmi_init(0)` is called
- **THEN** it returns success and the socket count is zero, because the flags
  are a discovery selector rather than a validated argument; unrecognized bits
  are likewise ignored rather than rejected

#### Scenario: Every handle-taking call refuses to run uninitialized

- **WHEN** any discovery or query function is called before the first
  `amdsmi_init()` or after the final `amdsmi_shut_down()`
- **THEN** it returns `AMDSMI_STATUS_NOT_INIT` before touching its arguments,
  with `amdsmi_get_lib_version()` and the status-string helpers the only
  deliberate exceptions

### Requirement: Handles Are Opaque, Library-Owned, And Validated By Lookup

`amdsmi_socket_handle`, `amdsmi_processor_handle`, and `amdsmi_node_handle`
SHALL be `void*` typedefs over library-internal objects. The caller SHALL NOT
allocate, free, dereference, or outlive them; their storage is owned by the
library and destroyed by the shutdown that drops the reference count to zero.
Before using a handle the library SHALL look it up in its live containers and
SHALL NOT dereference one it does not find. A null handle, and a non-null
socket handle that is not a member, SHALL yield `AMDSMI_STATUS_INVAL`; a
non-null processor handle that is not a member SHALL yield
`AMDSMI_STATUS_NOT_FOUND`.

This is the whole of the handle contract a binding or a C caller may rely on.
Which handles exist on a given machine, and what they are named after, is
[amdsmi-device-discovery].

#### Scenario: A fabricated handle is rejected rather than dereferenced

- **WHEN** an arbitrary non-null pointer is passed as a socket or processor
  handle
- **THEN** the call returns `AMDSMI_STATUS_INVAL` or `AMDSMI_STATUS_NOT_FOUND`
  respectively instead of faulting, because validation is a membership test on
  the live container rather than a cast

#### Scenario: A handle held across a full shutdown is not reliably detected

- **WHEN** a caller retains a handle, shuts the library down completely,
  re-initializes, and reuses the handle
- **THEN** the membership test may pass because the allocator handed the same
  address to a new object, and the call silently operates on a different device
  than intended — handles must be re-fetched after every teardown, since the
  library offers no generation counter to catch this

### Requirement: Concurrency Is Guarded Per Device, Not Per Library

Access to a GPU device SHALL be serialized by a per-device mutex, so concurrent
readers of the same device do not interleave driver access. The library-level
initialization reference count SHALL NOT be assumed safe against concurrent
`amdsmi_init()` and `amdsmi_shut_down()` calls; it is a plain counter with no
synchronization. Storage returned indirectly through an output structure SHALL
be documented as thread-local and treated as invalidated by the next call on
the same thread.

#### Scenario: Lifecycle calls must be sequenced by the caller

- **WHEN** two threads call `amdsmi_init()` and `amdsmi_shut_down()`
  concurrently
- **THEN** the reference count can be lost or double-counted, so a process that
  initializes from more than one thread must serialize the lifecycle calls
  itself; the per-device locking does not extend to library lifecycle

#### Scenario: A contended device reports busy only in the reserved test mode

- **WHEN** a device mutex cannot be acquired
- **THEN** the call blocks by default and returns `AMDSMI_STATUS_BUSY` only
  when the reserved internal init flag has switched the mutex to non-blocking
  mode, which exists so tests can observe the busy path deterministically

#### Scenario: Thread-local metrics storage expires on the next query

- **WHEN** a caller reads APU metrics through the pointer embedded in the GPU
  metrics structure and then queries metrics for a second device on the same
  thread
- **THEN** the first pointer no longer refers to the first device's data,
  regardless of which device was queried, so the structure must be copied
  immediately after the call that produced it

### Requirement: Structures And Enumerations Evolve Additively

Public structures SHALL carry trailing `reserved` arrays sized so that future
fields can be added by consuming reserved space without moving any existing
member or changing the structure size. New enumerators SHALL be appended and
existing enumerator values SHALL NOT be renumbered. When a type or enumerator
is renamed, the previous spelling SHALL be retained as an alias with an
unchanged value. Deprecation SHALL be expressed in documentation only: a
deprecated function keeps its declaration, its behavior, and its exported
symbol until a major release removes it.

#### Scenario: A field is added without disturbing the layout

- **WHEN** a new member is introduced into a structure whose last field is
  `uint64_t reserved[N]`
- **THEN** it is taken from the reserved space and `N` is reduced accordingly,
  so every existing field keeps its offset and the structure keeps its size — a
  caller compiled against the older header continues to read the same bytes

#### Scenario: Renaming an enumerator does not invalidate existing sources

- **WHEN** an unprefixed enumerator such as `CLK_LIMIT_MIN` is superseded by
  `AMDSMI_CLK_LIMIT_MIN`
- **THEN** the old name remains defined with the identical value and is marked
  deprecated, so existing sources compile unchanged and no compiled constant
  changes meaning

#### Scenario: A deprecated function remains callable and exported

- **WHEN** an API is superseded, as the `compute_partition` family was by the
  `accelerator_partition` family
- **THEN** the old entry point is still declared, still exported from the
  shared library, and delegates to its replacement, so deprecation alone never
  breaks a linked binary

#### Scenario: Widening a field is a break even though the field name survives

- **WHEN** accumulator counters in the GPU metrics structure are widened from
  32 to 64 bits ahead of the driver's table
- **THEN** every subsequent field offset moves and the change is a
  major-version break requiring recompilation, which is why such changes are
  batched into a major release rather than landed as an additive fix — and
  every copy site feeding the widened field must carry the sentinel across at
  the same time, or the widening silently changes what an unreported value
  looks like as well as where it sits

### Requirement: The Compile-Time Surface Is Conditioned On ENABLE_ESMI_LIB

The public header SHALL gate the CPU and HSMP portion of the API — its handle
typedef, its structures, and its function declarations — behind an
`ENABLE_ESMI_LIB` conditional. The build SHALL define this macro when the ESMI
backend is compiled in, which is the default on x86_64, and SHALL NOT propagate
it as an interface definition through the exported CMake target. A consumer
that needs the CPU API SHALL define the macro itself, matching the
configuration of the library it links against.

The option's own configure-time contract — its per-architecture default and the
pinned ESMI source it gates — is [amdsmi-build-configuration].

#### Scenario: A C consumer cannot see an API the library exports

- **WHEN** a downstream project links the installed x86_64 library and includes
  `<amd_smi/amdsmi.h>` without defining `ENABLE_ESMI_LIB`
- **THEN** the CPU functions are invisible at compile time even though the
  library exports them, so the call fails to compile rather than failing to
  link — and the failure looks like a missing feature rather than a
  configuration mismatch

#### Scenario: Only one of the two binding generators plumbs the macro

- **WHEN** the Python wrapper and the Rust FFI module are each regenerated
- **THEN** `py-interface` passes the build's own `ENABLE_ESMI_LIB` setting
  through to the generator's clang arguments, while `rust-interface/build.rs`
  gives `bindgen` no such argument — so the Python wrapper tracks the
  configuration the library was built with, and the committed Rust bindings
  declare exactly the header's 178 unconditional functions and none of the 68
  the conditional block adds, on every architecture and at every setting of the
  option; what that costs a Rust caller is [amdsmi-language-bindings]

### Requirement: ABI Break Detection Is Advisory, Not A Gate

The project SHALL compare the public header against its merge base on every
pull request touching AMD SMI, in two modes: a permissive mode that flags
removed symbols, data-type problems, and an incompatible verdict, and a strict
mode that additionally flags source-compatible changes. The result SHALL be
surfaced by applying or removing a `MAJOR ABI BREAKAGE` or `MINOR ABI BREAKAGE`
label and by publishing the report as an artifact. The check SHALL NOT block a
merge.

#### Scenario: A deliberate break is labelled and still mergeable

- **WHEN** a pull request widens a struct field or removes an API
- **THEN** the workflow labels the pull request and uploads the report, but the
  job does not fail the merge, so the label is a prompt for human judgement
  about whether the accompanying major-version bump was made — nothing verifies
  that pairing automatically

#### Scenario: The header-only comparison cannot see the built library

- **WHEN** the check runs
- **THEN** it compares two copies of `amdsmi.h` and never inspects a built
  `.so`, so it cannot detect that the version script stopped exporting a
  symbol, that an internal `amdsmi_`-prefixed helper became part of the ABI, or
  that the SOVERSION disagrees with the header

#### Scenario: The conditionally compiled CPU surface is outside the check

- **WHEN** a change alters a declaration inside the `ENABLE_ESMI_LIB` block
- **THEN** the comparison runs with no project build configuration, so a large
  part of the surface the shipped x86_64 library actually exports is not
  evaluated
