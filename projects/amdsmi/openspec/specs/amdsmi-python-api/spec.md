# amdsmi-python-api Specification

## Purpose

Defines what a caller who writes `import amdsmi` may rely on: which names the
package exposes, how every non-success `amdsmi_status_t` becomes a Python
exception, what shapes, key names and units the functions return, how processor
and socket handles are obtained and how long they remain valid, and the rules
governing the generated ctypes layer the whole package sits on.

This is one capability because all of it is decided in a single place — the
four hand-written files in `py-interface/` plus the generated wrapper they call
— and because the parts are not separable in practice: the exception hierarchy
is what makes an unsupported field on one ASIC survivable, the `"N/A"`
convention is what makes a dictionary's key set stable across ASICs, and both
only mean anything relative to the handle and init lifecycle that produced the
call.

How the module and the native `libamd_smi*.so` behind it are *found* is
deliberately excluded: that is specified in [amdsmi-python-loader], and the
per-channel detail in [amdsmi-python-wheel] and [amdsmi-python-system-package].
Where the files land on disk is [amdsmi-install-layout]. The header, the
exported symbols and the status and sentinel semantics this layer translates are
[amdsmi-c-api-abi]. The `amd-smi` command-line tool is a separate consumer of
this API with its own contract, [amdsmi-cli].

## Requirements

### Requirement: Three Layers With A Single Public One

The `amdsmi` package SHALL be built from exactly three layers, and a caller
SHALL depend only on the third:

| Layer | Files | Role | Caller-facing |
| ----- | ----- | ---- | ------------- |
| Generated bindings | `amdsmi_wrapper.py` | ctypes structs, unions, enum constants, `argtypes`/`restype` declarations, and the native-library load | No |
| Pythonic interface | `amdsmi_interface.py`, `amdsmi_exception.py`, `amdsmi_interface_utils.py` | typed functions, `IntEnum` mirrors of the C enums, ctypes marshalling, status-to-exception translation | Only through re-export |
| Public surface | `__init__.py` | the re-export list | Yes |

The hand-written layer SHALL be the only caller of the generated layer: every
public function allocates the ctypes object, calls the generated binding, and
converts the result, so no caller has to touch `ctypes` to use the API.
`amdsmi_interface_utils.py` SHALL hold only private helpers, named with a
leading underscore and listed in an `__all__` that exists solely so
`amdsmi_interface.py` can star-import them.

#### Scenario: Submodules are reachable but are not the API

- **WHEN** a caller reaches `amdsmi.amdsmi_wrapper`, `amdsmi.amdsmi_interface`,
  `amdsmi.amdsmi_interface_utils` or `amdsmi.amdsmi_exception`
- **THEN** the attribute resolves, because `__init__.py` imports from those
  modules and the package declares no `__all__` — but nothing in them is
  covered by this capability, and the underscore-prefixed helpers in
  `amdsmi_interface_utils.py` are documented in the module itself as not part
  of the public API

### Requirement: The Public API Is The `__init__.py` Re-Export List

`__init__.py` SHALL be the definition of the public API. It re-exports, by
explicit `from .amdsmi_interface import <name>` lines: the initialization pair,
the discovery functions, the GPU and CPU query and control functions, the
`AmdSmiEventReader` class, and the `IntEnum` types callers must pass as
arguments; plus `__version__` and `__commit__` from `_version.py` and the
exception classes `AmdSmiException`, `AmdSmiLibraryException`,
`AmdSmiRetryException`, `AmdSmiTimeoutException`, `AmdSmiParameterException`
and `AmdSmiKeyException`. A function that exists in
`amdsmi_interface.py` but is absent from that list SHALL NOT be treated as
public.

New public API SHALL be added to `amdsmi_interface.py` and re-exported, never
exposed only through a submodule.

#### Scenario: An in-tree function is not automatically public

- **WHEN** a caller reaches for the NIC, AI-NIC or switch helpers, or for
  `amdsmi_get_gpu_device_bdf_bdf`, all of which are defined in
  `amdsmi_interface.py`
- **THEN** they are not present on the `amdsmi` namespace, because they are not
  in the re-export list; reaching them requires importing the submodule, which
  places the caller outside the supported surface

#### Scenario: The test suite enforces top-level availability

- **WHEN** a test declares the APIs it needs via the shared `_skip_if_missing`
  helper in `tests/python/common/common.py`
- **THEN** the check is `hasattr(amdsmi, name)` against the top-level package,
  so an API added to `amdsmi_interface.py` but never re-exported is reported as
  missing rather than silently reached through a submodule

#### Scenario: A defined exception class is not necessarily catchable by name

- **WHEN** a caller writes `except amdsmi.AmdSmiBdfFormatException`
- **THEN** it fails with `AttributeError`, because that class is defined in
  `amdsmi_exception.py` but not re-exported; the class is still reachable
  through the base `AmdSmiException`, which is the supported way to catch it

#### Scenario: Deprecation is signalled in-band

- **WHEN** a caller invokes a superseded function such as
  `amdsmi_get_gpu_device_bdf_bdf`
- **THEN** it emits a `DeprecationWarning` naming the replacement and still
  returns its historical result, so a script keeps working while the warning
  surfaces under `-W error::DeprecationWarning` in CI

### Requirement: Every Non-Success Status Becomes An Exception

The interface SHALL NOT return status codes. Every call into the generated
layer SHALL be passed through `_check_res`, which raises when the returned
`amdsmi_status_t` is anything other than `AMDSMI_STATUS_SUCCESS`:

| Returned status | Raised exception |
| --------------- | ---------------- |
| `AMDSMI_STATUS_SUCCESS` | none; the function returns its value |
| `AMDSMI_STATUS_RETRY` | `AmdSmiRetryException` |
| `AMDSMI_STATUS_TIMEOUT` | `AmdSmiTimeoutException` |
| any other non-zero | `AmdSmiLibraryException` |

The hierarchy SHALL be: `AmdSmiException` derived from `Exception` as the base
every other class shares; `AmdSmiLibraryException` derived from it and carrying
a library status code; `AmdSmiRetryException` and `AmdSmiTimeoutException`
derived from `AmdSmiLibraryException`; and `AmdSmiParameterException`,
`AmdSmiKeyException` and `AmdSmiBdfFormatException` derived directly from
`AmdSmiException` with no status code.

Codes and messages SHALL come from the library, not from Python.
`AmdSmiLibraryException.err_code` is the numeric `amdsmi_status_t` the library
returned, whose meanings are defined in [amdsmi-c-api-abi], and the
human-readable text comes from the C library's own
`amdsmi_status_code_to_string` and `amdsmi_get_esmi_err_msg`.

#### Scenario: One except clause covers the whole API

- **WHEN** a caller wraps its work in `except amdsmi.AmdSmiException`
- **THEN** every failure mode of this API is caught — library status codes,
  argument-type rejections, dictionary-key and BDF-format errors — because all
  of them derive from that base

#### Scenario: A caller branches on the code, not the message

- **WHEN** a caller catches `AmdSmiLibraryException` and calls
  `get_error_code()`
- **THEN** it gets the integer status (for example `2` for
  `AMDSMI_STATUS_NOT_SUPPORTED`, `32` for `AMDSMI_STATUS_NOT_INIT`), directly
  comparable against the `AmdSmiStatus` enum, so no caller has to parse the
  human-readable text; `get_error_info(detailed=False)` yields just the
  `AMDSMI_STATUS_*` name when a short label is wanted

#### Scenario: A status newer than the Python table still reports truthfully

- **WHEN** the library returns a status code that the description table in
  `amdsmi_exception.py` does not yet list
- **THEN** an `AmdSmiLibraryException` is still raised carrying the real
  numeric code, and only the human-readable description degrades to the
  unknown-error text — so a library ahead of the Python layer cannot turn a
  failure into a silent success

### Requirement: Unsupported Is A Routine Outcome, Not A Fault

A caller enumerating GPUs SHALL expect that some queries are unsupported on
some parts and SHALL treat these as normal control flow rather than as errors:

| Status | Code | Meaning for the caller |
| ------ | ---- | ---------------------- |
| `AMDSMI_STATUS_NOT_SUPPORTED` | 2 | this ASIC or firmware has no such field; skip it |
| `AMDSMI_STATUS_NOT_YET_IMPLEMENTED` | 3 | the library has no backend for it yet; skip it |
| `AMDSMI_STATUS_NO_HSMP_MSG_SUP` | 49 | this HSMP message is absent on this CPU; skip it |

Every other non-success status SHALL be treated as a real fault, including the
ones a caller is most tempted to lump in with the above:
`AMDSMI_STATUS_NOT_INIT` (32), `AMDSMI_STATUS_DRIVER_NOT_LOADED` (34),
`AMDSMI_STATUS_NO_PERM` (10) and `AMDSMI_STATUS_INVAL` (1). Their meanings are
defined in [amdsmi-c-api-abi]; what this capability fixes is that they arrive
as exceptions rather than as return values.

Support SHALL be discovered by calling and catching. The API provides no
per-function capability query.

#### Scenario: A workstation part supports some queries and not others

- **WHEN** the same script runs against a Radeon Pro W6800 (gfx1030) and calls
  `amdsmi_get_gpu_asic_info`, `amdsmi_get_power_info`, `amdsmi_get_clock_info`,
  `amdsmi_get_gpu_kfd_info` and `amdsmi_get_gpu_total_ecc_count` alongside
  `amdsmi_get_gpu_compute_partition`, `amdsmi_get_violation_status` and
  `amdsmi_get_soc_pstate`
- **THEN** the first group returns data and the second group raises
  `AmdSmiLibraryException` with code 2, so a portable script must guard each
  call rather than deciding once per device that it is "supported"

### Requirement: Arguments Are Validated Before The Library Is Called

Every public function taking a handle or an enum SHALL `isinstance`-check it and
raise `AmdSmiParameterException` before entering the generated layer, so a
mistyped argument cannot reach ctypes and be reinterpreted as a pointer.
`amdsmi_init` SHALL additionally reject a flag value that is not
`AMDSMI_INIT_ALL_PROCESSORS` and does not consist solely of bits drawn from
`AmdSmiInitFlags`.

`AmdSmiParameterException` SHALL carry the received value's type and the
expected type, and SHALL have `err_code` set to `None` — it does not originate
from the library and has no status code.

#### Scenario: A wrong handle type is rejected, not dereferenced

- **WHEN** a caller passes `None`, an `int`, or a socket handle where a
  processor handle is required
- **THEN** `AmdSmiParameterException` is raised naming both the actual and the
  expected type, and the library is never called, which is why the test suite
  can use `None` as a portable "bad handle" on any platform

#### Scenario: Code branching on status must also handle its absence

- **WHEN** a caller catches both exception families and calls
  `get_error_code()`
- **THEN** it must first test for the method's presence, because
  `AmdSmiParameterException` does not provide it and touching `err_code`
  directly yields `None` rather than a status — the pattern the shared test
  helpers use

#### Scenario: An invalid init flag never reaches the library

- **WHEN** `amdsmi_init` is called with an integer that sets bits outside
  `AmdSmiInitFlags`
- **THEN** `AmdSmiParameterException` is raised naming `AmdSmiInitFlags` as the
  expected type, rather than the library being initialized with a partially
  meaningful flag word

### Requirement: Stable Return Shapes And Key Names

Return types SHALL follow one convention per kind of result and SHALL NOT vary
with the ASIC:

| Result kind | Shape | Examples |
| ----------- | ----- | -------- |
| A single measurement | a bare scalar | `amdsmi_get_temp_metric` → `int`, `amdsmi_get_gpu_bad_page_threshold` → `int`, `amdsmi_is_gpu_power_management_enabled` → `bool` |
| A group of related fields | a `dict` with a fixed key set | `amdsmi_get_gpu_asic_info`, `amdsmi_get_power_info`, `amdsmi_get_clock_info` |
| An enumeration of records | a `list` of `dict` | `amdsmi_get_gpu_bad_page_info`, `amdsmi_get_gpu_ras_block_features_enabled`, `amdsmi_get_gpu_pm_metrics_info` |
| A discovery result | a `list` of handles, or a `dict` wrapping the list with its count | `amdsmi_get_processor_handles`, `amdsmi_get_cpu_handles` |
| A formatted identifier | a `str` | `amdsmi_get_gpu_device_bdf`, `amdsmi_get_gpu_device_uuid` |

The key names of returned dictionaries SHALL be treated as part of the API
surface: they are consumed by user scripts and by the `amd-smi` CLI shipped
from the same tree, whose own key contract is [amdsmi-cli]. Renaming one is a
breaking change even when the underlying C field or entry point is renamed.

A field the hardware does not report SHALL still be present under its usual key
with the value `"N/A"`, never omitted and never left as the raw all-ones
sentinel. The substitution is an exact comparison against the maximum of the
width the field is declared with in `include/amd_smi/amdsmi.h` — `UINT64_MAX`
for a `uint64_t` field, and so on — plus the rule that an activity percentage
above 100 is unavailable; empty strings decoded from the library become `"N/A"`
as well.

That exactness is why this layer depends on the widening obligation in
[amdsmi-c-api-abi]: a sentinel that arrives narrower than the field it is
carried in does not compare equal, and the raw integer is passed through as if
it were a reading. This layer SHALL NOT compensate by accepting a narrower
sentinel in a wider field, because doing so would make a genuine reading of
`0xFFFFFFFF` in a 64-bit counter unrepresentable.

An enumeration that legitimately found nothing SHALL return an empty list, not
an exception and not a placeholder string.

#### Scenario: A dictionary's key set does not depend on the part

- **WHEN** `amdsmi_get_gpu_asic_info` is called on a part that reports no OAM id
- **THEN** the returned dictionary still contains `oam_id`, with the value
  `"N/A"`, so a caller can index every documented key unconditionally instead of
  writing `.get()` with defaults everywhere

#### Scenario: A caller must accept a string where a number is expected

- **WHEN** `amdsmi_get_power_info` runs on a part that does not report instant
  socket power, or `amdsmi_get_gpu_metrics_info` runs on a part where many
  fields are unpopulated
- **THEN** those keys hold the string `"N/A"` while their neighbours hold
  integers, so arithmetic on a metrics value must be guarded — on a W6800, 23 of
  the 85 `amdsmi_get_gpu_metrics_info` keys come back `"N/A"`

#### Scenario: A lost sentinel is unrecoverable at this layer

- **WHEN** `gfx_activity_acc` is declared `uint64_t` and the library hands up
  `0x00000000FFFFFFFF` because a 32-bit sentinel was zero-extended into it
- **THEN** the comparison against `UINT64_MAX` does not match and the key holds
  `4294967295` instead of `"N/A"`; every consumer of this API then reports that
  number as a real measurement, and none of them — see [amdsmi-cli] — can
  recover the distinction, which is why the sentinel has to survive the
  widening below

#### Scenario: Handle discovery reports its own count

- **WHEN** a caller uses `amdsmi_get_processor_handles_by_type` or
  `amdsmi_get_cpu_handles`
- **THEN** it receives a dictionary carrying both the handle list and the count,
  rather than a bare list, because those functions front a fixed-capacity C
  array whose fill level the library reports separately

### Requirement: Units Are The Library's Units, Not Normalized

The Python layer SHALL NOT normalize units. A GPU query returns the value in
whatever unit `include/amd_smi/amdsmi.h` documents for that field, as a bare
number with no unit suffix, and two fields in the same dictionary may therefore
use different scales.

Three families SHALL keep their existing, deliberate conversions:

| Family | Conversion |
| ------ | ---------- |
| CPU/HSMP scalar getters | value formatted into a unit-suffixed string, for example `"3600 MHz"`, `"1234 mW"` |
| `amdsmi_get_hsmp_metrics_table` | fixed-point Q10/UQ10/UQ16 fields decoded, two's-complemented where signed, rounded, and formatted with `°C`, `W`, `kJ`, `MHz`, `GHz` suffixes |
| `apu_metrics.*` inside `amdsmi_get_gpu_metrics_info` | centidegrees divided by 100 to °C and milliwatts divided by 1000 to W, rounded to two decimals; clocks and activity percentages are passed through unchanged |

#### Scenario: One dictionary mixes scales

- **WHEN** a caller reads `amdsmi_get_power_info` on a W6800 and sees
  `socket_power` of `20` next to `power_limit` of `213000000`
- **THEN** both are correct in their own documented units, because the Python
  layer copies the struct fields verbatim; a caller that assumes one unit for a
  whole dictionary will be wrong

#### Scenario: A CPU getter returns a string where a GPU getter returns a number

- **WHEN** a caller reads a CPU frequency or the HSMP metrics table
- **THEN** the value is a formatted string such as `"3600 MHz"`, so it must be
  parsed before use — the same script cannot treat GPU and CPU getters
  interchangeably

### Requirement: Handles Are Obtained Only From The Library

Processor, socket and CPU-core handles SHALL be obtainable only from the
library, through the discovery functions (`amdsmi_get_socket_handles`,
`amdsmi_get_processor_handles`, `amdsmi_get_processor_handles_by_type`,
`amdsmi_get_cpu_handles`, `amdsmi_get_cpucore_handles`) or from
`amdsmi_get_processor_handle_from_bdf`. The Python layer only type-checks them
and never frees them — `amdsmi_shut_down` does. Their opaqueness and validity
window are specified in [amdsmi-c-api-abi] and the tree they point into is
[amdsmi-device-discovery]; the consequence here is that a caller SHALL NOT
cache handles, or an index-to-handle mapping, across a shutdown.

#### Scenario: A handle used after shutdown raises rather than crashes

- **WHEN** a caller keeps a processor handle past `amdsmi_shut_down` and passes
  it to a query
- **THEN** `AmdSmiLibraryException` with code 32
  (`AMDSMI_STATUS_NOT_INIT`) is raised, because the C entry points check the
  initialization state first and the handle-to-processor lookup validates by
  membership in the live registry instead of dereferencing the stale pointer

#### Scenario: Handles are not stable across an init cycle

- **WHEN** a process calls `amdsmi_init`, records the handle list, shuts down,
  re-initializes, and lists again
- **THEN** the handles are not the same list — on a two-GPU host the two
  library-owned allocations come back in the opposite order — so a cached
  "GPU 0 is handle X" mapping can silently address the *other* device after a
  re-init; a durable identity must be derived from
  `amdsmi_get_gpu_device_bdf` or `amdsmi_get_gpu_device_uuid`

#### Scenario: A handle is not an index

- **WHEN** a caller wants the library's ordinal for a processor
- **THEN** it calls `amdsmi_get_processor_info`, which returns the zero-based
  position in the library's processor list as a decimal string, rather than
  interpreting the pointer value

### Requirement: Reference-Counted Initialization And Shutdown

`amdsmi_init` SHALL default to `AmdSmiInitFlags.INIT_AMD_GPUS` rather than
require a flag argument. The reference-counting semantics both functions
inherit are specified in [amdsmi-c-api-abi]; what this layer adds is that both
SHALL return `None` on success and raise on failure, so no caller should test
their return value.

#### Scenario: An unbalanced shutdown is not an error

- **WHEN** a `finally:` block calls `amdsmi_shut_down` on a path where
  initialization never happened, or one more time than it initialized
- **THEN** the call returns normally, so defensive cleanup code does not have to
  track whether it owns the initialization

#### Scenario: Initialization failure is reported as an exception

- **WHEN** `amdsmi_init` runs with no `amdgpu` driver loaded
- **THEN** it raises `AmdSmiLibraryException`, and callers distinguish "no
  hardware here" from a real fault by inspecting `get_error_code()` for
  `AMDSMI_STATUS_NOT_INIT` or `AMDSMI_STATUS_DRIVER_NOT_LOADED` — the check the
  shared test harness performs before skipping

### Requirement: The ctypes Wrapper Is Generated And Never Hand-Edited

`py-interface/amdsmi_wrapper.py` SHALL be produced by `tools/generator.py` from
`include/amd_smi/amdsmi.h` and SHALL NOT be edited by hand. It is committed to
the tree so that consumers need no clang at install time, and it is excluded
from the repository's Python formatting and lint hooks precisely because its
contents are not authored.

Regeneration SHALL require clang 16.0 or newer and `ctypeslib2` 2.4.0, and SHALL
be driven either by configuring with `BUILD_WRAPPER=ON` or by
`tools/update_wrapper.sh`, which runs the same CMake path inside a container
image tagged by its Dockerfile hash so two developers produce the same file.
Regeneration is incompatible with `BUILD_PYTHON_WHEEL=ON`, as specified in
[amdsmi-python-wheel].

Regeneration SHALL guarantee that every struct layout, union, enum constant,
`argtypes` and `restype` matches the header the native library was built from,
and that the loader block and `_AMDSMI_LIB_SONAME` are re-emitted from the
generator's fixed template as specified in [amdsmi-python-loader].

Every `amdsmi_*` symbol binding except a small stable set SHALL be wrapped in
`try: ... except AttributeError: pass`, so a symbol absent from the loaded
library leaves the name undefined on the wrapper module rather than breaking
`import amdsmi`.

#### Scenario: A hand edit is silently reverted and desyncs the ABI

- **WHEN** a developer patches a struct field or an `argtypes` list directly in
  the committed wrapper
- **THEN** the change disappears on the next regeneration, and until then the
  Python-side layout disagrees with the header the `.so` was compiled from,
  which corrupts data rather than failing loudly — the reason the file carries a
  never-edit rule

#### Scenario: An older library missing a newer symbol still imports

- **WHEN** a wrapper from one ROCm release is loaded against an older
  `libamd_smi.so` that does not export every symbol
- **THEN** the guarded bindings skip the missing ones and the common symbols
  keep working, while a call into a genuinely missing symbol raises
  `AttributeError` rather than silently returning success — the guarantee
  `tests/python/test_abi_compat.py` pins

#### Scenario: A new unguarded binding is rejected

- **WHEN** a change introduces a top-level `amdsmi_*` binding outside a
  `try`/`except AttributeError`
- **THEN** the ABI test's AST scan of the wrapper fails unless the symbol is in
  the curated stable set, because an unguarded binding makes
  `import amdsmi` itself fail against any library that predates that symbol

#### Scenario: Structure layout is pinned for newer CPython

- **WHEN** the generator emits a packed structure
- **THEN** it also emits `_layout_ = 'ms'` beside every `_pack_`, preserving the
  pre-3.14 MSVC-compatible layout that CPython 3.14 deprecates and 3.19 makes an
  error; generation aborts if the counts of `_pack_` and inserted `_layout_`
  lines differ, rather than shipping a wrapper whose ABI silently moved

### Requirement: CPython 3.6.8 Floor Through The Newest Supported Release

The generated wrapper and all four hand-written modules SHALL remain importable
and usable on CPython 3.6.8 and on every release up to the newest supported
minor, currently 3.14. The floor exists because the system package must work on
the platform interpreter of an el8-family host, which is Python 3.6: the Debian
package declares `python3 (>= 3.6.8)`, the wheel metadata declares
`requires-python >= 3.6`, and the Python-interface CMake requires a 3.6
interpreter.

#### Scenario: No syntax or typing newer than the floor

- **WHEN** a change introduces 3.7-or-later syntax, or PEP 585 builtin generics
  such as `dict[str, int]` in an annotation
- **THEN** it breaks the floor: the modules today parse under 3.6 and annotate
  exclusively with `typing.Dict`/`List`/`Tuple`/`Union` without
  `from __future__ import annotations`, so annotations evaluated at import time
  stay valid on the oldest interpreter

#### Scenario: The range is verified without hardware

- **WHEN** the Python-version job runs
  `tests/run_amdsmi_python_versions_test.py`
- **THEN** each discovered interpreter from 3.6 to 3.14 imports the wrapper and
  runs the ABI-compatibility and packaging guard tests, so a break confined to
  one end of the range is caught on a GPU-less runner
