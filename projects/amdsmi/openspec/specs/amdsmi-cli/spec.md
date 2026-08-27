# amdsmi-cli Specification

## Purpose

Defines the user-facing contract of the `amd-smi` command-line tool: the
subcommand surface, the three output formats, how a device is named, how a value
that the hardware or driver does not supply is rendered, which stream carries
what, and which exit code a failure produces.

`amd-smi` is the interface that scripts, CI jobs, and monitoring agents across
ROCm bind against. Those consumers parse `--json`/`--csv` output and branch on
`$?`; they do not read the source. Anything on this page is therefore a
compatibility surface rather than an implementation detail, even where the
implementation is a thin argparse wrapper. The tool itself is documented as
example code (`amdsmi_cli/README.md`), which describes the *quality* guarantee,
not the *stability* guarantee — the shapes below have downstream consumers
regardless.

The implementation is `amdsmi_cli/amdsmi_cli.py` (entry point and top-level
error handling), `amdsmi_parser.py` (surface), `amdsmi_commands.py` plus
`subcommands/` (data collection), `amdsmi_logger.py` (rendering),
`amdsmi_helpers.py` (device resolution, unit formatting),
`amdsmi_cli_exceptions.py` (exit codes), and `amdsmi_init.py` (library
initialization).

Neighbouring concerns are deliberately out of scope. How `amd-smi` locates the
`amdsmi` Python package, and what happens when it cannot, is specified in
[amdsmi-python-loader]; the `bin` → `libexec` → `share` relationships the entry
point relies on are in [amdsmi-install-layout]. The library statuses and
all-ones field sentinels this tool renders are defined in [amdsmi-c-api-abi]
and reach it already translated by [amdsmi-python-api]. Nothing here constrains
the C API or the packaging channels.

## Requirements

### Requirement: Stable Subcommand Surface

`amd-smi` SHALL accept exactly the subcommands below as its first positional
token, plus the top-level flags `-h`/`--help`, `--rocm-smi`, and `--loglevel`.
Invoking `amd-smi` with no arguments SHALL run the `default` summary view.

| Subcommand | Alias | Gated on |
| ---------- | ----- | -------- |
| `version` | — | always |
| `list` | — | amdgpu initialized |
| `static` | — | always |
| `firmware` | `ucode` | amdgpu initialized |
| `bad-pages` | — | Linux baremetal, amdgpu initialized |
| `metric` | — | always |
| `process` | — | not a hypervisor, amdgpu initialized |
| `profile` | — | Windows hypervisor only |
| `event` | — | amdgpu initialized |
| `topology` | — | amdgpu initialized |
| `set` | — | Linux |
| `reset` | — | Linux, amdgpu initialized |
| `monitor` | `dmon` | Linux, amdgpu initialized |
| `xgmi` | — | amdgpu initialized |
| `partition` | — | amdgpu initialized |
| `ras` | — | always |
| `node` | — | not a virtualized OS |
| `fabric` | — | amdgpu initialized |

Subcommand names SHALL NOT be abbreviated: the first token is matched against
the full name list before argparse is invoked. Aliases SHALL be exact
alternative spellings, not prefixes.

#### Scenario: An abbreviation is rejected rather than guessed

- **WHEN** a user runs `amd-smi stat`
- **THEN** the tool reports an invalid AMD-SMI command and exits 10, because a
  prefix match would make the surface silently expand every time a new
  subcommand is added

#### Scenario: A gated subcommand is distinguished from a typo

- **WHEN** a subcommand that exists in the name list is not registered on this
  platform — for example `bad-pages` outside Linux baremetal
- **THEN** the tool reports that the command is not supported on the system and
  exits 7, rather than reporting an invalid command, so a script can tell
  "wrong platform" from "wrong spelling"

#### Scenario: Output modifiers are not top-level flags

- **WHEN** a user runs `amd-smi --json`
- **THEN** it fails as an invalid command with exit 10, because `--json`,
  `--csv`, and `--file` are registered per subcommand, not on the root parser

### Requirement: Command-Line Case Normalization

Before parsing, the tool SHALL lowercase every long option and every positional
token, SHALL preserve the case of short options, and SHALL preserve the case of
the values following `--folder`, `--file`, `--gpu`, `--cpu`, `--core`,
`--profile`, and `--cper-file` (in both the space-separated and `--opt=value`
forms). Preserving short-option case is load-bearing because several
subcommands distinguish `-g` from `-G` and `-v` from `-V`.

#### Scenario: A path argument keeps its case

- **WHEN** a user passes `--file /tmp/MyRun.Json`
- **THEN** the file is created with that exact name, since case folding is
  suppressed for the value of a path-bearing option

#### Scenario: The subcommand token itself must be lowercase

- **WHEN** a user writes `amd-smi LIST`
- **THEN** the invocation fails, because parser construction inspects the
  original `sys.argv` to decide which subparser to register while parsing sees
  the folded token — the two disagree and the folded name is reported as an
  unsupported command

### Requirement: Three Output Formats, Two Of Them Machine-Readable

Every subcommand that renders data SHALL offer human-readable output (the
default), `--json`, and `--csv`, with `--json` and `--csv` mutually exclusive.
The machine-readable formats SHALL be a parsing contract:

- `--json` emits either a top-level array of per-device objects or, for the
  subcommands that report more than one kind of record (`static`, `metric`,
  `partition`, `xgmi`), a top-level object whose keys (`gpu_data`, `cpu_data`,
  `current_partition`, `xgmi_metric`, …) each hold such an array; each
  per-device object carries `gpu` — or `cpu`, `core`, `brcm_nic`, `ai_nic`,
  `brcm_switch` — as its device key;
- `--csv` emits a header row followed by one row per device;
- human-readable output emits an indented, upper-cased, YAML-like rendering and
  is explicitly not a parsing contract.

Units SHALL be carried differently per format: `--json` wraps a measured value
as `{"value": <n>, "unit": "<u>"}`, `--csv` emits the bare number and drops the
unit into the column name's implied meaning, and human-readable output emits
`<n> <u>`.

#### Scenario: Neither machine format requires splitting a string

- **WHEN** a script reads `power.socket_power` from `amd-smi metric`
- **THEN** `--json` gives `{"value": 12, "unit": "W"}` and `--csv` gives the
  bare `12`, either of which is usable as a number directly, whereas the
  human-readable `12 W` would have to be split

#### Scenario: Format selection is resolved before the parser runs

- **WHEN** a parse-time error occurs and `--json` was requested
- **THEN** the error is still rendered as JSON, because the output format is
  recovered by scanning `sys.argv` (accepting the abbreviations `--j` and
  `--c`) rather than from the parsed namespace

#### Scenario: Human-readable output suppresses empty clock data

- **WHEN** every field of a clock domain is unsupported
- **THEN** human-readable output collapses the domain to a single `N/A` while
  `--csv` still emits each column, so the two formats deliberately differ in
  density and only the machine-readable one is stable

### Requirement: Unsupported Values Render As The N/A Sentinel

When the library cannot supply a field — the ASIC lacks the sensor, the driver
does not export it, or the query returns an error — the CLI SHALL substitute the
string `N/A` for that field and SHALL NOT fail the command. In `--json` the
sentinel replaces the whole value, including a `{"value", "unit"}` object; in
`--csv` the cell is the bare token `N/A`; inside a list the sentinel replaces
the individual element. The underlying library error SHALL be recorded at debug
log level only.

For a field the library returns inside a successful call, the substitution is
performed by [amdsmi-python-api] against the all-ones sentinel for that field's
declared width, so `N/A` appears here only when the C layer presented the
sentinel at that width — the obligation stated in [amdsmi-c-api-abi]. A field
that loses its sentinel below this layer is rendered as a number, not as `N/A`,
and no part of the CLI can recover it.

#### Scenario: A consumer-class GPU reports most enterprise fields as N/A

- **WHEN** `amd-smi metric --json` runs on a workstation RDNA2 part
- **THEN** fields such as `usage.jpeg_activity`, `usage.vcn_busy`, and
  `power.ubb_power` are the string `"N/A"` while the supported fields carry
  their `{"value", "unit"}` objects, and the command still exits 0

#### Scenario: One unsupported field does not hide the rest

- **WHEN** a single query in a multi-field subcommand raises a library
  exception
- **THEN** only that field becomes `N/A`; the remaining fields are collected and
  printed, so a partially supported device still yields usable telemetry

#### Scenario: Type stability is sacrificed for presence

- **WHEN** a JSON consumer reads a numeric field
- **THEN** it must accept either an object/number or the string `"N/A"`, because
  the key is always present rather than omitted — omission would break
  fixed-schema consumers worse than a type change

#### Scenario: A sentinel lost below the CLI is printed as data

- **WHEN** an unreported 64-bit counter reaches this layer holding `4294967295`
  because its 32-bit sentinel was widened by a plain cast
- **THEN** the CLI prints `4294967295`, not `N/A` — the sentinel comparison is
  the only thing that distinguishes "not reported" from a reading, and it has
  already failed by the time the value arrives here, so `N/A` rendering is a
  consequence of the contract below the CLI and not an independent check

### Requirement: CSV Flattens Nested Data And Unifies Columns

`--csv` SHALL flatten nested dictionaries into a single level, joining a nested
key to its parent with `_`, and SHALL emit a single header whose columns are the
union of the keys seen across all rendered devices, filling any key a given
device lacks with `N/A`. List-valued fields SHALL be emitted as their Python
list representation in one cell.

#### Scenario: Heterogeneous devices still produce one rectangular table

- **WHEN** two devices in the same invocation expose different key sets
- **THEN** the header is their union and the missing cells are `N/A`, so the
  output stays loadable as a single table rather than changing width mid-file

### Requirement: Rendered Keys Are A Stable Part Of The Output Schema

The key a value is rendered under SHALL be part of the machine-readable output
contract, on the same footing as the value itself. One value SHALL carry one
key in every subcommand that emits it, and a key SHALL NOT change as a side
effect of an internal rename — neither a renamed C entry point, a renamed
Python function, nor a renamed argparse destination moves the key a consumer
reads. Where the two would disagree, the emitted key is what is held stable and
the internal name is free to move.

Changing a key is a breaking change to every consumer that indexes it, so it
SHALL be made deliberately: as its own change, moving the flag, every emitting
subcommand and the documentation together, and recorded in `CHANGELOG.md`.
Per-format naming differences are permitted only where they are specified, as
the `list` identity columns below are.

#### Scenario: An API rename does not move the key underneath a consumer

- **WHEN** the memory allocation mode moves from the `compute_partition` C API
  family to the `accelerator_partition` family, and `amd-smi set` is updated to
  call the new entry point
- **THEN** `amd-smi set --json` still emits `compute_partition_mem_alloc_mode`,
  matching what `amd-smi static --partition --json` emits for the same value
  and what `--compute-partition-mem-alloc-mode` is called on the command line;
  had the key followed the C rename, one field would have had two names
  depending on which subcommand produced it

#### Scenario: CSV renames the identity columns of `list`

- **WHEN** `amd-smi list --csv` runs
- **THEN** the BDF and UUID columns are named `gpu_bdf` and `gpu_uuid`, while
  `--json` and human-readable output name them `bdf` and `uuid`; the CSV names
  are deliberately aligned with the host-side tool and must not be "fixed" to
  match the other formats

### Requirement: Device Selection By Index, BDF, UUID, Or CUID

`-g`/`--gpu` SHALL accept one or more device designators, each of which is an
enumeration index, a PCI BDF, a UUID, or a CUID, plus the literal `all`, which
selects every processor handle. BDF comparison SHALL be structural, accepting
both `BB:DD.F` and `SSSS:BB:DD.F` and treating a missing segment as `0000`.
UUID and CUID comparison SHALL be case-insensitive. Which of these designators
survives a reboot, and which is a kernel-assigned ordinal that does not, is
specified in [amdsmi-device-discovery]. Sibling device classes —
`-U`/`--cpu`, `-O`/`--core`, `-N`/`--nic`, `--switch` — SHALL be mutually
exclusive with `--gpu` and SHALL be registered only when their driver is
initialized. Omitting the device argument SHALL select all devices of the
subcommand's default class.

#### Scenario: A short BDF resolves to the same device as the full BDF

- **WHEN** a user passes `--gpu 03:00.0` for a device reported as
  `0000:03:00.0`
- **THEN** the same device is selected, because the segment defaults to zero
  rather than the strings being compared literally

#### Scenario: An absent device is distinguished from a malformed one

- **WHEN** `--gpu 9` names an index that does not exist
- **THEN** the tool reports "Can not find a device: GPU '9'" and exits 3,
  whereas `--gpu zzz` — which is neither a digit, a UUID, nor a parseable BDF —
  reports an invalid value and exits 5, so automation can tell a hot-unplugged
  or renumbered device from a bad script variable

#### Scenario: Device flags absent from the help are absent from the machine

- **WHEN** the `amd_hsmp` driver is not loaded
- **THEN** `-U`/`--cpu` is not registered at all, so `--help` output is an
  accurate description of what this host accepts rather than a superset

### Requirement: Exit Codes Derived From The Exception Hierarchy

Every CLI-level failure SHALL be an `AmdSmiException` subclass carrying a
negative `value`, and the process SHALL exit with the absolute value of that
number. The mapping SHALL be:

| Condition | Exception | Exit |
| --------- | --------- | ---- |
| Unrecognized command name | `AmdSmiInvalidCommandException` | 1 |
| Unrecognized or misused option | `AmdSmiInvalidParameterException` | 2 |
| Device designator not found | `AmdSmiDeviceNotFoundException` | 3 |
| Output/input path unusable | `AmdSmiInvalidFilePathException` | 4 |
| Option value out of range or wrong type | `AmdSmiInvalidParameterValueException` | 5 |
| Option given without its value | `AmdSmiMissingParameterValueException` | 6 |
| Command not supported on this system | `AmdSmiCommandNotSupportedException` | 7 |
| Option not supported on this system | `AmdSmiParameterNotSupportedException` | 8 |
| Subcommand needs a target argument | `AmdSmiRequiredCommandException` | 9 |
| First token is not a subcommand | `AmdSmiInvalidSubcommandException` | 10 |
| Operation requires elevation | `AmdSmiPermissionDeniedException` | 11 |
| Unclassified failure | `AmdSmiUnknownErrorException` | 100 |
| Library returned a status code `n` | `AmdSmiLibraryErrorException` | `1000 + n` |

An `amdsmi` library exception escaping a subcommand SHALL be re-wrapped as
`AmdSmiLibraryErrorException`, and a raw `PermissionError` SHALL be re-wrapped
as `AmdSmiPermissionDeniedException`, so no failure reaches the user as an
unhandled traceback.

#### Scenario: A script branches on the reason, not on the message

- **WHEN** a monitoring job runs `amd-smi static --gpu $BDF` against a device
  that has disappeared
- **THEN** it sees exit 3 specifically, and can retry discovery instead of
  treating the failure like the exit 2 it would get from a stale flag

#### Scenario: Library error codes exceed the byte a shell can report

- **WHEN** the library fails with status `n` and the tool exits `1000 + n`
- **THEN** the shell observes that value modulo 256, so the machine-readable
  `code` field in the error payload — not `$?` — is the reliable carrier of a
  library status

#### Scenario: Tracebacks are suppressed unless debugging

- **WHEN** any failure occurs at a log level other than `DEBUG`
- **THEN** the traceback limit is set to `-1` and the user sees a single
  diagnostic line; `--loglevel DEBUG` restores a bounded traceback

### Requirement: Diagnostics On stdout, Logs On stderr

The tool SHALL write both data and error diagnostics to the logger destination —
stdout by default — and SHALL reserve stderr for the Python logging stream
(`--loglevel`). An error diagnostic SHALL be rendered in the currently selected
output format: a JSON object with `error` and `code` keys under `--json`, an
`error,code` header plus one row under `--csv`, and a sentence ending in
`Error code: <n>` otherwise. Each diagnostic line SHALL be prefixed with the
fully qualified exception class name.

#### Scenario: A JSON consumer can parse the failure as well as the success

- **WHEN** `amd-smi static --gpu 9 --json` fails
- **THEN** the payload after the class-name prefix is
  `{"error": "Can not find a device: GPU '9'", "code": -3}`, so a wrapper can
  extract a structured reason instead of scraping prose

#### Scenario: Redirecting stdout does not lose the error

- **WHEN** a caller redirects only stdout to a file
- **THEN** the error text lands in that file alongside where the data would
  have gone, and stderr stays empty for a clean run — which also means a
  caller that watches only stderr sees nothing on failure

#### Scenario: Parse-time failures ignore `--file`

- **WHEN** a command specifies `--file` but fails while the command line is
  still being parsed
- **THEN** the diagnostic goes to stdout, because the logger destination is
  switched to the file only after the whole command line parses successfully

### Requirement: Continuous Watch Modes

`metric`, `process`, and `monitor` SHALL support `-w`/`--watch INTERVAL` to
repeat the command every INTERVAL seconds, bounded optionally by
`-W`/`--watch_time TIME` or `-i`/`--iterations N`. `-W` and `-i` SHALL be
rejected unless `-w` is also given. With neither bound, the loop SHALL run until
interrupted. `ras --follow` SHALL provide the equivalent for CPER entries, and
`event` SHALL stream until `q` is entered or the process is signalled. All
interval, time, and iteration values SHALL be positive integers.

#### Scenario: A bound without an interval is an error, not a silent single run

- **WHEN** a user runs `amd-smi monitor -i 2` without `-w`
- **THEN** the command fails with exit 2, because silently running once would
  make a mistyped monitoring job look healthy

#### Scenario: Watch output announces itself on stdout

- **WHEN** a watch loop starts
- **THEN** `'CTRL' + 'C' to stop watching output:` is printed before the first
  record, so a consumer that pipes watch output must skip that line

#### Scenario: SIGINT and SIGTERM end the loop cleanly

- **WHEN** the process is interrupted or terminated during a watch loop
- **THEN** the installed handlers exit through the normal path, which runs the
  registered `amdsmi_shut_down()` at exit rather than leaving the library
  initialized

### Requirement: File Destination With Explicit Collision Handling

`--file PATH` SHALL redirect rendered output to PATH. When PATH is an existing
directory, the tool SHALL create `<epoch>-amdsmi-output` in it with the
extension matching the selected format. When PATH is an existing file, the tool
SHALL require `--overwrite` or `--append`; absent either, it SHALL prompt on an
interactive terminal and SHALL fail rather than prompt when stdin is not a TTY.
When PATH's parent directory does not exist, the tool SHALL fail with exit 4.
For a watch run the file SHALL be rewritten from the accumulated buffer on
completion rather than appended to per iteration.

#### Scenario: Automation never blocks on a prompt

- **WHEN** a CI job runs `amd-smi metric --json --file results.json` with stdin
  piped or closed and the file already exists
- **THEN** the command fails immediately with exit 4 and a message naming
  `--overwrite` and `--append`, instead of hanging forever on a read from a
  pipe that will never deliver a line

#### Scenario: Appending produces concatenated documents, not one document

- **WHEN** two `--json --file out.json --append` runs are made against the same
  path
- **THEN** the file holds two JSON arrays back to back and is not a single
  parseable document, so consumers that need one document must use
  `--overwrite` or one file per run

#### Scenario: A watch run leaves one header, not one per iteration

- **WHEN** `--csv --file` is combined with `-w`
- **THEN** the file is truncated and written once at loop completion with a
  single header row, so an interrupted-and-restarted watch does not interleave
  headers into the data

### Requirement: Initialization Failure Behavior

Before any subcommand runs, the CLI SHALL probe for the amdgpu, HSMP, ionic, and
bnxt_en drivers and initialize the library with a flag set derived from what is
live. Initialization SHALL run on a worker thread with a 60-second timeout.
Failure SHALL be reported as a log line on stderr and a non-zero exit, never as
a traceback:

| Situation | Behavior |
| --------- | -------- |
| No supported driver live, or the library reports not-initialized / driver-not-loaded | Log "Drivers not loaded (…)" and exit 255 |
| `amdsmi_init()` does not return within 60 s | Log a timeout naming an unresponsive driver and exit 2 |
| Any other library exception during init | Re-raised and reported through the library-error path |
| Driver live but zero devices enumerated | Print the `version` output, then exit 255 |

#### Scenario: An unresponsive driver does not hang a monitoring agent forever

- **WHEN** `amdsmi_init()` blocks in the kernel
- **THEN** the CLI abandons the init thread after 60 seconds and exits 2, which
  bounds the damage a wedged GPU does to a scheduled job

#### Scenario: The version view still works when no devices are found

- **WHEN** the driver is live but enumeration yields nothing
- **THEN** the tool prints version information before exiting non-zero, so the
  user can see which AMD SMI and ROCm they are running while diagnosing

#### Scenario: Cache lifetimes are tunable but defaulted

- **WHEN** the CLI starts
- **THEN** it sets `AMDSMI_GPU_METRICS_CACHE_MS=100`,
  `AMDSMI_ASIC_INFO_CACHE_MS=10000`, and `AMDSMI_PROCESS_INFO_CACHE_MS=100`
  only if unset, so a caller can widen the caches for a tight watch loop
  without patching the tool

### Requirement: Mutating Subcommands Require Elevation And Say So

Nearly every `set` option, and `reset`, writes a sysfs node the kernel exposes
to root only. When the library refuses such a write with
`AMDSMI_STATUS_NO_PERM`, the CLI SHALL surface the refusal as a failed command
— raised as a `PermissionError`, re-wrapped as
`AmdSmiPermissionDeniedException`, exit 11 — and SHALL NOT report the setting
as applied. A refusal is therefore a failure, not a skip, and an unprivileged
invocation of a mutating subcommand is expected to fail.

That makes the CLI test suite the thing enforcing this contract, and its
obligation is symmetric: the `set` coverage SHALL skip when the suite runs with
a non-zero effective UID, and SHALL exercise the commands when it runs
privileged, which is the mode the suite's own runner requires. Neither half is
optional — skipping unconditionally would leave the mutating surface untested,
and running unconditionally turns correct tool behavior into a test failure.
Only the few options that additionally need interactive input are excluded from
the privileged run as well.

#### Scenario: An unprivileged set fails rather than silently doing nothing

- **WHEN** a non-root user runs `amd-smi set --gpu 0 --perf-level high`
- **THEN** the command exits 11 reporting that elevation is required, and no
  output claims the level was set, so a configuration script fails at the point
  of refusal instead of continuing as though the device had been reconfigured

#### Scenario: An unprivileged suite run skips rather than fails

- **WHEN** the CLI tests are run without root
- **THEN** the `set` coverage is skipped and the run is green, because the
  commands it drives are all expected to be refused; a suite that reported
  those refusals as failures would be red on every developer workstation and
  would stop signalling anything

### Requirement: Device-Node Permission Advisory

When not running as root, subcommands that touch device state SHALL check read
access to `/dev/kfd` and each `/dev/dri/renderD*`, and SHALL print, once per
invocation, a plain-text advisory naming the owning groups and the `usermod`
command that would grant access. The check SHALL be advisory: the command
continues and renders whatever it can.

#### Scenario: The advisory is plain text in every output format

- **WHEN** the advisory triggers under `--json`
- **THEN** it is printed as plain text ahead of the JSON document, so a
  consumer that runs unprivileged must tolerate a preamble before the first
  `[`

### Requirement: Bounded rocm-smi Compatibility Mode

`amd-smi --rocm-smi` SHALL render the concise device table of
`rocm-smi --showallconcise` and nothing else. The compatibility layer SHALL be a
separate module (`amdsmi_rocm_smi_compat.py`) that performs its own
`amdsmi_init()` and `amdsmi_shut_down()`, SHALL read clocks, power, and
utilization from sysfs first to avoid waking a runtime-suspended GPU, and SHALL
warn when a device is in a low-power state. It SHALL NOT implement any other
`rocm-smi` subcommand or option, and SHALL NOT support JSON or CSV output.

#### Scenario: The mode is a table, not a rocm-smi emulator

- **WHEN** a script expects `rocm-smi --showtemp` or `--showmeminfo` to work
  through this flag
- **THEN** it does not; only the concise table is provided, so migration off
  `rocm-smi` still requires mapping each query onto an `amd-smi` subcommand

#### Scenario: Machine-readable output is rejected, not silently ignored

- **WHEN** `amd-smi --rocm-smi --json` is run
- **THEN** the invocation fails with exit 2, so a script cannot mistake a
  human-readable table for the JSON it asked for

#### Scenario: The compatibility layer carries its own version identity

- **WHEN** the module reports a version
- **THEN** it is `4.0.0+amdsmi`, tracking the emulated rocm-smi generation
  rather than the AMD SMI tool version, so the two must not be conflated
