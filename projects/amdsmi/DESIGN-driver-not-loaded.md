# AMD SMI CLI: Device-Agnostic Driver Handling

> **Temporary review copy. Delete this file before PR #12365 merges.** It is in the PR only so
> reviewers can comment on the design inline.

| | |
|---|---|
| **Status** | Implemented in PR #12365 and validated on a Helios MI455X system |
| **Scope** | `projects/amdsmi/amdsmi_cli` only: Python CLI, no library or ABI change, no automatic driver loading |
| **Decided by** | Adam Pryor with Maisam Arif, 2026-10-08 |

## Problem

amd-smi starts with whichever AMD drivers are loaded: amdgpu (GPU), hsmp_acpi or amd_hsmp (CPU),
ionic (AI NIC) and bnxt_en (Broadcom NIC). When one is missing but its device is present, the result
depends on the command, and none of them names the driver. Measured on develop on Helios (GPU,
Pensando AI NIC, EPYC CPU):

| Missing driver | Today |
|---|---|
| amdgpu | `list`, `firmware`, `process` and the other GPU commands say "not supported" (200), even `list -N all` while the AI NIC driver is loaded. `-g` and GPU options such as `static --asic` say "Parameter is invalid" (195). `static` and `metric` silently leave out the GPU. Bare `amd-smi` prints nothing |
| ionic | `list` and `static` silently leave out the AI NIC. `-N` says "Parameter is invalid" (195) |
| CPU driver | `static` and `metric` silently leave out the CPU. `-U` and `-O` say "Parameter is invalid" (195) |
| any | `version` shows "N/A", or leaves the line out |

## Decisions

1. **One rule set for every device type.** GPU, CPU, AI NIC and Broadcom NIC are handled the same
   way, and each command checks only the drivers it uses.
2. **Run with what is loaded.** A command shows the devices whose drivers are loaded and prints a
   note for each missing driver. The exit status stays 0. Not taken: a non-zero exit, which would
   fail monitoring on nodes that simply lack an optional driver.
3. **Notes go to stderr in every output format,** so JSON, CSV, pipes and `--file` output stay
   clean. Not taken: notes inside the JSON or CSV output, which changes its shape.
4. **Fail only when the command has nothing to show,** or when the user asks for a device type or
   option whose driver is missing, and only if that hardware is present. The error names the driver
   and reuses exit code 206 (`DRIVERS_NOT_LOADED`). Not taken: a new code 208, or one code per device type.
5. **No hardware, no change.** Without the device, keep "not supported" (200) or "Parameter is
   invalid" (195). Platform restrictions (hypervisor, VM, Windows, Linux-only) also keep "not
   supported". Hardware detection reads Linux sysfs, so Windows and WSL are unchanged.
6. **A CPU note needs the ACPI HSMP device (`AMDI0097`).** A driver counts as loaded when amd-smi
   initialized it or the kernel shows it loaded, so a CPU driver turned off with
   `AMDSMI_DISABLE_CPU_INIT=1` produces no notes or errors. Not taken: noting any EPYC CPU (noisy on
   GPU servers that never load amd_hsmp), or noting only when `-U` or `-O` is used.
7. **`metric` also notes a missing ionic driver,** although it shows no AI NIC data yet. Not taken:
   noting only the drivers `metric` reports.
8. **Broadcom NIC is included, untested** (no hardware): a Broadcom network device with no driver
   bound counts as present. Broadcom switches are left out because there is no reliable way to
   detect them without the driver.
9. **`version` shows "not loaded"** instead of "N/A" for a driver that is not loaded.
10. **Everything ships in one PR,** including `version` and the Broadcom options in `firmware`,
    `monitor` and `topology`. Not taken: phased delivery.

## Behavior

| Command | Device types | New behavior |
|---|---|---|
| `list` | GPU, AI NIC, Broadcom NIC | Runs with any of its drivers (was amdgpu only); notes for the rest |
| `static`, `metric` | GPU, CPU, AI NIC, Broadcom NIC | Notes for missing drivers |
| `firmware`, `monitor`, `topology` | GPU, Broadcom NIC | Without amdgpu, run only their Broadcom options, otherwise the amdgpu error; notes |
| `set` | GPU, CPU | Options name the missing driver; no notes (action command) |
| `process`, `event`, `reset`, `xgmi`, `partition`, `fabric`, `bad-pages`, `ras` | GPU | The amdgpu error instead of "not supported" |
| bare `amd-smi` | GPU | The amdgpu error instead of empty output |
| `version` | all | "not loaded" |
| `node`, `profile` | - | Unchanged |

Notes cover only the device types the user selected with `-g`, `-U`/`-O` or `-N`, if any. "Options"
means device selectors (`-g`, `-U`, `-O`, `-N`) and driver-specific options such as `static --asic`
or `set --fan`.

| Case | Text |
|---|---|
| Note (stderr) | `Note: GPU 0001:01:00.0: amdgpu driver not loaded (sudo modprobe amdgpu)` |
| Command error | `Command 'process' requires the amdgpu driver but it is not loaded. Check amdgpu version and module status (sudo modprobe amdgpu).` |
| Option error | `Parameter '-N' requires the ionic driver but it is not loaded. Check ionic version and module status (sudo modprobe ionic).` |
| Several drivers | `Command 'list' requires the amdgpu or ionic driver but none is loaded. ...` |
| Bare `amd-smi` | `amd-smi requires the amdgpu driver but it is not loaded. ...` |
| `version` | `amdgpu version: not loaded` |

Errors keep today's human, JSON and CSV error format on stdout.

## How it works

- **Driver table and hardware checks** (`amdsmi_helpers.py`): one table maps each driver to its
  device label and to how its hardware is found without the driver. GPUs and NICs come from the PCI
  vendor and class in sysfs: AMD 0x1002 with class 0x03 or 0x12 for GPUs, Pensando 0x1dd8 or
  Broadcom 0x14e4 with class 0x02 for NICs. The CPU comes from the ACPI device `AMDI0097`. Devices
  bound to another driver (e.g. vfio-pci, tg3) don't count. The checks use no new names from
  `amdsmi_init`, which many tests replace with a stub.
- **Explaining rejected input** (`amdsmi_parser.py`): when argparse rejects a command or option,
  `error()` rebuilds the parser once per missing driver, with that driver reported as loaded and
  having no devices. If the command or option exists in the rebuilt parser, the error names that
  driver. There is no option list to maintain, the rebuild runs only after argparse has already
  failed, and any failure during it keeps today's error. This replaces the hand-kept command list of
  the PR's first version.
- **Availability:** `list` is built when amdgpu or a NIC driver is loaded. `firmware`, `monitor` and
  `topology` are also built with only a Broadcom NIC or switch, and then run only their Broadcom
  options until amdgpu is loaded.
- **Notes** (`amdsmi_cli.py`): each read command declares the drivers it uses, and the CLI prints
  the notes to stderr before running the command.
- **Errors:** one exception, `AmdSmiDriverNotLoadedException` (206), for commands, options and
  bare `amd-smi`.

## Compatibility

- Systems with every relevant driver loaded: no change.
- Exit codes change only when a present device's driver is missing: 200 to 206 (GPU commands), 195
  to 206 (options), 200 to 0 (`list` that can still show an AI NIC), 0 to 206 (bare `amd-smi`).
- Notes are new and go to stderr; stdout is unaffected.
- `version` shows "not loaded" instead of "N/A" for a driver that is not loaded.

## Validation

Helios MI455X (GPU, Pensando AI NIC, EPYC with hsmp_acpi), ROCm 10.2.0 CLI with the changed files.
Each driver was hidden from amd-smi in a private mount namespace while staying loaded for other users;
output checked in human, JSON and CSV.

| Hidden | Result |
|---|---|
| amdgpu | `list` shows the AI NIC with a GPU note, exit 0, valid JSON. `process`, `firmware` and bare `amd-smi` exit 206. `list -g 0`, `static --asic` and `set --fan 50` exit 206. `static` and `metric` add a GPU note. `version` shows "not loaded". `profile` still exits 200 |
| ionic | `list`, `static` and `metric` note the AI NIC. `-N all` exits 206 naming ionic |
| `/dev/hsmp` | `static` and `metric` note the CPU. `-U all` and `-O all` exit 206 naming hsmp_acpi. `list` is unchanged |
| CPU init off, driver loaded | No notes; `-U all` still exits 195 |
| All PCI devices | No notes; `process` exits 200, `-g 0` exits 195 |
| Nothing | No change |

Unit tests pass on Python 3.6, 3.8, 3.13 and 3.14, the full unit suite has no new failures against
develop, and each of seven deliberate breakages of the new behavior fails a test. Architecture,
test, skeptic and docs reviews were run and their suggestions applied.

## Limitations and follow-ups

- Abbreviated long options (e.g. `--mon`) still get "Parameter is invalid".
- The Broadcom NIC paths are unit-tested only; Broadcom switches need a reliable hardware signal.
- `metric` has no AI NIC data yet.
- Found on develop and left unchanged: `list --json` with both GPUs and AI NICs prints two JSON
  arrays, so the output is not valid JSON; `version` prints `ionic version: N/A, ionic.<version>`.
