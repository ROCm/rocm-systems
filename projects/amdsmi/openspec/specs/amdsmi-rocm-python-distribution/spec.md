# amdsmi-rocm-python-distribution Specification

## Purpose

Defines how the `core-amdsmi` artifact becomes part of the ROCm pip distribution
— the `rocm-sdk-core` wheel installed by `pip install rocm[...]` — and what that
transformation does to AMD SMI's on-disk layout.

This is a *different* Python channel from the standalone `amdsmi` PyPI wheel in
[amdsmi-python-wheel]. That wheel ships AMD SMI alone with a private
`libamd_smi_python.so` and makes `import amdsmi` work. This one ships AMD SMI as
part of a whole relocated ROCm tree inside site-packages, reaches the library
through the relative path contract in [amdsmi-install-layout], and delivers the
`amd-smi` command rather than an importable module. Artifact slicing is in
[amdsmi-therock-artifact]; the native packages cut from the same artifact are
[amdsmi-rocm-os-packages].

The transformation is not a copy: runtime wheels may contain no symlinks, so
every symlink in the artifact is resolved, dropped, stubbed, or flattened. AMD
SMI has one of each interesting case, and the resulting `amd-smi` invocation
path passes through four separate resolution steps before it loads the library.

## Requirements

### Requirement: AMD SMI Lands In The Core Package

The `lib` and `run` components of `core-amdsmi` SHALL be selected into the
`rocm-sdk-core` wheel. AMD SMI SHALL NOT appear in the arch-specific
`rocm-sdk-libraries-*` or `rocm-sdk-device-*` wheels.

#### Scenario: AMD SMI installs with the required core package

- **WHEN** a user installs any ROCm pip configuration
- **THEN** AMD SMI is present, because `rocm-sdk-core` is a required package of
  the `rocm` selector rather than an optional extra

#### Scenario: No per-GPU duplication

- **WHEN** a user installs device extras for several GPU targets
- **THEN** AMD SMI is installed once, because its artifact is target-neutral and
  its components are routed to the arch-neutral core package

#### Scenario: Files land in the core platform directory

- **WHEN** the wheel is installed
- **THEN** the flattened prefix (`bin/`, `lib/`, `libexec/`, `share/`) lives
  under a version-nonce-suffixed platform package directory (`_rocm_sdk_core…`)
  alongside the pure `rocm_sdk_core` package, and both are siblings of
  `rocm_sdk` in the same site-packages

### Requirement: Runtime Wheels Contain No Symlinks

Every symlink in the artifact SHALL be resolved during wheel population, using
one of four rules: a dangling or directory symlink is dropped; a shared-library
symlink is materialized only when its own name equals the target's SONAME; a
symlink to an ELF executable becomes a compiled launcher stub; anything else is
copied as a regular file.

The same SONAME rule SHALL apply to regular `.so` files: a shared library whose
own filename differs from its SONAME is recorded as an alias rather than
materialized. Together the two rules mean exactly one file per SONAME survives.

#### Scenario: Only the SONAME library file survives

- **WHEN** the artifact contains `lib/libamd_smi.so.<X.Y.Z>` (the real file,
  whose SONAME is `libamd_smi.so.<MAJOR>`) plus the symlinks
  `libamd_smi.so.<MAJOR>` and `libamd_smi.so`
- **THEN** the wheel contains exactly one regular file,
  `lib/libamd_smi.so.<MAJOR>`, holding the resolved contents — the versioned
  real name and the unversioned dev link are both recorded as SONAME aliases and
  not materialized

#### Scenario: The loader's hardcoded SONAME is the only name that resolves

- **WHEN** the wrapper looks for `<root>/lib/<SONAME>` in a ROCm pip install
- **THEN** it finds the file, because the loader's `_AMDSMI_LIB_SONAME` is
  precisely the one name the wheel materialized; an unversioned `libamd_smi.so`
  lookup would find nothing

#### Scenario: The unversioned dev link is restored only in the devel package

- **WHEN** `rocm-sdk-devel` is installed
- **THEN** it re-creates `lib/libamd_smi.so` as a symlink to the SONAME alias
  already materialized by the core package, so `find_package(amd_smi)` and
  link-time consumers work

### Requirement: The bin/amd-smi Symlink Becomes A Copied Python Script

`bin/amd-smi` is a symlink to `../libexec/amdsmi_cli/amdsmi_cli.py`, a Python
script rather than an ELF executable. It SHALL therefore be materialized by the
copy rule, not the executable-stub rule.

#### Scenario: No launcher stub is compiled for amd-smi

- **WHEN** the wheel is populated
- **THEN** the file-type probe, which recognizes an executable only by the ELF
  magic description, classifies the link target as a script, so `bin/amd-smi`
  becomes a regular copy of `amdsmi_cli.py` retaining its
  `#!/usr/bin/env python3` shebang

#### Scenario: The copy still finds its sibling CLI modules

- **WHEN** the copied `bin/amd-smi` runs, so `sys.path[0]` is `<platform>/bin`
  and the flat `from amdsmi_init import *` imports fail
- **THEN** the CLI's fallback appends `<platform>/bin/../libexec/amdsmi_cli` to
  `sys.path` and the imports succeed — this fallback is what makes the copied
  entry point work at all

#### Scenario: Moving the CLI out of libexec would break the wheel

- **WHEN** the CLI's install location changes relative to `bin/`
- **THEN** the `../libexec/amdsmi_cli` fallback no longer resolves and `amd-smi`
  fails in the ROCm pip channel, even though the deb/rpm channel is unaffected

### Requirement: The amd-smi Console Script Re-Execs The Copied Entry Point

The `rocm-sdk-core` wheel SHALL declare a `console_scripts` entry point named
`amd-smi`, available on Linux only, which resolves the core platform directory
and `exec`s `bin/amd-smi` from it, forwarding all arguments. It SHALL request no
devel -tree expansion.

#### Scenario: amd-smi is on PATH inside a virtual environment

- **WHEN** ROCm is pip-installed into a venv
- **THEN** the venv's `bin/amd-smi` shim is on `PATH`, and running it transfers
  control to the copied CLI inside the platform directory

#### Scenario: The system-info tool does not pay the devel expansion cost

- **WHEN** `amd-smi` is invoked in an environment that also has `rocm-sdk-devel`
  installed but not yet expanded
- **THEN** the devel tarball is not expanded, because AMD SMI needs only runtime
  files — unlike compiler entry points, which default to triggering expansion

#### Scenario: The process is replaced rather than wrapped

- **WHEN** the shim runs on Linux
- **THEN** it `execv`s the target, so exit status, signals, and terminal
  behavior are those of the CLI itself

### Requirement: The Full CLI Resolution Chain Closes Inside site-packages

From the console script to the loaded library, the chain SHALL resolve entirely
within the installed platform directory, with no reference to `/opt/rocm`:

1. `amd-smi` console script → `<platform>/bin/amd-smi`;
2. CLI import fallback → `<platform>/libexec/amdsmi_cli`;
3. CLI module resolution → `<platform>/share/amd_smi`;
4. wrapper library lookup → `<platform>/lib/<SONAME>`.

#### Scenario: A venv install gives the command but not the import

- **WHEN** ROCm is pip-installed into a venv on a machine with no `/opt/rocm`
  and no `ld.so.conf.d` entry
- **THEN** `amd-smi` works, provided the amdgpu kernel driver is present, but a
  plain `import amdsmi` still fails: step 3 above is performed by the CLI
  itself, and the wheel installs no `.pth` file and no site-packages copy of the
  module. A script must add `<platform>/share/amd_smi` to `sys.path` first

#### Scenario: Each hop is a fixed relative path

- **WHEN** the platform directory is relocated (a different site-packages, a
  moved venv)
- **THEN** every hop still resolves, because each is computed relative to the
  file performing the lookup rather than from an absolute prefix

#### Scenario: Breaking any hop breaks only this channel

- **WHEN** one of the four relative relationships changes
- **THEN** the ROCm pip channel fails while the deb/rpm and standalone-wheel
  channels continue to work, so this chain needs its own verification

### Requirement: The Library Is Registered For Programmatic Lookup

The distribution SHALL register AMD SMI as a known library of the core package,
under the short name `amd_smi` with the glob `libamd_smi.so*`, so it can be
located and preloaded through the `rocm_sdk` API.

#### Scenario: A framework resolves the library path

- **WHEN** a consumer calls the SDK's library-finding entry point with the
  `amd_smi` short name
- **THEN** it receives an OS-independent absolute path to the installed
  `libamd_smi.so*`, without needing to know the platform directory layout

#### Scenario: Preloading works alongside other ROCm libraries

- **WHEN** a framework's initialization preloads ROCm libraries by short name
- **THEN** `amd_smi` is resolvable through the same mechanism as the other core
  libraries

### Requirement: The Installation Self-Test Exercises amd-smi

The bundled `rocm-sdk` self-test SHALL include `amd-smi` in its Linux console
script checks and SHALL treat its absence or failure as a test failure.

#### Scenario: The self-test proves the whole chain

- **WHEN** `rocm-sdk test` runs on Linux
- **THEN** it invokes `amd-smi` with no arguments and requires `AMD-SMI` in the
  output — which passes only if all four resolution hops and the library load
  succeeded

#### Scenario: The check is mandatory, not best-effort

- **WHEN** `amd-smi` is missing from the installation
- **THEN** the test fails rather than skipping, because the entry is marked
  required

### Requirement: Package Versions Follow The ROCm Release Channel

The wheels carrying AMD SMI SHALL be versioned with the ROCm SDK version for
their release channel, not with AMD SMI's own library version:

| Release type | Wheel version |
| ------------ | ------------- |
| release | `X.Y.Z` |
| prerelease | `X.Y.ZrcN` |
| nightly | `X.Y.ZaYYYYMMDD` |
| ci, dev | `X.Y.Z.dev0+<commit>` |

Only the nightly BKC variant adds a `+bkc.<date>` segment
(`X.Y.ZaYYYYMMDD+bkc.<date>`); the dev BKC variant is deliberately identical to
the plain `ci`/`dev` form above and adds no segment. The native package
versions cut from the same artifact use a different scheme — see
[amdsmi-rocm-os-packages].

#### Scenario: AMD SMI's library version is independent of the wheel version

- **WHEN** a user inspects an installed ROCm pip distribution
- **THEN** the wheel version reports the ROCm SDK version while
  `amd-smi version` reports AMD SMI's own library version, and the two are
  expected to differ

#### Scenario: Channels are separated by index

- **WHEN** a user installs from a nightly or dev index rather than the stable
  one
- **THEN** they receive that channel's version series, and version ordering
  (stable > nightly > dev) is preserved by the PEP 440 forms used

### Requirement: Binary Compatibility Comes From The manylinux Build

The AMD SMI binaries shipped in these wheels SHALL be produced by TheRock's
manylinux build, so they run on any glibc-based distribution at or above the
image's glibc baseline without further alteration. They carry the vendored
sysdeps described in [amdsmi-therock-subproject], so a host `libdrm` or `libnl`
loaded into the same process cannot collide with them.

#### Scenario: Building the wheels outside the container is discouraged

- **WHEN** the packaging step runs on a host rather than in the portable
  container
- **THEN** the result risks referencing too-new glibc symbols, so the documented
  practice is to build packages in the same container that built the SDK
