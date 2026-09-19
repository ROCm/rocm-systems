# amdsmi-python-loader Specification

## Purpose

Defines how a Python process resolves the `amdsmi` module and the native
`libamd_smi*.so` behind it, across every delivery channel AMD SMI ships.

The loader lives in `py-interface/amdsmi_wrapper.py`, emitted as a fixed
template by `tools/generator.py` and committed to the tree. Every ctypes binding
in that file keys on the dictionary entry `_libraries['libamd_smi.so']`, so the
loader is the single point where a delivery channel is distinguished. It
deliberately implements a short, fixed sequence: no `ROCM_PATH`/`ROCM_HOME`
ladders, no `LD_LIBRARY_PATH` probing, no `.pth` tricks, and no combining of
paths from two channels.

Detailed behavior of each channel lives in the sibling capabilities:
[amdsmi-python-wheel], [amdsmi-python-system-package],
[amdsmi-rocm-python-distribution], and [amdsmi-rocm-os-packages]. The
prefix-relative tree that step 3 resolves against is [amdsmi-install-layout];
this capability ends the moment a handle to the library exists.

## Requirements

### Requirement: Supported Delivery Channels

AMD SMI SHALL deliver the Python module through exactly the four channels below,
and each channel SHALL be self-contained — a user installs one of them.

| Channel | Native library | Python module location | Resolution mechanism |
| ------- | -------------- | ---------------------- | -------------------- |
| `amd-smi-lib` deb/rpm | `<prefix>/lib/libamd_smi.so.<MAJOR>` plus an `ld.so.conf.d` entry | system interpreter's `site-packages`/`dist-packages` **and** `<prefix>/share/amd_smi/amdsmi` | bare SONAME via the dynamic linker |
| `amdsmi` PyPI wheel | bundled `libamd_smi_python.so` inside the package directory | interpreter's `site-packages` | the bundled `.so`; system fallback disabled |
| ROCm pip (`rocm-sdk-core`) | `<root>/lib/libamd_smi.so.<MAJOR>` | `<root>/share/amd_smi/amdsmi`, not on `sys.path` | fixed path relative to the wrapper |
| `amdrocm-amdsmi` deb/rpm and tarballs | `<root>/lib/libamd_smi.so.<MAJOR>` | `<root>/share/amd_smi/amdsmi`, not on `sys.path` | fixed path relative to the wrapper |

#### Scenario: The TheRock channels put nothing on sys.path

- **WHEN** a user installs the ROCm pip distribution, installs `amdrocm-amdsmi`,
  or extracts a tarball, and then runs `import amdsmi`
- **THEN** the import fails, because those channels ship the module only under
  `share/amd_smi` and install no `.pth` file and no site-packages copy; the
  `amd-smi` CLI still works because it puts that directory on `sys.path` itself

#### Scenario: No channel combines paths from another

- **WHEN** the loader runs in any channel
- **THEN** it consults only that channel's own location and the fixed sequence
  below, and never walks up to a ROCm root, reads `ROCM_PATH`/`ROCM_HOME`, or
  searches `LD_LIBRARY_PATH`

### Requirement: Fixed Library Resolution Order

`_load_library()` SHALL attempt to load the native library in exactly this
order, returning the first success together with the path it resolved:

1. the path in the `AMDSMI_LIB_OVERRIDE` environment variable;
2. `libamd_smi_python.so` in the same directory as the wrapper;
3. `<wrapper>.parents[3]/lib/<SONAME>`, when that file exists;
4. the bare `<SONAME>`, resolved by the dynamic linker.

Steps 3 and 4 SHALL be skipped when `_AMDSMI_ALLOW_SYSTEM_FALLBACK` is `False`.
The library SHALL be opened with `RTLD_LOCAL`.

#### Scenario: Override takes precedence over every other candidate

- **WHEN** `AMDSMI_LIB_OVERRIDE` is set to an absolute path
- **THEN** that library is loaded, even when a bundled or system library is also
  present

#### Scenario: Wheel resolves its bundled library

- **WHEN** `libamd_smi_python.so` sits next to the wrapper
- **THEN** the bundled library is loaded and no later step is attempted

#### Scenario: Step 3 is one fixed location, not a search

- **WHEN** `<wrapper>.parents[3]/lib/<SONAME>` does not exist
- **THEN** the loader proceeds directly to the dynamic linker rather than
  walking further up the tree looking for a ROCm root, so a wrapper imported
  from an unexpected depth cannot silently bind to an unrelated prefix

#### Scenario: An unloadable relocatable library does not shadow the linker

- **WHEN** `<wrapper>.parents[3]/lib/<SONAME>` exists but `ctypes.CDLL` raises
  `OSError` on it (for example, a missing transitive dependency)
- **THEN** the loader falls through to the bare-SONAME lookup instead of failing

#### Scenario: System package resolves through the dynamic linker

- **WHEN** no override, bundled, or relocatable candidate applies and fallback
  is allowed
- **THEN** the bare `<SONAME>` is loaded via the dynamic linker, which finds it
  through the `ld.so.conf.d` entry the package installed

### Requirement: SONAME Pinned To The Library Major Version

The loader SHALL reference the system library by the SONAME
`libamd_smi.so.<MAJOR>`, where `<MAJOR>` is the library major derived from the
header. The generator SHALL parse it from the header it is already processing
rather than hardcode it, so the loader cannot carry a major the library does
not have. The value SHALL remain the *system* SONAME even when the wrapper is
generated against the wheel-private library.

#### Scenario: Generating against the wheel library does not change the SONAME

- **WHEN** the wrapper generator is invoked with `-l libamd_smi_python.so`
- **THEN** `_AMDSMI_LIB_SONAME` still names `libamd_smi.so.<MAJOR>`, because the
  wheel reaches its library through the bundled-file step, not the SONAME step

#### Scenario: A library major bump propagates through regeneration

- **WHEN** `AMDSMI_LIB_VERSION_MAJOR` changes in `amdsmi.h` and the wrapper is
  regenerated
- **THEN** `_AMDSMI_LIB_SONAME` picks up the new major automatically, so a bump
  cannot silently desync the loader from the installed library

#### Scenario: Generation aborts when the major cannot be parsed

- **WHEN** `tools/generator.py` cannot find `AMDSMI_LIB_VERSION_MAJOR` in the
  header it is processing
- **THEN** it exits with an error instead of emitting a wrapper with an empty or
  stale SONAME

### Requirement: System Fallback Flag

The wrapper SHALL carry a module-level boolean `_AMDSMI_ALLOW_SYSTEM_FALLBACK`
that gates steps 3 and 4 of the resolution order. The committed wrapper and
every channel except the PyPI wheel SHALL keep it `True`. The wheel build SHALL
set it `False` in its staged copy, as specified in [amdsmi-python-wheel].

#### Scenario: A wheel missing its bundled library fails loudly

- **WHEN** the flag is `False` and no `libamd_smi_python.so` sits next to the
  wrapper
- **THEN** the loader raises `OSError` naming the missing bundled library and
  stating that it refuses to fall back to a system `libamd_smi.so`

#### Scenario: A wheel never loads a system library

- **WHEN** a wheel is installed on a host that also has `/opt/rocm/lib` on the
  linker path
- **THEN** the system library is never loaded, so a differently-versioned
  `libamd_smi.so` cannot be pulled into a process that already loaded another
  copy (for example, inside PyTorch or JAX)

### Requirement: Import Tolerance With A Missing-Library Sentinel

Importing the module SHALL NOT raise when the native library cannot be loaded.
The loader SHALL install a `_MissingLibrary` sentinel in place of the library
handle, and any *call* of a wrapped `amdsmi_*` symbol SHALL raise `OSError`
carrying the underlying `ctypes.CDLL` error and a remediation hint.

#### Scenario: Documentation and lint tooling import without a ROCm install

- **WHEN** `import amdsmi` runs in an environment with no loadable library
- **THEN** the import succeeds, so doc builds, linters, and multi-stage
  container builds work without a runtime ROCm install

#### Scenario: Attribute assignment on the sentinel is absorbed

- **WHEN** the generated bindings set `restype` and `argtypes` on a symbol taken
  from the sentinel, as they do unconditionally at import time
- **THEN** those assignments are swallowed rather than raising, which is what
  keeps module import tolerant all the way to the end of the file

#### Scenario: The first API call reports the real cause

- **WHEN** a wrapped symbol is called after the sentinel was installed
- **THEN** the raised `OSError` states that the shared library could not be
  loaded, quotes the underlying `dlopen` error, and points at installing
  `amd-smi-lib` or pip-installing the `amdsmi` wheel

### Requirement: Explicit Library Override For Testing

`AMDSMI_LIB_OVERRIDE` SHALL be the single supported escape hatch for loading an
alternate library, and SHALL be honored in every channel including the wheel.

#### Scenario: ABI-compatibility tests point the wrapper at a curated library

- **WHEN** a test sets `AMDSMI_LIB_OVERRIDE` to an alternate `libamd_smi*.so`
- **THEN** the wrapper loads exactly that file, allowing an old wrapper to be
  exercised against a new library and vice versa

#### Scenario: In-tree development against a local build

- **WHEN** a developer sets `AMDSMI_LIB_OVERRIDE=$PWD/build/libamd_smi.so`
- **THEN** `import amdsmi` binds to the freshly built library without installing
  anything

### Requirement: CLI Module Resolution Is Independent Of The Loader

The `amd-smi` CLI SHALL resolve the `amdsmi` module in its own order, distinct
from a plain `import amdsmi` in a user script:

1. `$ROCM_PATH`/`$ROCM_HOME` + `share/amd_smi`, when either variable is set;
2. `share/amd_smi` resolved relative to the CLI's own location
   (`<file>/../../../share/amd_smi`, the CLI running from
   `<prefix>/libexec/amdsmi_cli/`);
3. whatever a normal import finds — a pip install, then the system copy.

The first candidate directory that actually contains an `amdsmi` package SHALL
be inserted at `sys.path[0]`; candidates that do not are skipped.

#### Scenario: A pip install does not change CLI behavior

- **WHEN** a host has both a system `amd-smi-lib` install and a pip-installed
  `amdsmi`
- **THEN** the CLI uses the modules it shipped with, from its own
  `share/amd_smi`, because a pip install is meant for Python scripting rather
  than for altering the CLI

#### Scenario: Multiple ROCm installs each run their own CLI

- **WHEN** two ROCm prefixes are installed and one prefix's `amd-smi` is invoked
- **THEN** it resolves the module from its own prefix, not from the other

#### Scenario: An unset ROCM_PATH still resolves

- **WHEN** neither `ROCM_PATH` nor `ROCM_HOME` is set
- **THEN** the CLI resolves `share/amd_smi` relative to its own file, since it
  runs from `<root>/libexec/amdsmi_cli/`

#### Scenario: The CLI reports a missing module actionably

- **WHEN** no `amdsmi` package is found by any of the three steps
- **THEN** the CLI prints the import error plus an instruction to install
  `amd-smi-lib` (rpm/deb) or pip-install the wheel, and exits non-zero

### Requirement: Channel Coexistence And Precedence

When two channels are installed together, the resulting `import amdsmi` SHALL be
determined by ordinary `sys.path` precedence between install locations, and
uninstalling one channel SHALL NOT remove another channel's files.

| Installed together | `import amdsmi` resolves to | Package uninstall removes the wheel? |
| ------------------ | --------------------------- | ------------------------------------ |
| deb + pip wheel | wheel in `/usr/local/.../dist-packages` | No |
| rpm + pip `--user` wheel | wheel in `~/.local/.../site-packages` | No |
| deb/rpm + venv wheel | wheel in the venv | No |

#### Scenario: A pip wheel wins over a system package

- **WHEN** both `amd-smi-lib` and a PyPI wheel are installed for the same
  interpreter
- **THEN** the wheel's copy is imported, because its install location precedes
  the system one on `sys.path`, and the wheel's bundled library is used

#### Scenario: Removing the system package leaves the wheel intact

- **WHEN** `amd-smi-lib` is removed while a pip wheel is installed
- **THEN** the wheel remains functional, because it is self-contained (bundled
  `.so`, fallback disabled) and the package manager removes only its own files

#### Scenario: A global PYTHONPATH overrides install-location precedence

- **WHEN** `PYTHONPATH=/opt/rocm/share/amd_smi` is set — as some ROCm container
  images do
- **THEN** the `share/amd_smi` copy is imported even when a pip wheel is
  installed, because Python searches `PYTHONPATH` before any install location

#### Scenario: A TheRock channel coexists with any pip install

- **WHEN** `amdrocm-amdsmi` is installed, or a tarball extracted, on a host that
  also has a pip-installed `amdsmi`
- **THEN** the module comes from pip and the ROCm prefix's `.so` is simply
  unused by that module, because the TheRock channels put nothing on `sys.path`

#### Scenario: Two system packages of different major SOVERSION are unsupported

- **WHEN** two `amd-smi-lib` packages with different major SOVERSIONs target one
  prefix
- **THEN** the configuration is unsupported; the loader pins a single SONAME and
  the packages co-own the same paths

### Requirement: Interpreter-Version-Independent Loader Behavior

The loader SHALL behave identically across every CPython version AMD SMI
supports — 3.6.8 through the latest release — and this SHALL be verified without
GPU hardware.

#### Scenario: The loader contract is exercised across the interpreter matrix

- **WHEN** the Python-version CI job runs
- **THEN** `tests/run_amdsmi_python_versions_test.py` confirms, under each of
  CPython 3.6 … 3.14, that the wrapper imports and either resolves a path or
  degrades to the sentinel without raising, and runs the ABI-compat and
  dual-copy guard unit tests under that interpreter

#### Scenario: Loader tests need no GPU

- **WHEN** the loader contract tests run
- **THEN** they use a fake `ctypes.CDLL`, so no amdgpu driver or device is
  required
