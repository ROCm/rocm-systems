# amdsmi-python-wheel Specification

## Purpose

Defines the standalone, pip-installable `amdsmi` wheel: the isolated native
library it bundles, how the staged package tree is assembled, how the wheel is
tagged and versioned, and the build pipeline that produces a manylinux artifact.

The wheel is the only channel that ships its own copy of the native library. It
exists so a user can `pip install amdsmi` into a venv or container without
`/opt/rocm` being present, and so that copy can never collide with a system
`libamd_smi.so` already loaded in the same process.

Loader behavior is specified in [amdsmi-python-loader]; this capability covers
only how the artifact is built and what it contains. This wheel is a different
artifact from the `rocm-sdk-core` wheel that carries AMD SMI as part of a whole
relocated ROCm tree; that one is [amdsmi-rocm-python-distribution].

## Requirements

### Requirement: Wheel Build Is Opt-In

`BUILD_PYTHON_WHEEL` SHALL default to `OFF`. A default build SHALL produce only
the system-package layout and no `.whl` artifact. Turning it `ON` SHALL enable
both the isolated `libamd_smi_python.so` target and the `python_wheel` target,
which is part of `ALL`.

#### Scenario: Default ROCm CI build produces no wheel

- **WHEN** the project is configured with no extra flags
- **THEN** no `.whl` is emitted and the isolated library target is not built

#### Scenario: Wheel build requires a shared library build

- **WHEN** `BUILD_PYTHON_WHEEL=ON` is combined with `BUILD_SHARED_LIBS=OFF`
- **THEN** configuration fails with a fatal error, because the wheel ships a
  `.so` and the `py-interface` subdirectory only enters the build when shared
  libraries are enabled

#### Scenario: A separate pip invocation would skip the loader-flag step

- **WHEN** a caller runs `pip wheel` against the staging tree instead of
  building the `python_wheel` target
- **THEN** the staged wrapper is zipped with the system fallback still enabled,
  because the flag flip is a step of that target — which is why the wheel is
  produced by the build, not by an out-of-band packaging command

### Requirement: Wrapper Regeneration Is Incompatible With The Wheel Build

`BUILD_WRAPPER=ON` combined with `BUILD_PYTHON_WHEEL=ON` SHALL be rejected at
configure time. The wheel SHALL reuse the committed wrapper rather than
regenerate one.

#### Scenario: The combination is refused instead of producing a broken wrapper

- **WHEN** both options are `ON`
- **THEN** configuration fails with a fatal error directing the user to
  regenerate the wrapper with `BUILD_PYTHON_WHEEL=OFF` and then build the wheel
  from the committed wrapper

#### Scenario: Regenerating against the wheel library would mismatch the loader

- **WHEN** the generator is fed `-l libamd_smi_python.so`
- **THEN** it would emit bindings keyed on `libamd_smi_python.so` while the
  loader populates `_libraries['libamd_smi.so']`, making every API call raise
  `KeyError` — which is why the combination is refused

### Requirement: SONAME-Isolated Wheel Library

When `BUILD_PYTHON_WHEEL=ON`, the build SHALL produce a second library target
alongside `libamd_smi` from the same sources, with:

- output name `amd_smi_python`, so the file is `libamd_smi_python.so`;
- `SOVERSION` equal to the library major and `VERSION` equal to the full library
  version;
- the same linker version script as `libamd_smi`;
- `-Bsymbolic-functions`, binding the library's own internal `amdsmi_*` calls at
  link time.

#### Scenario: Both libraries can be loaded in one process

- **WHEN** a process has both the system `libamd_smi.so.<MAJOR>` and the wheel's
  `libamd_smi_python.so.<MAJOR>` loaded
- **THEN** the distinct SONAMEs prevent the dynamic linker from interposing one
  on the other, and `-Bsymbolic-functions` keeps each library's internal calls
  bound to its own definitions — without it, `amdsmi_init()` initializes one
  copy's state while enumeration reads the other's and reports zero processors

#### Scenario: The SONAME split is verified from built artifacts

- **WHEN** `tests/run_amdsmi_pkg_conflict_test.py` runs against a build tree
  containing both libraries
- **THEN** it asserts the two `.so` files carry different SONAMEs, so a
  double-load cannot crash during static C++ initialization

### Requirement: Staged Package Tree

The build SHALL assemble a staging tree under
`<build>/py-interface/python_package/` containing a generated `pyproject.toml`
and `setup.py` at its root and an `amdsmi/` package directory holding:
`__init__.py`, `_version.py`, `amdsmi_wrapper.py`, `amdsmi_interface.py`,
`amdsmi_interface_utils.py`, `amdsmi_exception.py`, `README.md`, `LICENSE`, and
the bundled `libamd_smi_python.so`.

Package metadata SHALL declare the distribution name `amdsmi`, a single package
`amdsmi`, `requires-python >= 3.6`, `zip-safe = false`, and package data `*.so`.

#### Scenario: Python sources are staged by copy-if-different

- **WHEN** a rebuild runs with unchanged Python sources
- **THEN** the staged files are not re-copied, so downstream packaging steps are
  not needlessly re-run

#### Scenario: The wheel carries its own README and LICENSE

- **WHEN** the wheel is built
- **THEN** `amdsmi/README.md` and `amdsmi/LICENSE` are present and referenced by
  the project metadata as the readme and license files

### Requirement: System Fallback Disabled In The Staged Wrapper

Before the wheel is zipped, `tools/disable_system_fallback.py` SHALL flip
`_AMDSMI_ALLOW_SYSTEM_FALLBACK` from `True` to `False` in the **staged** copy of
`amdsmi_wrapper.py` only. The committed wrapper in the source tree SHALL remain
`True`.

#### Scenario: Exactly one occurrence is required

- **WHEN** the staged wrapper does not contain exactly one
  `_AMDSMI_ALLOW_SYSTEM_FALLBACK = True`
- **THEN** the tool exits non-zero rather than build a wheel with an ambiguous
  loader fallback flag

#### Scenario: The flip is idempotent across rebuilds

- **WHEN** a rebuild reuses an already-flipped staged copy (copy-if-different
  did not re-copy the unchanged source)
- **THEN** the tool treats the already-disabled flag as a no-op and succeeds

### Requirement: Platform-Specific, ABI-Independent Wheel Tag

The wheel SHALL be tagged `py3-none-<platform>`. It SHALL NOT be tagged
`py3-none-any`, and it SHALL NOT carry a CPython ABI tag.

The distribution SHALL report itself as non-pure so the bundled `.so` is placed
in `platlib`.

#### Scenario: One wheel serves every supported CPython

- **WHEN** the wheel is built once
- **THEN** it installs on every supported CPython, because it carries no C
  extension and reaches the library through `ctypes`

#### Scenario: The platform tag prevents an install that would fail at load time

- **WHEN** a user on an incompatible platform attempts to install the wheel
- **THEN** pip rejects it on the platform tag, instead of installing an
  `any`-tagged wheel that would fail when the bundled glibc/arch-bound `.so` is
  loaded

#### Scenario: auditwheel can inspect the bundled library

- **WHEN** `auditwheel` processes the wheel
- **THEN** the `.so` is found in `platlib` because the distribution declares
  itself as having extension modules, and the concrete manylinux tag is stamped

### Requirement: Wheel Version Stamping

The wheel version SHALL be derived from the library version in
`include/amd_smi/amdsmi.h`. By default it SHALL be `<X.Y.Z>+<hash>`. With
`AMDSMI_WHEEL_RELEASE=ON` it SHALL be the clean PEP 440 `<X.Y.Z>`, suitable for
publishing to PyPI. The staged `amdsmi/_version.py` SHALL expose `__version__`
and `__commit__`.

#### Scenario: A development wheel carries the commit hash

- **WHEN** the wheel is built without `AMDSMI_WHEEL_RELEASE`
- **THEN** its version carries the `+<hash>` local-version segment, which PyPI
  rejects — marking it as a non-publishable build while keeping per-commit CI
  wheel filenames unique

#### Scenario: A release wheel is publishable

- **WHEN** `AMDSMI_WHEEL_RELEASE=ON` (equivalently `build_wheel.py --release`)
- **THEN** the version is a clean PEP 440 release version and the wheel is
  publishable

#### Scenario: The installed package reports its provenance

- **WHEN** a user reads `amdsmi.__version__` and `amdsmi.__commit__`
- **THEN** they get the library version and the source commit the wheel was
  built from, which is how a `--release` wheel stays traceable after the hash is
  dropped from its version

### Requirement: Reproducible Wheel Build Pipeline

`tools/build_wheel.py` SHALL be the single entry point for building wheels, on a
supported host distro or inside a manylinux container, and SHALL perform:

1. optional git `safe.directory` registration for the project and its bundled
   esmi tree, falling back to a per-build `GIT_CONFIG_GLOBAL` when the global
   gitconfig is not writable;
2. CMake configure with `BUILD_PYTHON_WHEEL=ON`;
3. build;
4. one `pip wheel --no-deps --no-build-isolation` invocation;
5. a build-smoke of the package with the oldest and newest available
   interpreters;
6. an optional `auditwheel repair`.

It SHALL remain runnable under Python 3.6.

#### Scenario: A developer reproduces a CI wheel failure locally

- **WHEN** a developer runs the same `build_wheel.py` invocation CI runs, inside
  the same container image
- **THEN** the build reproduces, because the pipeline lives in the script rather
  than in workflow YAML

#### Scenario: Interpreters are selected deterministically

- **WHEN** interpreters are not given explicitly
- **THEN** the script uses `/opt/python/cp3*-cp3*/bin/python3` when present
  (manylinux images) and otherwise the system `python3`; the building
  interpreter is `cp310` when available, else the oldest found

#### Scenario: The wheel is built once and only smoke-checked elsewhere

- **WHEN** several interpreters are available
- **THEN** the wheel is produced once — it is identical for every CPython — and
  the oldest and newest interpreters only build-smoke the package to catch
  setuptools compatibility breakage

#### Scenario: A failed repair never ships an un-audited wheel

- **WHEN** `--repair` is requested and `auditwheel` is missing or its repair
  fails
- **THEN** the script aborts, rather than copying the un-audited wheel to the
  output directory

#### Scenario: Stale build state is cleared before configuring

- **WHEN** a build directory contains `CMakeCache.txt`/`CMakeFiles` or the
  source tree contains a stale `esmi_ib_library_temp`
- **THEN** those are removed before configuring, so a previous configuration
  cannot leak into the wheel build

### Requirement: Wheel Build Verification In CI

Wheel builds SHALL be exercised in CI without GPU hardware, on both an
RPM-family manylinux image and a Debian-family host.

#### Scenario: manylinux wheel leg

- **WHEN** the manylinux job runs
- **THEN** it builds inside `quay.io/pypa/manylinux_2_28_x86_64` with
  `--os-variant AlmaLinux8 --repair --release`, producing a manylinux-tagged,
  PyPI-publishable wheel

#### Scenario: Debian wheel leg

- **WHEN** the Debian job runs
- **THEN** it builds on Ubuntu 22.04 without `--repair` and without `--release`,
  so the Debian leg proves the build works off a plain host without claiming to
  produce a publishable artifact

#### Scenario: Both legs check the SONAME split

- **WHEN** either leg completes
- **THEN** both built libraries are read back from the shared build directory
  and their SONAMEs are asserted to differ

#### Scenario: The shipped wheel is checked for the disabled fallback

- **WHEN** either leg installs the wheel it just built
- **THEN** it asserts that the imported `amdsmi` resolves under pip's own
  location and that `_AMDSMI_ALLOW_SYSTEM_FALLBACK` is `False`, so a build-step
  ordering regression cannot ship a wheel that silently loads a system `.so`
