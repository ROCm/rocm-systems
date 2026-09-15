# amdsmi-therock-subproject Specification

## Purpose

Defines how TheRock superbuild consumes AMD SMI as a CMake subproject: where the
source comes from, which options TheRock does and does not set, what AMD SMI
links against in a bundled-sysdeps build, and the stage install contract.

This is the layer where AMD SMI's own packaging assumptions meet TheRock's. The
single most consequential fact is that TheRock installs with
`CMAKE_INSTALL_PREFIX=<stage>` and **never sets `DESTDIR`**, so the system
site-packages install described in [amdsmi-python-system-package] is
deliberately skipped in every TheRock build and `share/amd_smi` is the only copy
of the Python module that exists. Everything the two TheRock channels
([amdsmi-rocm-python-distribution], [amdsmi-rocm-os-packages]) can ship follows
from that. The artifact slicing built on top of this stage tree is specified in
[amdsmi-therock-artifact].

## Requirements

### Requirement: Source Comes From The rocm-systems Submodule

TheRock SHALL build AMD SMI from `projects/amdsmi` inside the `rocm-systems` git
submodule, declared with `EXTERNAL_SOURCE_DIR`. TheRock SHALL NOT vendor a
separate copy of the AMD SMI sources.

#### Scenario: The submodule pins the source revision

- **WHEN** TheRock is checked out and sources are fetched
- **THEN** `rocm-systems` is a submodule tracking branch `develop`, and its
  pinned commit determines exactly which AMD SMI revision is built

#### Scenario: The subproject binary and stage dirs are TheRock-owned

- **WHEN** the subproject is declared
- **THEN** its build tree lives under TheRock's `core/amdsmi/build` and its
  install tree under `core/amdsmi/stage`, leaving the source tree clean

### Requirement: The Subproject Is Linux-Only And Feature-Gated

The AMD SMI subproject SHALL be declared only when `THEROCK_ENABLE_CORE_AMDSMI`
is enabled and the host is not Windows. The `CORE_AMDSMI` feature SHALL declare
`windows` as a disabled platform and SHALL require the `BASE`, `SYSDEPS`,
`SYSDEPS_LIBNL`, `SYSDEPS_LIBMNL`, and `COMPILER` features.

#### Scenario: A Windows build omits AMD SMI entirely

- **WHEN** TheRock configures on Windows
- **THEN** no AMD SMI subproject or artifact is declared, and downstream
  subprojects that optionally use it build without it

#### Scenario: Disabling the feature removes the dependency edge

- **WHEN** `THEROCK_ENABLE_CORE_AMDSMI` is off
- **THEN** subprojects that list AMD SMI as an optional dependency (for example
  rocrtst, RCCL, and rocSHMEM, which each guard on `if(TARGET amdsmi)`)
  configure without it rather than failing

### Requirement: TheRock Passes A Minimal Option Set

TheRock SHALL pass only `CMAKE_VERBOSE_MAKEFILE` (as `OFF`) and `BUILD_TESTS` to
the AMD SMI subproject. All other AMD SMI options SHALL take their in-project
defaults. The subproject SHALL be compiled with TheRock's own `amd-llvm`
toolchain rather than the host compiler.

#### Scenario: No wheel is produced by a TheRock build

- **WHEN** TheRock builds AMD SMI
- **THEN** `BUILD_PYTHON_WHEEL` stays at its default `OFF`, so neither
  `libamd_smi_python.so` nor a `.whl` is produced — the pip wheel described in
  [amdsmi-python-wheel] is built by a separate, AMD SMI-owned pipeline

#### Scenario: The committed wrapper is used as-is

- **WHEN** TheRock builds AMD SMI
- **THEN** `BUILD_WRAPPER` stays `OFF`, so no clang/ctypeslib toolchain is
  required and the committed `amdsmi_wrapper.py` is copied into the staging tree
  unchanged

#### Scenario: Test binaries follow TheRock's master testing switch

- **WHEN** `THEROCK_BUILD_TESTING` is on (it follows CMake's `BUILD_TESTING`,
  which defaults on)
- **THEN** AMD SMI is configured with `BUILD_TESTS=ON` and its test tree
  installs under `share/amd_smi/tests`

### Requirement: Bundled Sysdeps Replace The System Netlink And DRM Libraries

In a bundled-sysdeps build (`THEROCK_BUNDLE_SYSDEPS=ON`, the default), AMD SMI
SHALL declare runtime dependencies on TheRock's bundled `libdrm`, `libmnl`, and
`libnl` rather than resolving them from the host, and SHALL additionally depend
on `rocm-core`.

#### Scenario: The built library carries no host libnl or libmnl dependency

- **WHEN** AMD SMI is built in the bundled-sysdeps configuration
- **THEN** its `DT_NEEDED` entries name `librocm_sysdeps_nl_3`,
  `librocm_sysdeps_nl_genl_3`, and `librocm_sysdeps_mnl`, whose symbols carry
  the `AMDROCM_SYSDEPS_1.0` version, so they cannot collide with a host copy
  loaded into the same process

#### Scenario: libdrm is reached by dlopen, not by linking

- **WHEN** `amdsmi_init()` needs `drmCommandWrite`
- **THEN** the library `dlopen`s the first of its SONAME candidates that loads —
  the configure-time SONAME, `libdrm_amdgpu.so`, then
  `librocm_sysdeps_drm_amdgpu.so.1` — so a TheRock install resolves the vendored
  copy through the origin-relative search path rather than through a link-time
  dependency

#### Scenario: This differs from the AMD SMI-native package

- **WHEN** the same sources are built for the standalone `amd-smi-lib` deb/rpm
- **THEN** they resolve the host `libdrm`/`libnl`/`libmnl` and the package
  declares `libdrm-dev`/`libdrm-devel` dependencies — the two delivery paths
  make opposite choices about system libraries

### Requirement: Install Uses A Stage Prefix And Never DESTDIR

TheRock SHALL install each subproject by configuring
`CMAKE_INSTALL_PREFIX=<stage dir>` and invoking `cmake --install <build dir>`.
It SHALL NOT set `DESTDIR`.

#### Scenario: The absolute site-packages install is skipped

- **WHEN** AMD SMI's install rules run under TheRock
- **THEN** the site-packages copy is skipped with a status message, because that
  install is guarded on `DESTDIR` being set — which prevents an install into the
  build host's real `/usr` and keeps the stage tree relocatable

#### Scenario: share/amd_smi is the only module copy in the stage tree

- **WHEN** the stage tree is inspected after install
- **THEN** the `amdsmi` Python module appears only under
  `stage/share/amd_smi/amdsmi`, which is why [amdsmi-therock-artifact] must
  capture it, why neither TheRock channel makes a plain `import amdsmi` work,
  and why AMD SMI's dual-copy requirement exists at all

#### Scenario: Detection still runs and can fail the build

- **WHEN** AMD SMI configures under TheRock on an image with neither
  `/usr/libexec/platform-python` nor `/usr/bin/python3`
- **THEN** configuration fails with AMD SMI's fatal detection error, even though
  the detected path would never be installed to — TheRock's build image must
  therefore provide a default `python3`

### Requirement: Relocatable RPATHs Within The Install Prefix

The subproject SHALL declare `lib` as both its install RPATH directory and its
interface install RPATH directory, so binaries and libraries resolve their
dependencies relative to `$ORIGIN` within the prefix. The `amdsmitst` test
binary SHALL declare an RPATH origin of `share/amd_smi/tests`.

#### Scenario: The test binary finds the library from a different subdirectory

- **WHEN** `amdsmitst` is installed to `share/amd_smi/tests` while
  `libamd_smi.so` is under `lib`
- **THEN** its origin-relative search path is computed from
  `share/amd_smi/tests` — `$ORIGIN/../../../lib` plus the sysdeps directory
  beneath it — so it loads the library from the same prefix without
  `LD_LIBRARY_PATH`

#### Scenario: A relocated prefix keeps working

- **WHEN** the flattened distribution tree is moved or extracted elsewhere
- **THEN** binaries still resolve their libraries, because every search path is
  origin-relative

### Requirement: AMD SMI Is Exported As A find_package Provider

The subproject SHALL provide the `amd_smi` CMake package from `lib/cmake`, so
dependent TheRock subprojects can `find_package(amd_smi)`.

#### Scenario: A dependent subproject links AMD SMI

- **WHEN** a subproject such as RDC, a profiler, or a benchmark declares AMD SMI
  as a dependency
- **THEN** `find_package(amd_smi CONFIG)` resolves through the provided package
  path without the dependent needing to know AMD SMI's stage layout

#### Scenario: Build ordering follows the declared dependencies

- **WHEN** the build graph is computed
- **THEN** AMD SMI builds after `therock-googletest` (its build dependency),
  after `rocm-core` and the bundled sysdeps (its runtime dependencies), and
  after the `amd-llvm` toolchain it compiles with — and before every subproject
  that lists the `core-amdsmi` artifact as a dependency
