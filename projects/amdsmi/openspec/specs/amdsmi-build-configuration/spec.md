# amdsmi-build-configuration Specification

## Purpose

Defines the configure-time surface of AMD SMI's CMake build: every option a
packager or a downstream build may set, what each one turns on, which
combinations are rejected before a single object file is compiled, which values
configure derives rather than accepts, and what the installed `amd_smi` CMake
package promises a `find_package` consumer.

A distro packager building `amd-smi-lib`, TheRock building the subproject, and a
developer running `cmake ..` in a scratch directory all drive the same option
set, so it is one capability rather than a section of each delivery channel.
Defaults are part of the contract: a plain `cmake ..` with no flags must produce
a defined result, and the parts of that result that surprise people — an
unoptimized library, no wheel, no static archive, no Python module when shared
libraries are off — are stated here rather than discovered.

The distinction that organizes most of this capability is between options and
derivations. An option is something a packager sets. A derivation is something
configure computes from the source tree or the build host and then freezes into
the artifact: the library version, the site-packages destination, the
`libdrm_amdgpu` SONAME, the paths baked into the maintainer scriptlets. A
derivation is a contract precisely because the packager must not be able to set
a value that silently disagrees with its source.

This capability owns the option set and the configure-time rules. It does not
own what the options produce downstream: the wheel pipeline is
[amdsmi-python-wheel], the CPack packaging and maintainer scriptlets are
[amdsmi-python-system-package] and [amdsmi-rocm-os-packages], TheRock's
invocation and the option subset it passes are [amdsmi-therock-subproject], the
installed tree shape is [amdsmi-install-layout], and the exported C surface —
including the consumer-visible half of the `ENABLE_ESMI_LIB` gate and the
`SOVERSION` the version macros produce — is [amdsmi-c-api-abi].

## Requirements

### Requirement: Option Inventory And Defaults

The build SHALL expose the following user-settable options. A configure run that
passes none of them SHALL produce the "Default" column exactly.

| Option | Default | What it turns on |
| ------ | ------- | ---------------- |
| `BUILD_SHARED_LIBS` | `ON` | `libamd_smi.so` with the linker version script; gates the entire `py-interface`, `amdsmi_cli`, and `rust-interface` surface |
| `ENABLE_ESMI_LIB` | `ON` on `x86_64`, `OFF` otherwise | Fetches and compiles the pinned ESMI sources and defines `ENABLE_ESMI_LIB=1` for the whole project |
| `BUILD_CLI` | `ON`, forced `OFF` without shared libraries | Builds and installs the CLI tree and the `bin/amd-smi` symlink |
| `ENABLE_LDCONFIG` | `ON`, forced `OFF` without shared libraries | Emits the `ld.so.conf.d` registration into the maintainer scriptlets |
| `BUILD_TESTS` | `OFF` | Builds `amdsmitst` and the packaged Python test tree; transitively enables the static library |
| `BUILD_PYTHON_WHEEL` | `OFF` | Builds `libamd_smi_python.so` and the `python_wheel` target, and suppresses the site-packages install |
| `AMDSMI_WHEEL_RELEASE` | `OFF` | Drops the `+<hash>` local-version segment from the wheel version |
| `BUILD_WRAPPER` | `OFF`, forced `OFF` without shared libraries | Regenerates the committed `amdsmi_wrapper.py` using clang and `ctypeslib2` |
| `BUILD_RUST_WRAPPER` | `OFF`, forced `OFF` without shared libraries | Adds `rust-interface`, which requires `cargo` |
| `BUILD_RUST_EXAMPLES` | `OFF`, declared only when the Rust wrapper is enabled | Builds and installs the Rust example binaries |
| `REGENERATE_RUST_WRAPPER` | `OFF`, declared only when the Rust wrapper is enabled | Regenerates the Rust binding during the cargo build |
| `BUILD_EXAMPLES` | `OFF` | Compiles `example/`; the example *sources* are installed either way |
| `AUTO_BUILD_STATIC_LIBS` | `ON` when `NOT BUILD_SHARED_LIBS OR BUILD_TESTS`, else forced `OFF` | Permits a static library to be built |
| `BUILD_BOTH_LIBS` | `ON` when `BUILD_SHARED_LIBS AND AUTO_BUILD_STATIC_LIBS`, else forced `OFF` | Builds and installs `libamd_smi.a` alongside the shared library and points tests and examples at it |
| `ENABLE_WSL_BACKEND` | `OFF` | Compiles the experimental WSL/WDDM backend and its tests |
| `BRCM_NIC` | `OFF` | Compiles the Broadcom NIC sources and defines `BRCM_NIC` |
| `BUILD_CUID` | `OFF` | Links `libamdcuid_static` and `OpenSSL::Crypto` into every `amd_smi` library target |
| `ENABLE_ASAN_PACKAGING` | `OFF` | Packages only the `asan` component; changes no compiler flag |
| `ADDRESS_SANITIZER` | unset | Adds `-fsanitize=address` and removes the hardening flag set |
| `ROCM_DEP_ROCMCORE` | unset | Adds a `rocm-core` dependency to every produced package |
| `INSTALL_GTEST` | `OFF`, declared only when tests are enabled | Installs a fetched GTest alongside the tests |
| `CMAKE_POSITION_INDEPENDENT_CODE` | `ON`, marked advanced | `-fPIC` for every target, so a static `libamd_smi.a` can be embedded in a shared library |
| `CMAKE_VERBOSE_MAKEFILE` | `ON` | Verbose build output; inverts CMake's own default |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | `ON` | `compile_commands.json` for linters and language servers; inverts CMake's own default |

What the three Rust options build, and the fact that no delivery channel
enables any of them, are [amdsmi-language-bindings].

An option consumed by more than one directory of the main build SHALL be
declared once, at the root.

#### Scenario: A plain configure builds one shared library and nothing optional

- **WHEN** `cmake ..` runs with no flags on x86_64
- **THEN** the build produces `libamd_smi.so` with the version script, the ESMI
  backend, the Python module staging tree, and the CLI — and no static archive,
  no `.whl`, no tests, no compiled examples, no Rust or WSL surface

#### Scenario: A default build is unoptimized, debug-annotated, and unstripped

- **WHEN** `CMAKE_BUILD_TYPE` is left unset, which is what CMake does by default
- **THEN** the library is compiled `-ggdb -O0 -DDEBUG` and the Release-only
  strip step does not run, so a packager who omits
  `-DCMAKE_BUILD_TYPE=Release` ships a debug build without any diagnostic
  saying so

#### Scenario: An option is not redeclared in the subdirectory that uses it

- **WHEN** `BUILD_PYTHON_WHEEL` is consulted by `src/`, `py-interface/`, and the
  root
- **THEN** it is declared only at the root, so a single `-D` switch reaches
  every consumer regardless of `add_subdirectory()` ordering — a redeclaration
  in a subdirectory would make the option's effect depend on which directory
  CMake descended into first

#### Scenario: Example sources ship whether or not they are compiled

- **WHEN** `BUILD_EXAMPLES` is left `OFF`
- **THEN** `example/` is not compiled but its headers, sources, and
  `CMakeLists.txt` are still installed under the share directory, because the
  installed examples exist to be built by a user against the installed package,
  not by the project

### Requirement: Configure Destinations Default To The ROCm Prefix

`CMAKE_INSTALL_PREFIX` SHALL default to `ROCM_DIR`, itself defaulting to
`/opt/rocm`, and `CPACK_PACKAGING_INSTALL_PREFIX` SHALL follow it. The
generators SHALL default to `DEB;RPM`. The following destinations SHALL be
cache variables a packager may redirect:

| Variable | Default |
| -------- | ------- |
| `ROCM_DIR` | `/opt/rocm` |
| `CMAKE_INSTALL_PREFIX` | `${ROCM_DIR}` |
| `CPACK_PACKAGING_INSTALL_PREFIX` | `${CMAKE_INSTALL_PREFIX}` |
| `SHARE_INSTALL_PREFIX` | `<datarootdir>/amd_smi` |
| `PY_CLI_INSTALL_DIR` | `${CMAKE_INSTALL_LIBEXECDIR}` |
| `RUST_WRAPPER_INSTALL_DIR` | `${SHARE_INSTALL_PREFIX}/rust-wrapper` |
| `AMDSMI_SYSTEM_PYTHON_SITELIB` | empty, meaning auto-detect |
| `CPACK_GENERATOR` | `DEB;RPM` |

The package version and release fields SHALL NOT be cache variables at all: they
are read from the environment, as specified in
[amdsmi-rocm-systems-integration]. A `-D` switch cannot set them.

#### Scenario: A packager who sets no prefix does not install to /usr/local

- **WHEN** configure runs with no `CMAKE_INSTALL_PREFIX`
- **THEN** the prefix is `/opt/rocm`, because the default is seeded into the
  cache before `project()` runs and therefore before CMake would install its own
  `/usr/local` default

#### Scenario: The share directory does not follow the CMake project name

- **WHEN** the share destination is computed
- **THEN** it is `share/amd_smi`, seeded at the root before the packaging helper
  runs — the helper would otherwise derive `share/amd_smi_lib` from the CMake
  project name, silently relocating the Python module and every downstream tool
  that reaches it through `ROCM_PATH/share/amd_smi` as specified in
  [amdsmi-install-layout]

#### Scenario: A locally built package outranks every released one

- **WHEN** a developer builds without `ROCM_LIBPATCH_VERSION`
- **THEN** the package version ends in `.99999`, which sorts above any real ROCm
  libpatch, so a hand-built package installed for testing is never displaced by
  a subsequent official update; CI matches its uploaded package artifacts on
  exactly that `99999-local` pattern

### Requirement: The Shared-Library Switch Gates The Entire Non-C Surface

`BUILD_SHARED_LIBS=OFF` SHALL reduce the build to the C/C++ library alone. The
`py-interface`, `amdsmi_cli`, and `rust-interface` subdirectories SHALL NOT be
added, and `BUILD_WRAPPER`, `BUILD_CLI`, `BUILD_RUST_WRAPPER`, and
`ENABLE_LDCONFIG` SHALL be forced `OFF` regardless of what the user requested.

#### Scenario: A static build ships no Python module and no CLI

- **WHEN** `BUILD_SHARED_LIBS=OFF`
- **THEN** neither the site-packages copy nor the `share/amd_smi` copy of the
  `amdsmi` module is produced and no `bin/amd-smi` exists, because both are
  staged by `py-interface`, which never enters the build

#### Scenario: An explicitly requested option is silently overridden

- **WHEN** a packager configures `-DBUILD_SHARED_LIBS=OFF -DBUILD_CLI=ON`
- **THEN** the CLI is not built and configure reports no error, because the
  dependent-option mechanism forces the value and hides the entry rather than
  rejecting the request — unlike the combinations below, which are refused

#### Scenario: The static library exports more than the shared one

- **WHEN** only the static library is built
- **THEN** the linker version script is not generated or applied, so the archive
  still carries the internal `rsmi_*` and `amd::smi::*` symbols that the shared
  library hides; a consumer that statically links AMD SMI into a process that
  also loads `librocm_smi64.so` can therefore collide in a way the shared
  library is specifically built to prevent

### Requirement: Enabling Tests Changes The Installed Library Set

`BUILD_TESTS=ON` SHALL, in a default shared-library build, additionally enable
`AUTO_BUILD_STATIC_LIBS` and therefore `BUILD_BOTH_LIBS`. The static
`libamd_smi.a` SHALL then be built and installed into the library directory
under the same `dev` component as the shared library, and tests and examples
SHALL link the static library in preference to the shared one.

#### Scenario: Asking for tests adds an archive to the runtime package

- **WHEN** a packaging build sets `BUILD_TESTS=ON` to obtain the `-tests`
  package
- **THEN** the main package also gains `libamd_smi.a`, because the static target
  installs into the same component — the test switch is not confined to the test
  package

#### Scenario: The gtest suite does not exercise the shipped library

- **WHEN** `amdsmitst` runs in CI
- **THEN** it is linked against `libamd_smi.a`, so a passing run says nothing
  about the version script, the exported symbol set, or the SOVERSION of the
  `libamd_smi.so` that actually ships; those properties are covered by
  [amdsmi-c-api-abi] instead

### Requirement: Rejected Configurations Fail At Configure Time

Rather than emit a broken artifact, configuration SHALL fail with a fatal error
in each of the following cases:

| Condition | What the failure protects against |
| --------- | --------------------------------- |
| `BUILD_PYTHON_WHEEL=ON` with `BUILD_SHARED_LIBS=OFF` | a wheel target with no `.so` to bundle, because `py-interface` never enters a static build |
| `BUILD_WRAPPER=ON` with `BUILD_PYTHON_WHEEL=ON` | a regenerated wrapper keyed on `libamd_smi_python.so` while the loader keys on `libamd_smi.so` |
| `BUILD_WRAPPER=ON` with no clang, or clang older than 16.0 | a wrapper regenerated by a toolchain that produces different bindings from the committed one |
| `ENABLE_WSL_BACKEND=ON` without the sibling `rocr-runtime/libhsakmt` WSL header | a WSL build that compiles against absent `rocdxg` declarations |
| `BUILD_CUID=ON` without `libamdcuid_static`, `amd_cuid.h`, or OpenSSL | a library that claims CUID support and cannot link |
| `BUILD_RUST_WRAPPER=ON` with no `cargo` on `PATH` or under `~/.cargo/bin` | a Rust target that fails deep into the build instead of at configure |
| `libdrm`, `libdrm_amdgpu`, `libnl-3.0`, `libnl-genl-3.0`, or `libmnl` not resolvable by pkg-config or bundled config | a build that silently drops device or link telemetry backends |
| the `libdrm_amdgpu` SONAME not readable from the resolved library | a runtime `dlopen` list whose first candidate is empty |
| GCC older than 5.4.0 | a compiler that cannot build the C++17 sources |
| a Debian build host without `gzip` or `date` | a `.deb` missing the compressed changelog and timestamped copyright that Debian policy requires |
| a system-package build whose site-packages destination cannot be determined | a module installed where the target host's `python3` never looks |

The rationale for the two wheel rows is in [amdsmi-python-wheel]; the
site-packages row is specified in [amdsmi-python-system-package]. This
capability owns the obligation that each is a configure-time failure rather
than a build-time or install-time one.

#### Scenario: A refusal is cheaper than a bad artifact

- **WHEN** any condition in the table holds
- **THEN** configure stops before compiling, so the failure is a one-line CMake
  error a packager reads immediately, rather than a link error, a broken
  package, or — worst — a package that installs cleanly and misbehaves on a
  user's machine

#### Scenario: Optional features fail loudly rather than degrade

- **WHEN** an opt-in backend such as CUID, the WSL backend, or the Rust wrapper
  is requested and its external dependency is absent
- **THEN** configuration fails; the build never quietly drops the feature the
  caller explicitly asked for, because a silently reduced library is
  indistinguishable at runtime from one whose hardware simply has nothing to
  report

#### Scenario: Mandatory dependencies are not optional in disguise

- **WHEN** the netlink libraries are missing
- **THEN** configure fails even though the code paths that use them tolerate
  runtime absence, so the shipped library's dependency set is decided once at
  configure and cannot vary with what happened to be installed on a build host

### Requirement: ESMI Is An Architecture-Defaulted Gate Over A Pinned Source

`ENABLE_ESMI_LIB` SHALL default to `ON` on `x86_64` and `OFF` on every other
processor. When on, configure SHALL obtain the ESMI sources at a pinned,
immutable commit hash, compile them into every `amd_smi` library target, and
define `ENABLE_ESMI_LIB=1` for the whole project. The pin SHALL be enforced: an
existing checkout at a different commit SHALL be discarded and re-fetched, and
an existing checkout at the pinned commit SHALL be reused without network
access.

The macro SHALL be a directory-scope compile definition, not an interface
property of any exported target.

#### Scenario: The same header describes two different libraries

- **WHEN** the project is built on aarch64, where the gate defaults off
- **THEN** the built library exports no CPU or HSMP entry points while the
  installed `amdsmi.h` still contains their declarations behind the
  conditional — the consumer-visible consequences of that split are specified in
  [amdsmi-c-api-abi]

#### Scenario: A pin bump cannot build stale sources

- **WHEN** the pinned ESMI commit is changed and a developer reconfigures an
  existing tree
- **THEN** the previously fetched checkout is removed and re-fetched, because a
  reused checkout would compile the old sources indefinitely and no build output
  would say which ESMI revision was used

#### Scenario: An offline or air-gapped build reuses what is present

- **WHEN** the ESMI sources are already checked out at the pinned commit
- **THEN** no fetch is attempted, so a build host with no network access still
  configures

#### Scenario: The installed package does not record which way the gate went

- **WHEN** a `find_package(amd_smi)` consumer inspects the imported targets
- **THEN** nothing tells it whether the library it is about to link was built
  with ESMI, because the definition is directory-scope and is not propagated
  through the export; the consumer must define the macro itself, which is the
  compile-time hazard [amdsmi-c-api-abi] specifies

### Requirement: The Version Is Derived From The Header And Cannot Be Overridden

The library version SHALL be extracted from the `AMDSMI_LIB_VERSION_MAJOR`,
`AMDSMI_LIB_VERSION_MINOR`, and `AMDSMI_LIB_VERSION_RELEASE` macros in
`include/amd_smi/amdsmi.h` and SHALL drive the library `SOVERSION` and `VERSION`
properties, the CPack package version, the CMake package version file, and the
wheel version.

The only version input a packager supplies SHALL be the fourth CPack component
and the release fields, all of which come from the environment rather than the
cache — see [amdsmi-rocm-systems-integration]. The three leading components
SHALL NOT be settable from the command line: configure recomputes them from the
header into ordinary variables that shadow any cache entry a `-D` switch would
create.

The informational package version string SHALL additionally carry the commit
count since the previous version tag, the build identifier, and the short commit
hash, with the hash suffixed `-dirty` when the working tree has uncommitted
changes and replaced by `unknown` when git is unavailable.

#### Scenario: The package version can never disagree with the SONAME

- **WHEN** a packager attempts `-DCPACK_PACKAGE_VERSION_MAJOR=<n>`
- **THEN** the value is ignored, so the `.deb`/`.rpm` version and the
  `libamd_smi.so.<MAJOR>` SONAME the loader resolves are guaranteed to come from
  the same header macros

#### Scenario: A build from a dirty tree is marked as such

- **WHEN** the working tree has uncommitted modifications
- **THEN** the version hash carries a `-dirty` suffix, so an artifact built from
  an unreproducible tree is identifiable after the fact

#### Scenario: Version-carrying mirrors are repaired, not trusted

- **WHEN** the Rust binding's version constants lag the header
- **THEN** configure rewrites them; the mechanics of that repair and the set of
  mirrors it covers are specified in [amdsmi-c-api-abi]

### Requirement: Configure Freezes Host-Resolved Values Into The Artifact

Configure SHALL resolve the following facts about the build host and embed them
into generated files that become part of the artifact:

| Value | Resolved from | Frozen into |
| ----- | ------------- | ----------- |
| `libdrm_amdgpu` SONAME | `objdump` on the pkg-config-resolved library | a generated config header, as the first runtime `dlopen` candidate |
| system site-packages directory | the host's default `python3` | the install rule and the maintainer scriptlets |
| Python major.minor ABI | the resolved site-packages path | the RPM interpreter dependency |
| library, libexec, share, and packaging prefixes | the configured install destinations | the maintainer scriptlets |
| `ENABLE_LDCONFIG` | the option | the maintainer scriptlets |

The Python ABI SHALL be recomputed on every configure and forced into the cache,
so it can never be set to a value that disagrees with the destination path it
describes. The site-packages destination itself SHALL remain overridable,
because a cross-distro packager may legitimately know better than detection.

#### Scenario: A package cannot be relocated after it is built

- **WHEN** a `.deb` or `.rpm` is produced
- **THEN** its maintainer scriptlets contain absolute paths chosen at configure
  time, so the package is bound to the prefix and the interpreter of the host it
  was configured on — which is why the packaging CI jobs build and test inside
  the same distro image, as [amdsmi-python-system-package] specifies

#### Scenario: The dlopen candidate list is decided at configure, not at runtime

- **WHEN** the library later looks for `libdrm_amdgpu`
- **THEN** its first candidate is the exact SONAME captured from the build host,
  ahead of the bare filename and the vendored sysdeps name; the ordered
  resolution itself is specified in [amdsmi-device-discovery]

#### Scenario: The interpreter CMake finds is not the interpreter that decides

- **WHEN** `find_package(Python3)` resolves to a container's build interpreter
- **THEN** the module destination is still detected from the host's default
  `python3`, because the two are routinely different and the destination must
  match the interpreter the installed CLI and a plain `import amdsmi` will use;
  `find_package(Python3)` is only invoked at all when `BUILD_PYTHON_WHEEL` or
  `BUILD_WRAPPER` is on, so a C-only build requires no interpreter

### Requirement: The Installed CMake Package Contract

The build SHALL install, under `<libdir>/cmake/amd_smi` and into the `dev`
component, a config file, a version file, and a targets file. A consumer that
calls `find_package(amd_smi CONFIG)` SHALL be able to rely on:

- the imported target `amd_smi`, carrying `<prefix>/include` as its interface
  include directory, so `#include <amd_smi/amdsmi.h>` resolves;
- the imported target `amdsminic`, the static NIC helper, in every
  configuration;
- the imported target `amd_smi_static` **only** when the producing build had
  `BUILD_BOTH_LIBS` on;
- the variables `AMD_SMI_INCLUDE_DIR`, `AMD_SMI_INCLUDE_DIRS`,
  `AMD_SMI_LIB_DIR`, `AMD_SMI_LIB_DIRS`, `AMD_SMI_LIBRARIES`, and
  `AMD_SMI_LIBRARY`, plus their lowercase-prefixed spellings;
- `SameMajorVersion` compatibility against the four-component CPack package
  version.

Targets SHALL be exported without a namespace, and the config file SHALL skip
including the targets file when an `amd_smi` target already exists in the
consuming project.

#### Scenario: The package is what downstream actually consumes

- **WHEN** the installed examples or the Go shim are built standalone
- **THEN** each resolves AMD SMI through `find_package(amd_smi CONFIG REQUIRED)`
  rather than by guessing `lib` versus `lib64`, and the examples path is
  exercised on every distro in the packaging matrix — so a regression in the
  exported package fails CI rather than only downstream

#### Scenario: A consumer must probe for the static target

- **WHEN** a consumer prefers static linkage
- **THEN** it has to test `if(TARGET amd_smi_static)` and fall back, because the
  target exists only in builds that enabled both libraries — a default build
  exports the shared target alone, and hard-coding the static name fails to
  configure

#### Scenario: An in-tree target wins over the installed package

- **WHEN** a superbuild that already defines an `amd_smi` target calls
  `find_package(amd_smi CONFIG)`
- **THEN** the config sets its variables but does not include the targets file,
  so the in-tree target is used and CMake does not error on a duplicate imported
  target — which is what makes the same `find_package` call work in both a
  standalone and a superbuild context, as [amdsmi-therock-subproject] relies on

#### Scenario: The static target's link interface carries build-host paths

- **WHEN** a consumer links `amd_smi_static` from an installed package
- **THEN** it inherits the absolute netlink library paths recorded at the
  producing build's configure time; those are deliberately absolute rather than
  pkg-config imported targets, which cannot appear in an exported link
  interface, so a consumer on a host with a different netlink layout must supply
  its own

### Requirement: Sanitizer Flags And ASAN Packaging Are Independent Switches

`ADDRESS_SANITIZER` SHALL control compilation only: it adds
`-fno-omit-frame-pointer -fsanitize=address` to the C and C++ flags and
`-fsanitize=address` to the executable and shared linker flags, selecting
`-shared-libsan` for Clang shared builds and `-static-libsan` for static builds.
`ENABLE_ASAN_PACKAGING` SHALL control packaging only: it restricts the packaged
components to `asan`, relocates the installed license, and renames the Debian
lintian override file.

A sanitizer build SHALL NOT receive the hardening flag set — `_FORTIFY_SOURCE`,
`-fstack-protector-all`, and the `noexecstack`/`relro`/`now` linker options —
because the two sets are mutually exclusive branches of the same decision.

#### Scenario: An asan package with no sanitizer in it

- **WHEN** only `ENABLE_ASAN_PACKAGING=ON` is set
- **THEN** an `amd-smi-lib-asan` package is produced containing a library with
  no instrumentation whatsoever, since nothing couples the packaging switch to
  the compilation switch

#### Scenario: A sanitizer build is not a hardened build

- **WHEN** `ADDRESS_SANITIZER` is set
- **THEN** the hardening flags are omitted, so the resulting library is a
  diagnostic artifact and SHALL NOT be treated as a drop-in replacement for the
  normally built one

### Requirement: Configurations Exercised By CI

AMD SMI's own workflows SHALL exercise the following configurations. They do not
cover the full option matrix, and anything absent from this table is unverified
by them. TheRock builds the same tree with a different, minimal option set,
specified in [amdsmi-therock-subproject].

| CI leg | Options passed |
| ------ | -------------- |
| manylinux build gate | `ENABLE_ESMI_LIB=ON`, `BUILD_PYTHON_WHEEL=ON`, `CMAKE_BUILD_TYPE=Release`, explicit `Python3_EXECUTABLE` |
| nine-distro package build and GPU test | `BUILD_TESTS=ON`, `ENABLE_ESMI_LIB=ON` |
| standalone example build on each of those distros | `ENABLE_ESMI_LIB=OFF` against the installed CMake package |
| manylinux and Debian wheel legs | `BUILD_TESTS=OFF`, `ENABLE_ESMI_LIB=ON`, `BUILD_PYTHON_WHEEL=ON`, `AMDSMI_WHEEL_RELEASE` per leg, `CMAKE_BUILD_TYPE`, explicit `Python3_EXECUTABLE` |
| SLES packaging | `CMAKE_BUILD_TYPE=Release`, `ENABLE_ESMI_LIB=ON`, explicit GCC 12 compilers |
| upgrade/downgrade | `CMAKE_BUILD_TYPE=Release`, `ENABLE_ESMI_LIB=ON`, then reconfigured at two `ROCM_LIBPATCH_VERSION` values |
| DME integration | `BUILD_TESTS=OFF`, `ENABLE_ESMI_LIB=OFF`, explicit prefix and build type |

#### Scenario: No leg builds the default configuration

- **WHEN** AMD SMI's own workflows are enumerated
- **THEN** every leg overrides at least `ENABLE_ESMI_LIB`, so a plain `cmake ..`
  — the configuration a first-time contributor runs — is not covered by any job

#### Scenario: The packaging matrix builds a debug library

- **WHEN** the nine-distro packaging job configures
- **THEN** it passes no `CMAKE_BUILD_TYPE`, despite accepting a build-type
  argument, so the `.deb`/`.rpm` artifacts that job uploads for manual review
  contain a `-O0 -DDEBUG` library rather than the Release build the wheel and
  SLES legs produce

#### Scenario: The ESMI default is never relied on

- **WHEN** any packaging or wheel leg configures
- **THEN** it sets `ENABLE_ESMI_LIB` explicitly rather than depending on the
  architecture default, so the CPU surface a job builds does not change if the
  runner architecture does

#### Scenario: A large part of the option surface is unverified

- **WHEN** a change touches static builds, wrapper regeneration, the WSL or
  Broadcom NIC backends, CUID, the Rust wrapper, sanitizer instrumentation, or
  ASAN packaging
- **THEN** no job configures with those options, so the change is verified
  only by review — which is why the rejected-configuration failures above are
  worth having: they are the only automatic check some of these options get
