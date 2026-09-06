# amdsmi-rocm-os-packages Specification

## Purpose

Defines the `amdrocm-amdsmi` deb/rpm that TheRock builds from the `core-amdsmi`
artifact, and the tarballs cut from the same source.

There are **two independent OS package families** carrying AMD SMI, and
conflating them is the main hazard this capability exists to prevent:

| | `amd-smi-lib` | `amdrocm-amdsmi` |
| --- | --- | --- |
| Built by | AMD SMI's own CMake/CPack ([amdsmi-python-system-package]) | TheRock, from the artifact |
| Source of contents | the CMake install tree | the `core-amdsmi` artifact components |
| Prefix | `/opt/rocm` | `/opt/rocm/core-X.Y` |
| Python module | site-packages **and** `share/amd_smi` | `share/amd_smi` only |
| System libraries | host `libdrm`/`libnl`/`libmnl` | vendored `lib/rocm_sysdeps` |
| Maintainer scriptlets | postinst/prerm/postun, including `ldconfig` registration | none |
| Version | AMD SMI library version + ROCm libpatch | ROCm SDK version |

Artifact slicing is specified in [amdsmi-therock-artifact]; the build that
produces the staged tree is [amdsmi-therock-subproject]; the wheel channel built
from the same artifact is [amdsmi-rocm-python-distribution].

## Requirements

### Requirement: The Package Is Declared In The Native Package Descriptor

`amdrocm-amdsmi` SHALL be defined as an entry in TheRock's native package
descriptor, drawing its contents from the `core-amdsmi` artifact's `lib`, `run`,
`dev`, and `doc` components under the `amdsmi` artifact subdirectory.

#### Scenario: All four components ship in one package

- **WHEN** the package is built
- **THEN** the library, the CLI and Python module, the headers, and the docs are
  delivered together — unlike the pip channel, which takes only `lib`+`run` into
  the core wheel and routes the rest through devel

#### Scenario: Adding a component requires a descriptor edit

- **WHEN** a new component is added to the artifact
- **THEN** it is absent from the package until the descriptor's component list
  names it

#### Scenario: The `dbg` component becomes a debug package

- **WHEN** packages are generated
- **THEN** a corresponding debuginfo/dbgsym package is produced alongside,
  unless the entry disables it

### Requirement: The Package Is Architecture-Neutral

`amdrocm-amdsmi` SHALL be declared as not GPU-architecture-specific, so exactly
one package is produced regardless of how many GPU families a build targets.

#### Scenario: Arch-specific metapackages share one AMD SMI package

- **WHEN** a build produces per-arch core metapackages for several targets
- **THEN** every one of them depends on the same non-arch `amdrocm-amdsmi`
  package, rather than an arch-suffixed variant

#### Scenario: The package name carries no gfx suffix

- **WHEN** the package file is produced
- **THEN** its name has no `-gfxNNNN` suffix, matching the artifact's
  target-neutral declaration

### Requirement: Versioned And Non-Versioned Package Pairs

For the package entry, the build SHALL produce a versioned package
`amdrocm-amdsmi<X.Y>` carrying the contents, and a non-versioned
`amdrocm-amdsmi` meta package that carries no payload and depends on the
versioned one.

#### Scenario: A user can pin or float the ROCm version

- **WHEN** a user installs `amdrocm-amdsmi`
- **THEN** they get the latest available ROCm version through the meta package;
  installing `amdrocm-amdsmi<X.Y>` instead pins that release series

#### Scenario: Side-by-side minor releases coexist

- **WHEN** two ROCm minor releases are installed
- **THEN** each versioned package installs under its own `/opt/rocm/core-X.Y`
  prefix, and only major.minor pairs — not patch versions — coexist

### Requirement: Installation Prefix Is Version-Scoped

Packages SHALL install under `/opt/rocm/core-<major>.<minor>`, derived from the
ROCm version passed to the packaging step.

#### Scenario: The AMD SMI package alone does not put the CLI on PATH

- **WHEN** only `amdrocm-amdsmi` is installed
- **THEN** `amd-smi` is not invocable until `<prefix>/bin` is prepended to
  `PATH`, because the package ships no maintainer scriptlets at all; it is the
  `amdrocm-core` metapackage's post-install that registers `/usr/bin/amd-smi` as
  an `update-alternatives` link to the selected prefix

#### Scenario: The relative resolution chain still closes

- **WHEN** `amd-smi` is run from the versioned prefix
- **THEN** the CLI resolves `share/amd_smi` and then `lib/<SONAME>` relative to
  its own location, so the version-scoped prefix needs no extra configuration —
  and no `ld.so.conf.d` entry is written by any of these packages

### Requirement: Dependencies Reflect The Vendored System Libraries

The package SHALL depend on `amdrocm-sysdeps` and a loose `python3 >= 3.6.8`,
plus `libc6` on Debian. It SHALL NOT depend on the host `libdrm` development
packages. The sysdeps dependency SHALL be emitted with the same version suffix
as the package itself, so a versioned package pulls its own release's sysdeps.

#### Scenario: Netlink and DRM come from the ROCm sysdeps package

- **WHEN** the package is installed
- **THEN** its library resolves the vendored `libdrm`/`libnl`/`libmnl` from
  `amdrocm-sysdeps` rather than the distribution's own copies

#### Scenario: No versioned interpreter dependency is needed

- **WHEN** the package is installed on an RPM distro
- **THEN** a loose `python3 >= 3.6.8` requirement suffices, because the module
  ships in the prefix-relative `share/amd_smi` rather than in a version-specific
  site-packages directory — the constraint that forces `python(abi) = X.Y` on
  the `amd-smi-lib` package does not apply here

#### Scenario: The two package families are not interchangeable

- **WHEN** a host has `amd-smi-lib` installed and the user installs
  `amdrocm-amdsmi`
- **THEN** they occupy different prefixes and different package namespaces; a
  plain `import amdsmi` continues to resolve the `amd-smi-lib` site-packages
  copy, and would resolve nothing at all on a host with only `amdrocm-amdsmi`,
  unless the user puts the ROCm prefix's `share/amd_smi` on `sys.path`

### Requirement: RPATH Normalization At Packaging Time

`RUNPATH` entries in packaged binaries and libraries SHALL be converted to
`RPATH` by default, with a `--runpath-pkg` opt-out available.

#### Scenario: Transitive dependency resolution behaves predictably

- **WHEN** the packaged `libamd_smi.so` and `amdsmitst` are loaded
- **THEN** their origin-relative search paths apply to transitive dependencies
  too, which `RUNPATH` alone would not guarantee

#### Scenario: RUNPATH can be retained deliberately

- **WHEN** the packaging step is invoked with the runpath option
- **THEN** the conversion is skipped, for cases where `LD_LIBRARY_PATH` override
  behavior is wanted

### Requirement: Package Versions Follow The Release Channel

Native package versions SHALL use the ROCm SDK version with channel-specific
suffixes, which differ between deb and rpm:

| Release type | deb | rpm |
| ------------ | --- | --- |
| release | `X.Y.Z` | `X.Y.Z` |
| prerelease | `X.Y.Z~preN` | `X.Y.Z~rcN` |
| nightly | `X.Y.Z~YYYYMMDD` | `X.Y.Z~YYYYMMDD` |
| ci, dev | `X.Y.Z~devYYYYMMDD` | `X.Y.Z~YYYYMMDDg<short-sha>` |

The corresponding wheel versions are specified in
[amdsmi-rocm-python-distribution]; both come from the same version computation
and both report the ROCm SDK version rather than AMD SMI's library version.

#### Scenario: Package managers order the channels correctly

- **WHEN** two versions from different channels are compared by apt/dnf/zypper
- **THEN** the `~` forms sort below the corresponding stable release, so a
  nightly never shadows a stable upgrade

### Requirement: Packages Are Published Per Distribution Channel

Built packages SHALL be uploaded to the storage bucket and repository URL that
matches their release type — ci, dev, nightly, prerelease, or release — with RPM
repositories nested under an architecture subdirectory.

#### Scenario: A user adds the matching repository

- **WHEN** a user follows the install instructions for a channel
- **THEN** they configure the repository URL for that channel and receive AMD
  SMI along with the rest of the ROCm Core SDK

#### Scenario: Fork CI uploads are isolated

- **WHEN** packages are built from a fork or external repository
- **THEN** they are uploaded under an `<owner>-<repo>/` prefix in a separate
  external bucket, so they cannot be mistaken for first-party artifacts

### Requirement: Tarballs Are Cut From The Same Artifacts

Per-family and, where applicable, combined multi-arch tarballs SHALL be produced
by flattening the fetched artifacts into a single install-prefix layout.

#### Scenario: The tarball provides the library and CLI but not an import

- **WHEN** a user extracts a distribution tarball
- **THEN** `bin/amd-smi`, `lib/libamd_smi.so.<MAJOR>`, `libexec/amdsmi_cli/`,
  and `share/amd_smi/amdsmi/` are present and the CLI runs, but `import amdsmi`
  from an arbitrary interpreter still fails because nothing was added to
  `sys.path`

#### Scenario: AMD SMI appears once in a multi-arch tarball

- **WHEN** a combined multi-arch tarball is produced
- **THEN** AMD SMI's target-neutral artifacts contribute a single copy shared by
  all targets

#### Scenario: Test content is excluded by default

- **WHEN** tarballs are generated without `--include-test-tarballs`
- **THEN** the `test` component is excluded — but AMD SMI's own test tree is
  claimed by its `run` component, so `share/amd_smi/tests` ships in the default
  tarball regardless
