# amdsmi-therock-artifact Specification

## Purpose

Defines the `core-amdsmi` artifact: the slice of AMD SMI's TheRock stage tree
that every downstream distribution channel is cut from. Python wheels
([amdsmi-rocm-python-distribution]), OS packages ([amdsmi-rocm-os-packages]),
and tarballs are all assembled from these component directories, never from the
stage tree directly. The stage tree itself is specified in
[amdsmi-therock-subproject].

An artifact is a per-component extract of the build tree that preserves the
`.../stage` structure plus an `artifact_manifest.txt`. Components are processed
through a fixed *extends chain* so each file is claimed exactly once. AMD SMI's
descriptor is short, but the interaction between its explicit includes and the
per-component defaults determines which channel every file reaches — including
some routing that is not obvious from reading the descriptor alone.

## Requirements

### Requirement: A Single Target-Neutral Artifact

AMD SMI SHALL provide exactly one artifact, named `core-amdsmi`, declared
`TARGET_NEUTRAL` and assembled from the `amdsmi` subproject's stage directory.
It SHALL declare the components `lib`, `run`, `dbg`, `dev`, and `doc`.

#### Scenario: Artifact directories carry the generic target suffix

- **WHEN** artifacts are built
- **THEN** they are named `core-amdsmi_{component}_generic` under
  `build/artifacts`, because AMD SMI contains no GPU-target-specific device code

#### Scenario: One artifact per GPU family is never produced

- **WHEN** a multi-family build runs
- **THEN** a single `_generic` set of AMD SMI artifacts is shared by every
  family, rather than being duplicated per target

#### Scenario: AMD SMI is its own artifact rather than part of base

- **WHEN** a consumer wants only GPU management tooling
- **THEN** it can depend on `core-amdsmi` alone, which is why AMD SMI is split
  out from the other core artifacts

### Requirement: Component Routing Follows The Extends Chain

Components SHALL be processed in the order `lib → run → dbg → dev → doc → test`,
each skipping files already claimed by its predecessors, so the components are
disjoint. Explicit `include` patterns in the descriptor SHALL be **added to**
each component's default patterns rather than replacing them, because
`default_patterns` defaults to true.

For AMD SMI this yields:

| Component | Descriptor includes | Effective claim |
| --------- | ------------------- | --------------- |
| `lib` | `lib/**` | everything under `lib/`, plus the default `**/*.so`, `**/*.so.*` patterns |
| `run` | `bin/**`, `libexec/**`, `share/amd_smi/**` | those three trees (the `run` defaults are empty) |
| `dbg` | none | the default `.build-id/**/*.debug` |
| `dev` | none | the defaults (`**/*.a`, `**/cmake/**`, `**/include/**`, `**/pkgconfig/**`, …) minus anything already claimed |
| `doc` | none | the default `**/share/doc/**` |

#### Scenario: The whole lib directory lands in the lib component

- **WHEN** the descriptor gives `lib` an explicit `include = ["lib/**"]`
- **THEN** `lib` claims everything beneath `lib/` — including the CMake package
  config under `lib/cmake/amd_smi/` and the static libraries — before `dev`'s
  default `**/cmake/**` and `**/*.a` patterns are considered, so those files
  ship in the `lib` component rather than `dev`

#### Scenario: Headers are what actually reaches dev

- **WHEN** the `dev` component is computed
- **THEN** it effectively receives only the `include/**` trees (the public
  `amd_smi` headers, the vendored `e_smi` headers, and the two Go shim
  headers), because `lib/**` already consumed the package config and the
  archives

#### Scenario: The Go shim rides along in lib and dev

- **WHEN** the `lib` and `dev` claims are applied to the stage tree
- **THEN** they also sweep up `libgoamdsmi_shim64.so*` and the Go shim headers,
  because the build produces them unconditionally and both claims are
  path-based — nothing in the descriptor mentions Go, yet every TheRock-derived
  channel ships the binding

#### Scenario: An empty run entry would swallow the later components

- **WHEN** a descriptor edit drops `run`'s explicit include list
- **THEN** `run` becomes a catch-all — it has no default patterns — and claims
  every file `lib` did not, leaving `dev` and `doc` empty and silently emptying
  the headers out of every downstream channel

#### Scenario: The test tree ships inside the run component

- **WHEN** AMD SMI is built with tests enabled
- **THEN** `share/amd_smi/tests/**` is claimed by `run`, because the artifact
  declares no `test` component and `run`'s `share/amd_smi/**` include matches
  the test subtree — so the gtest binary and the Python test suite travel with
  the runtime component into every downstream channel

#### Scenario: Package documentation is not swallowed by run

- **WHEN** AMD SMI installs its license and docs under `share/doc/amd-smi-lib`
- **THEN** `run`'s `share/amd_smi/**` pattern does not match, and the files fall
  through to `doc` as intended

### Requirement: The Run Component Carries The Complete Python Delivery

The `run` component SHALL include `share/amd_smi/**`, `libexec/**`, and
`bin/**`, which together are the entire Python-facing surface of AMD SMI in a
TheRock build. The interpreter's site-packages tree is an absolute system path
outside the stage tree and cannot be captured — see [amdsmi-therock-subproject].

#### Scenario: The three pieces must travel together

- **WHEN** any downstream channel takes the `run` component
- **THEN** it gets `bin/amd-smi`, `libexec/amdsmi_cli/`, and
  `share/amd_smi/amdsmi/` together, which is exactly what the CLI's fallback
  chain and the wrapper's relative library lookup require

#### Scenario: lib and run together are the minimum runnable set

- **WHEN** a consumer installs only `core-amdsmi_lib` and `core-amdsmi_run`
- **THEN** `amd-smi` runs, because `lib` supplies `libamd_smi.so.<MAJOR>` and
  `run` supplies the module and CLI

### Requirement: Artifacts Preserve Build-Tree Structure And A Manifest

Each artifact directory SHALL mirror the build tree's `.../stage` layout and
SHALL contain an `artifact_manifest.txt` listing the relative stage directories
it was drawn from. Archives SHALL write that manifest first.

#### Scenario: Flattening produces an install prefix

- **WHEN** `fileset_tool.py artifact-flatten` processes the artifact
- **THEN** the listed stage directories are merged into a single install-prefix
  layout (`bin/`, `lib/`, `libexec/`, `share/`), which is the form every
  packaging step consumes

#### Scenario: Manifest-first ordering is a precondition for flattening

- **WHEN** an artifact archive is created as `.tar.xz` or `.tar.zst`
- **THEN** `artifact_manifest.txt` is the first member, so a streaming consumer
  can flatten without reading the whole archive first

#### Scenario: Artifacts can bootstrap a partial build

- **WHEN** artifact directories are copied into a build tree with the
  appropriate marker files
- **THEN** the build system reuses the staged output instead of configuring and
  building AMD SMI, which is how multi-stage CI parallelizes at subproject
  granularity

### Requirement: The Artifact Is Registered In TheRock's Mapping Tables

The `core-amdsmi` artifact SHALL be registered so tooling can map between
subproject, artifact, artifact group, and build stage:

| Table | Entry |
| ----- | ----- |
| `build_tools/project_mappings.json` | the `amdsmi` rocm-systems project maps to `core-amdsmi` |
| `build_tools/artifact_subprojects.json` | `core-amdsmi` maps back to the `amdsmi` subproject |
| `BUILD_TOPOLOGY.toml` `[artifacts.core-amdsmi]` | artifact group `core-amdsmi`, type `target-neutral`, artifact deps `base`, `sysdeps`, `sysdeps-libnl`, `sysdeps-libmnl`, `amd-llvm`, feature `CORE_AMDSMI` in group `CORE` |
| `BUILD_TOPOLOGY.toml` `[artifact_groups.core-amdsmi]` | source set `rocm-systems`, group deps `base`, `third-party-sysdeps`, `compiler` |
| `BUILD_TOPOLOGY.toml` `[build_stages.compiler-runtime]` | includes the `core-amdsmi` artifact group |

#### Scenario: A change under projects/amdsmi selects the right CI work

- **WHEN** CI determines which artifacts a rocm-systems pull request affects
- **THEN** the `amdsmi` → `core-amdsmi` mapping selects the `core-amdsmi`
  artifact group and the `compiler-runtime` build stage

#### Scenario: Downstream artifacts declare the dependency explicitly

- **WHEN** RDC, rocprofiler-systems, rocprofiler-compute, or another consumer is
  built
- **THEN** it lists `core-amdsmi` in its artifact dependencies, so a stage-aware
  fetch pulls AMD SMI's artifacts before building it

#### Scenario: Users can install the artifact selectively

- **WHEN** a developer runs the artifact installer
- **THEN** `core-amdsmi_run` and `core-amdsmi_lib` are part of the base pattern
  set, so a test environment gets AMD SMI without installing the full SDK

### Requirement: Adding Or Renaming Components Requires Updating Consumers

A change to the artifact's component set SHALL be propagated to the tools that
name those components explicitly.

#### Scenario: A new component is invisible until consumers are updated

- **WHEN** a component is added to the artifact descriptor
- **THEN** the Python package filter, the OS package descriptor, and the
  artifact installer must each be updated to reference it, or its files are
  silently absent from those channels

#### Scenario: Moving files between components changes every channel

- **WHEN** a descriptor edit moves files from `run` to a new `test` component
- **THEN** those files leave the Python core wheel and the OS package at the
  same time, because both select components by name — and they additionally
  start being dropped from tarballs, which exclude the `test` component by
  default
