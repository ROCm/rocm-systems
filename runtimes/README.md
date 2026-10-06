<!-- SPDX-License-Identifier: MIT -->

# ROCm Runtime Components

The `runtimes` directory owns one Cargo workspace and a standalone CMake build
for runtime components developed together in this repository.

ROCm Systems is currently in a transitional state as key runtime components
move into a monorepo layout. This directory will be built out over time as that
migration progresses. Components here are early-access software and are not
part of the repository's default build or installation yet.

## Contents

- [`api-headers`](api-headers/README.md) contains the vendored AMDF, HSA, DRM,
  KFD, and UDMABUF API headers used by runtime implementations. Its README
  records their primary sources and synchronization policy.
- [`ddi/rocddi`](ddi/rocddi/README.md) is the early-access, private,
  implementation-neutral AMD GPU device-interface layer.
- [`ddi/libamdf`](ddi/libamdf/README.md) implements the AMDF C table ABI and
  builds a static library. Its shared ABI is an alias of the shared image.
- [`hsa/libhsa`](hsa/libhsa/README.md) implements the HSA C ABI and builds a
  shared ABI in the same image. It consumes rocddi directly, not through AMDF.

Component directories own implementation and local tests. The root owns the
shared-image linker package, lockfile, toolchain, Rust profiles, and common
CMake integration. Grouping directories do not introduce additional
Cargo workspaces or CMake projects.
Rocddi remains an internal `rlib`; it has no standalone native ABI.

## Shared image

The workspace root is the `rocddi-frontends` Cargo package. It implements no
third frontend. Its small `src/lib.rs` makes Cargo link the AMDF and HSA `rlib`s
into one `cdylib`, so both public ABIs use one copy of rocddi's process state.
The root `build.rs` retains both entry point sets, applies HSA symbol versions
and SONAME, and marks the image `NODELETE` while native KFD and DRM owners may
remain live. `cmake/SharedImage.cmake` owns the corresponding CMake targets.

The CMake build below stages one physical `libhsa-runtime64.so.1` with
`libhsa-runtime64.so`, `libhsa_runtime64.so`, `libamdf.so.0`, and
`libamdf.so` aliases. The AMDF static archive is built separately.

## Build with CMake

Provision the pinned toolchain from `rust-toolchain.toml` (Rust/Cargo 1.98.0,
rustfmt, and Clippy) before configuring. CMake 3.25+, Ninja, Python 3.11+, native
C/C++ compilers, and `ld.lld` must also be available. Existing generated bindings
are checked in; bindgen and libclang are not needed. Nothing is downloaded by
the CMake configure or build.

From the rocm-systems repository root:

```sh
cmake -S runtimes -B build/runtimes -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/runtimes
ctest --test-dir build/runtimes --output-on-failure
```

Use `ROCM_RUNTIMES_CARGO`, `ROCM_RUNTIMES_RUSTC`, and `ROCM_RUNTIMES_LLD` to
provide explicit executable paths. An already-installed rustup toolchain also
works; configure disables implicit toolchain installation. The selected LLD
directory is added to Cargo's command environment so HSA's `-fuse-ld=lld` works.
For example, an existing ROCm installation can supply `llvm/bin/ld.lld`.

Debug, Release, RelWithDebInfo, and MinSizeRel map to workspace Cargo profiles.
Use separate CMake build directories for each configuration. All CMake-driven
Cargo output is under the binary directory's `cargo/`; selected native artifacts
are copied into `lib/` only when their contents change. Build-tree SONAME
symlinks are provided there. The `rocm_runtime_rust` target invokes Cargo on each
build; Cargo decides what needs recompilation.

Build-tree consumers can link `rocm_runtime::amdf_shared`, `rocm_runtime::amdf_static`, or
`rocm_runtime::hsa_shared`. The two shared targets resolve to one image and
retain native process state together. These targets include headers and build ordering. The AMDF
static target propagates rustc's reported native link requirements through a
generated linker response file. `rocm_runtime::runtime_headers` exposes the source
header tree for internal tests, not an installed SDK contract.

The library helper takes keyword arguments for the Cargo package/target, native
output name, type, and ABI version. Component `project(... VERSION ...)` calls and the shared-image module
own native versions; the helper handles filenames, symlinks, SONAME settings,
and per-static-library link requirements. Platform-specific conventions live
in the helper, with explicit errors for Windows/Darwin until implemented.
The configure helper recognizes native Linux x86-64, AArch64, PPC64/PPC64LE,
and RISC-V64 toolchains. This is not runtime backend support: rocddi currently
supports only Linux x86-64/AArch64 and rejects other architectures at compile
time in `ddi/rocddi/src/driver/builtin.rs`. Only x86-64 has been validated locally.
Python's standard-library TOML parser requires Python 3.11 or newer.
`runtime-rust-config.cmake` in the build directory records the generated Cargo
command and environment for inspection.

With `BUILD_TESTING=ON` (the default), CTest runs build-helper tests, the Rust unit suite, AMDF C ABI
layout and shared/static negotiation checks, and an HSA pre-initialization C ABI
check. These tests do not require GPU activation. Device execution examples
remain explicit qualification tools, not automatic build tests.

## Dependency inputs

All CMake-driven Cargo metadata/build/test commands run frozen and offline.
The current lockfile has no external dependencies. Future dependency inputs
must be prepared before configuration; missing inputs fail rather than fetch.

- `ROCM_RUNTIMES_CARGO_HOME`: writable Cargo home, defaulting to `cargo-home/`
  in the binary directory. Can point to a prepared registry/Git cache.
- `ROCM_RUNTIMES_CARGO_CONFIG`: optional path to prepared Cargo configuration,
  for example source replacement pointing to vendored crates. Relative paths
  are made absolute against the configure helper's invocation directory before
  running Cargo, without resolving symlinks. The absolute path is retained in
  the CMake cache and generated commands so builds and regeneration do not
  depend on the caller's working directory.
- `ROCM_RUNTIMES_CARGO_JOBS`: optional positive parallel-job limit.

Native C libraries should be discovered by CMake and explicitly supplied to
Rust consumers when needed. No native package-discovery abstraction is added
before there is a concrete dependency requiring it.

## Direct Rust development

From `runtimes/`, with the pinned tools and LLD on `PATH`:

```sh
cargo build --workspace --frozen
cargo test --workspace --all-targets --all-features --frozen
cargo clippy --workspace --all-targets --all-features --frozen -- -D warnings
cargo fmt --all --check
```

Direct Cargo uses the workspace `target/` directory unless `CARGO_TARGET_DIR`
is set. Its output tree is separate from the CMake build. GPU behavior retains the qualification limits in component docs.
