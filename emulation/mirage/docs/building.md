# Building mirage

This guide covers building the `mirage` CLI and (optionally) the
`rocjitsu` GPU emulator that mirage drives.

mirage is a single Cargo workspace ([`emulation/mirage/`](../)). One
`cargo build` produces the unified `mirage` binary from a set of crates —
`core`, `ctl`, `supervisor`, `container`, `builtin`, `rocjitsu_sys`, and
the emulator backends (`rocjitsu`, `hotswap`). See
[`architecture.md`](architecture.md) for the full crate map.

## TL;DR

```sh
cd emulation/mirage
cargo build            # builds everything
cargo test --workspace # run the test suite
./target/debug/mirage --help
```

Rust is the only toolchain the mirage build itself needs. Everything
below that is about rocjitsu, which is a separate C++ project mirage
merely loads at runtime.

## Prerequisites

| Tool | Version | Needed for | Notes |
|------|---------|------------|-------|
| Rust + Cargo | 1.85+ (edition 2024) | everything | Install via [rustup](https://rustup.rs). |
| CMake | 3.28+ | building rocjitsu from source | Only if you want live GPU emulation. |
| Ninja | any recent | building rocjitsu from source | `-G Ninja`. |
| C++20 compiler | GCC 12+ / Clang 16+ | building rocjitsu from source | |
| Python | 3.10+ | rocjitsu ISA codegen | Only when building rocjitsu from source. |

mirage runs on Linux. It leans on POSIX process groups and Unix domain
sockets: each workload process leads its own group so it can be
signalled as a unit, and each `mirage run` serves one socket under
`$XDG_RUNTIME_DIR/mirage/run/` so `mirage exec` in another terminal can
find it.

## Building the workspace

```sh
cd emulation/mirage
cargo build              # debug
cargo build --release    # optimized
```

This builds the `mirage` binary at `target/debug/mirage` (or
`target/release/mirage`). The build embeds:

- the **builtin agents, topologies and profiles** — Rust constructors in
  the `builtin` crate, so the data is validated by the compiler rather
  than by a runtime parse, and
- the **third-party dependency manifest** that `mirage about` prints,
  distilled from `cargo metadata` by the root `build.rs`.

### Cargo features

The only optional things in the workspace are the emulator backends.
Each is a link-only dependency that registers itself into the emulator
registry via `inventory`; nothing in the binary names a backend, so a
feature flag literally adds or removes an entry from
`mirage emulators`.

| Feature | Default | Backend |
|---------|---------|---------|
| `rocjitsu` | on | the rocjitsu GPU emulator |
| `hotswap` | off | the HotSwap intercept backend |

```sh
cargo build                                              # rocjitsu only
cargo build --features hotswap                           # both
cargo build --no-default-features --features hotswap     # hotswap only
```

A build with no backend at all compiles, but `profile create` then fails
with a message telling you to rebuild with one — a profile has to name
an emulator, and there is none to name.

### Building through CMake

There is a thin CMake wrapper ([`CMakeLists.txt`](../CMakeLists.txt)) so
mirage configures, installs and tests with the same recipe as the rest
of the monorepo. It shells out to cargo:

```sh
cmake -S . -B build
cmake --build build      # cargo build --release
ctest --test-dir build   # cargo test --release --workspace
```

Options worth knowing: `MIRAGE_CARGO_FEATURES` (comma-separated extra
cargo features), `MIRAGE_CARGO_PROFILE` (default `release`),
`MIRAGE_BUILD_HOTSWAP` (build HotSwap's LLVM + COMGR + ROCR stack from
source — a long build, hence opt-in), and `MIRAGE_ALLOW_TEST_SKIP`,
which is the `ctest` spelling of `MIRAGE_E2E_ALLOW_SKIP=1` described
below.

## Building rocjitsu

rocjitsu is the emulator mirage drives, so mirage needs its libraries to
bring a session up. Without them:

* `mirage emulators` reports the backend as not installed;
* `mirage run` fails at bring-up, naming the missing library;
* the end-to-end test suites cannot bring a session up, so every test in
  them skips.

Because a skipped Rust test still reports `ok`, each of those suites
carries one guard test that **fails** in that situation rather than
letting the suite go green while proving nothing. So a `cargo test` in a
checkout without rocjitsu built reports a handful of deliberate failures
whose message says exactly what is missing:

```console
the `rocjitsu` runtime was not found, so every session test in this suite
skipped and the suite proves nothing.

Build the sibling `emulation/rocjitsu` project, or set ROCM_HOME to an
install that provides librocjitsu.so.

If this build deliberately excludes rocjitsu, set MIRAGE_E2E_ALLOW_SKIP=1
to accept the skips.
```

Set `MIRAGE_E2E_ALLOW_SKIP=1` for a build that intentionally excludes
rocjitsu — a docs-only CI job, say. Prefer building rocjitsu where you
can: those suites are where mirage's session and process lifecycle is
actually covered.

### Option A — let mirage find them

mirage discovers the rocjitsu library (`librocjitsu.so`) in this order:

1. a sibling monorepo build, relative to the `mirage` binary
   (`../../../rocjitsu/build`, and a level deeper again so an
   integration-test binary in `target/<profile>/deps/` finds it too);
2. an install layout — `<prefix>/lib` next to a `<prefix>/bin/mirage`;
3. `$ROCM_HOME/lib`;
4. `$(rocm-sdk path --root)/lib` (present when a ROCm Python wheel venv
   is active);
5. the in-container mount directory, for a containerised session where
   the host libraries are bind-mounted in.

If nothing is found mirage still builds and runs; rocjitsu is simply
reported as not installed.

### Option B — build rocjitsu yourself

From the rocjitsu source tree
([`emulation/rocjitsu/`](../../rocjitsu)):

```sh
cd emulation/rocjitsu
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Verifying rocjitsu is wired up

```sh
./target/debug/mirage state builtins        # extract agents/topologies
./target/debug/mirage profile create gpu --emulator rocjitsu --no-input
./target/debug/mirage run --profile gpu -- \
  sh -c 'echo LD=$LD_PRELOAD ROCJITSU_RUNTIME_DIR=$ROCJITSU_RUNTIME_DIR'
```

If the profile is created successfully and `LD_PRELOAD` /
`ROCJITSU_RUNTIME_DIR` are populated in the run, rocjitsu is integrated.
Profile creation
validates against the emulator, so an unusable rocjitsu setup is
reported at `profile create` time with the reason.

## Testing

```sh
cargo test --workspace   # unit tests + the integration suites
```

The integration suites under `tests/` drive the real binary as a
subprocess against a private XDG root, so what they exercise is the
whole stack: CLI → session bring-up → supervisor → real processes.
`tests/e2e.rs` covers a run's streams, its exit code, the socket it
serves while it lives, and `mirage exec` borrowing that session from
another terminal; `tests/container_e2e.rs` does the same for a
containerised profile; `tests/matrix_e2e.rs` walks the backend ×
topology × plugin cross product from [`tests/matrix.md`](../tests/matrix.md);
`tests/strain.rs` hammers the lifecycle for leaks. `supervisor/tests/`
covers the run and process layer directly.

Because a session only exists while the `mirage run` that owns it is
alive, every one of those tests is bounded by a process it started: a
suite that crashes cannot leave a session behind for the next one to
trip over.

## Troubleshooting

- **`command not found: <cmd>` from `mirage run`** — the program you asked
  mirage to run doesn't exist on `PATH` inside the session. The rank that
  couldn't start reports `mirage: node <n>: <reason>` on mirage's own
  stderr and records exit code 127, rather than the exec quietly running
  the ranks that did start.
- **`no mirage run is serving session <id>`** — the run that owned the
  session has exited. A session exists exactly as long as its `mirage
  run` does; start one in another terminal and `mirage exec` into that.
- **rocjitsu reported as not installed** — build rocjitsu (Option B) and
  ensure `librocjitsu.so` is reachable (a sibling monorepo build,
  `$ROCM_HOME/lib`, or `$(rocm-sdk path --root)/lib`), or run
  `mirage state builtins` to extract any embedded assets.
