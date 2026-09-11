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

### Upgrading Existing Agents

The complete builtin `vm` and component `topology` are read from the
RocJITsu configs and validated at build time. Every GPU RocJITsu has a
config for becomes one builtin agent: `mi300x`, `mi350x` and `mi450x`
from `gfx942_cdna3.json`, `gfx950_mi355x.json` and `gfx1250_mi455x.json`
(including their 320, 288 and 256 CU topologies), and `gfx90a_mi210_kmd`,
`gfx1100_w7900`, `gfx1151` and `gfx1201_r9700` under their config's own
name. Where RocJITsu ships several configs for one GPU — a `_kmd`
variant, an `_Ngpu` one — Mirage takes the plainest, because the GPU
count is a Mirage topology setting (`--gpus-per-node`) that would
override a baked-in one anyway. Mirage no longer maintains separate
hardware values or uniform 256-CU layouts.

Builtin **profiles** are not read from disk or written to it. Each is
derived from the agent it pins, whenever it is asked for, so there is
nothing to seed and nothing to go stale on upgrade. A profile file is
always one you wrote, and it shadows the builtin of that name until you
delete it.

RocJITsu refuses a device that has SDMA engines and no queues on them,
and it refuses it while loading the config — so a session dies at daemon
start rather than at profile validation. Mirage never wrote
`vm.gpu.device.num_sdma_queues_per_engine` before this release, so an
agent an older Mirage left on disk is one RocJITsu now rejects. On
startup Mirage completes that one field, in place, but only for a stored
agent that is otherwise byte-for-byte the one this Mirage ships: put the
field back and the file *is* the shipped document, so filling it in is
finishing Mirage's own work and the result is the shipped agent whole.
The shipped MI300X and MI350X presets use 8 queues per engine, and MI450X
uses 2.

Any other file is left exactly as it is, because Mirage has no value to
offer it. An agent seeded by an older release and an agent you edited
look alike from here, and both are yours to keep — and this GPU's queue
count describes neither, so writing it in would produce a machine that is
neither what was on disk nor what Mirage ships. An MI350X seeded before
this release has five SDMA engines and four CUs per shader array; filling
in today's eight queues per engine would leave a document belonging to no
GPU either party has heard of. An explicit `0` is likewise an answer, and
Mirage does not argue with it — RocJITsu will.

Such a file is not left silent, though. `mirage state builtins` names it
along with every other builtin document that differs from the shipped
one, with its path and how to take the shipped version instead
(`mirage agent delete <name>`, then run `mirage state builtins` again).
Profile validation refuses the incomplete pair with the reason, at
`profile create`, `profile import` and `run` — while the document can
still be edited, rather than at daemon start.

Agents with older or customized layouts are not overwritten. Missing fields
are reported against the selected RocJITsu preset during profile validation
and config synthesis. Preserve any customizations before deleting an old
builtin agent; the next invocation recreates a missing builtin from the
current preset.

### Profile Compatibility

Additional JSON fields on an agent are preserved recursively and passed to
RocJITsu. Additional profile fields may be placed inside `emulator` or at the
profile root; root extras are normalized into `emulator` when saved. For example:

```json
{
  "name": "custom",
  "emulator": {
    "emulator": "rocjitsu",
    "plugins": {},
    "exec_mode": "Functional",
    "options": {},
    "topology": "MI350X-1x1",
    "max_ticks": 200000,
    "vm": { "gpu": { "device": { "capability2": 1 } } }
  }
}
```

Objects are merged recursively over the agent configuration; scalar values
and arrays replace the corresponding values. Mirage's per-node GPU count
and explicitly selected plugins remain authoritative. Duplicate root and
emulator overrides are rejected. Mirage-only container and system-topology
controls still reject unknown fields *within* them. Replacement stops at
the shape of a field Mirage has a name for: `vm` still has to describe a
device after merging, so a scalar over `vm.gpu.device` or a string over
`num_sdma_engines` is refused when the profile is written.

Passthrough is not validation, and RocJITsu is not a backstop: it parses
the synthesised config with `skip_unexpected_fields_in_json` set, so a key
it does not recognise is dropped without a word (see
`config_common.h`). A misspelled key therefore survives Mirage and
vanishes in RocJITsu, and the machine that comes up is quietly not the one
the file describes. What Mirage can still tell you is that the key was not
one of *its own*: a profile-root field Mirage does not recognise is
forwarded to the emulator with a warning naming it, which is the signal to
read for `containerise`, `descriptoin` and the like. Inside `emulator` and
inside an agent there is no such signal — check those against the preset.

Missing profile fields produce warnings on stderr. Omitted `plugins`,
`exec_mode` and `options` use their defaults; required structural fields
still cause a parse error. Missing agent fields stay absent in the generated
JSON so RocJITsu's schema defaults apply. Warnings identify fields present
in the matching preset, including component config entries; they do not
replace RocJITsu validation. Explicit zeroes are preserved, not mistaken
for missing values. `--config` continues to use the supplied file verbatim.

## Prerequisites

| Tool | Version | Needed for | Notes |
|------|---------|------------|-------|
| Rust + Cargo | 1.88+ (edition 2024) | everything | Install via [rustup](https://rustup.rs). 1.88 is the floor for let-chains, which the workspace uses. |
| CMake | 3.22+ | building rocjitsu from source | Only if you want live GPU emulation. mirage's own wrapper needs 3.20. |
| Ninja | any recent | building rocjitsu from source | `-G Ninja`. |
| C++20 compiler | GCC 12+ / Clang 16+ | building rocjitsu from source | |
| Python | 3.10+ | regenerating rocjitsu's ISA sources | Not a build prerequisite: the generated sources are checked in, and nothing in either CMake build invokes Python. Only needed to re-run the `amdisa` generator. |

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

- the **builtin agents**, one per GPU in `rocjitsu/configs/`, read from
  that directory and validated by the `builtin` build script — plus
  Mirage's system topologies, and the builtin profiles generated from
  those agents at run time, and
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
| `rocjitsu` | on | two entries: `rocjitsu`, the GPU emulator, and `rocjitsu-dbt`, which translates a GPU's code objects to run on a different physical GPU |
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
ctest --test-dir build   # the test suite, plus clippy and rustfmt
```

`ctest` runs three cases, not one: `cargo_test`, and the two lint gates
described under [Linting](#linting).

Options worth knowing: `MIRAGE_CARGO_FEATURES` (comma-separated extra
cargo features), `MIRAGE_CARGO_PROFILE` (default `release`),
`MIRAGE_BUILD_HOTSWAP` (build HotSwap's LLVM + COMGR + ROCR stack from
source — a long build, hence opt-in), `MIRAGE_LINT_TESTS` (on; turn it
off to register only `cargo_test`), and `MIRAGE_ALLOW_TEST_SKIP`, which
is the `ctest` spelling of `MIRAGE_E2E_ALLOW_SKIP=1` described below.

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

Every backend resolves its runtime library through one shared search
policy, so what works for one works for the others. For rocjitsu
(`librocjitsu.so`) the order is:

1. `$ROCJITSU_LIB`, which names the `.so` **file** itself rather than a
   directory;
2. every directory on `$LD_LIBRARY_PATH`;
3. an install layout — `<prefix>/lib` next to a `<prefix>/bin/mirage`.
   This outranks everything below it on purpose: a prefix that ships its
   own library must use it, even when the prefix happens to sit inside a
   checkout whose `emulation/rocjitsu/build` also has one;
4. a sibling monorepo build, found by walking up from the `mirage`
   binary and looking for a rocjitsu build under each ancestor — so an
   integration-test binary in `target/<profile>/deps/` finds it just as
   the CLI does, without anybody counting `..`s;
5. `$ROCM_HOME/lib`, then `$ROCM_PATH/lib`;
6. `$(rocm-sdk path --root)/lib` (present when a ROCm Python wheel venv
   is active);
7. the standard system directories: `/opt/rocm/lib`, `/usr/local/lib`,
   `/usr/lib`, `/usr/lib/x86_64-linux-gnu`;
8. the in-container mount directory, for a containerised session where
   the host libraries are bind-mounted in.

The first path that is actually a file wins. `$ROCJITSU_LIB` naming
something that is not there is skipped rather than fatal, so an
environment left over from another checkout degrades to the search rather
than breaking the build.

The DBT backend follows the same policy for `librocjitsu_hooks.so`, with
`ROCJITSU_HOOKS_LIB` in place of `ROCJITSU_LIB`. HotSwap is the one that
opts out: it takes `HOTSWAP_HOME` (an install root, with
`lib/libhotswap_intercept.so` under it) and does not consult
`$LD_LIBRARY_PATH` or the ROCm variables at all.

Reach for the file overrides — `ROCJITSU_LIB`, `ROCJITSU_HOOKS_LIB` — when
you have a library in a place no search would guess, or when you want to
pin one build while another sits in the way. They are also what you need
when the prefix itself is the problem: step 3 outranks
`ROCM_HOME`/`ROCM_PATH`, so those two select an ordinary install root but
cannot override a `<prefix>/lib` beside the `mirage` you are running —
only `ROCJITSU_LIB` or `$LD_LIBRARY_PATH` can.

You do not have to reason about any of this in the dark. `mirage
emulators -l` prints, per backend, the library it resolved — or, when it
found none, every path it tried and the variables that would fix it:

```sh
$ mirage emulators -l
hotswap
  ...
  installed: no
  runtime:   not found (libhotswap_intercept.so)
  searched:  /path/that/was/tried/libhotswap_intercept.so
  set:       HOTSWAP_HOME=<install root, with lib/libhotswap_intercept.so under it>
```

If nothing is found mirage still builds and runs; the backend is simply
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

The release lane of [RocJITsu CI](../../../.github/workflows/rocjitsu-corpus-tests.yml)
runs `cargo test --locked --workspace --no-fail-fast -- --test-threads=4 --nocapture`
against that job's freshly built RocJITsu libraries, with missing-emulator
skips disabled. Changes to either project trigger the workflow. This
includes daemon startup for every builtin agent — the list is taken from
`mirage agent list`, so a GPU added to RocJITsu is covered without
editing a test — legacy-agent upgrades, the container and lifecycle
suites, and the software-emulator matrix.
Hardware-dependent DBT cases remain capability-gated.

## Linting

The workspace lint policy lives in [`Cargo.toml`](../Cargo.toml) under
`[workspace.lints]`: `unsafe_code` is forbidden everywhere except the
`rocjitsu_sys` FFI crate, `clippy::all` is denied, and `unwrap_used`,
`expect_used`, `panic`, `exit`, `todo`, `unimplemented`, `dbg_macro` and a
few concurrency hazards (`await_holding_lock`, `mem_forget`) are denied
outside test modules.

The table has two halves and only one of them is free. The `rust` half —
`unsafe_code`, `unused_must_use`, `rust_2018_idioms` — is applied by
rustc, so `cargo build` already enforces it. The clippy half is applied
only when clippy is what you ran, and `cargo test` never runs clippy. So
run it:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Every flag is load-bearing. `--all-targets` reaches the integration test
targets under `tests/`, which are where most of the `unwrap` temptation
lives; `--all-features` reaches the `hotswap` backend that the default
feature set leaves uncompiled; and `-D warnings` is what turns the
warn-level entries in the policy (`unreachable_pub`,
`unused_qualifications`, `missing_debug_implementations`) into failures
rather than output you scroll past.

Both checks are also registered as `ctest` cases (`cargo_clippy` and
`cargo_fmt`) unless `MIRAGE_LINT_TESTS` is turned off, so a CMake-driven
build catches a lint regression the same way it catches a failing test.
`cmake --build build --target mirage_lint` runs just those two without the
test suite.

## Troubleshooting

- **`command not found: <cmd>` from `mirage run`** — the program you asked
  mirage to run doesn't exist on `PATH` inside the session. The rank that
  couldn't start reports `mirage: node <n>: <reason>` on mirage's own
  stderr and records exit code 127, rather than the exec quietly running
  the ranks that did start.
- **`no mirage run is serving session <id>`** — the run that owned the
  session has exited. A session exists exactly as long as its `mirage
  run` does; start one in another terminal and `mirage exec` into that.
- **A backend reported as not installed** — run `mirage emulators -l`
  first. It prints every path that was searched for that backend's
  library and the environment variables that would resolve it, which is
  faster than guessing. Then either build rocjitsu (Option B), or point
  `ROCJITSU_LIB` (or `ROCJITSU_HOOKS_LIB`, or `HOTSWAP_HOME`) at a
  library you already have.
