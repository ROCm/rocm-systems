# Building mirage

This guide covers building the `mirage` CLI/daemon, its embedded web
dashboard, and (optionally) the `rocjitsu` GPU emulator that mirage
drives.

mirage is a single Cargo workspace
([`emulation/mirage/`](../)) made of six crates: `core`, `ctl`,
`daemon`, `dashboard`, `host`, and `rocjitsu` (the `mirage_rocjitsu`
asset-embedding crate). One `cargo build` produces the unified
`mirage` binary.

## TL;DR

```sh
cd emulation/mirage
cargo build            # builds everything, including the dashboard SPA
cargo test --workspace # run the test suite
./target/debug/mirage --help
```

The first build also compiles the React dashboard, which requires
**Node.js 20.19+** and **npm** on your `PATH` (see below). If you don't
have them, see [Building without Node.js](#building-without-nodejs).

## Prerequisites

| Tool | Version | Needed for | Notes |
|------|---------|------------|-------|
| Rust + Cargo | 1.85+ (edition 2024) | everything | Install via [rustup](https://rustup.rs). |
| Node.js | **20.19+** (or 22.12+) | the dashboard SPA | Vite 8 / React 19 toolchain. Checked by `dashboard/build.rs`. |
| npm | bundled with Node | the dashboard SPA | Runs `npm ci` + `npm run build`. |
| CMake | 3.28+ | building rocjitsu from source | Only if you want live GPU emulation. |
| Ninja | any recent | building rocjitsu from source | `-G Ninja`. |
| C++20 compiler | GCC 12+ / Clang 16+ | building rocjitsu from source | |
| Python | 3.10+ | rocjitsu ISA codegen | Only when building rocjitsu from source. |

mirage runs on Linux. The per-session host uses PTYs, FIFOs, and POSIX
process groups.

### Installing Node.js

The dashboard `build.rs` runs `node --version` up front and fails with
an actionable message if Node is missing or too old. Recommended ways to
get a current Node:

```sh
# nvm (per-user, no root)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
nvm install 22

# or your distro / nodesource, etc.
node --version   # must be >= 20.19.0
```

If `node`/`npm` live at a non-standard path, point the build at them:

```sh
export NODE=/opt/node/bin/node
export NPM=/opt/node/bin/npm
```

## Building the workspace

```sh
cd emulation/mirage
cargo build              # debug
cargo build --release    # optimized
```

This builds the `mirage` binary at `target/debug/mirage` (or
`target/release/mirage`). The build embeds:

- the **dashboard SPA** (compiled by `dashboard/build.rs` via `npm`),
- the **builtin agents / topologies** (from `agents/*.json`), and
- the **rocjitsu runtime assets** when available (see below).

### Building the dashboard SPA

You normally don't build the SPA by hand — `cargo build` does it through
`dashboard/build.rs`, which:

1. verifies Node.js is **20.19+**,
2. runs `npm ci` when `web/package-lock.json` is newer than the
   installed `node_modules`, and
3. runs `npm run build` (`tsc -b && vite build`) and embeds the output.

To iterate on the front end directly:

```sh
cd dashboard/web
npm ci
npm run dev      # Vite dev server with HMR
npm run build    # production build
npm test         # vitest unit tests
npm run test:e2e # Playwright end-to-end tests
```

### Building without Node.js

If the SPA is already built (e.g. in CI, or a container layer that
prebuilt it), you can skip the Node toolchain entirely:

```sh
MIRAGE_DASHBOARD_SKIP_NPM_CI=1 cargo build
```

This skips the Node version check, `npm ci`, and `npm run build`. The
crate then embeds whatever SPA assets were produced by a previous build.

## Building rocjitsu (optional)

mirage works without rocjitsu (the `noop` emulator runs commands
directly). To get real GPU emulation you need the rocjitsu libraries.

### Option A — let mirage find them

mirage discovers the rocjitsu assets in this order:

1. explicit paths in `ROCJITSU_KMD_LIB`
2. the rocjitsu source tree at `$ROCJITSU_ROOT` or the sibling checkout
   [`../../rocjitsu`](../../rocjitsu) (relative to the `rocjitsu` crate),
   using prebuilt artifacts under `<root>/build/`;

If nothing is found, empty placeholders are staged and mirage still
compiles; rocjitsu is simply reported as not installed.

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
./target/debug/mirage state builtins        # extract agents/topologies/assets
./target/debug/mirage profile create gpu --emulator rocjitsu
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
cargo test --workspace        # Rust unit + e2e tests
cd dashboard/web && npm test  # dashboard unit tests (vitest)
cd dashboard/web && npm run test:e2e  # dashboard Playwright e2e
```

The Rust end-to-end tests in `tests/e2e.rs` and `tests/daemon_e2e.rs`
exercise the full lifecycle (create session → start exec → attach →
signal → stop) through the public CLI and HTTP surfaces.

## Troubleshooting

- **`Node.js X.Y is too old…`** — upgrade Node to 20.19+ (see
  [Installing Node.js](#installing-nodejs)), or set
  `MIRAGE_DASHBOARD_SKIP_NPM_CI=1` if the SPA is prebuilt.
- **`could not run node --version…`** — Node/npm aren't on `PATH`; install
  them or set `NODE`/`NPM`.
- **`command not found: <cmd>` from `mirage run`** — the program you asked
  mirage to run doesn't exist on `PATH` inside the session; the exec ends
  with exit code 127 and the message is shown on its stdout.
- **rocjitsu reported as not installed** — build rocjitsu (Option B) and
  rebuild mirage with `ROCJITSU_ROOT` set, or run
  `mirage state builtins` to extract any embedded assets.
