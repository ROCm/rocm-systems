# mirage

`mirage` is a user-facing UX (CLI + dashboard) for the [`rocjitsu`][rocjitsu]
GPU emulator and other emulator backends. It is designed so that:

* **All state lives on disk** in standard [XDG locations][xdg].  There is
  no required daemon to keep state alive — the CLI and the per-session
  host processes communicate exclusively through well-defined files.
* **Sessions are long-lived** processes that may host an emulator, manage
  containers, and shepherd the lifecycle of user execs (commands).
* **Execs are individual command invocations** within a session, with
  fully-redirected stdio (FIFO for stdin, plain files for stdout/stderr)
  and a published per-node `pid`/`exit_code`.

This makes mirage easy to inspect (`ls`, `cat`, `tail -f`), easy to
script (`--json` on every list/show), and easy to recover from crashes
(restart the host; it picks up where it left off).

```text
$ mirage profile create cdna3 --emulator rocjitsu
$ mirage run --profile cdna3 -- ./my-rocm-app --flag
```

See:

* [`docs/cli.md`](docs/cli.md) — full CLI reference.
* [`docs/state-layout.md`](docs/state-layout.md) — on-disk layout reference.
* [`docs/host.md`](docs/host.md) — what the per-session host does and
  how to extend it.
* [`docs/architecture.md`](docs/architecture.md) — design overview.

[rocjitsu]: ../rocjitsu/
[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

## Building

```sh
cargo build --workspace
```

Produces the unified `mirage` binary under `target/debug/`. The build
also compiles the embedded web dashboard, which needs **Node.js 20.19+**
and **npm** on your `PATH`.

See [`docs/building.md`](docs/building.md) for the full guide:
prerequisites (Rust, Node, optional rocjitsu/CMake), building the
dashboard SPA, building rocjitsu, environment variables, and
troubleshooting.

## Testing

```sh
cargo test --workspace
```

The end-to-end tests in `tests/e2e.rs` cover the full lifecycle
(create session → start exec → attach → signal → stop) using only the
public CLI surface.
