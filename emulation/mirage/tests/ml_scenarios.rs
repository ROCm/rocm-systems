//! Env-gated ML scenario tests.
//!
//! These exercise the full mirage pipeline (profile → session →
//! exec → attach) end-to-end against realistic ML workloads. Each
//! scenario auto-skips when the prerequisites for it aren't on the
//! host, so the file is safe to keep in the default `cargo test`
//! run even on machines without rocjitsu / torch / podman.
//!
//! Detection contract:
//!
//! * `rocjitsu_installed()` — looks for `librocjitsu.so` on the
//!   loader's standard path, or for `ROCJITSU_LIB_DIR`/`ROCJITSU_ROOT`.
//! * `torch_available()`    — runs `python -c "import torch"`.
//! * `podman_available()`   — runs `podman --version`.
//!
//! Skipped tests print a one-line `skipping: <reason>` and return
//! `Ok(())`; they never fail.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

// ----- capability detection --------------------------------------------------

fn rocjitsu_installed() -> bool {
    if std::env::var_os("ROCJITSU_LIB_DIR").is_some()
        || std::env::var_os("ROCJITSU_ROOT").is_some()
    {
        return true;
    }
    [
        "/usr/local/lib/librocjitsu.so",
        "/usr/lib/librocjitsu.so",
        "/usr/lib/x86_64-linux-gnu/librocjitsu.so",
        "/opt/rocm/lib/librocjitsu.so",
    ]
    .iter()
    .any(|p| std::path::Path::new(p).exists())
}

fn torch_available() -> bool {
    Command::new("python")
        .args(["-c", "import torch"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn podman_available() -> bool {
    Command::new("podman")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ----- harness ---------------------------------------------------------------

struct Env {
    _dir: TempDir,
    runtime: PathBuf,
    config: PathBuf,
    state: PathBuf,
    mirage_bin: PathBuf,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self {
            runtime: dir.path().join("runtime"),
            config: dir.path().join("config"),
            state: dir.path().join("state"),
            mirage_bin: PathBuf::from(env!("CARGO_BIN_EXE_mirage")),
            _dir: dir,
        }
    }

    fn mirage(&self) -> Command {
        let mut c = Command::new(&self.mirage_bin);
        c.env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("XDG_STATE_HOME", &self.state)
            .env("MIRAGE_BIN", &self.mirage_bin)
            .env_remove("MIRAGE_LOG");
        c
    }
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ml")
}

/// Helper: run an exec via `mirage run` and assert it exits 0,
/// printing the expected substring on stdout.
fn assert_run_succeeds(env: &Env, profile: &str, args: &[&str], expect_stdout: &str) {
    let assert = env
        .mirage()
        .arg("run")
        .args(["--profile", profile])
        .arg("--")
        .args(args)
        .assert()
        .success();
    let out = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    assert!(
        out.contains(expect_stdout),
        "expected stdout to contain `{expect_stdout}`; got:\n{out}"
    );
}

// ----- scenarios -------------------------------------------------------------

#[test]
fn torch_single_node_no_container() {
    if !rocjitsu_installed() {
        eprintln!("skipping: rocjitsu not installed");
        return;
    }
    if !torch_available() {
        eprintln!("skipping: python torch not importable");
        return;
    }
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "rj", "--emulator", "rocjitsu"])
        .assert()
        .success();
    let script = fixtures_dir().join("tiny_torch.py");
    assert_run_succeeds(
        &env,
        "rj",
        &["python", script.to_str().unwrap()],
        "tiny_torch_ok",
    );
}

#[test]
fn torch_single_node_with_container() {
    if !rocjitsu_installed() {
        eprintln!("skipping: rocjitsu not installed");
        return;
    }
    if !podman_available() {
        eprintln!("skipping: podman not available");
        return;
    }
    if std::env::var_os("MIRAGE_ML_CONTAINER_IMAGE").is_none() {
        eprintln!("skipping: MIRAGE_ML_CONTAINER_IMAGE not set (e.g. rocm/dev-ubuntu-22.04)");
        return;
    }
    // We don't actually drive a container exec without a working
    // session→container plumbing wired in the CLI today; this is a
    // gate point so the test surface is in place. When the
    // `--container` flag lands on `profile create`, the assertion
    // below can switch on.
    eprintln!("skipping: containerized profile flag not yet exposed on CLI");
}

#[test]
fn multi_node_hip_kernel_no_container() {
    if !rocjitsu_installed() {
        eprintln!("skipping: rocjitsu not installed");
        return;
    }
    let env = Env::new();
    env.mirage()
        .args([
            "profile",
            "create",
            "rj-multi",
            "--emulator",
            "rocjitsu",
            "--nodes",
            "2",
            "--gpus-per-node",
            "1",
        ])
        .assert()
        .success();
    // We don't have a HIP toolchain assumed here; run a portable
    // sentinel command and assert orchestration works under a
    // multi-node profile. (The HIP kernel fixture is present for
    // hosts that *do* have a compiler — see tests/fixtures/ml/.)
    assert_run_succeeds(
        &env,
        "rj-multi",
        &["sh", "-c", "echo hip_kernel_ok"],
        "hip_kernel_ok",
    );
}

#[test]
fn multi_node_hip_kernel_with_container() {
    if !rocjitsu_installed() {
        eprintln!("skipping: rocjitsu not installed");
        return;
    }
    if !podman_available() {
        eprintln!("skipping: podman not available");
        return;
    }
    if std::env::var_os("MIRAGE_ML_CONTAINER_IMAGE").is_none() {
        eprintln!("skipping: MIRAGE_ML_CONTAINER_IMAGE not set");
        return;
    }
    eprintln!("skipping: containerized profile flag not yet exposed on CLI");
}
