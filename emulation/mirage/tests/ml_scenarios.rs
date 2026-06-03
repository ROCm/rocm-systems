//! Env-gated ML scenario tests.
//!
//! These exercise real ML workloads against rocjitsu's simulated
//! GPU via the `librocjitsu_kmd.so` LD_PRELOAD interposer. The
//! tests do **not** fall back to running the workload natively when
//! the simulator is missing — they either run against the simulator
//! or print `skipping: <reason>` and return.
//!
//! Detection contract:
//!
//! * [`rocjitsu_env()`] — returns `Some((kmd, config, schema))`
//!   when all three prerequisites for an interposed workload are
//!   present on disk. Otherwise `None`.
//! * [`torch_available()`] — runs `python -c "import torch"`.
//! * [`hipcc_available()`] — checks for `hipcc` on `$PATH`.
//! * [`podman_available()`] — runs `podman --version`.
//!
//! Skipped tests print exactly one line `skipping: <reason>` and
//! return; they do not silently pass without exercising anything.

use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

// ----- capability detection --------------------------------------------------

/// Returns `Some((kmd_preload_so, simulation_config_json, schema_fbs))`
/// when the full rocjitsu KMD-interposer pipeline is present.
fn rocjitsu_env() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let kmd = mirage_rocjitsu::kmd_preload()?;
    // Build a minimal EmulatorDef referencing a builtin agent so
    // rocjitsu can synthesise the sim config.
    let _ = mirage_core::agent::store::ensure_builtins(false).ok()?;
    let agents = mirage_core::agent::store::list().ok()?;
    let agent_name = agents.into_iter().next()?;
    let def = mirage_core::emulator::EmulatorDef {
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: mirage_core::emulator::ExecMode::Functional,
        options: Default::default(),
        topology: mirage_core::common::MaybeRef::Owned(mirage_core::topology::TopologyDef {
            racks: 1,
            nodes_per_rack: 1,
            gpus_per_node: 1,
            agent: mirage_core::common::MaybeRef::Ref(agent_name),
        }),
    };
    let (cfg, schema) = mirage_rocjitsu::kmd_config(&def).ok()?;
    Some((kmd, cfg, schema))
}

fn torch_available() -> bool {
    Command::new("python")
        .args(["-c", "import torch"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn hipcc_available() -> bool {
    Command::new("hipcc")
        .arg("--version")
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

/// Compile `tiny_hip.hip` with `hipcc` into `out_dir/tiny_hip`,
/// targeting the offload arch that matches our MI350X KMD config.
/// Returns the binary path on success.
fn compile_tiny_hip(out_dir: &Path) -> PathBuf {
    let src = fixtures_dir().join("tiny_hip.hip");
    let bin = out_dir.join("tiny_hip");
    let status = Command::new("hipcc")
        .arg("--offload-arch=gfx950")
        .arg("-std=c++20")
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .status()
        .expect("failed to invoke hipcc");
    assert!(status.success(), "hipcc failed to compile {}", src.display());
    bin
}

// ----- direct (no-mirage) interposer scenarios -------------------------------
//
// These verify rocjitsu's KMD interposer works for realistic ML
// workloads. They bypass `mirage run` so that a regression in the
// CLI surface doesn't mask a regression in the simulator pipeline.

#[test]
fn hip_vector_add_runs_against_simulated_gpu() {
    let Some((kmd, cfg, schema)) = rocjitsu_env() else {
        eprintln!("skipping: rocjitsu KMD interposer / config / schema not all present");
        return;
    };
    if !hipcc_available() {
        eprintln!("skipping: hipcc not on PATH");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let bin = compile_tiny_hip(tmp.path());
    let output = Command::new(&bin)
        .env("LD_PRELOAD", &kmd)
        .env("RJ_CONFIG", &cfg)
        .env("RJ_SCHEMA", &schema)
        .output()
        .expect("failed to spawn tiny_hip");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "tiny_hip failed under simulator (status {:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );
    assert!(
        stdout.contains("hip_kernel_ok"),
        "expected `hip_kernel_ok` sentinel; stdout was:\n{stdout}"
    );
}

#[test]
fn torch_runs_on_simulated_gpu_via_kmd_preload() {
    let Some((kmd, cfg, schema)) = rocjitsu_env() else {
        eprintln!("skipping: rocjitsu KMD interposer / config / schema not all present");
        return;
    };
    if !torch_available() {
        eprintln!("skipping: python torch not importable");
        return;
    }
    let script = fixtures_dir().join("tiny_torch.py");
    let output = Command::new("python")
        .arg(&script)
        .env("LD_PRELOAD", &kmd)
        .env("RJ_CONFIG", &cfg)
        .env("RJ_SCHEMA", &schema)
        .output()
        .expect("failed to spawn python");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "torch failed under simulator (status {:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );
    assert!(
        stdout.contains("tiny_torch_ok"),
        "expected `tiny_torch_ok` sentinel; stdout was:\n{stdout}\nstderr:\n{stderr}"
    );
}

// ----- mirage-orchestrated scenarios -----------------------------------------
//
// These exercise the full mirage pipeline (profile -> session ->
// exec -> attach) and additionally verify that the user can inject
// the KMD interposer through `mirage run --env`. They are the
// integration counterpart to the direct tests above.

#[test]
fn mirage_run_drives_hip_kernel_under_kmd_preload() {
    let Some((kmd, cfg, schema)) = rocjitsu_env() else {
        eprintln!("skipping: rocjitsu KMD interposer / config / schema not all present");
        return;
    };
    if !hipcc_available() {
        eprintln!("skipping: hipcc not on PATH");
        return;
    }
    let env = Env::new();
    let tmp = tempfile::tempdir().unwrap();
    let bin = compile_tiny_hip(tmp.path());

    env.mirage()
        .args(["profile", "create", "rj", "--emulator", "rocjitsu"])
        .assert()
        .success();

    let assert = env
        .mirage()
        .arg("run")
        .args(["--profile", "rj"])
        .arg("--env")
        .arg(format!("LD_PRELOAD={}", kmd.display()))
        .arg("--env")
        .arg(format!("RJ_CONFIG={}", cfg.display()))
        .arg("--env")
        .arg(format!("RJ_SCHEMA={}", schema.display()))
        .arg("--")
        .arg(&bin)
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    assert!(
        stdout.contains("hip_kernel_ok"),
        "expected `hip_kernel_ok`; got:\n{stdout}"
    );
}

#[test]
fn mirage_run_drives_torch_under_kmd_preload() {
    let Some((kmd, cfg, schema)) = rocjitsu_env() else {
        eprintln!("skipping: rocjitsu KMD interposer / config / schema not all present");
        return;
    };
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
    let assert = env
        .mirage()
        .arg("run")
        .args(["--profile", "rj"])
        .arg("--env")
        .arg(format!("LD_PRELOAD={}", kmd.display()))
        .arg("--env")
        .arg(format!("RJ_CONFIG={}", cfg.display()))
        .arg("--env")
        .arg(format!("RJ_SCHEMA={}", schema.display()))
        .arg("--")
        .arg("python")
        .arg(&script)
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    assert!(
        stdout.contains("tiny_torch_ok"),
        "expected `tiny_torch_ok`; got:\n{stdout}"
    );
}

#[test]
fn mirage_run_rejects_malformed_env_flag() {
    // No prerequisites needed: this is a pure CLI contract test.
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    let assert = env
        .mirage()
        .arg("run")
        .args(["--profile", "p", "--env", "NOEQUALS"])
        .arg("--")
        .arg("true")
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr);
    assert!(
        stderr.contains("--env expects KEY=VALUE"),
        "expected helpful --env error; got:\n{stderr}"
    );
}

// ----- containerized scenario ------------------------------------------------
//
// Verifies the rocjitsu KMD interposer also works inside a podman
// container, when an image with the ROCm runtime is provided via
// `MIRAGE_ML_CONTAINER_IMAGE`. This bypasses the mirage CLI (the
// host does not yet wire `ExecArgs` through a container provider).
// When implemented end-to-end it will move into the mirage pipeline.

#[test]
fn hip_kernel_runs_inside_container_with_kmd_preload() {
    let Some((kmd, cfg, schema)) = rocjitsu_env() else {
        eprintln!("skipping: rocjitsu KMD interposer / config / schema not all present");
        return;
    };
    if !podman_available() {
        eprintln!("skipping: podman not available");
        return;
    }
    if !hipcc_available() {
        eprintln!("skipping: hipcc not on PATH (needed to build the test binary)");
        return;
    }
    let Some(image) = std::env::var_os("MIRAGE_ML_CONTAINER_IMAGE") else {
        eprintln!("skipping: MIRAGE_ML_CONTAINER_IMAGE not set");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let bin = compile_tiny_hip(tmp.path());

    let output = Command::new("podman")
        .args(["run", "--rm"])
        .arg("-v")
        .arg(format!("{}:/work/tiny_hip:ro", bin.display()))
        .arg("-v")
        .arg(format!("{}:/work/kmd.so:ro", kmd.display()))
        .arg("-v")
        .arg(format!("{}:/work/cfg.json:ro", cfg.display()))
        .arg("-v")
        .arg(format!("{}:/work/schema.fbs:ro", schema.display()))
        .args(["-e", "LD_PRELOAD=/work/kmd.so"])
        .args(["-e", "RJ_CONFIG=/work/cfg.json"])
        .args(["-e", "RJ_SCHEMA=/work/schema.fbs"])
        .arg(&image)
        .args(["/work/tiny_hip"])
        .output()
        .expect("failed to spawn podman");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "containerized tiny_hip failed (status {:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );
    assert!(
        stdout.contains("hip_kernel_ok"),
        "expected `hip_kernel_ok`; stdout was:\n{stdout}"
    );
}

