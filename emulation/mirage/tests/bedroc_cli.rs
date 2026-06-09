//! End-to-end tests for the `mirage bedroc` CLI surface.
//!
//! These drive the unified `mirage` binary as a subprocess, exercising
//! the full path: argument parsing -> catalogue loading -> planning ->
//! (simulated) execution -> JSON rendering. They double as a worked
//! example of the command-line workflow described in `docs/bedroc.md`.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::str;
use serde_json::Value;
use tempfile::TempDir;

struct Env {
    dir: TempDir,
    config: PathBuf,
    runtime: PathBuf,
    mirage_bin: PathBuf,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config");
        let runtime = dir.path().join("runtime");
        let mirage_bin = PathBuf::from(env!("CARGO_BIN_EXE_mirage"));
        Self {
            dir,
            config,
            runtime,
            mirage_bin,
        }
    }

    fn mirage(&self) -> Command {
        let mut c = Command::new(&self.mirage_bin);
        c.env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("XDG_STATE_HOME", self.dir.path().join("state"))
            .env("XDG_CACHE_HOME", self.dir.path().join("cache"))
            .env_remove("MIRAGE_LOG");
        c
    }
}

#[test]
fn tools_lists_the_builtin_catalogue() {
    let env = Env::new();
    env.mirage()
        .args(["bedroc", "tools"])
        .assert()
        .success()
        .stdout(str::contains("waitcheck"))
        .stdout(str::contains("rocjitsu-emulate"));
}

#[test]
fn tools_json_is_a_valid_manifest_array() {
    let env = Env::new();
    let out = env
        .mirage()
        .args(["--json", "bedroc", "tools"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let tools: Value = serde_json::from_slice(&out.stdout).unwrap();
    let arr = tools.as_array().expect("tools JSON is an array");
    assert!(arr.iter().any(|t| t["id"] == "waitcheck"));
    // Every manifest declares the current schema version.
    assert!(arr.iter().all(|t| t["schema_version"] == 1));
}

#[test]
fn plan_proves_hazards_without_any_environment() {
    let env = Env::new();
    // The static hazard check needs neither a GPU nor an emulator, so a
    // hazard proof is always reachable, even in a bare test sandbox.
    let out = env
        .mirage()
        .args([
            "--json",
            "bedroc",
            "plan",
            "--source",
            "kernel.hip",
            "--target",
            "mi350",
            "--prove",
            "no-data-hazards",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let proof: Value = serde_json::from_slice(&out.stdout).unwrap();
    let goals = proof["goals"].as_array().unwrap();
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0]["fact"], "no_hazards:gfx950");
    assert_eq!(goals[0]["proven"], true);
    // The plan compiles then runs waitcheck.
    let plan = proof["plan"].as_array().unwrap();
    let manifests: Vec<&str> = plan
        .iter()
        .map(|s| s["manifest_id"].as_str().unwrap())
        .collect();
    assert!(manifests.contains(&"compile-hip"));
    assert!(manifests.contains(&"waitcheck"));
}

#[test]
fn run_exits_success_for_a_provable_goal() {
    let env = Env::new();
    env.mirage()
        .args([
            "bedroc",
            "run",
            "--source",
            "kernel.hip",
            "--target",
            "mi350",
            "--prove",
            "no-data-hazards",
        ])
        .assert()
        .success();
}

#[test]
fn fuzz_reports_no_inconsistencies() {
    let env = Env::new();
    let out = env
        .mirage()
        .args(["--json", "bedroc", "fuzz", "--iterations", "50", "--seed", "1"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let summary: Value = serde_json::from_slice(&out.stdout).unwrap();
    let inconsistencies = summary["inconsistencies"].as_array().unwrap();
    assert!(inconsistencies.is_empty(), "fuzzing found inconsistencies");
}

#[test]
fn tools_dir_extends_the_catalogue() {
    let env = Env::new();
    let tools_dir = env.dir.path().join("extra-tools");
    std::fs::create_dir_all(&tools_dir).unwrap();
    std::fs::write(
        tools_dir.join("bankcheck.json"),
        r#"{
            "schema_version": 1,
            "id": "bankcheck",
            "name": "LDS Bank-Conflict Checker",
            "description": "Detects shared-memory bank conflicts.",
            "category": "analyze",
            "cost": 120,
            "produces": ["no_bank_conflicts:${target}"],
            "requires": ["emulated:${target}"],
            "per_target": true,
            "needs_tools": ["rocjitsu"]
        }"#,
    )
    .unwrap();

    env.mirage()
        .args(["bedroc", "tools", "--tools-dir"])
        .arg(&tools_dir)
        .assert()
        .success()
        .stdout(str::contains("bankcheck"));
}

#[test]
fn unknown_property_is_rejected() {
    let env = Env::new();
    env.mirage()
        .args([
            "bedroc",
            "plan",
            "--source",
            "kernel.hip",
            "--target",
            "mi350",
            "--prove",
            "definitely-not-a-real-property",
        ])
        .assert()
        .failure();
}
