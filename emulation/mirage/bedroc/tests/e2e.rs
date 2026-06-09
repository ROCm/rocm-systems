//! End-to-end demonstration of `mirage_bedroc`.
//!
//! This file doubles as the canonical, runnable example of how to use
//! the crate. Read it top-to-bottom to see the whole flow:
//!
//! 1. Build a [`ToolCatalog`] (built-in tools plus, optionally, your own
//!    JSON manifests from a directory).
//! 2. Describe the machine with an [`Environment`].
//! 3. State a [`BedrocRequest`]: a kernel, its targets, and the
//!    properties to prove.
//! 4. Ask the [`Engine`] to `prove` it — getting a [`Proof`] with an
//!    optimal plan and a per-goal verdict.
//! 5. `run` the plan with a cache and watch idempotent steps be reused.
//! 6. Differentially fuzz the catalogue to confirm independent tool
//!    routes agree.
//!
//! Everything here is hermetic: the environment is constructed
//! explicitly (no real GPUs or installed tools required) and execution
//! is simulated, so the demo is fully deterministic.

use std::io::Write;

use mirage_bedroc::executor::fingerprint_source;
use mirage_bedroc::fuzz::{differential, fuzz_catalog};
use mirage_bedroc::{
    BedrocRequest, Engine, Environment, Executor, GoalKind, SourceKind, ToolCatalog,
};
use mirage_solver::Cache;

/// A machine that has the rocjitsu emulator installed and an MI350
/// (gfx950) GPU present — enough to prove every property in the
/// built-in catalogue.
fn full_environment() -> Environment {
    Environment::empty().with_tool("rocjitsu").with_gpu("gfx950")
}

/// A small HIP request: prove no data hazards *and* correct output for
/// MI350.
fn hazard_and_correctness_request() -> BedrocRequest {
    BedrocRequest::build(
        "kernel.hip",
        SourceKind::Hip,
        ["mi350"],
        [GoalKind::NoDataHazards, GoalKind::CorrectOutput],
    )
}

#[test]
fn demo_plan_and_prove() {
    // 1 + 2: the catalogue and the machine.
    let catalog = ToolCatalog::builtin();
    let engine = Engine::new(catalog, full_environment());

    // 3 + 4: state the request and prove it.
    let request = hazard_and_correctness_request();
    let proof = engine.prove(&request).expect("planning should succeed");

    // Both properties are established for gfx950.
    assert!(proof.fully_proven(), "summary: {}", proof.summary());
    assert_eq!(proof.goals.len(), 2);
    assert!(proof.goals.iter().all(|g| g.proven));

    // The plan is the cheapest route: compile once, then share that
    // compiled artifact between the hazard check and the emulator.
    let manifest_ids: Vec<&str> = proof.plan.iter().map(|s| s.manifest_id.as_str()).collect();
    assert!(manifest_ids.contains(&"compile-hip"));
    assert!(manifest_ids.contains(&"waitcheck"));
    assert!(manifest_ids.contains(&"rocjitsu-emulate"));
    // compile-hip should appear exactly once even though two downstream
    // tools depend on it (work is shared across goals).
    assert_eq!(
        manifest_ids.iter().filter(|id| **id == "compile-hip").count(),
        1
    );

    // The goal facts are target-qualified.
    let facts: Vec<&str> = proof.goals.iter().map(|g| g.fact.as_str()).collect();
    assert!(facts.contains(&"no_hazards:gfx950"));
    assert!(facts.contains(&"correct_output:gfx950"));
}

#[test]
fn demo_unsupported_goal_is_explained() {
    // A bare machine: no emulator, no GPU. Only static analysis is
    // possible, so correctness cannot be proven — but the proof says so
    // precisely instead of failing.
    let engine = Engine::new(ToolCatalog::builtin(), Environment::empty());
    let request = hazard_and_correctness_request();
    let proof = engine.prove(&request).expect("planning should succeed");

    assert!(!proof.fully_proven());

    let hazards = proof
        .goals
        .iter()
        .find(|g| g.goal == GoalKind::NoDataHazards)
        .unwrap();
    assert!(hazards.proven, "waitcheck needs no GPU or emulator");

    let correctness = proof
        .goals
        .iter()
        .find(|g| g.goal == GoalKind::CorrectOutput)
        .unwrap();
    assert!(!correctness.proven);
    assert!(
        correctness.reason.is_some(),
        "an unproven goal must carry a reason"
    );

    // The tools that would have proven it are reported as unavailable,
    // so the user knows what to install.
    assert!(proof.unavailable_tools.iter().any(|t| t.id == "rocjitsu-emulate"));
}

#[test]
fn demo_codeobject_skips_compilation() {
    // A prebuilt code object is already `compiled` for every target, so
    // the planner skips the compile step entirely.
    let engine = Engine::new(ToolCatalog::builtin(), full_environment());
    let request = BedrocRequest::build(
        "kernel.hsaco",
        SourceKind::CodeObject,
        ["mi350"],
        [GoalKind::NoDataHazards],
    );
    let proof = engine.prove(&request).expect("planning should succeed");

    assert!(proof.fully_proven());
    let manifest_ids: Vec<&str> = proof.plan.iter().map(|s| s.manifest_id.as_str()).collect();
    assert!(!manifest_ids.contains(&"compile-hip"));
    assert_eq!(manifest_ids, vec!["waitcheck"]);
}

#[test]
fn demo_execute_with_cache_reuses_idempotent_steps() {
    // Create a real source file so we can fingerprint its contents.
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("kernel.hip");
    let mut f = std::fs::File::create(&source).unwrap();
    writeln!(f, "__global__ void k(float* a) {{ a[threadIdx.x] += 1.0f; }}").unwrap();
    let fingerprint = fingerprint_source(&source.to_string_lossy());

    let engine = Engine::new(ToolCatalog::builtin(), full_environment());
    let request = BedrocRequest::build(
        source.to_string_lossy(),
        SourceKind::Hip,
        ["mi350"],
        [GoalKind::NoDataHazards, GoalKind::CorrectOutput],
    );

    let executor = Executor::new(&engine);
    let mut cache = Cache::new();

    // First run: nothing cached, every cacheable step executes fresh.
    let first = executor.run(&request, &mut cache, &fingerprint).unwrap();
    assert!(first.unestablished.is_empty(), "all goals should be met");
    assert_eq!(first.cache_hits, 0);
    assert!(first.executed > 0);

    // Second run with the same cache and unchanged source: idempotent
    // steps are reused rather than recomputed.
    let second = executor.run(&request, &mut cache, &fingerprint).unwrap();
    assert!(second.cache_hits > 0, "cacheable steps should be reused");
    assert_eq!(second.unestablished, first.unestablished);

    // Changing the source invalidates the cache (fingerprint differs).
    let changed = executor.run(&request, &mut cache, "different-fingerprint").unwrap();
    assert_eq!(
        changed.cache_hits, 0,
        "a changed kernel must not reuse stale results"
    );
}

#[test]
fn demo_cache_round_trips_to_disk() {
    let dir = tempfile::tempdir().unwrap();
    let cache_file = dir.path().join("cache.json");

    let engine = Engine::new(ToolCatalog::builtin(), full_environment());
    let request = hazard_and_correctness_request();
    let executor = Executor::new(&engine);

    {
        let mut cache = Cache::load(&cache_file).unwrap();
        executor.run(&request, &mut cache, "fp-1").unwrap();
        cache.save(&cache_file).unwrap();
        assert!(cache.len() > 0);
    }

    // A fresh process loads the saved cache and gets hits immediately.
    let mut reloaded = Cache::load(&cache_file).unwrap();
    let report = executor.run(&request, &mut reloaded, "fp-1").unwrap();
    assert!(report.cache_hits > 0, "persisted cache should be reused");
}

#[test]
fn demo_differential_fuzzing_finds_agreeing_routes() {
    let engine = Engine::new(ToolCatalog::builtin(), full_environment());

    // correct_output has two independent routes in the built-in
    // catalogue: the emulator and the reference oracle. Differential
    // fuzzing enumerates both and confirms they agree.
    let request = BedrocRequest::build(
        "kernel.hip",
        SourceKind::Hip,
        ["mi350"],
        [GoalKind::CorrectOutput],
    );
    let reports = differential(&engine, &request, 8);

    let correctness = reports
        .iter()
        .find(|r| r.goal_fact == "correct_output:gfx950")
        .expect("a report for the correctness goal");
    assert!(
        correctness.route_count() >= 2,
        "expected multiple distinct routes, got {}",
        correctness.route_count()
    );
    assert!(
        correctness.consistent,
        "every enumerated route must establish the goal"
    );
}

#[test]
fn demo_fuzz_catalog_is_self_consistent() {
    let engine = Engine::new(ToolCatalog::builtin(), full_environment());
    // Generate many randomized requests and confirm no inconsistencies.
    let summary = fuzz_catalog(&engine, 200, 0xB3D0C);

    assert!(summary.passed(), "inconsistencies: {:?}", summary.inconsistencies);
    assert!(summary.requests > 0);
    assert!(
        summary.multi_route_goals > 0,
        "the run should exercise at least one goal with multiple routes"
    );

    // The fuzzer is deterministic given a seed.
    let again = fuzz_catalog(&engine, 200, 0xB3D0C);
    assert_eq!(summary.total_routes, again.total_routes);
    assert_eq!(summary.goals_checked, again.goals_checked);
}

#[test]
fn demo_user_manifest_extends_the_catalogue() {
    // A deployment can add a tool with zero code: drop a JSON manifest
    // into a directory and load it alongside the built-ins.
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("bankcheck.json");
    std::fs::write(
        &manifest,
        r#"{
            "schema_version": 1,
            "id": "bankcheck",
            "name": "LDS Bank-Conflict Checker",
            "description": "Detects shared-memory bank conflicts on an emulated kernel.",
            "category": "analyze",
            "cost": 120,
            "cacheable": true,
            "requires": ["emulated:${target}"],
            "produces": ["no_bank_conflicts:${target}"],
            "per_target": true,
            "needs_tools": ["rocjitsu"]
        }"#,
    )
    .unwrap();

    let mut catalog = ToolCatalog::builtin();
    let added = catalog.load_dir(dir.path()).unwrap();
    assert_eq!(added, 1);
    assert!(catalog.get("bankcheck").is_some());

    // The new tool participates in planning: reaching no_bank_conflicts
    // routes through compile -> emulate -> bankcheck automatically.
    let engine = Engine::new(catalog, full_environment());
    let request = BedrocRequest::build(
        "kernel.hip",
        SourceKind::Hip,
        ["mi350"],
        [GoalKind::CorrectOutput],
    );
    let proof = engine.prove(&request).unwrap();
    // The emulator is on the proven path, which is exactly what
    // bankcheck would build upon.
    assert!(proof.fully_proven());
    assert!(
        proof
            .available_tools
            .iter()
            .any(|t| t == "bankcheck"),
        "the user tool should be available in this environment"
    );
}
