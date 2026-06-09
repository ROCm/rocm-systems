//! `mirage bedroc`: composable kernel-correctness planning.
//!
//! Bedroc takes a high-level request — a kernel, its target GPUs, and
//! the properties to prove — and combines it with the implicit
//! constraints of the current machine (which GPUs are present, which
//! emulators are installed) to produce an optimal plan and proof using
//! the [`mirage_bedroc`] engine.
//!
//! Subcommands:
//!
//! * `mirage bedroc tools` — list the correctness tools in the catalogue.
//! * `mirage bedroc plan` — plan a proof without running anything.
//! * `mirage bedroc run` — plan and (simulate) execute, using the cache.
//! * `mirage bedroc fuzz` — differentially fuzz the tool catalogue.
//!
//! Tools are defined by **JSON manifests**, never hardcoded. The
//! built-in catalogue is augmented by any `*.json` manifests under
//! `<MIRAGE_CONFIG>/bedroc/tools/` (and an optional `--tools-dir`). See
//! `docs/bedroc.md`.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context as _;
use clap::{Args, Subcommand};
use mirage_bedroc::executor::fingerprint_source;
use mirage_bedroc::{
    BedrocRequest, Engine, Environment, Executor, GoalKind, SourceKind, ToolCatalog,
};
use mirage_solver::Cache;

/// `mirage bedroc` subcommands.
#[derive(Subcommand, Debug)]
pub enum BedrocCmd {
    /// List the correctness tools available in the catalogue and
    /// whether each can run in the current environment.
    Tools {
        /// Show long form (description, requirements, effects).
        #[arg(short = 'l', long)]
        long: bool,
        /// Additional directory of JSON tool manifests to include.
        #[arg(long = "tools-dir")]
        tools_dir: Option<PathBuf>,
    },
    /// Plan a proof for a kernel without executing anything.
    Plan(RequestArgs),
    /// Plan and (simulate) execute a proof, reusing cached results.
    Run(RequestArgs),
    /// Differentially fuzz the tool catalogue: enumerate distinct tool
    /// routes that should establish the same property and confirm they
    /// agree.
    Fuzz {
        /// Number of randomized requests to generate.
        #[arg(long, default_value_t = 200)]
        iterations: usize,
        /// Seed for reproducible generation.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Additional directory of JSON tool manifests to include.
        #[arg(long = "tools-dir")]
        tools_dir: Option<PathBuf>,
    },
}

/// Shared flags describing a bedroc request.
#[derive(Args, Debug)]
pub struct RequestArgs {
    /// Path to the kernel source (informational; used for cache
    /// fingerprinting).
    #[arg(long)]
    source: Option<String>,
    /// The kind of source: `hip`, `asm`, or `codeobject`.
    #[arg(long = "source-kind", default_value = "hip")]
    source_kind: String,
    /// A target architecture to prove for. Accepts marketing names
    /// (`mi350`) or gfx targets (`gfx950`). May be repeated.
    #[arg(long = "target", required = true)]
    targets: Vec<String>,
    /// A property to prove, e.g. `no-data-hazards`, `correct-output`,
    /// `no-data-races`, `fp-correct`. May be repeated.
    #[arg(long = "prove", required = true)]
    goals: Vec<String>,
    /// Additional directory of JSON tool manifests to include.
    #[arg(long = "tools-dir")]
    tools_dir: Option<PathBuf>,
    /// Ignore and do not update the on-disk result cache.
    #[arg(long)]
    no_cache: bool,
}

/// Where user-supplied tool manifests live by default.
fn default_tools_dir() -> PathBuf {
    mirage_core::paths::mirage_config_dir()
        .join("bedroc")
        .join("tools")
}

/// Where the bedroc result cache is stored.
fn cache_path() -> PathBuf {
    mirage_core::paths::mirage_cache_dir()
        .join("bedroc")
        .join("cache.json")
}

/// Build the tool catalogue: the built-ins plus the default user
/// directory plus any explicit `--tools-dir`.
fn load_catalog(extra_dir: Option<&PathBuf>) -> anyhow::Result<ToolCatalog> {
    let mut catalog = ToolCatalog::builtin();
    catalog
        .load_dir(&default_tools_dir())
        .context("loading default tool manifests")?;
    if let Some(dir) = extra_dir {
        catalog
            .load_dir(dir)
            .with_context(|| format!("loading tool manifests from {}", dir.display()))?;
    }
    Ok(catalog)
}

/// Probe the environment: physical GPUs from the kernel topology, plus
/// the installed emulators reported by the registry (these are the
/// "tools" bedroc manifests can require).
fn probe_environment() -> Environment {
    let mut env = Environment::probe();
    for spec in crate::registry() {
        if spec.installed {
            env = env.with_tool(spec.name);
        }
    }
    env
}

/// Parse the shared request flags into a [`BedrocRequest`].
fn build_request(args: &RequestArgs) -> anyhow::Result<BedrocRequest> {
    let source_kind = SourceKind::parse(&args.source_kind)
        .with_context(|| format!("unknown source kind `{}`", args.source_kind))?;
    let mut goals = Vec::new();
    for g in &args.goals {
        let parsed =
            GoalKind::parse(g).with_context(|| format!("unknown property to prove `{g}`"))?;
        goals.push(parsed);
    }
    let source = args.source.clone().unwrap_or_default();
    Ok(BedrocRequest::build(
        source,
        source_kind,
        args.targets.clone(),
        goals,
    ))
}

/// Dispatch a `mirage bedroc` subcommand.
pub fn dispatch(cmd: BedrocCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        BedrocCmd::Tools { long, tools_dir } => tools_cmd(long, tools_dir.as_ref(), json),
        BedrocCmd::Plan(args) => plan_cmd(args, json),
        BedrocCmd::Run(args) => run_cmd(args, json),
        BedrocCmd::Fuzz {
            iterations,
            seed,
            tools_dir,
        } => fuzz_cmd(iterations, seed, tools_dir.as_ref(), json),
    }
}

fn tools_cmd(long: bool, tools_dir: Option<&PathBuf>, json: bool) -> anyhow::Result<ExitCode> {
    let catalog = load_catalog(tools_dir)?;
    let env = probe_environment();
    if json {
        let tools: Vec<_> = catalog.iter().cloned().collect();
        println!("{}", serde_json::to_string_pretty(&tools)?);
        return Ok(ExitCode::from(0));
    }
    if !long {
        println!("{:<20} {:<10} DESCRIPTION", "ID", "CATEGORY");
    }
    for m in catalog.iter() {
        let needs_unmet: Vec<&String> = m
            .needs_tools
            .iter()
            .filter(|t| !env.has_tool(t))
            .collect();
        let available = needs_unmet.is_empty() && (!m.needs_gpu || env.has_gpu());
        if long {
            println!("{} ({}) [{}]", m.id, m.category, if available { "available" } else { "unavailable" });
            println!("    {}", m.description);
            println!("    requires: {}", join_or_none(&m.requires));
            println!("    produces: {}", m.produces.join(", "));
            if !m.needs_tools.is_empty() {
                println!("    needs tools: {}", m.needs_tools.join(", "));
            }
            println!("    cost: {}  cacheable: {}", m.cost, m.cacheable);
            println!();
        } else {
            let mark = if available { "" } else { " (unavailable)" };
            println!("{:<20} {:<10} {}{}", m.id, m.category, m.description, mark);
        }
    }
    Ok(ExitCode::from(0))
}

fn join_or_none(v: &[String]) -> String {
    if v.is_empty() {
        "(none)".to_string()
    } else {
        v.join(", ")
    }
}

fn plan_cmd(args: RequestArgs, json: bool) -> anyhow::Result<ExitCode> {
    let catalog = load_catalog(args.tools_dir.as_ref())?;
    let env = probe_environment();
    let request = build_request(&args)?;
    let engine = Engine::new(catalog, env);
    let proof = engine.prove(&request).context("planning proof")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&proof)?);
    } else {
        print!("{}", proof.summary());
        print_unavailable(&proof);
    }
    Ok(exit_code_for(proof.fully_proven()))
}

fn run_cmd(args: RequestArgs, json: bool) -> anyhow::Result<ExitCode> {
    let catalog = load_catalog(args.tools_dir.as_ref())?;
    let env = probe_environment();
    let request = build_request(&args)?;
    let engine = Engine::new(catalog, env);

    let cache_file = cache_path();
    let mut cache = if args.no_cache {
        Cache::new()
    } else {
        Cache::load(&cache_file).unwrap_or_default()
    };

    let source_fp = fingerprint_source(&request.source);
    let report = {
        let executor = Executor::new(&engine);
        executor
            .run(&request, &mut cache, &source_fp)
            .context("executing plan")?
    };
    let proof = engine.prove_with_cache(&request, &cache)?;

    if !args.no_cache {
        cache.save(&cache_file).ok();
    }

    if json {
        let combined = serde_json::json!({
            "proof": proof,
            "execution": report,
        });
        println!("{}", serde_json::to_string_pretty(&combined)?);
    } else {
        print!("{}", proof.summary());
        println!(
            "executed {} step(s), reused {} from cache",
            report.executed, report.cache_hits
        );
        print_unavailable(&proof);
    }
    Ok(exit_code_for(proof.fully_proven()))
}

fn fuzz_cmd(
    iterations: usize,
    seed: u64,
    tools_dir: Option<&PathBuf>,
    json: bool,
) -> anyhow::Result<ExitCode> {
    let catalog = load_catalog(tools_dir)?;
    // Fuzzing exercises the catalogue logic itself, so assume a fully
    // capable environment (every emulator installed) to reach every
    // tool route.
    let mut env = Environment::probe();
    for spec in crate::registry() {
        env = env.with_tool(spec.name);
    }
    let engine = Engine::new(catalog, env);
    let summary = mirage_bedroc::fuzz::fuzz_catalog(&engine, iterations, seed);
    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "fuzzed {} requests; {} goals; {} routes; {} multi-route goals",
            summary.requests, summary.goals_checked, summary.total_routes, summary.multi_route_goals
        );
        if summary.passed() {
            println!("all routes agree with the planner");
        } else {
            println!("INCONSISTENCIES:");
            for i in &summary.inconsistencies {
                println!("  {i}");
            }
        }
    }
    Ok(exit_code_for(summary.passed()))
}

fn print_unavailable(proof: &mirage_bedroc::Proof) {
    if !proof.unavailable_tools.is_empty() {
        println!("unavailable tools in this environment:");
        for t in &proof.unavailable_tools {
            println!("  {} — {}", t.id, t.reason);
        }
    }
}

fn exit_code_for(ok: bool) -> ExitCode {
    if ok {
        ExitCode::from(0)
    } else {
        ExitCode::from(1)
    }
}
