//! Performance + accuracy benchmark across the mirage emulator
//! backends (`noop`, `rocjitsu`, `hotswap`).
//!
//! This is an *integration benchmark*: it drives the real `mirage`
//! CLI exactly as a user would (`profile create` → `run`), executing a
//! real HIP GPU workload under each installed-and-supported emulator
//! and measuring two things per backend:
//!
//! * **Accuracy** — the workload kernel validates its own numeric
//!   result on-device (`c[i] == 3*i`) and only prints the
//!   `hip_kernel_ok` sentinel + exits `0` when every element is
//!   correct. A run counts as *correct* iff exit code is `0` **and**
//!   the sentinel is present. This is a genuine functional-accuracy
//!   check: under rocjitsu's software simulator a wrong answer means
//!   the simulator mis-modelled the GPU; under hotswap a wrong answer
//!   means the load-time ISA rewrite produced incorrect code.
//! * **Performance** — wall-clock latency of the end-to-end
//!   `mirage run`, sampled over N iterations (after one warm-up), with
//!   min / mean / median / max / stddev reported. A `/bin/true` run is
//!   also timed per emulator to isolate fixed framework + emulator
//!   bring-up overhead from the workload cost.
//!
//! The benchmark is **capability-gated**: emulators that are not
//! installed or not supported on this host are skipped (and noted in
//! the report), and the whole benchmark no-ops with a single
//! `skipping: <reason>` line when `hipcc` is unavailable — it never
//! silently passes without measuring anything.
//!
//! On completion it writes a polished Markdown report to
//! `target/emulator-benchmark/report.md` (override with
//! `MIRAGE_BENCH_REPORT`). Iteration count is configurable via
//! `MIRAGE_BENCH_ITERS` (default 5).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use tempfile::TempDir;

/// Emulator backends to compare, in report order.
const EMULATORS: &[&str] = &["noop", "rocjitsu", "hotswap"];

/// GPU architectures baked into the benchmark fat binary so that the
/// same executable runs natively (`noop`), under the software
/// simulator (`rocjitsu`), and after a load-time rewrite (`hotswap`,
/// which retargets `gfx1250` code onto the physical card).
const OFFLOAD_ARCHS: &[&str] = &["gfx942", "gfx950", "gfx1250"];

// ----- capability detection --------------------------------------------------

fn hipcc_available() -> bool {
    Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// One row of `mirage emulators --json`.
#[derive(serde::Deserialize)]
struct EmulatorRow {
    name: String,
    installed: bool,
    support: SupportRow,
}

#[derive(serde::Deserialize)]
struct SupportRow {
    supported: bool,
    reason: String,
}

// ----- harness ---------------------------------------------------------------

struct Env {
    _dir: TempDir,
    config: PathBuf,
    runtime: PathBuf,
    state: PathBuf,
    mirage_bin: PathBuf,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self {
            config: dir.path().join("config"),
            runtime: dir.path().join("runtime"),
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

    /// The emulator registry as mirage sees it on this host.
    fn registry(&self) -> Vec<EmulatorRow> {
        let out = self
            .mirage()
            .args(["emulators", "--json"])
            .output()
            .expect("failed to run `mirage emulators --json`");
        serde_json::from_slice(&out.stdout).expect("malformed emulators JSON")
    }
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ml")
}

fn iters() -> usize {
    std::env::var("MIRAGE_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(5)
}

fn report_path() -> PathBuf {
    std::env::var("MIRAGE_BENCH_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target/emulator-benchmark/report.md")
        })
}

/// Compile `tiny_hip.hip` into a multi-arch fat binary so it can run
/// under every backend. Returns the binary path on success.
fn compile_fat_binary(out_dir: &Path) -> PathBuf {
    let src = fixtures_dir().join("tiny_hip.hip");
    let bin = out_dir.join("bench_hip");
    let mut cmd = Command::new("hipcc");
    for arch in OFFLOAD_ARCHS {
        cmd.arg(format!("--offload-arch={arch}"));
    }
    let status = cmd
        .arg("-std=c++20")
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .status()
        .expect("failed to invoke hipcc");
    assert!(status.success(), "hipcc failed to build {}", src.display());
    bin
}

// ----- measurement -----------------------------------------------------------

/// Outcome of a single timed `mirage run`.
struct Sample {
    wall_ms: f64,
    exit_ok: bool,
    correct: bool,
}

/// Run `mirage run --profile <emu> -- <args...>` once, timing the full
/// invocation and checking the accuracy sentinel.
fn timed_run(env: &Env, emulator: &str, args: &[&Path], sentinel: Option<&str>) -> Sample {
    let mut cmd = env.mirage();
    cmd.arg("run").args(["--profile", emulator]).arg("--");
    for a in args {
        cmd.arg(a);
    }
    let start = Instant::now();
    let out = cmd.output().expect("failed to spawn `mirage run`");
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let exit_ok = out.status.success();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let correct = match sentinel {
        Some(s) => exit_ok && stdout.contains(s),
        None => exit_ok,
    };
    Sample {
        wall_ms,
        exit_ok,
        correct,
    }
}

/// Aggregate statistics over a set of samples.
struct Stats {
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    stddev: f64,
}

impl Stats {
    fn from(times: &[f64]) -> Self {
        let n = times.len();
        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sum: f64 = sorted.iter().sum();
        let mean = sum / n as f64;
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let var = sorted.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64;
        Stats {
            min: sorted[0],
            max: sorted[n - 1],
            mean,
            median,
            stddev: var.sqrt(),
        }
    }
}

/// Per-emulator collected result.
struct EmulatorResult {
    name: String,
    /// `None` when the emulator was skipped; carries the reason in
    /// `skip_reason`.
    workload: Option<Stats>,
    overhead: Option<Stats>,
    runs: usize,
    correct: usize,
    exit_ok: usize,
    skip_reason: Option<String>,
}

// ----- the benchmark ---------------------------------------------------------

#[test]
fn benchmark_emulators_and_write_report() {
    if !hipcc_available() {
        eprintln!("skipping: hipcc not on PATH (cannot build the GPU workload)");
        return;
    }

    let env = Env::new();
    let registry = env.registry();

    // Build the workload once; shared by every backend.
    let bin_dir = tempfile::tempdir().unwrap();
    let fat = compile_fat_binary(bin_dir.path());
    let true_bin = PathBuf::from("/bin/true");

    let n = iters();
    let mut results: Vec<EmulatorResult> = Vec::new();

    for &emu in EMULATORS {
        let row = registry.iter().find(|r| r.name == emu);
        // Skip emulators that aren't usable on this host, recording why.
        let skip = match row {
            None => Some("not present in the registry".to_string()),
            Some(r) if !r.installed => Some("not installed".to_string()),
            Some(r) if !r.support.supported => {
                Some(format!("unsupported: {}", r.support.reason))
            }
            Some(_) => None,
        };
        if let Some(reason) = skip {
            results.push(EmulatorResult {
                name: emu.to_string(),
                workload: None,
                overhead: None,
                runs: 0,
                correct: 0,
                exit_ok: 0,
                skip_reason: Some(reason),
            });
            continue;
        }

        // Fresh profile per backend, fully non-interactive.
        env.mirage()
            .args(["profile", "create", emu, "--emulator", emu, "--no-input"])
            .output()
            .expect("failed to create profile");

        // Warm-up (asset extraction, first-touch caches) — not timed.
        let _ = timed_run(&env, emu, &[&fat], Some("hip_kernel_ok"));

        let mut wl_times = Vec::with_capacity(n);
        let mut oh_times = Vec::with_capacity(n);
        let mut correct = 0usize;
        let mut exit_ok = 0usize;
        for _ in 0..n {
            let s = timed_run(&env, emu, &[&fat], Some("hip_kernel_ok"));
            if s.correct {
                correct += 1;
            }
            if s.exit_ok {
                exit_ok += 1;
            }
            wl_times.push(s.wall_ms);

            // Framework + emulator bring-up overhead baseline.
            let o = timed_run(&env, emu, &[&true_bin], None);
            oh_times.push(o.wall_ms);
        }

        results.push(EmulatorResult {
            name: emu.to_string(),
            workload: Some(Stats::from(&wl_times)),
            overhead: Some(Stats::from(&oh_times)),
            runs: n,
            correct,
            exit_ok,
            skip_reason: None,
        });
    }

    let report = render_report(&results, n);
    let path = report_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, &report).expect("failed to write benchmark report");
    eprintln!("benchmark report written to {}", path.display());
    // Echo to test output too, so `cargo test -- --nocapture` shows it.
    eprintln!("\n{report}");

    // Contract: at least one emulator must have actually run, and every
    // emulator that ran must have been functionally correct on every
    // iteration (no silent accuracy regressions).
    let ran: Vec<&EmulatorResult> = results.iter().filter(|r| r.runs > 0).collect();
    assert!(
        !ran.is_empty(),
        "no emulator was runnable on this host; benchmark measured nothing"
    );
    for r in ran {
        assert_eq!(
            r.correct, r.runs,
            "emulator `{}` produced an incorrect result in {}/{} runs \
             (accuracy regression)",
            r.name,
            r.runs - r.correct,
            r.runs
        );
    }
}

// ----- report rendering ------------------------------------------------------

fn fmt_ms(v: f64) -> String {
    format!("{v:.1}")
}

fn render_report(results: &[EmulatorResult], n: usize) -> String {
    let mut s = String::new();
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");

    writeln!(s, "# Mirage Emulator Benchmark Report").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "_Generated {now}_").unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "Comparison of the mirage emulator backends running an identical real \
         HIP GPU workload (a vector-add kernel that self-validates its result \
         on-device) end-to-end through the `mirage` CLI."
    )
    .unwrap();
    writeln!(s).unwrap();

    // Methodology.
    writeln!(s, "## Methodology").unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "- **Workload:** `tiny_hip.hip`, a HIP vector-add (`c[i] = a[i] + b[i]`) \
         compiled as a fat binary for `{}`. The kernel checks every element \
         on-device and prints the `hip_kernel_ok` sentinel only when all \
         results are correct.",
        OFFLOAD_ARCHS.join("`, `")
    )
    .unwrap();
    writeln!(
        s,
        "- **Driver:** each run is a full `mirage run --profile <emulator> -- \
         <binary>` invocation (profile create → session bring-up → exec → \
         attach → teardown), exactly as an end user would invoke it."
    )
    .unwrap();
    writeln!(
        s,
        "- **Samples:** {n} timed iterations per backend after one untimed \
         warm-up. Wall-clock latency is reported as min / mean / median / max \
         / stddev."
    )
    .unwrap();
    writeln!(
        s,
        "- **Overhead baseline:** a `mirage run ... -- /bin/true` is timed \
         alongside each workload run to separate fixed framework + emulator \
         bring-up cost from the workload itself."
    )
    .unwrap();
    writeln!(
        s,
        "- **Accuracy:** a run is *correct* iff it exits `0` **and** emits the \
         `hip_kernel_ok` sentinel — a genuine functional-correctness check of \
         the emulated numeric result."
    )
    .unwrap();
    writeln!(s).unwrap();

    // Backends under test.
    writeln!(s, "## Backends").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "| Emulator | Role |").unwrap();
    writeln!(s, "| --- | --- |").unwrap();
    writeln!(
        s,
        "| `noop` | Pass-through baseline — runs the workload directly on the \
         physical GPU with no emulation. |"
    )
    .unwrap();
    writeln!(
        s,
        "| `rocjitsu` | Software GPU emulator — executes the workload against a \
         simulated AMD GPU (no physical GPU required). |"
    )
    .unwrap();
    writeln!(
        s,
        "| `hotswap` | Load-time ISA rewriter — retargets device code built for \
         another GPU (e.g. `gfx1250`) onto the physical card at load time. |"
    )
    .unwrap();
    writeln!(s).unwrap();

    // Accuracy table.
    writeln!(s, "## Accuracy").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "| Emulator | Correct runs | Exit-0 runs | Result |").unwrap();
    writeln!(s, "| --- | --- | --- | --- |").unwrap();
    for r in results {
        if let Some(reason) = &r.skip_reason {
            writeln!(s, "| `{}` | — | — | skipped ({reason}) |", r.name).unwrap();
        } else {
            let verdict = if r.correct == r.runs {
                "✅ accurate"
            } else {
                "❌ incorrect"
            };
            writeln!(
                s,
                "| `{}` | {}/{} | {}/{} | {verdict} |",
                r.name, r.correct, r.runs, r.exit_ok, r.runs
            )
            .unwrap();
        }
    }
    writeln!(s).unwrap();

    // Performance table.
    writeln!(s, "## Performance (end-to-end `mirage run`, milliseconds)").unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "| Emulator | Min | Mean | Median | Max | Stddev | Overhead (median) | Workload Δ (median) |"
    )
    .unwrap();
    writeln!(s, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |").unwrap();
    for r in results {
        match (&r.workload, &r.overhead) {
            (Some(w), Some(o)) => {
                let delta = w.median - o.median;
                writeln!(
                    s,
                    "| `{}` | {} | {} | {} | {} | {} | {} | {} |",
                    r.name,
                    fmt_ms(w.min),
                    fmt_ms(w.mean),
                    fmt_ms(w.median),
                    fmt_ms(w.max),
                    fmt_ms(w.stddev),
                    fmt_ms(o.median),
                    fmt_ms(delta),
                )
                .unwrap();
            }
            _ => {
                let reason = r.skip_reason.as_deref().unwrap_or("skipped");
                writeln!(
                    s,
                    "| `{}` | — | — | — | — | — | — | skipped ({reason}) |",
                    r.name
                )
                .unwrap();
            }
        }
    }
    writeln!(s).unwrap();
    writeln!(
        s,
        "_“Workload Δ” is the median workload latency minus the median \
         `/bin/true` overhead for the same backend — an estimate of the time \
         attributable to executing the kernel under that emulator._"
    )
    .unwrap();
    writeln!(s).unwrap();

    // Relative performance (vs noop), when noop ran.
    if let Some(base) = results
        .iter()
        .find(|r| r.name == "noop")
        .and_then(|r| r.workload.as_ref())
        .map(|w| w.median)
    {
        writeln!(s, "## Relative latency (vs `noop` baseline)").unwrap();
        writeln!(s).unwrap();
        writeln!(s, "| Emulator | Median | Slowdown vs `noop` |").unwrap();
        writeln!(s, "| --- | ---: | ---: |").unwrap();
        for r in results {
            if let Some(w) = &r.workload {
                let ratio = w.median / base;
                writeln!(
                    s,
                    "| `{}` | {} ms | {:.2}× |",
                    r.name,
                    fmt_ms(w.median),
                    ratio
                )
                .unwrap();
            }
        }
        writeln!(s).unwrap();
    }

    writeln!(
        s,
        "> Note: latencies are dominated by fixed `mirage` session \
         orchestration; for this small kernel the per-emulator compute \
         difference is best read from the *Workload Δ* column. Larger \
         workloads widen the gap, especially for the software simulator."
    )
    .unwrap();

    s
}
