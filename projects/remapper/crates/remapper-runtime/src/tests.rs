use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use anyhow::{Context, Result};
use remapper_rewrite::{manifest::Manifest, mode::Mode};

use super::*;

const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../remapper-rewrite/tests/fixtures/wgremap"
);

fn tools_available() -> bool {
    let ok = remapper_rewrite::llvm_tool("llvm-objdump").is_ok()
        && remapper_rewrite::llvm_tool("llvm-mc").is_ok();
    if !ok {
        eprintln!("skipping session tests: ROCm LLVM tools unavailable");
    }
    ok
}

/// Device memory plus a model of the remap prologue for ordinary (non-cluster) table-model launches: workgroup `w`
/// lands on key `placement(w)` and takes the next turn there.
struct FakeGpu {
    memory: BTreeMap<u64, Vec<u8>>,
    next: u64,
    ctl_offset: u64,
    /// Passes (by order of `run`) whose placement differs from the natural one.
    drift: BTreeSet<u32>,
    runs: u32,
    keys: u32,
}

impl FakeGpu {
    fn new() -> Self {
        Self {
            memory: BTreeMap::new(),
            next: 0x1000_0000,
            ctl_offset: 0,
            drift: BTreeSet::new(),
            runs: 0,
            keys: 64,
        }
    }

    fn region(&mut self, va: u64, len: u64) -> Result<(&mut Vec<u8>, usize)> {
        let (&base, buf) = self
            .memory
            .range_mut(..=va)
            .next_back()
            .context("unmapped address")?;
        let off = (va - base) as usize;
        anyhow::ensure!(
            off + len as usize <= buf.len(),
            "access past the end of a buffer"
        );
        Ok((buf, off))
    }

    fn u32_at(&mut self, va: u64) -> u32 {
        u32::from_le_bytes(self.read(va, 4).unwrap().try_into().unwrap())
    }

    fn u64_at(&mut self, va: u64) -> u64 {
        u64::from_le_bytes(self.read(va, 8).unwrap().try_into().unwrap())
    }

    /// Execute a pass the way the rewritten kernel would.
    fn run(&mut self, plan: &PassPlan) {
        let drifted = self.drift.contains(&self.runs);
        self.runs += 1;
        let Some(d) = plan.dispatch else { return };
        let ctl = self.u64_at(d.kernarg_address + self.ctl_offset);
        let mode = self.u32_at(ctl);
        let max_k = self.u32_at(ctl + 4);
        let (arrival, table, counters, log) = (
            self.u64_at(ctl + 8),
            self.u64_at(ctl + 16),
            self.u64_at(ctl + 24),
            self.u64_at(ctl + 32),
        );
        let n = (0..3)
            .map(|i| d.grid_size[i] / d.workgroup_size[i])
            .product::<u32>();
        let (mut overflow, mut logged) = (0u32, 0u32);
        let mut turns = vec![0u32; self.keys as usize];
        for w in 0..n {
            let key = if drifted && w < 8 { 0 } else { w % self.keys };
            let turn = turns[key as usize];
            turns[key as usize] += 1;
            self.write(arrival + key as u64 * 4, &turns[key as usize].to_le_bytes())
                .unwrap();
            let logical = if mode == Mode::Table.code() || mode == Mode::MarkTable.code() {
                let e = if turn < max_k {
                    self.u32_at(table + (key * max_k + turn) as u64 * 8)
                } else {
                    u32::MAX
                };
                if e == u32::MAX {
                    overflow += 1;
                    w
                } else {
                    e
                }
            } else {
                w
            };
            if mode == Mode::Record.code() || mode == Mode::Table.code() {
                let r = [key, turn, logical, 0, 1, 0, w, 0];
                let bytes: Vec<u8> = r.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.write(log + logged as u64 * 32, &bytes).unwrap();
                logged += 1;
            }
        }
        let c: Vec<u8> = [overflow, logged]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        self.write(counters, &c).unwrap();
    }
}

/// The session owns one handle; the test keeps another to run passes.
#[derive(Clone)]
struct Gpu(Rc<RefCell<FakeGpu>>);

impl Device for Gpu {
    fn alloc(&mut self, bytes: u64) -> Result<u64> {
        Ok(self.0.borrow_mut().alloc(bytes))
    }

    fn write(&mut self, va: u64, data: &[u8]) -> Result<()> {
        self.0.borrow_mut().write(va, data)
    }

    fn read(&mut self, va: u64, len: u64) -> Result<Vec<u8>> {
        self.0.borrow_mut().read(va, len)
    }

    fn load_kernel(&mut self, code: &[u8], symbol: &str) -> Result<LoadedKernel> {
        let a = remapper_rewrite::analyze::analyze_code_object(code, Some(symbol))?;
        self.0.borrow_mut().ctl_offset = a.kd.kernarg_size as u64 - 8;
        Ok(LoadedKernel {
            kernel_object: 0xdead_0000,
            kernarg_segment_size: a.kd.kernarg_size,
            group_segment_size: a.kd.group_segment_fixed_size,
            private_segment_size: 0,
        })
    }
}

impl FakeGpu {
    fn alloc(&mut self, bytes: u64) -> u64 {
        let va = self.next;
        self.next += bytes.max(8).next_multiple_of(4096);
        self.memory.insert(va, vec![0xcd; bytes.max(8) as usize]);
        va
    }

    fn write(&mut self, va: u64, data: &[u8]) -> Result<()> {
        let (buf, off) = self.region(va, data.len() as u64)?;
        buf[off..off + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn read(&mut self, va: u64, len: u64) -> Result<Vec<u8>> {
        let (buf, off) = self.region(va, len)?;
        Ok(buf[off..off + len as usize].to_vec())
    }
}

fn app_dispatch(gpu: &Gpu, workgroups: u32) -> Dispatch {
    let kernarg = gpu.0.borrow_mut().alloc(512);
    Dispatch {
        kernel_object: 0xabcd_0000,
        kernarg_address: kernarg,
        workgroup_size: [256, 1, 1],
        grid_size: [256 * workgroups, 1, 1],
        cluster_size: [1, 1, 1],
        group_segment_size: 0,
        private_segment_size: 0,
    }
}

fn config(dir: &std::path::Path) -> Config {
    Config {
        output_dir: dir.to_path_buf(),
        launch_model: LaunchChoice::Table,
        ..Config::default()
    }
}

/// Drive a whole replay loop; returns the plans in pass order.
fn drive(session: &mut Session<Gpu>, gpu: &Gpu) -> Vec<PassPlan> {
    let mut plans = vec![];
    for pass in 0.. {
        let plan = session.pass_enter(pass).unwrap();
        gpu.0.borrow_mut().run(&plan);
        plans.push(plan);
        if !session.pass_exit(pass).unwrap() {
            break;
        }
        assert!(pass < 1000, "schedule did not end");
    }
    plans
}

fn gpu(drift: &[u32]) -> Gpu {
    let mut g = FakeGpu::new();
    g.drift = drift.iter().copied().collect();
    Gpu(Rc::new(RefCell::new(g)))
}

fn begin(cfg: &Config, gpu: &Gpu, workgroups: u32) -> Session<Gpu> {
    let code = std::fs::read(format!("{FIXTURES}/v8.hsaco")).unwrap();
    let symbol = remapper_rewrite::analyze::analyze_code_object(&code, None)
        .unwrap()
        .symbol;
    let app = app_dispatch(gpu, workgroups);
    let info = KernelInfo {
        dispatch_id: 7,
        symbol: &symbol,
        code_object: &code,
        app,
        shader_engines: 8,
    };
    Session::begin(cfg, &info, gpu.clone()).unwrap()
}

#[test]
fn table_schedule_records_checks_traces_and_ends_with_the_application_dispatch() {
    if !tools_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let gpu = gpu(&[1, 5]);
    let cfg = config(dir.path());
    let mut session = begin(&cfg, &gpu, 512);
    let plans = drive(&mut session, &gpu);
    assert!(!session.failed(), "{}", session.report());
    assert_eq!(session.report()["status"], "ok");
    let stages: Vec<&str> = session.passes().iter().map(|p| p.stage.as_str()).collect();
    let count = |s: &str| stages.iter().filter(|&&x| x == s).count();
    // 3 agreeing records plus the drifted one; 4 checks plus the drifted one; then 2 + 2; 3 ATT runs of 2 sweeps x 4.
    assert_eq!(count("record"), 4);
    assert_eq!(count("table"), 5);
    assert_eq!((count("mark_table"), count("off")), (2, 2));
    assert_eq!(
        (
            count("att-table"),
            count("att-natural"),
            count("att-original")
        ),
        (8, 8, 8)
    );
    assert_eq!(*stages.last().unwrap(), "final");
    assert_eq!(plans.len(), 4 + 5 + 2 + 2 + 24 + 1);
    let last = plans.last().unwrap();
    assert!(last.dispatch.is_none() && last.trace.is_none());
    // ATT passes trace SIMD 0..3 in each sweep; the original run submits the application's dispatch.
    let att: Vec<_> = session
        .passes()
        .iter()
        .filter(|p| p.stage == "att-table")
        .collect();
    assert_eq!(
        att.iter()
            .map(|p| p.trace.unwrap().simd)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 0, 1, 2, 3]
    );
    assert_eq!(
        att.iter().map(|p| p.sweep.unwrap()).collect::<Vec<_>>(),
        [0, 0, 0, 0, 1, 1, 1, 1]
    );
    for (p, plan) in session.passes().iter().zip(&plans) {
        assert_eq!(plan.trace.is_some(), p.stage.starts_with("att-"));
        assert_eq!(
            plan.dispatch.is_none(),
            p.stage == "att-original" || p.stage == "final"
        );
        if let Some(d) = plan.dispatch {
            assert_eq!(d.kernel_object, 0xdead_0000);
            assert_eq!(d.grid_size, [256 * 512, 1, 1]);
        }
    }
    let rejected: Vec<u64> = session
        .passes()
        .iter()
        .filter(|p| !p.accepted)
        .map(|p| p.pass)
        .collect();
    assert_eq!(
        rejected,
        [5],
        "drifted table check is discarded; the odd record pass is just outvoted"
    );
    assert_eq!(
        session.report()["drift"]["record"]["distinct_placements"],
        2
    );
    assert_eq!(session.report()["placement"].as_array().unwrap().len(), 4);
    let (m, _) = Manifest::load(&session.dir().join("manifest.json")).unwrap();
    assert!(m.table.is_some());
    let passes: serde_json::Value =
        serde_json::from_slice(&std::fs::read(session.dir().join("passes.json")).unwrap()).unwrap();
    assert_eq!(passes["passes"].as_array().unwrap().len(), plans.len());
}

#[test]
fn drifted_att_cells_are_retried_and_the_retry_is_traced_on_the_same_simd() {
    if !tools_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let gpu = gpu(&[11]);
    let cfg = Config {
        natural_baseline: false,
        original_baseline: false,
        repetitions: 1,
        ..config(dir.path())
    };
    let mut session = begin(&cfg, &gpu, 512);
    drive(&mut session, &gpu);
    assert!(!session.failed(), "{}", session.report());
    let att: Vec<_> = session
        .passes()
        .iter()
        .filter(|p| p.stage == "att-table")
        .collect();
    assert_eq!(att.len(), 5);
    assert_eq!(
        att.iter()
            .map(|p| (p.trace.unwrap().simd, p.attempt.unwrap(), p.accepted))
            .collect::<Vec<_>>(),
        [
            (0, 0, false),
            (0, 1, true),
            (1, 0, true),
            (2, 0, true),
            (3, 0, true)
        ]
    );
    assert_eq!(session.report()["att"]["table"]["discarded"], 1);
}

#[test]
fn unstable_placement_fails_over_to_the_application_dispatch() {
    if !tools_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let gpu = gpu(&[1, 3]);
    let cfg = Config {
        drift_retries: 1,
        ..config(dir.path())
    };
    let mut session = begin(&cfg, &gpu, 512);
    let plans = drive(&mut session, &gpu);
    assert!(session.failed());
    assert_eq!(session.report()["status"], "failed");
    assert!(session.report()["reason"]
        .as_str()
        .unwrap()
        .contains("per-key turn counts changed"));
    // Four record passes (records + drift retries) then the application's dispatch.
    assert_eq!(plans.len(), 5);
    assert!(plans[4].dispatch.is_none() && plans[4].trace.is_none());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(session.dir().join("report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "failed");
}

#[test]
fn passes_must_alternate_enter_and_exit() {
    if !tools_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let gpu = gpu(&[]);
    let cfg = config(dir.path());
    let mut session = begin(&cfg, &gpu, 512);
    assert!(session.pass_exit(0).is_err());
    session.pass_enter(0).unwrap();
    assert!(session.pass_enter(1).is_err());
    assert!(session.pass_exit(1).is_err());
}

#[test]
fn options_parse_into_the_config() {
    let mut c = Config::default();
    c.apply_options("records=5, drift_retries=0,launch-model=admission,prefetch=wave0,natural-baseline=0,loop-markers=cluster,admission-slack=64").unwrap();
    assert_eq!(
        (c.records, c.drift_retries, c.launch_model),
        (5, 0, LaunchChoice::Admission)
    );
    assert_eq!(c.prefetch, Some(remapper_rewrite::rewrite::Prefetch::Wave0));
    assert!(!c.natural_baseline && c.original_baseline);
    assert_eq!(
        c.loop_markers,
        Some(remapper_rewrite::markers::LoopMarkers::Cluster)
    );
    assert_eq!(c.admission_slack, Some(64));
    c.apply_options("").unwrap();
    for bad in [
        "records",
        "records=x",
        "bogus=1",
        "launch-model=dense",
        "natural-baseline=2",
        "records=99999999999",
    ] {
        assert!(Config::default().apply_options(bad).is_err(), "{bad}");
    }
    let mut zero = Config::default();
    zero.apply_options("checks=0").unwrap();
    assert!(zero.validate().is_err());
}

#[test]
fn padded_grids_must_be_whole_workgroups_and_clusters() {
    let mut d = Dispatch {
        workgroup_size: [64, 1, 1],
        ..Default::default()
    };
    assert!(setup::check_padded_grid(&d, [32768, 1, 1]).is_ok());
    assert!(setup::check_padded_grid(&d, [1000, 1, 1]).is_err());
    d.cluster_size = [4, 1, 1];
    assert!(setup::check_padded_grid(&d, [64 * 512, 1, 1]).is_ok());
    assert!(setup::check_padded_grid(&d, [64 * 514, 1, 1]).is_err());
    d.cluster_size = [1, 1, 1];
    assert!(setup::check_padded_grid(&d, [64, 65536, 1]).is_err());
    assert_eq!(setup::cluster_dims(&Dispatch::default()), [1, 1, 1]);
}

#[test]
fn remapper_counts_dispatches_and_writes_a_report_when_one_cannot_be_remapped() {
    let dir = tempfile::tempdir().unwrap();
    let mut r = Remapper::new(config(dir.path())).unwrap();
    let gpu = gpu(&[]);
    let info = KernelInfo {
        dispatch_id: 3,
        symbol: "missing",
        code_object: b"not an elf",
        app: app_dispatch(&gpu, 4),
        shader_engines: 8,
    };
    assert!(r.begin_dispatch(&info, gpu.clone()).is_err());
    assert!(!r.wants_more());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("dispatch-3/report.json")).unwrap())
            .unwrap();
    assert_eq!(report["status"], "failed");
    assert!(r.begin_dispatch(&info, gpu).is_err());
}
