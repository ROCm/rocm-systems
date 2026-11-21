//! One remapped dispatch: the pass schedule of a kernel replay loop.
//!
//! The replay loop submits one pass at a time. Before each, [`Session::pass_enter`] writes the control block (mode,
//! table, buffer pointers) and says which dispatch to submit and whether to trace it; after each,
//! [`Session::pass_exit`] reads the counters and logs, before the SDK restores device memory, and moves the
//! schedule on. The stages follow the aegis `att-remap` pipeline:
//!
//! - table model: record passes until `records` of them place workgroups identically, build the table, check
//!   placement in `checks` table passes, then two mark-table and two off passes;
//! - admission model: one record pass becomes the schedule, then the table, mark-table and off checks;
//! - sparse model: probe passes, assignment, then the table, mark-table and off checks;
//!
//! then the ATT sweeps (remapped with markers, natural placement with markers, and the application's kernel), each
//! tracing SIMD 0 to 3 in turn, with drifted table passes discarded and retried. The loop always ends with the
//! application's own dispatch, untraced, so its results are the ones the application keeps. A failed stage skips
//! straight to that pass.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use remapper_rewrite::{
    manifest::{AssignedWorkgroup, Manifest, TableRef, FIXED_CONTROL_BYTES},
    mode::{Mode, Outcome},
    rewrite, sparse_table,
    table::{self, Table},
};
use serde::Serialize;
use serde_json::{json, Value};

use crate::{
    device::{Device, LoadedKernel},
    setup, Config, Dispatch,
};

/// Unmodified launches put a workgroup on a traced WGP in only some passes; sparse launches sample more of them.
const SPARSE_ORIGINAL_SWEEPS: u32 = 6;
pub const SIMDS: u32 = 4;

/// What rocprofv3 should trace in a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct TraceTarget {
    pub target_cu: u32,
    pub simd: u32,
    pub shader_engine_mask: u64,
}

/// The dispatch a pass submits and whether it is traced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassPlan {
    /// `None` submits the application's dispatch.
    pub dispatch: Option<Dispatch>,
    pub trace: Option<TraceTarget>,
}

/// The kernel and dispatch a session remaps.
pub struct KernelInfo<'a> {
    pub dispatch_id: u64,
    pub symbol: &'a str,
    /// The application's code object (an ELF for this agent).
    pub code_object: &'a [u8],
    pub app: Dispatch,
    pub shader_engines: u32,
}

#[derive(Clone, Copy, Debug)]
enum After {
    Nothing,
    TablePlacement,
    SparsePlacement,
}

#[derive(Clone, Copy, Debug)]
struct AttRun {
    name: &'static str,
    mode: Option<Mode>,
    sweeps: u32,
}

#[derive(Clone, Copy, Debug)]
enum Step {
    /// Table model: record until `records` passes agree on per-key turn counts.
    Record,
    /// Admission model: one record pass becomes the schedule.
    AdmissionRecord,
    /// Sparse model: `records` probe passes.
    Probe,
    /// `want` passes of `mode`; with `drift`, drifted passes are discarded within the drift budget.
    Collect {
        mode: Mode,
        want: u32,
        drift: bool,
        after: After,
    },
    Att(AttRun),
    /// The application's dispatch, untraced.
    Final,
}

impl Step {
    fn name(&self) -> String {
        match self {
            Step::Record | Step::AdmissionRecord => "record".into(),
            Step::Probe => "probe".into(),
            Step::Collect { mode, .. } => mode.name().into(),
            Step::Att(r) => format!("att-{}", r.name),
            Step::Final => "final".into(),
        }
    }

    fn mode(&self) -> Option<Mode> {
        match self {
            Step::Record | Step::AdmissionRecord => Some(Mode::Record),
            Step::Probe => Some(Mode::Probe),
            Step::Collect { mode, .. } => Some(*mode),
            Step::Att(r) => r.mode,
            Step::Final => None,
        }
    }
}

/// One pass as the merge and the report see it.
#[derive(Clone, Debug, Serialize)]
pub struct PassRecord {
    pub pass: u64,
    pub stage: String,
    pub mode: Option<&'static str>,
    /// ATT run (`table`, `natural`, `original`), sweep, SIMD and attempt of a traced pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sweep: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attempt: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceTarget>,
    /// Whether the pass's results were kept; drifted passes are discarded.
    pub accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<Outcome>,
}

#[derive(Default)]
struct StepState {
    kept: Vec<u64>,
    rejected: u32,
    runs: u32,
    groups: Vec<(Vec<u32>, Vec<u64>)>,
    sweep: u32,
    simd: u32,
    attempt: u32,
    notes: Vec<Value>,
}

#[derive(Clone, Copy)]
struct Buffers {
    control: u64,
    arrival: u64,
    table: u64,
    table_capacity: u64,
    counters: u64,
    log: u64,
    kernarg: u64,
}

pub struct Session<D: Device> {
    cfg: Config,
    dir: PathBuf,
    device: D,
    dispatch_id: u64,
    m: Manifest,
    app: Dispatch,
    kernel: LoadedKernel,
    shader_engines: u32,
    buffers: Buffers,
    extra_va: Vec<u64>,
    table: Option<Table>,
    table_bytes: Vec<u8>,
    max_k: u32,
    sparse: Option<(BTreeSet<u32>, Vec<AssignedWorkgroup>)>,
    steps: std::collections::VecDeque<Step>,
    state: StepState,
    current: Option<(u64, Step, PassRecord)>,
    passes: Vec<PassRecord>,
    report: Value,
    failed: bool,
    done: bool,
}

fn read_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

struct AdmissionCounts {
    unclaimed: u64,
    parked: u64,
    held: u64,
}

/// `parked` and `held` are reported in workgroups; a cluster leader counts once for its `cluster_size` members.
fn admission_counts(path: &Path, cluster_size: u64) -> Result<AdmissionCounts> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if b.len() != rewrite::ADMISSION_WORDS * 4 {
        bail!(
            "{}: {} bytes, expected {}",
            path.display(),
            b.len(),
            rewrite::ADMISSION_WORDS * 4
        );
    }
    let w: Vec<u64> = read_u32s(&b).into_iter().map(u64::from).collect();
    let n = rewrite::ADMISSION_ENGINES;
    Ok(AdmissionCounts {
        unclaimed: w[..n].iter().sum(),
        parked: w[n..2 * n].iter().sum::<u64>() * cluster_size,
        held: w[2 * n..].iter().sum::<u64>() * cluster_size,
    })
}

impl<D: Device> Session<D> {
    /// Rewrite the kernel, load it, and allocate and fill every buffer the passes use.
    pub fn begin(cfg: &Config, info: &KernelInfo<'_>, mut device: D) -> Result<Self> {
        cfg.validate()?;
        let se = match cfg.shader_engines {
            Some(n) if n != info.shader_engines => bail!(
                "configured shader engines {n} differ from the agent's {}; cluster keys would address the wrong \
                 buffers",
                info.shader_engines
            ),
            _ => info.shader_engines,
        };
        let dir = cfg
            .output_dir
            .join(format!("dispatch-{}", info.dispatch_id));
        if dir.exists() && std::fs::read_dir(&dir)?.next().is_some() {
            bail!("{} already exists and is not empty", dir.display());
        }
        std::fs::create_dir_all(dir.join("passes"))?;
        let mut report =
            json!({"status": "running", "version": 1, "dispatch_id": info.dispatch_id});
        let m = setup::prepare(cfg, info, &dir, &mut report)?;
        let code = std::fs::read(dir.join(&m.code_object))?;
        let kernel = device.load_kernel(&code, &m.symbol)?;
        let ctl = m.control_kernarg_offset as u64;
        if kernel.kernarg_segment_size as u64 != ctl + 8 {
            bail!(
                "rewritten kernel takes {} kernarg bytes; the control pointer is at {ctl}",
                kernel.kernarg_segment_size
            );
        }
        let mut kernarg = device
            .read(info.app.kernarg_address, ctl)
            .context("reading the application's kernel arguments")?;
        for x in &m.extra_buffers {
            if x.control_offset < FIXED_CONTROL_BYTES
                || x.control_offset + 8 > m.control_bytes as usize
            {
                bail!(
                    "extra buffer '{}' pointer at control offset {} is outside {FIXED_CONTROL_BYTES}..{}",
                    x.name,
                    x.control_offset,
                    m.control_bytes
                );
            }
        }
        let needed = m.num_keys * m.num_log_records.max(1) * 8;
        let table_capacity = needed.min(cfg.table_capacity_bytes);
        let log_bytes = m.num_log_records * m.log_words * 4;
        let buffers = Buffers {
            control: device.alloc(m.control_bytes)?,
            arrival: device.alloc(m.num_keys * 4)?,
            table: device.alloc(table_capacity)?,
            table_capacity,
            counters: device.alloc(8)?,
            log: device.alloc(log_bytes)?,
            kernarg: device.alloc(ctl + 8)?,
        };
        let extra_va = m
            .extra_buffers
            .iter()
            .map(|x| device.alloc(x.bytes))
            .collect::<Result<Vec<_>>>()?;
        kernarg.extend_from_slice(&buffers.control.to_le_bytes());
        device.write(buffers.kernarg, &kernarg)?;
        let mut steps = std::collections::VecDeque::new();
        let reps = cfg.repetitions;
        let (sparse, admission) = (m.is_sparse(), m.is_admission());
        if sparse {
            steps.push_back(Step::Probe);
            steps.push_back(Step::Collect {
                mode: Mode::Table,
                want: cfg.checks,
                drift: false,
                after: After::SparsePlacement,
            });
            steps.push_back(Step::Collect {
                mode: Mode::MarkTable,
                want: 2,
                drift: false,
                after: After::Nothing,
            });
        } else {
            steps.push_back(if admission {
                Step::AdmissionRecord
            } else {
                Step::Record
            });
            steps.push_back(Step::Collect {
                mode: Mode::Table,
                want: cfg.checks,
                drift: true,
                after: After::TablePlacement,
            });
            steps.push_back(Step::Collect {
                mode: Mode::MarkTable,
                want: 2,
                drift: true,
                after: After::Nothing,
            });
        }
        steps.push_back(Step::Collect {
            mode: Mode::Off,
            want: 2,
            drift: false,
            after: After::Nothing,
        });
        steps.push_back(Step::Att(AttRun {
            name: "table",
            mode: Some(Mode::MarkTable),
            sweeps: reps,
        }));
        if cfg.natural_baseline && !sparse {
            steps.push_back(Step::Att(AttRun {
                name: "natural",
                mode: Some(Mode::MarkNatural),
                sweeps: reps,
            }));
        } else if cfg.natural_baseline {
            report["natural_baseline"] =
                json!({"skipped": "a sparse launch has no natural placement on the traced WGPs"});
        }
        if cfg.original_baseline {
            let sweeps = if sparse {
                reps.max(SPARSE_ORIGINAL_SWEEPS)
            } else {
                reps
            };
            steps.push_back(Step::Att(AttRun {
                name: "original",
                mode: None,
                sweeps,
            }));
        }
        steps.push_back(Step::Final);
        let session = Self {
            cfg: cfg.clone(),
            dir,
            device,
            dispatch_id: info.dispatch_id,
            m,
            app: info.app,
            kernel,
            shader_engines: se,
            buffers,
            extra_va,
            table: None,
            table_bytes: Vec::new(),
            max_k: 0,
            sparse: None,
            steps,
            state: StepState::default(),
            current: None,
            passes: Vec::new(),
            report,
            failed: false,
            done: false,
        };
        session.save()?;
        Ok(session)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn done(&self) -> bool {
        self.done
    }

    pub fn failed(&self) -> bool {
        self.failed
    }

    pub fn report(&self) -> &Value {
        &self.report
    }

    pub fn passes(&self) -> &[PassRecord] {
        &self.passes
    }

    fn save(&self) -> Result<()> {
        write_json(&self.dir.join("report.json"), &self.report)?;
        write_json(
            &self.dir.join("passes.json"),
            &json!({"dispatch_id": self.dispatch_id, "passes": self.passes}),
        )
    }

    fn pass_dir(&self, pass: u64) -> PathBuf {
        self.dir.join("passes").join(format!("{pass:05}"))
    }

    fn fail(&mut self, e: anyhow::Error) {
        log::warn!(
            "remapper: dispatch {}: {e:#}; finishing with the application's dispatch",
            self.dispatch_id
        );
        self.failed = true;
        self.report["status"] = json!("failed");
        self.report["reason"] = json!(format!("{e:#}"));
        self.steps.clear();
        self.steps.push_back(Step::Final);
        self.state = StepState::default();
    }

    /// The dispatch and trace target of pass `pass`.
    pub fn pass_enter(&mut self, pass: u64) -> Result<PassPlan> {
        if self.done {
            bail!("pass {pass} after the schedule ended");
        }
        if self.current.is_some() {
            bail!("pass {pass} entered before the previous pass exited");
        }
        let plan = match self.plan(pass) {
            Ok(plan) => plan,
            Err(e) => {
                self.fail(e);
                self.plan(pass)?
            }
        };
        Ok(plan)
    }

    fn plan(&mut self, pass: u64) -> Result<PassPlan> {
        let step = *self.steps.front().context("empty schedule")?;
        let mode = step.mode();
        let mut record = PassRecord {
            pass,
            stage: step.name(),
            mode: mode.map(Mode::name),
            run: None,
            sweep: None,
            attempt: None,
            trace: None,
            accepted: true,
            outcome: None,
        };
        if let Step::Att(run) = step {
            let target = TraceTarget {
                target_cu: self.cfg.target_cu,
                simd: self.state.simd,
                shader_engine_mask: if self.shader_engines >= 64 {
                    u64::MAX
                } else {
                    (1u64 << self.shader_engines) - 1
                },
            };
            record.run = Some(run.name);
            record.sweep = Some(self.state.sweep);
            record.attempt = Some(self.state.attempt);
            record.trace = Some(target);
        }
        let dispatch = match mode {
            Some(mode) => {
                self.before(mode)?;
                Some(self.remapped_dispatch(mode))
            }
            None => None,
        };
        let trace = record.trace;
        self.current = Some((pass, step, record));
        Ok(PassPlan { dispatch, trace })
    }

    fn remapped_dispatch(&self, mode: Mode) -> Dispatch {
        let mut d = self.app;
        d.kernel_object = self.kernel.kernel_object;
        d.kernarg_address = self.buffers.kernarg;
        // Admission launches pad only the table modes; the others run the application's grid with hardware ids.
        if let Some(l) = self
            .m
            .launch
            .as_ref()
            .filter(|l| l.model != "admission" || mode.uses_table())
        {
            d.grid_size = l.padded_grid_size;
        }
        d.group_segment_size = d
            .group_segment_size
            .max(self.m.min_group_segment_bytes)
            .max(self.kernel.group_segment_size);
        d.private_segment_size = d.private_segment_size.max(self.kernel.private_segment_size);
        d
    }

    fn check_mode(&self, mode: Mode) -> Result<()> {
        let sparse = self.m.is_sparse();
        match (sparse, mode) {
            (true, Mode::Record | Mode::MarkNatural) => {
                bail!(
                    "mode '{}' would run the body with padded hardware ids",
                    mode.name()
                )
            }
            (false, Mode::Probe) => bail!("mode 'probe' needs a sparse launch"),
            _ => {}
        }
        if (mode.uses_table() || (sparse && mode == Mode::Off)) && self.max_k == 0 {
            bail!(
                "mode '{}' needs a table, and none has been built",
                mode.name()
            );
        }
        Ok(())
    }

    /// Reset the pass's tool buffers and write its control block and table.
    fn before(&mut self, mode: Mode) -> Result<()> {
        self.check_mode(mode)?;
        let b = self.buffers;
        let uses_table = mode.uses_table() || (self.m.is_sparse() && mode == Mode::Off);
        let max_k = if uses_table { self.max_k } else { 0 };
        if uses_table {
            self.device.write(b.table, &self.table_bytes)?;
        }
        let mut control = vec![0u8; self.m.control_bytes as usize];
        let words = [
            mode.code() as u64 | (max_k as u64) << 32,
            b.arrival,
            b.table,
            b.counters,
            b.log,
        ];
        for (i, w) in words.iter().enumerate() {
            control[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        for (x, &va) in self.m.extra_buffers.iter().zip(&self.extra_va) {
            control[x.control_offset..x.control_offset + 8].copy_from_slice(&va.to_le_bytes());
            let mut fill: Vec<u8> = x
                .fill
                .to_le_bytes()
                .iter()
                .copied()
                .cycle()
                .take(x.bytes as usize)
                .collect();
            for (i, w) in x.init.iter().flatten().enumerate() {
                fill[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            self.device.write(va, &fill)?;
        }
        self.device.write(b.control, &control)?;
        self.device
            .write(b.arrival, &vec![0u8; self.m.num_keys as usize * 4])?;
        self.device.write(b.counters, &[0u8; 8])?;
        let log_bytes = self.m.num_log_records * self.m.log_words * 4;
        self.device.write(b.log, &vec![0u8; log_bytes as usize])?;
        Ok(())
    }

    /// Read what the pass left in the tool buffers into `passes/NNNNN/`.
    fn after(&mut self, pass: u64, mode: Mode) -> Result<Outcome> {
        let b = self.buffers;
        let counters = read_u32s(&self.device.read(b.counters, 8)?);
        let outcome = Outcome {
            overflow: counters[0],
            logged: counters[1],
            expected_log_records: if mode.logs() {
                self.m.num_log_records as u32
            } else {
                0
            },
            golden_ok: None,
        };
        let dir = self.pass_dir(pass);
        std::fs::create_dir_all(&dir)?;
        if mode.logs() {
            let log_bytes = self.m.num_log_records * self.m.log_words * 4;
            std::fs::write(dir.join("log.bin"), self.device.read(b.log, log_bytes)?)?;
            std::fs::write(
                dir.join("arrival.bin"),
                self.device.read(b.arrival, self.m.num_keys * 4)?,
            )?;
            for (x, &va) in self.m.extra_buffers.iter().zip(&self.extra_va) {
                std::fs::write(
                    dir.join(format!("{}.bin", x.name)),
                    self.device.read(va, x.bytes)?,
                )?;
            }
        }
        write_json(&dir.join("outcome.json"), &outcome)?;
        Ok(outcome)
    }

    /// Finish pass `pass`; returns whether the loop runs another pass.
    pub fn pass_exit(&mut self, pass: u64) -> Result<bool> {
        let (entered, step, mut record) = self
            .current
            .take()
            .context("pass exit without a pass enter")?;
        if entered != pass {
            bail!("pass {pass} exited, but pass {entered} entered");
        }
        let result = (|| -> Result<()> {
            let outcome = match step.mode() {
                Some(mode) => Some(self.after(pass, mode)?),
                None => None,
            };
            record.outcome = outcome;
            record.accepted = self.observe(step, pass, outcome)?;
            Ok(())
        })();
        self.passes.push(record);
        if let Err(e) = result {
            self.fail(e);
        }
        if self.steps.is_empty() {
            self.done = true;
            if !self.failed {
                self.report["status"] = json!("ok");
            }
        }
        if let Err(e) = self.save() {
            log::warn!("remapper: dispatch {}: {e:#}", self.dispatch_id);
        }
        Ok(!self.done)
    }

    fn advance(&mut self) {
        self.steps.pop_front();
        self.state = StepState::default();
    }

    fn log(&self, pass: u64) -> Result<table::Log> {
        table::read_log(&self.pass_dir(pass).join("log.bin"), &self.m)
    }

    fn set_table(&mut self, table: Table) -> Result<()> {
        let path = self.dir.join("table.bin");
        table.write(&path)?;
        let bytes = std::fs::read(&path)?;
        self.install_table(bytes, table.max_k)?;
        self.table = Some(table);
        Ok(())
    }

    fn install_table(&mut self, bytes: Vec<u8>, max_k: u32) -> Result<()> {
        if max_k == 0 || bytes.len() as u64 != self.m.table_bytes(max_k) {
            bail!(
                "table has {} bytes, expected {} (max_k={max_k})",
                bytes.len(),
                self.m.table_bytes(max_k)
            );
        }
        if bytes.len() as u64 > self.buffers.table_capacity {
            bail!(
                "the placement table needs {} bytes (max_k={max_k}); its buffer holds {}; raise the table capacity",
                bytes.len(),
                self.buffers.table_capacity
            );
        }
        self.m.table = Some(TableRef {
            file: "table.bin".into(),
            max_k,
        });
        self.m.save(&self.dir)?;
        self.table_bytes = bytes;
        self.max_k = max_k;
        Ok(())
    }

    /// Account for a finished pass; returns whether its results were kept.
    fn observe(&mut self, step: Step, pass: u64, outcome: Option<Outcome>) -> Result<bool> {
        let name = step.name();
        let valid = |mode: Mode, o: &Outcome| -> Result<()> {
            if !o.valid(mode) {
                bail!("{name} pass {pass}: overflow or missing log records ({o:?})");
            }
            Ok(())
        };
        match step {
            Step::Record => {
                let o = outcome.context("record pass without an outcome")?;
                valid(Mode::Record, &o)?;
                let counts = table::counts(&self.m, &self.log(pass)?)?;
                match self.state.groups.iter_mut().find(|g| g.0 == counts) {
                    Some(g) => g.1.push(pass),
                    None => self.state.groups.push((counts, vec![pass])),
                }
                self.state.runs += 1;
                let best = self
                    .state
                    .groups
                    .iter()
                    .map(|g| g.1.len() as u32)
                    .max()
                    .unwrap_or(0);
                let (records, retries) = (self.cfg.records, self.cfg.drift_retries);
                if best >= records {
                    let group = self
                        .state
                        .groups
                        .iter()
                        .max_by_key(|g| g.1.len())
                        .unwrap()
                        .1
                        .clone();
                    let table = Table::build(&self.m, &self.log(group[0])?)?;
                    for &p in group.iter().take(records as usize) {
                        table.check_record(&self.m, &self.log(p)?)?;
                    }
                    self.report["drift"]["record"] = json!({
                        "runs": self.state.runs,
                        "distinct_placements": self.state.groups.len(),
                        "modal_runs": group.len(),
                        "modal_passes": group,
                    });
                    self.report["record_invariants"] = json!({
                        "every_tile_once": true, "stable_counts": true, "counts": table.counts, "max_k": table.max_k,
                    });
                    self.set_table(table)?;
                    self.advance();
                } else if self.state.runs >= records + retries {
                    bail!(
                        "per-key turn counts changed between record passes: no {records} of {} passes placed \
                         workgroups identically ({} distinct placements, largest group {best}); placement is too \
                         unstable for a fixed table (drift retries {retries}); the admission launch model enforces \
                         one recorded placement instead",
                        self.state.runs,
                        self.state.groups.len()
                    );
                }
                Ok(true)
            }
            Step::AdmissionRecord => {
                let o = outcome.context("record pass without an outcome")?;
                valid(Mode::Record, &o)?;
                let table = Table::build(&self.m, &self.log(pass)?)?;
                let mut tiles = vec![0u32; rewrite::ADMISSION_ENGINES];
                let keys_per_engine = self.m.num_keys as usize / rewrite::ADMISSION_ENGINES;
                for (key, &c) in table.counts.iter().enumerate() {
                    // Cluster keys are SE (traced) and NUM_SE + SE (filler): both count toward the engine's tiles.
                    let se = match &self.m.cluster {
                        Some(cl) => key % cl.shader_engines as usize,
                        None => key / keys_per_engine,
                    };
                    *tiles
                        .get_mut(se)
                        .context("placement key outside 32 shader engines")? += c;
                }
                self.m
                    .extra_buffers
                    .iter_mut()
                    .find(|x| x.name == "admission")
                    .context("admission manifest without its buffer")?
                    .init = Some(tiles.clone());
                self.report["record_invariants"] =
                    json!({"every_tile_once": true, "counts": table.counts, "max_k": table.max_k});
                self.report["admission"] = json!({"tiles_per_engine": tiles});
                self.set_table(table)?;
                self.advance();
                Ok(true)
            }
            Step::Probe => {
                let o = outcome.context("probe pass without an outcome")?;
                valid(Mode::Probe, &o)?;
                self.state.kept.push(pass);
                if self.state.kept.len() as u32 == self.cfg.records {
                    let logs = self
                        .state
                        .kept
                        .iter()
                        .map(|&p| self.log(p))
                        .collect::<Result<Vec<_>>>()?;
                    let se = self.shader_engines;
                    let locations =
                        sparse_table::validate_probe(&self.m, &logs, se, self.cfg.target_cu)?;
                    let (assignment, traced) = sparse_table::assign(
                        &locations,
                        self.m.workgroups,
                        se,
                        self.cfg.target_cu,
                    )?;
                    let entries =
                        sparse_table::build_table(self.m.num_keys, &locations, &assignment)?;
                    let launch = self
                        .m
                        .launch
                        .as_mut()
                        .context("sparse manifest without a launch layout")?;
                    launch.assignment = assignment.clone();
                    launch.traced_locations = traced.clone();
                    let bytes = sparse_table::table_bytes(&entries);
                    std::fs::write(self.dir.join("table.bin"), &bytes)?;
                    self.install_table(bytes, 1)?;
                    let traced_engines: Vec<u32> = traced.iter().map(|l| l / 32).collect();
                    self.report["probe_invariants"] = json!({
                        "wgps": locations.len(), "one_workgroup_per_wgp": true, "same_wgps_every_probe": true,
                    });
                    self.report["launch"] = json!({
                        "padded_workgroups": locations.len(),
                        "workgroups": self.m.num_workgroups,
                        "traced_engines_with_work": traced_engines,
                        "excluded_engines": (0..se).filter(|s| !traced_engines.contains(s)).collect::<Vec<_>>(),
                    });
                    self.sparse = Some((locations, assignment));
                    self.advance();
                }
                Ok(true)
            }
            Step::Collect {
                mode,
                want,
                drift,
                after,
            } => {
                let o = outcome.context("pass without an outcome")?;
                if o.drifted() {
                    if !drift {
                        bail!(
                            "{name} pass {pass}: placement drifted from the table (overflow={})",
                            o.overflow
                        );
                    }
                    self.state.rejected += 1;
                    if self.state.rejected > self.cfg.drift_retries {
                        bail!(
                            "{name}: {} of {} passes drifted from the recorded placement (drift retries {}); \
                             placement of this launch is too unstable to replay a fixed table",
                            self.state.rejected,
                            self.state.rejected + self.state.kept.len() as u32,
                            self.cfg.drift_retries
                        );
                    }
                    return Ok(false);
                }
                valid(mode, &o)?;
                if self.m.is_admission() && mode == Mode::Table {
                    let cluster_size = self.m.cluster.as_ref().map_or(1, |c| c.size as u64);
                    let w =
                        admission_counts(&self.pass_dir(pass).join("admission.bin"), cluster_size)?;
                    if w.unclaimed != 0 {
                        let slack = self.m.launch.as_ref().map_or(0, |l| l.slack);
                        bail!(
                            "admission slack exhausted: {} tiles were never claimed ({} arrivals parked, {} of \
                             them holding a WGP); raise the admission slack (now {slack})",
                            w.unclaimed,
                            w.parked,
                            w.held
                        );
                    }
                    self.state
                        .notes
                        .push(json!({"parked": w.parked, "held": w.held}));
                }
                self.state.kept.push(pass);
                if (self.state.kept.len() as u32) < want {
                    return Ok(true);
                }
                let key = format!("{}_discarded", mode.name());
                self.report["drift"][key] = json!(self.state.rejected);
                match after {
                    After::Nothing => {}
                    After::TablePlacement => {
                        let table = self
                            .table
                            .as_ref()
                            .context("table checks without a table")?;
                        let mut placement = vec![];
                        for &p in &self.state.kept {
                            let slot_path = self.pass_dir(p).join("slot.bin");
                            let slot = match slot_path.exists() {
                                true => Some(read_u32s(&std::fs::read(&slot_path)?)),
                                false => None,
                            };
                            let r = table.check(&self.m, &self.log(p)?, slot.as_deref())?;
                            if !r.valid() {
                                bail!("placement mismatch in pass {p}: {r:?}");
                            }
                            placement.push(r);
                        }
                        self.report["placement"] = serde_json::to_value(&placement)?;
                        if self.m.is_admission() {
                            let held: Vec<u64> = self
                                .state
                                .notes
                                .iter()
                                .filter_map(|v| v["held"].as_u64())
                                .collect();
                            let launch = self
                                .m
                                .launch
                                .as_ref()
                                .context("admission without a launch layout")?;
                            self.report["admission"]["slack"] = json!(launch.slack);
                            self.report["admission"]["padded_workgroups"] =
                                json!(launch.padded_workgroups);
                            self.report["admission"]["check_passes"] = json!(self.state.notes);
                            self.report["admission"]["max_held"] = json!(held.iter().max());
                            self.report["admission"]["slack_used_pct"] = json!(held
                                .iter()
                                .max()
                                .map(|p| 100.0 * *p as f64 / launch.slack.max(1) as f64));
                        }
                    }
                    After::SparsePlacement => {
                        let (locations, assignment) = self
                            .sparse
                            .as_ref()
                            .context("sparse checks before the probe")?;
                        let mut placement = vec![];
                        for &p in &self.state.kept {
                            placement.push(sparse_table::check_placement(
                                &self.m,
                                &self.log(p)?,
                                locations,
                                assignment,
                            )?);
                        }
                        self.report["placement"] = serde_json::to_value(&placement)?;
                    }
                }
                self.advance();
                Ok(true)
            }
            Step::Att(run) => {
                if let (Some(mode), Some(o)) = (run.mode, outcome) {
                    if mode.uses_table() && o.drifted() {
                        if self.state.attempt >= self.cfg.drift_retries {
                            bail!(
                                "ATT {} sweep {} SIMD {}: placement drifted from the table in all {} passes \
                                 (raise the drift retries)",
                                run.name,
                                self.state.sweep,
                                self.state.simd,
                                self.state.attempt + 1
                            );
                        }
                        self.state.attempt += 1;
                        self.state.rejected += 1;
                        return Ok(false);
                    }
                    valid(mode, &o)?;
                }
                self.state.attempt = 0;
                self.state.simd += 1;
                if self.state.simd == SIMDS {
                    self.state.simd = 0;
                    self.state.sweep += 1;
                }
                if self.state.sweep == run.sweeps {
                    self.report["att"][run.name] =
                        json!({"sweeps": run.sweeps, "discarded": self.state.rejected});
                    self.advance();
                }
                Ok(true)
            }
            Step::Final => {
                self.advance();
                Ok(true)
            }
        }
    }
}
