use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use remapper_rewrite::{markers::LoopMarkers, rewrite::Prefetch};

/// Which launch model a remapped dispatch uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchChoice {
    /// Sparse for non-cluster dispatches with fewer workgroups than WGPs, table otherwise.
    Auto,
    /// The application's grid; the k-th workgroup at a location takes `table[location][k]`.
    Table,
    /// One workgroup per WGP; the kernel's workgroups run on the traced WGPs.
    Sparse,
    /// The application's grid plus slack; one recorded placement is enforced by parking arrivals.
    Admission,
}

impl LaunchChoice {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "auto" => Self::Auto,
            "table" => Self::Table,
            "sparse" => Self::Sparse,
            "admission" => Self::Admission,
            _ => bail!("unknown launch model '{s}' (auto, table, sparse, admission)"),
        })
    }
}

/// Upper bound on the placement table buffer, which is allocated before the passes that size it run.
pub const DEFAULT_TABLE_CAPACITY_BYTES: u64 = 256 << 20;

#[derive(Clone, Debug)]
pub struct Config {
    /// Each remapped dispatch writes `dispatch-<id>/` here.
    pub output_dir: PathBuf,
    /// Record passes whose per-key turn counts must agree (the modal placement).
    pub records: u32,
    /// Table passes checked for placement.
    pub checks: u32,
    /// Drift budget per stage: extra record passes to find `records` identical placements, and drifted
    /// (table-overflow) passes to discard and replace in the checks and in each ATT cell.
    pub drift_retries: u32,
    /// ATT sweeps per run; each sweep traces SIMD 0 to 3 in turn.
    pub repetitions: u32,
    /// ATT target CU (WGP), which is also the WGP the table keys traced work to.
    pub target_cu: u32,
    pub wgps_per_se: u32,
    pub launch_model: LaunchChoice,
    /// `None`: no prefetch for ordinary dispatches, per-wave for clusters and sparse launches.
    pub prefetch: Option<Prefetch>,
    /// Trace the rewritten kernel at its natural placement too (not for sparse launches).
    pub natural_baseline: bool,
    /// Trace the application's kernel too.
    pub original_baseline: bool,
    pub admission_slack: Option<u64>,
    pub loop_markers: Option<LoopMarkers>,
    /// Shader engines; must match the agent when given.
    pub shader_engines: Option<u32>,
    /// Dispatches to remap; later ones run without replay.
    pub max_dispatches: u32,
    pub table_capacity_bytes: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("remap"),
            records: 3,
            checks: 4,
            drift_retries: 12,
            repetitions: 2,
            target_cu: 0,
            wgps_per_se: 16,
            launch_model: LaunchChoice::Auto,
            prefetch: None,
            natural_baseline: true,
            original_baseline: true,
            admission_slack: None,
            loop_markers: None,
            shader_engines: None,
            max_dispatches: 1,
            table_capacity_bytes: DEFAULT_TABLE_CAPACITY_BYTES,
        }
    }
}

fn flag(key: &str, value: &str) -> Result<bool> {
    match value {
        "1" | "true" | "on" => Ok(true),
        "0" | "false" | "off" => Ok(false),
        _ => bail!("{key}: expected 0 or 1, got '{value}'"),
    }
}

impl Config {
    /// Apply a comma-separated `key=value` list (the `remapper_create` options).
    pub fn apply_options(&mut self, options: &str) -> Result<()> {
        for item in options.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (key, value) = item
                .split_once('=')
                .with_context(|| format!("remapper option '{item}' is not key=value"))?;
            let (key, value) = (key.trim().replace('_', "-"), value.trim());
            let num = || {
                value
                    .parse::<u64>()
                    .with_context(|| format!("{key}: '{value}' is not a number"))
            };
            let num32 = || -> Result<u32> { Ok(u32::try_from(num()?)?) };
            match key.as_str() {
                "records" => self.records = num32()?,
                "checks" => self.checks = num32()?,
                "drift-retries" => self.drift_retries = num32()?,
                "repetitions" => self.repetitions = num32()?,
                "target-cu" => self.target_cu = num32()?,
                "wgps-per-se" => self.wgps_per_se = num32()?,
                "launch-model" => self.launch_model = LaunchChoice::parse(value)?,
                "prefetch" => {
                    self.prefetch = Some(Prefetch::parse(value).with_context(|| {
                        format!("unknown prefetch '{value}' (per-wave, wave0, none)")
                    })?)
                }
                "natural-baseline" => self.natural_baseline = flag(&key, value)?,
                "original-baseline" => self.original_baseline = flag(&key, value)?,
                "admission-slack" => self.admission_slack = Some(num()?),
                "loop-markers" => {
                    self.loop_markers = Some(LoopMarkers::parse(value).with_context(|| {
                        format!("unknown loop markers '{value}' (cluster, cluster-and-workgroup)")
                    })?)
                }
                "shader-engines" => self.shader_engines = Some(num32()?),
                "max-dispatches" => self.max_dispatches = num32()?,
                "table-capacity-bytes" => self.table_capacity_bytes = num()?,
                _ => bail!("unknown remapper option '{key}'"),
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.records == 0 || self.checks == 0 || self.repetitions == 0 || self.wgps_per_se == 0 {
            bail!("records, checks, repetitions and wgps_per_se must be positive");
        }
        Ok(())
    }
}
