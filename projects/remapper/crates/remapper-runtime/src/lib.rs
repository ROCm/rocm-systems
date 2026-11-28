//! Kernel replay side of the workgroup remapper: the pass schedule that turns one application dispatch into the
//! record, check and ATT passes of the remap, the device buffers those passes use, and the report and pass list the
//! ATT merge reads.
//!
//! A profiler drives it from its kernel replay callbacks: [`Remapper::begin_dispatch`] at CONFIG (before the loop),
//! [`Session::pass_enter`] and [`Session::pass_exit`] around each pass, and the session ends when `pass_exit`
//! returns `false`.

pub mod config;
pub mod device;
pub mod session;
pub mod setup;

pub use config::{Config, LaunchChoice};
pub use device::{Device, HsaDevice, LoadedKernel};
pub use session::{KernelInfo, PassPlan, PassRecord, Session, TraceTarget};

use anyhow::{bail, Result};
use serde::Serialize;

/// A kernel dispatch as a replay pass submits it (`rocprofiler_kernel_replay_dispatch_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Dispatch {
    pub kernel_object: u64,
    pub kernarg_address: u64,
    pub workgroup_size: [u32; 3],
    pub grid_size: [u32; 3],
    /// Workgroups per cluster, or zero for an ordinary dispatch packet.
    pub cluster_size: [u32; 3],
    pub group_segment_size: u32,
    pub private_segment_size: u32,
}

/// Remapper state across dispatches: which ones have been remapped.
pub struct Remapper {
    config: Config,
    begun: u32,
}

impl Remapper {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, begun: 0 })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Whether another dispatch may be remapped.
    pub fn wants_more(&self) -> bool {
        self.begun < self.config.max_dispatches
    }

    /// Start remapping a dispatch. On error the dispatch should run without replay.
    pub fn begin_dispatch<D: Device>(
        &mut self,
        info: &KernelInfo<'_>,
        device: D,
    ) -> Result<Session<D>> {
        if !self.wants_more() {
            bail!(
                "already remapped {} dispatches (the configured maximum)",
                self.begun
            );
        }
        self.begun += 1;
        let session = Session::begin(&self.config, info, device);
        if let Err(e) = &session {
            log::warn!(
                "remapper: dispatch {} not remapped: {e:#}",
                info.dispatch_id
            );
            let dir = self
                .config
                .output_dir
                .join(format!("dispatch-{}", info.dispatch_id));
            if std::fs::create_dir_all(&dir).is_ok() {
                let report = serde_json::json!({
                    "status": "failed", "version": 1, "dispatch_id": info.dispatch_id, "reason": format!("{e:#}"),
                    "symbol": info.symbol, "app_dispatch": info.app,
                });
                let _ = std::fs::write(
                    dir.join("report.json"),
                    serde_json::to_vec_pretty(&report).unwrap_or_default(),
                );
            }
        }
        session
    }
}

#[cfg(test)]
mod tests;
