//! `manifest.json` of a remap run directory: the rewritten kernel's layout, the tool buffers the replayer allocates,
//! and (once a record run has been turned into one) the placement table.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const FILE_NAME: &str = "manifest.json";
pub const VERSION: u32 = 1;
/// Log record size in dwords; the layout depends on the prologue (see `table.rs`).
pub const LOG_WORDS: u64 = 8;
/// Control block: 0 mode, 4 max_k, 8 arrival*, 16 table*, 24 counters*, 32 log*, then extra buffer pointers.
pub const CONTROL_BYTES: u64 = 64;
pub const FIXED_CONTROL_BYTES: usize = 40;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Manifest {
    pub version: u32,
    pub symbol: String,
    /// Rewritten code object, relative to the manifest directory.
    pub code_object: PathBuf,
    /// Captured code object it was built from, relative to the manifest directory.
    pub source_code_object: PathBuf,
    pub dispatch_index: u64,
    pub workgroups: [u64; 3],
    pub num_workgroups: u64,
    pub waves_per_workgroup: u32,
    /// Byte offset in the kernarg segment where the replayer appends the control-block address.
    pub control_kernarg_offset: usize,
    pub control_bytes: u64,
    /// Arrival counters: one per location, or one per cluster key.
    pub num_keys: u64,
    pub log_words: u64,
    /// Log records the prologue writes: one per workgroup, or one per cluster.
    pub num_log_records: u64,
    /// LDS the prologue needs to hand the chosen ids from wave 0 to the other waves.
    pub min_group_segment_bytes: u32,
    pub extra_buffers: Vec<ExtraBuffer>,
    pub cluster: Option<ClusterLayout>,
    pub table: Option<TableRef>,
    pub rewrite: RewriteInfo,
    /// Sparse launch model; absent for the table model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch: Option<LaunchLayout>,
}

/// Sparse launch: the replayer dispatches one workgroup per WGP. The workgroup at an assigned location runs the
/// body as that logical workgroup; every other one ends in the prologue. Admission launch (`model` "admission"):
/// table modes dispatch the captured workgroups plus slack, and arrivals past a location's quota park.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LaunchLayout {
    pub model: String,
    /// Padded grid in work-items.
    pub padded_grid_size: [u32; 3],
    pub padded_workgroups: u64,
    /// Filled after the probe.
    pub assignment: Vec<AssignedWorkgroup>,
    /// Traced locations that hold a real workgroup.
    pub traced_locations: Vec<u32>,
    /// Admission model: workgroups launched beyond the captured ones.
    #[serde(default)]
    pub slack: u64,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssignedWorkgroup {
    pub location: u32,
    pub ttmp9: u32,
    pub ttmp7: u32,
}

/// Table entry of a sparse location whose workgroup ends in the prologue.
pub const FAKE: u32 = 0xFFFF_FFFE;
/// Table entry of a location without an entry.
pub const NO_ENTRY: u32 = 0xFFFF_FFFF;

impl Manifest {
    pub fn is_sparse(&self) -> bool {
        self.launch.as_ref().is_some_and(|l| l.model == "sparse")
    }
    pub fn is_admission(&self) -> bool {
        self.launch.as_ref().is_some_and(|l| l.model == "admission")
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExtraBuffer {
    pub name: String,
    pub control_offset: usize,
    pub bytes: u64,
    pub fill: u32,
    /// Leading dwords written over the fill before every replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init: Option<Vec<u32>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ClusterLayout {
    pub size: u32,
    /// Workgroups per cluster in X, Y, Z; absent in one-dimensional manifests written before 2-D support.
    #[serde(default)]
    pub dims: [u32; 3],
    pub num_clusters: u64,
    pub shader_engines: u32,
    /// `loc % 32` of the traced WGP: `sa * 16 + wgp`.
    pub traced_wgp: u32,
}

impl ClusterLayout {
    pub fn dims(&self) -> [u32; 3] {
        if self.dims == [0; 3] {
            [self.size, 1, 1]
        } else {
            self.dims
        }
    }
    /// Whether `rank` (wg_x | wg_y << 4, as in ttmp6[7:0]) names a workgroup of the cluster. Rank renaming XORs
    /// both fields, so a valid XOR mask satisfies the same bounds.
    pub fn valid_rank(&self, rank: u32) -> bool {
        let [x, y, _] = self.dims();
        rank >> 8 == 0 && rank & 15 < x && rank >> 4 < y
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TableRef {
    /// `num_keys * max_k` int32 pairs, relative to the manifest directory.
    pub file: PathBuf,
    pub max_k: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RewriteInfo {
    pub prologue_vaddr: u64,
    pub prologue_bytes: u64,
    pub original_entry_vaddr: u64,
    pub kernel_vaddr: u64,
    pub kernel_size: u64,
    /// INST_PREF_SIZE lines: the original descriptor's, the patched descriptor's (prologue only), and the body
    /// lines the prologue prefetches.
    pub original_prefetch_lines: u32,
    pub descriptor_prefetch_lines: u32,
    pub body_prefetch: String,
    pub reserved_sgprs: BTreeMap<String, u32>,
    pub prologue_sgprs: String,
    pub workgroup_id_sites: Vec<PatchedSite>,
    pub cluster_rank_sites: Vec<RankSite>,
    pub preload_converted: bool,
    /// Opt-in loop markers (`--loop-markers`); absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loop_markers: Option<LoopMarkerInfo>,
}

/// Body barriers redirected through marker stubs placed after the prologue (see `markers.rs`).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoopMarkerInfo {
    pub sites_kind: String,
    /// SGPR the stubs save M0 in (the body never references it); `None` when the body never reads M0.
    pub save_sgpr: Option<u32>,
    pub stubs_vaddr: u64,
    pub stubs_bytes: u64,
    pub sites: Vec<MarkerSite>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MarkerSite {
    pub addr: u64,
    pub original: String,
    /// Marker word bits [31:24]: A3 cluster signal, A4 cluster wait, A5 workgroup wait.
    pub tag: u8,
    /// Marker word bits [23:16].
    pub index: u32,
    pub stub_vaddr: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PatchedSite {
    pub addr: u64,
    pub original: String,
    pub patched: String,
    pub bytes_changed: Vec<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct RankSite {
    pub addr: u64,
    pub text: String,
    /// `data` (reads the logical rank) or `multicast` (keeps the hardware rank).
    pub kind: String,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<(Self, PathBuf)> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let m: Manifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        if m.version != VERSION {
            bail!(
                "{}: manifest version {} (expected {VERSION})",
                path.display(),
                m.version
            );
        }
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Ok((m, dir))
    }

    pub fn save(&self, dir: &Path) -> Result<PathBuf> {
        let path = dir.join(FILE_NAME);
        std::fs::write(&path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }

    pub fn table_bytes(&self, max_k: u32) -> u64 {
        self.num_keys * max_k as u64 * 8
    }
}
