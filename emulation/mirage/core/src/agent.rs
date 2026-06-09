//! Agent definitions.
//!
//! An [`AgentDef`] is the hardware-level description of a single
//! device (typically one GPU): a recursive tree of [`ComponentDef`]s
//! plus the [`LinkDef`]s wiring them together. Agents are
//! hardware-not-emulator-specific: the same `cdna3` agent JSON can
//! be consumed by any backend that knows how to interpret it.
//!
//! Agents live on disk at `<MIRAGE_CONFIG>/agent/<name>.json`. The
//! system-level layout that arranges agents into racks/nodes lives
//! in [`crate::topology`].

use serde::{Deserialize, Serialize};

fn one() -> u32 {
    1
}

/// Key-value pair for component configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ConfigEntry {
    pub key: String,

    /// All values as strings, parsed by the factory.
    pub value: String,
}

/// Port definition for dynamic ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PortDef {
    pub name: String,

    /// "in" or "out".
    pub direction: String,

    /// "untyped", "memory_req", "memory_resp", "dispatch", etc.
    pub protocol: String,
}

/// Component definition (recursive for hierarchy).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ComponentDef {
    /// Name or range pattern like "xcd[0:7]".
    pub name: String,

    /// Registry type: "compute_unit", "l2_cache", etc.
    #[serde(rename = "type")]
    pub r#type: String,

    /// Component-specific parameters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config: Vec<ConfigEntry>,

    /// Child components (recursive).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ComponentDef>,

    /// Optional dynamic ports.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<PortDef>,
}

/// Range variable for link pattern expansion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ForRange {
    /// Variable name: "i", "j", "k".
    pub var_name: String,

    /// Range start (inclusive).
    pub start: u32,

    /// Range end (exclusive).
    pub end: u32,
}

/// Link definition (direct or pattern-based).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkDef {
    /// Direct source: "soc.xcd0.l2.hbm_out".
    #[serde(default)]
    pub src: String,

    /// Direct destination.
    #[serde(default)]
    pub dst: String,

    /// Pattern: "soc.xcd[i].l2 -> soc.iod[i/4].msc".
    #[serde(default)]
    pub pattern: String,

    /// Loop variables.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub for_ranges: Vec<ForRange>,

    /// Filter: "i != j".
    #[serde(default)]
    pub where_expr: String,

    #[serde(default = "one")]
    pub latency: u32,

    #[serde(default = "one")]
    pub weight: u32,
}

impl Default for LinkDef {
    fn default() -> Self {
        Self {
            src: String::new(),
            dst: String::new(),
            pattern: String::new(),
            for_ranges: Vec::new(),
            where_expr: String::new(),
            latency: 1,
            weight: 1,
        }
    }
}

/// Declarative component-tree topology for a single agent.
///
/// Mirrors the flatbuffer `TopologyDef` in
/// `rocjitsu/schemas/simulation_config.fbs`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AgentTopologyDef {
    pub root: ComponentDef,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<LinkDef>,
}

/// KFD device identity and topology properties for sysfs generation.
/// Mirrors `KfdDeviceInfo` in the rocjitsu flatbuffer schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KfdDeviceInfo {
    #[serde(default)]
    pub gpu_id: u32,
    #[serde(default)]
    pub gfx_target_version: u32,
    #[serde(default)]
    pub vendor_id: u32,
    #[serde(default)]
    pub device_id: u32,
    #[serde(default)]
    pub family_id: u32,
    #[serde(default)]
    pub unique_id: u64,
    #[serde(default)]
    pub marketing_name: String,
    #[serde(default)]
    pub drm_render_minor: u32,
    #[serde(default)]
    pub simd_count: u32,
    #[serde(default)]
    pub max_waves_per_simd: u32,
    #[serde(default)]
    pub num_shader_engines: u32,
    #[serde(default)]
    pub num_shader_arrays_per_engine: u32,
    #[serde(default)]
    pub num_cu_per_sh: u32,
    #[serde(default)]
    pub simd_per_cu: u32,
    #[serde(default)]
    pub wave_front_size: u32,
    #[serde(default)]
    pub max_slots_scratch_cu: u32,
    #[serde(default)]
    pub local_mem_size: u64,
    #[serde(default)]
    pub lds_size_kb: u32,
    #[serde(default)]
    pub mem_width: u32,
    #[serde(default)]
    pub mem_clk_max: u32,
    #[serde(default)]
    pub l1_size_kb: u32,
    #[serde(default)]
    pub l1_line_size: u32,
    #[serde(default)]
    pub l1_assoc: u32,
    #[serde(default)]
    pub l2_size_kb: u32,
    #[serde(default)]
    pub l2_line_size: u32,
    #[serde(default)]
    pub l2_assoc: u32,
    #[serde(default)]
    pub num_sdma_engines: u32,
    #[serde(default)]
    pub num_sdma_xgmi_engines: u32,
    #[serde(default)]
    pub num_cp_queues: u32,
    #[serde(default)]
    pub max_engine_clk_fcompute: u32,
}

/// AMDGPU memory configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GpuMemoryConfig {
    #[serde(default)]
    pub size_mb: u32,
    #[serde(default)]
    pub memory_side_cache_mb: u32,
}

/// AMDGPU top-level configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AmdgpuConfig {
    #[serde(default)]
    pub num_xcds: u32,
    #[serde(default)]
    pub num_iods: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<GpuMemoryConfig>,
    #[serde(default)]
    pub device: KfdDeviceInfo,
}

/// Virtual machine hardware model. Mirrors `VirtualMachineConfig`
/// in the rocjitsu flatbuffer schema. `programs` is intentionally
/// omitted: it's runtime workload configuration, not hardware.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VirtualMachineConfig {
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub gpu: AmdgpuConfig,
}

/// Top-level agent (single-device hardware) definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AgentDef {
    pub vm: VirtualMachineConfig,
    pub topology: AgentTopologyDef,
}

/// On-disk agent store backed by `<MIRAGE_CONFIG>/agent/`.
///
/// Agents may be stored as opaque JSON blobs by external backends
/// (e.g. rocjitsu's `cdna3`/`cdna4` flatbuffer configs). `get()`
/// will fail to parse those as [`AgentDef`]; callers that need raw
/// access should read the file at [`crate::paths::agent_path`].
pub mod store {
    use super::AgentDef;
    use crate::error::{MirageError, Result};
    use std::path::PathBuf;

    /// List the names of all agent files on disk.
    pub fn list() -> Result<Vec<String>> {
        let root = crate::paths::agent_root();
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&root).map_err(|e| MirageError::Io {
            path: root.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| MirageError::Io {
                path: root.clone(),
                source: e,
            })?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(stem) = name.strip_suffix(".json") {
                out.push(stem.to_string());
            }
        }
        out.sort();
        Ok(out)
    }

    /// Read an agent by name.
    pub fn get(name: &str) -> Result<AgentDef> {
        let p = crate::paths::agent_path(name);
        crate::state::read_json(&p)
    }

    /// Write an agent to disk.
    pub fn put(name: &str, agent: &AgentDef) -> Result<PathBuf> {
        let p = crate::paths::agent_path(name);
        crate::state::write_json(&p, agent)?;
        Ok(p)
    }
}
