//! Agent definitions.
//!
//! An [`AgentDef`] is the hardware-level description of a single
//! device (typically one GPU): a recursive tree of [`ComponentDef`]s
//! plus the [`LinkDef`]s wiring them together. Agents are
//! hardware-not-emulator-specific: the same `cdna3` agent JSON can
//! be consumed by any backend that knows how to interpret it.
//!
//! Agents live on disk at `<ROCJITSU_CLI_CONFIG_DIR>/agent/<name>.json`. The
//! system-level layout that arranges agents into racks/nodes lives
//! in [`crate::topology`].
//!
//! Every type here rejects unknown fields; see [`crate::profile`] for
//! why. It bites hardest on an agent, where the fields are the emulated
//! device's identity — a mistyped `l2_size_kb` that silently defaulted to
//! zero produced a GPU the workload could see and could not explain.

use serde::{Deserialize, Serialize};

fn one() -> u32 {
    1
}

/// Key-value pair for component configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConfigEntry {
    pub key: String,

    /// All values as strings, parsed by the factory.
    pub value: String,
}

/// Port definition for dynamic ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PortDef {
    pub name: String,

    /// "in" or "out".
    pub direction: String,

    /// "untyped", "memory_req", "memory_resp", "dispatch", etc.
    pub protocol: String,
}

/// Component definition (recursive for hierarchy).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ComponentDef {
    /// Name or range pattern like `"xcd[0:7]"`.
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct LinkDef {
    /// Direct source: "soc.xcd0.l2.hbm_out".
    #[serde(default)]
    pub src: String,

    /// Direct destination.
    #[serde(default)]
    pub dst: String,

    /// Pattern: `"soc.xcd[i].l2 -> soc.iod[i/4].msc"`.
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
#[serde(deny_unknown_fields)]
pub struct AgentTopologyDef {
    pub root: ComponentDef,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<LinkDef>,
}

/// KFD device identity and topology properties for sysfs generation.
/// Mirrors `KfdDeviceInfo` in the rocjitsu flatbuffer schema.
///
/// # Defaults are the schema's
///
/// `Default` is [hand-written](#impl-Default-for-KfdDeviceInfo) from
/// values the build script reads out of the schema, and `serde(default)`
/// is on the container so a field a document omits gets that rather than
/// `u32::default()`. Both halves are needed and neither is optional.
///
/// Every field here is serialised, so leaving one at zero does not omit
/// it — it writes a zero, and the emulator cannot tell that from a
/// deliberate one. Eighteen of these fields have a nonzero default in
/// the schema, and the derived `Default` overrode six of them in every
/// shipped profile: a 0 KB L1, 0-byte cache lines, 0-way associativity
/// and no scratch slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
// No `serde(default)` on any field below. It would mean the *field
// type's* default — zero — and take precedence over the container's,
// which is the whole point of this struct having one.
pub struct KfdDeviceInfo {
    pub gpu_id: u32,
    pub gfx_target_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub family_id: u32,
    pub unique_id: u64,
    pub marketing_name: String,
    pub drm_render_minor: u32,
    pub simd_count: u32,
    pub max_waves_per_simd: u32,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub num_cu_per_sh: u32,
    pub simd_per_cu: u32,
    pub wave_front_size: u32,
    pub max_slots_scratch_cu: u32,
    pub local_mem_size: u64,
    pub lds_size_kb: u32,
    pub mem_width: u32,
    pub mem_clk_max: u32,
    pub l1_size_kb: u32,
    pub l1_line_size: u32,
    pub l1_assoc: u32,
    pub l2_size_kb: u32,
    pub l2_line_size: u32,
    pub l2_assoc: u32,
    pub num_sdma_engines: u32,
    pub num_sdma_xgmi_engines: u32,
    /// Rejected by the emulator when zero and `num_sdma_engines` is not.
    pub num_sdma_queues_per_engine: u32,
    pub num_cp_queues: u32,
    pub max_engine_clk_fcompute: u32,
}

/// The schema's own defaults, read from `simulation_config.fbs` at build
/// time. See `core/build.rs`.
mod kfd_device_defaults {
    // The schema declares 40 fields and `KfdDeviceInfo` models 31, so
    // nine of these constants have no reader — `location_id`,
    // `hive_id`, `capability` and the rest, which this crate leaves to
    // the emulator's own defaults. Publishing the whole table is what
    // keeps the build script from having to know which subset is
    // modelled, and the failure that matters still fails: a field the
    // struct names and the schema drops takes the constant with it, and
    // the `Default` below stops compiling.
    #![allow(dead_code)]

    include!(concat!(env!("OUT_DIR"), "/kfd_device_defaults.rs"));
}

impl Default for KfdDeviceInfo {
    fn default() -> Self {
        use kfd_device_defaults as d;
        Self {
            gpu_id: d::GPU_ID,
            gfx_target_version: d::GFX_TARGET_VERSION,
            vendor_id: d::VENDOR_ID,
            device_id: d::DEVICE_ID,
            family_id: d::FAMILY_ID,
            unique_id: d::UNIQUE_ID,
            marketing_name: String::new(),
            drm_render_minor: d::DRM_RENDER_MINOR,
            simd_count: d::SIMD_COUNT,
            max_waves_per_simd: d::MAX_WAVES_PER_SIMD,
            num_shader_engines: d::NUM_SHADER_ENGINES,
            num_shader_arrays_per_engine: d::NUM_SHADER_ARRAYS_PER_ENGINE,
            num_cu_per_sh: d::NUM_CU_PER_SH,
            simd_per_cu: d::SIMD_PER_CU,
            wave_front_size: d::WAVE_FRONT_SIZE,
            max_slots_scratch_cu: d::MAX_SLOTS_SCRATCH_CU,
            local_mem_size: d::LOCAL_MEM_SIZE,
            lds_size_kb: d::LDS_SIZE_KB,
            mem_width: d::MEM_WIDTH,
            mem_clk_max: d::MEM_CLK_MAX,
            l1_size_kb: d::L1_SIZE_KB,
            l1_line_size: d::L1_LINE_SIZE,
            l1_assoc: d::L1_ASSOC,
            l2_size_kb: d::L2_SIZE_KB,
            l2_line_size: d::L2_LINE_SIZE,
            l2_assoc: d::L2_ASSOC,
            num_sdma_engines: d::NUM_SDMA_ENGINES,
            num_sdma_xgmi_engines: d::NUM_SDMA_XGMI_ENGINES,
            num_sdma_queues_per_engine: d::NUM_SDMA_QUEUES_PER_ENGINE,
            num_cp_queues: d::NUM_CP_QUEUES,
            max_engine_clk_fcompute: d::MAX_ENGINE_CLK_FCOMPUTE,
        }
    }
}

/// AMDGPU memory configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct GpuMemoryConfig {
    #[serde(default)]
    pub size_mb: u32,
    #[serde(default)]
    pub memory_side_cache_mb: u32,
}

/// AMDGPU top-level configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AmdgpuConfig {
    #[serde(default)]
    pub num_xcds: u32,
    #[serde(default)]
    pub num_iods: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<GpuMemoryConfig>,
    #[serde(default)]
    pub device: KfdDeviceInfo,

    /// Number of simulated GPU instances. Mirrors `num_gpus` in the
    /// rocjitsu flatbuffer schema (defaults to 1). The system-level
    /// per-node GPU count comes from
    /// [`crate::topology::TopologyDef::gpus_per_node`]; this field lets
    /// the synthesised rocjitsu config request that many devices.
    #[serde(default = "one")]
    pub num_gpus: u32,
}

/// Virtual machine hardware model. Mirrors `VirtualMachineConfig`
/// in the rocjitsu flatbuffer schema. `programs` is intentionally
/// omitted: it's runtime workload configuration, not hardware.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct VirtualMachineConfig {
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub gpu: AmdgpuConfig,
}

/// Top-level agent (single-device hardware) definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AgentDef {
    pub vm: VirtualMachineConfig,
    pub topology: AgentTopologyDef,
}

/// On-disk agent store backed by `<ROCJITSU_CLI_CONFIG_DIR>/agent/`.
///
/// Agents may be stored as opaque JSON blobs by external backends
/// (e.g. rocjitsu's `cdna3`/`cdna4` flatbuffer configs). `get()`
/// will fail to parse those as [`AgentDef`]; callers that need raw
/// access should read the file at [`crate::paths::agent_path`].
///
/// [`crate::store::agent_get`] is where a `MaybeRef::Ref` on a topology is
/// followed, so it is
/// also where that reference is checked, and where one that resolves to
/// nothing is reported — see [`crate::topology::store`].
pub mod store {
    use super::AgentDef;
    use crate::error::{Result, RocJITsuError};
    use crate::store::{DocKind, Referrer, dangling_ref, validate_name};
    use std::path::PathBuf;

    /// List the names of all agent files on disk.
    pub fn list() -> Result<Vec<String>> {
        let root = crate::paths::agent_root();
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&root).map_err(|e| RocJITsuError::Io {
            path: root.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| RocJITsuError::Io {
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

    /// Read an agent by name, for a caller that cannot say which
    /// topology sent it.
    ///
    /// Prefer [`get_referred_by`] wherever the referring topology is in
    /// scope; see [`crate::topology::store::get`] for why the name of the
    /// referring document is the half that makes the error actionable.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not a single path component, if
    /// there is no such agent — reported as the dangling reference it is,
    /// since a topology is what brought the name here — or if the
    /// document is malformed.
    pub fn get(name: &str) -> Result<AgentDef> {
        get_referred_by(Referrer::anonymous(DocKind::Topology), name)
    }

    /// Read an agent by name on behalf of the document that named it.
    ///
    /// The referrer may be a topology or the profile that carries one
    /// inline, which is why it is a value rather than the constant it
    /// used to be.
    ///
    /// # Errors
    ///
    /// As [`get`], with the referring document named in a dangling
    /// reference.
    pub fn get_referred_by(referrer: Referrer<'_>, name: &str) -> Result<AgentDef> {
        validate_name(DocKind::Agent, name)?;
        let p = crate::paths::agent_path(name);
        if !p.exists() {
            return Err(dangling_ref(referrer, DocKind::Agent, name));
        }
        crate::state::read_json(&p)
    }

    /// Write an agent to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not a single path component or the
    /// document cannot be written.
    pub fn put(name: &str, agent: &AgentDef) -> Result<PathBuf> {
        validate_name(DocKind::Agent, name)?;
        let p = crate::paths::agent_path(name);
        crate::state::write_json(&p, agent)?;
        Ok(p)
    }
}
