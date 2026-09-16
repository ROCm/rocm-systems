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
//!
//! Additional fields are retained for the emulator, including fields in
//! nested device and topology objects.

use serde::{Deserialize, Serialize};

fn one() -> u32 {
    1
}

/// Key-value pair for component configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ConfigEntry {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    pub key: String,

    /// All values as strings, parsed by the factory.
    pub value: String,
}

/// Port definition for dynamic ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PortDef {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    pub name: String,

    /// "in" or "out".
    pub direction: String,

    /// "untyped", "memory_req", "memory_resp", "dispatch", etc.
    pub protocol: String,
}

/// Component definition (recursive for hierarchy).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ComponentDef {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
pub struct ForRange {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
            extra: Default::default(),
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
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    pub root: ComponentDef,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<LinkDef>,
}

/// KFD device identity and topology properties for sysfs generation.
/// Mirrors `KfdDeviceInfo` in the rocjitsu flatbuffer schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KfdDeviceInfo {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
    pub num_sdma_queues_per_engine: u32,
    #[serde(default)]
    pub num_cp_queues: u32,
    #[serde(default)]
    pub max_engine_clk_fcompute: u32,
}

/// AMDGPU memory configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GpuMemoryConfig {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub size_mb: u32,
    #[serde(default)]
    pub memory_side_cache_mb: u32,
}

/// AMDGPU top-level configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AmdgpuConfig {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
/// in the rocjitsu flatbuffer schema. Fields without a typed accessor,
/// including `programs`, are retained in the passthrough map.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VirtualMachineConfig {
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub gpu: AmdgpuConfig,
}

/// Top-level agent (single-device hardware) definition.
///
/// # Absent is not zero
///
/// rocjitsu's schema has its own defaults, and several of them are not
/// Rust's: `drm_render_minor` defaults to 128 there and to 0 here, and a
/// 0 maps the emulated GPU onto a render node that does not exist, so
/// HSA aborts with `OUT_OF_RESOURCES`. A field the document did not
/// mention therefore has to stay unmentioned when mirage writes the
/// document back out, rather than being written as the Rust default.
///
/// [`omitted`](Self::omitted) is how that is remembered: every path
/// `Deserialize` had to fill in, with the value it filled in, so
/// `Serialize` can take it back out again. Two consequences a caller
/// has to know about, because the type cannot enforce either:
///
/// * The record is taken when the document is *parsed*. An `AgentDef`
///   built in Rust — [`Default::default`], a struct literal — has an
///   empty record and serializes every field explicitly, `drm_render_minor: 0`
///   included. Build agents by parsing, not by constructing, anywhere
///   the result reaches rocjitsu.
/// * The record is keyed by JSON pointer, so array paths are keyed by
///   *index*. Inserting into, removing from or reordering any `Vec` in
///   here after parsing re-points those paths at different elements,
///   and an element that happens to hold the recorded default loses
///   that field on the next serialize. Reparse rather than mutate a
///   `Vec`.
#[derive(Debug, Clone, Eq, Default)]
pub struct AgentDef {
    /// Document keys with no typed field of their own, kept so they
    /// reach the emulator. `vm` and `topology` are not among them and
    /// are overwritten on serialize by the fields below.
    pub extra: serde_json::Map<String, serde_json::Value>,
    pub vm: VirtualMachineConfig,
    pub topology: AgentTopologyDef,
    /// `(pointer, value)` for every field `Deserialize` defaulted
    /// because the document omitted it. See the type docs.
    omitted: Vec<(String, serde_json::Value)>,
}

/// Two agents are equal when they describe the same machine.
///
/// Deliberately not derived: `omitted` is provenance, not content, so a
/// terse document and a fully spelled-out one describing the same GPU
/// would otherwise compare unequal while serializing identically.
impl PartialEq for AgentDef {
    fn eq(&self, other: &Self) -> bool {
        self.extra == other.extra && self.vm == other.vm && self.topology == other.topology
    }
}

impl Serialize for AgentDef {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::Value::Object(self.extra.clone());
        value["vm"] = serde_json::to_value(&self.vm).map_err(serde::ser::Error::custom)?;
        value["topology"] =
            serde_json::to_value(&self.topology).map_err(serde::ser::Error::custom)?;
        for (path, default) in &self.omitted {
            if value.pointer(path) == Some(default)
                && let Some((parent, key)) = path.rsplit_once('/')
                && let Some(object) = value
                    .pointer_mut(parent)
                    .and_then(serde_json::Value::as_object_mut)
            {
                object.remove(&key.replace("~1", "/").replace("~0", "~"));
            }
        }
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AgentDef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            vm: VirtualMachineConfig,
            topology: AgentTopologyDef,
            #[serde(flatten)]
            extra: serde_json::Map<String, serde_json::Value>,
        }
        let original = serde_json::Value::deserialize(deserializer)?;
        let fields: Fields =
            serde_json::from_value(original.clone()).map_err(serde::de::Error::custom)?;
        let mut agent = Self {
            vm: fields.vm,
            topology: fields.topology,
            extra: fields.extra,
            omitted: Vec::new(),
        };
        let defaulted = serde_json::to_value(&agent).map_err(serde::de::Error::custom)?;
        collect_omitted(&original, &defaulted, "", &mut agent.omitted);
        Ok(agent)
    }
}

fn collect_omitted(
    original: &serde_json::Value,
    defaulted: &serde_json::Value,
    path: &str,
    omitted: &mut Vec<(String, serde_json::Value)>,
) {
    if let Some(object) = defaulted.as_object() {
        for (key, value) in object {
            let child_path = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
            if let Some(child) = original.get(key) {
                collect_omitted(child, value, &child_path, omitted);
            } else {
                omitted.push((child_path, value.clone()));
            }
        }
    } else if let (Some(original), Some(defaulted)) = (original.as_array(), defaulted.as_array()) {
        for (index, (child, value)) in original.iter().zip(defaulted).enumerate() {
            collect_omitted(child, value, &format!("{path}/{index}"), omitted);
        }
    }
}

/// On-disk agent store backed by `<MIRAGE_CONFIG>/agent/`.
///
/// Additional emulator fields are retained when an agent is read or written.
///
/// [`crate::store::agent_get`] is where a `MaybeRef::Ref` on a topology is
/// followed, so it is
/// also where that reference is checked, and where one that resolves to
/// nothing is reported — see [`crate::topology::store`].
pub mod store {
    use super::AgentDef;
    use crate::error::{MirageError, Result};
    use crate::store::{DocKind, Referrer, dangling_ref, validate_name};
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
