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

/// Top-level topology definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TopologyDef {
    pub root: ComponentDef,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<LinkDef>,
}
