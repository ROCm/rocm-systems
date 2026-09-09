//! The rocjitsu configs mirage ships, embedded by `build.rs`.
//!
//! One entry per GPU rocjitsu has a config for. Everything mirage
//! preloads is derived from this table: the builtin agents are the
//! `{vm, topology}` in each config ([`mod@crate::agents`]), and the
//! builtin profiles are one per entry, generated on demand
//! ([`mod@crate::profiles`]).

/// One shipped rocjitsu config, as mirage addresses it.
pub(crate) struct Preset {
    /// The name the agent and its profile are addressed by.
    pub name: &'static str,
    /// The `rocjitsu/configs/<stem>.json` it was read from, named in
    /// the profile's description so a user can go and read the source.
    pub stem: &'static str,
    /// `vm.gpu.device.marketing_name` — the GPU this describes.
    pub marketing_name: &'static str,
    /// `vm.arch` — the architecture rocjitsu emulates it as.
    pub arch: &'static str,
    /// The `{vm, topology}` document, as JSON. Parsed rather than
    /// constructed, which is what keeps a field the config omits
    /// omitted; see [`mirage_core::agent::AgentDef`].
    pub agent: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/presets.rs"));
