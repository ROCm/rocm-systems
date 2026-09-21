//! The parts of a rocjitsu preset the builtin agents must not drift from.
//!
//! Filled in by `build.rs`, which reads `rocjitsu/configs/*.json` — see
//! there for why these five values are read rather than written, and for
//! what is deliberately left to the [`agents`](mod@crate::agents)
//! module instead.

/// The arch name and per-CU limits of one rocjitsu preset.
///
/// Strings because that is what a component's `config` carries: rocjitsu
/// reads these back out of the JSON mirage writes, and parsing them to
/// integers here only to print them again would be a chance to change
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Preset {
    /// The preset file this came from, for error messages and doc.
    pub(crate) preset: &'static str,
    /// `vm.arch` — which ISA rocjitsu emulates, and therefore which
    /// limits it enforces on the values below.
    pub(crate) arch: &'static str,
    /// Wavefront slots per compute unit. The one that bites: `cdna5`
    /// caps it at 64 and rocjitsu refuses a larger value outright.
    pub(crate) num_wf_slots: &'static str,
    /// Scalar registers per wavefront.
    pub(crate) sgprs_per_wf: &'static str,
    /// Vector registers per wavefront.
    pub(crate) vgprs_per_wf: &'static str,
    /// Local data share per compute unit, in KiB.
    pub(crate) lds_size_kb: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/presets.rs"));
