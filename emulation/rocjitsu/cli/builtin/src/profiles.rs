//! Strongly-typed builtin [`ProfileDef`]s.
//!
//! These are the ready-to-use presets rocjitsu preloads into
//! `<ROCJITSU_CLI_CONFIG_DIR>/profile/` so a fresh install can `rocjitsu run
//! --profile <gpu>` without first hand-building a profile.
//!
//! Each profile pins a single GPU agent and is named after that GPU in
//! lowercase (e.g. `mi450x`). Every builtin profile targets the
//! rocjitsu software emulator, which works on every builtin agent.

use rj_core::common::{MaybeRef, SimpleMap};
use rj_core::emulator::{EmulatorDef, EmulatorKind, ExecMode};
use rj_core::profile::ProfileDef;
use rj_core::topology::TopologyDef;

/// All builtin profiles, keyed by the name written to disk.
///
/// Names are the target GPU in lowercase (e.g. `mi450x`) so the GPU
/// each profile targets is visible at a glance in `rocjitsu profile
/// list`. Every builtin targets the rocjitsu emulator.
pub fn profiles() -> Vec<(&'static str, ProfileDef)> {
    vec![
        ("mi300x", profile("mi300x", "rocjitsu", "MI300X")),
        ("mi350x", profile("mi350x", "rocjitsu", "MI350X")),
        ("mi450x", profile("mi450x", "rocjitsu", "MI450X")),
    ]
}

/// Build a single-GPU builtin profile pinning `agent`.
fn profile(name: &str, emulator: &str, agent: &str) -> ProfileDef {
    ProfileDef {
        name: name.to_string(),
        description: None,
        emulator: EmulatorDef {
            emulator: EmulatorKind::from(emulator),
            plugins: Default::default(),
            exec_mode: ExecMode::default(),
            options: SimpleMap::default(),
            topology: MaybeRef::Owned(TopologyDef {
                num_nodes: 1,
                gpus_per_node: 1,
                agent: MaybeRef::Ref(agent.to_string()),
            }),
        },
        containerize: None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn ships_expected_profiles() {
        let names: Vec<&str> = profiles().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["mi300x", "mi350x", "mi450x"]);
    }

    #[test]
    fn every_builtin_targets_rocjitsu() {
        for (name, p) in profiles() {
            assert_eq!(
                p.emulator.emulator, "rocjitsu",
                "builtin profile {name} must target rocjitsu"
            );
        }
    }

    #[test]
    fn names_are_lowercase() {
        for (name, _) in profiles() {
            assert_eq!(
                name,
                name.to_lowercase(),
                "builtin profile {name} must be lowercase"
            );
        }
    }

    #[test]
    fn no_profile_is_containerised() {
        // Builtin profiles are plain (non-containerised); container
        // settings are layered on at session-create time via CLI flags,
        // not baked into the builtin definitions.
        for (_, p) in profiles() {
            assert!(
                p.containerize.is_none(),
                "builtin profile {} must not be containerised",
                p.name
            );
        }
    }
}
