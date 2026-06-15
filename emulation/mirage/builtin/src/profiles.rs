//! Strongly-typed builtin [`ProfileDef`]s.
//!
//! These are the ready-to-use presets mirage preloads into
//! `<MIRAGE_CONFIG>/profile/` so a fresh install can `mirage run
//! --profile <emulator>-<gpu>` without first hand-building a profile.
//!
//! Each profile pins a single GPU agent (the GPU name is part of the
//! profile name) and runs every node inside the standard ROCm
//! development container. rocjitsu is a software emulator and works on
//! every builtin agent; HotSwap only supports `MI450X`, so it ships a
//! single profile.

use mirage_core::common::{MaybeRef, SimpleMap};
use mirage_core::emulator::{EmulatorDef, EmulatorKind, ExecMode};
use mirage_core::profile::ProfileDef;
use mirage_core::topology::TopologyDef;

/// All builtin profiles, keyed by the name written to disk.
///
/// Names follow `<emulator>-<gpu>` so the GPU each profile targets is
/// visible at a glance in `mirage profile list`.
pub fn profiles() -> Vec<(&'static str, ProfileDef)> {
    vec![
        (
            "rocjitsu-MI300X",
            profile("rocjitsu-MI300X", "rocjitsu", "MI300X"),
        ),
        (
            "rocjitsu-MI350X",
            profile("rocjitsu-MI350X", "rocjitsu", "MI350X"),
        ),
        (
            "rocjitsu-MI450X",
            profile("rocjitsu-MI450X", "rocjitsu", "MI450X"),
        ),
        // HotSwap only supports MI450X.
        (
            "hotswap-MI450X",
            profile("hotswap-MI450X", "hotswap", "MI450X"),
        ),
    ]
}

/// Build a single-GPU, containerised builtin profile pinning `agent`.
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
    use super::*;

    #[test]
    fn ships_expected_profiles() {
        let names: Vec<&str> = profiles().iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "rocjitsu-MI300X",
                "rocjitsu-MI350X",
                "rocjitsu-MI450X",
                "hotswap-MI450X",
            ]
        );
    }

    #[test]
    fn hotswap_only_targets_mi450x() {
        let hotswap: Vec<&str> = profiles()
            .iter()
            .filter(|(_, p)| p.emulator.emulator == "hotswap")
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(hotswap, vec!["hotswap-MI450X"]);
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
