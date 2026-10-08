//! The builtin [`ProfileDef`]s: one per rocjitsu config mirage ships.
//!
//! These are the ready-to-use presets that let a fresh install `mirage
//! run --profile <gpu>` without first hand-building a profile. Each
//! pins a single GPU agent, is named after that agent, and targets the
//! rocjitsu software emulator, which works on every builtin agent.
//!
//! # They are generated, not stored
//!
//! Nothing here is ever written to `<MIRAGE_CONFIG>/profile/`. A builtin
//! profile is derived from the agent it pins, every time it is asked
//! for, and [`mirage_core::store`] serves it from here when the profile
//! directory has no file of that name — see [`crate::builtin_profiles`].
//!
//! Writing them out had no upside and two costs. A profile is three
//! fields around an agent reference, so the file added nothing the agent
//! did not already say, and it went stale the moment a mirage upgrade
//! changed how the derivation works — leaving the user to run `mirage
//! state builtins` to pick up a definition mirage could simply have
//! produced. And they had to be seeded before they could be listed,
//! which made `mirage profile list` on an unwritable config directory
//! report a machine with no profiles at all.
//!
//! Writing your own profile under a builtin's name still works and still
//! wins: the file shadows the generated one, and deleting it brings the
//! generated one back.

use mirage_core::common::{MaybeRef, SimpleMap};
use mirage_core::emulator::{EmulatorDef, EmulatorKind, ExecMode};
use mirage_core::profile::ProfileDef;
use mirage_core::topology::TopologyDef;

use crate::presets::{PRESETS, Preset};

/// The emulator every builtin profile targets.
const EMULATOR: &str = "rocjitsu";

/// All builtin profiles, keyed by the name they are addressed by — the
/// same name as the agent each one pins.
pub fn profiles() -> Vec<(&'static str, ProfileDef)> {
    PRESETS
        .iter()
        .map(|preset| (preset.name, profile(preset)))
        .collect()
}

/// The builtin profile called `name`, if mirage ships one.
///
/// Case-insensitive, as profile names are everywhere else.
#[must_use]
pub fn profile_named(name: &str) -> Option<ProfileDef> {
    PRESETS
        .iter()
        .find(|preset| preset.name.eq_ignore_ascii_case(name))
        .map(profile)
}

/// Build the single-GPU profile for one shipped config.
fn profile(preset: &Preset) -> ProfileDef {
    ProfileDef {
        name: preset.name.to_string(),
        // Seven profiles need telling apart, and the name alone does not
        // do it: `gfx1151` names a file, not a machine. The GPU, the
        // architecture rocjitsu emulates it as, and the config it comes
        // from are the three things a user picking one wants, and the
        // last is where they go to read the rest.
        description: Some(format!(
            "{} ({}), from RocJITsu's configs/{}.json",
            preset.marketing_name, preset.arch, preset.stem
        )),
        emulator: EmulatorDef {
            extra: Default::default(),
            emulator: EmulatorKind::from(EMULATOR),
            plugins: Default::default(),
            exec_mode: ExecMode::default(),
            options: SimpleMap::default(),
            // One GPU on one node. Wider runs come from `--num-nodes`
            // and `--gpus-per-node`, which override this per run, so the
            // profile does not have to guess at a shape.
            topology: MaybeRef::Owned(TopologyDef {
                num_nodes: 1,
                gpus_per_node: 1,
                agent: MaybeRef::Ref(preset.name.to_string()),
            }),
        },
        containerize: None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// There is exactly one profile per builtin agent, and it pins that
    /// agent. A profile naming an agent mirage does not ship is a
    /// dangling reference the first `mirage run` would meet.
    #[test]
    fn every_profile_pins_the_agent_it_is_named_after() {
        let agents: Vec<&str> = crate::agents::agents()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        let profiles = profiles();
        assert_eq!(
            profiles.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            agents,
            "one profile per agent"
        );
        for (name, profile) in profiles {
            let MaybeRef::Owned(topology) = &profile.emulator.topology else {
                panic!("builtin profile {name} must carry its topology inline");
            };
            assert_eq!(topology.agent, MaybeRef::Ref(name.to_string()));
            assert_eq!(topology.total_gpus(), 1);
            assert_eq!(profile.name, name);
        }
    }

    #[test]
    fn every_builtin_targets_rocjitsu() {
        for (name, p) in profiles() {
            assert_eq!(
                p.emulator.emulator, EMULATOR,
                "builtin profile {name} must target rocjitsu"
            );
        }
    }

    /// Profiles are stored lowercase, so a builtin whose name is not
    /// already lowercase could never be written over by the user.
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

    /// Each description says which GPU it is and where it came from —
    /// the two questions `mirage profile list -l` exists to answer.
    #[test]
    fn descriptions_name_the_gpu_and_its_config() {
        for preset in PRESETS {
            let description = profile_named(preset.name).unwrap().description.unwrap();
            assert!(description.contains(preset.marketing_name), "{description}");
            assert!(
                description.contains(&format!("configs/{}.json", preset.stem)),
                "{description}"
            );
        }
    }

    #[test]
    fn profiles_are_addressed_case_insensitively() {
        assert_eq!(profile_named("MI350X"), profile_named("mi350x"));
        assert!(profile_named("mi350x").is_some());
        assert!(profile_named("no-such-gpu").is_none());
    }
}
