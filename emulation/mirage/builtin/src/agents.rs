//! The builtin [`AgentDef`]s: one per rocjitsu config mirage ships.
//!
//! Each is the `{vm, topology}` of `rocjitsu/configs/<stem>.json`,
//! embedded and validated at build time — see `build.rs` for which
//! config becomes which agent, and why one GPU yields one agent when
//! rocjitsu ships several configs for it.

use mirage_core::agent::AgentDef;

use crate::presets::PRESETS;

/// All builtin agents, keyed by the name written to disk.
pub fn agents() -> Vec<(&'static str, AgentDef)> {
    PRESETS
        .iter()
        .map(|preset| (preset.name, from_preset(preset.agent)))
        .collect()
}

/// The builtin agent called `name`, if mirage ships one.
///
/// Case-insensitively, as every other way of naming an agent is:
/// `mirage agent show MI350X` and `mirage agent show mi350x` are the
/// same document, and a lookup here that disagreed would be a second
/// spelling rule for the same names.
#[must_use]
pub fn agent(name: &str) -> Option<AgentDef> {
    PRESETS
        .iter()
        .find(|preset| preset.name.eq_ignore_ascii_case(name))
        .map(|preset| from_preset(preset.agent))
}

/// One builtin, parsed from the config `build.rs` embedded.
///
/// The `expect` is the workspace's one production opt-out of
/// `expect_used`, and it is here because there is nothing to report:
/// `build.rs` parses the same string into the same [`AgentDef`] and
/// fails the build if it cannot, so a panic here means the crate was
/// linked against a `mirage_core` it was not built against. There is no
/// user input on this path and no configuration that reaches it.
#[allow(clippy::expect_used)]
fn from_preset(json: &str) -> AgentDef {
    serde_json::from_str(json).expect("builtin agent validated by build.rs")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// Every agent is the config it names, in full and unchanged.
    ///
    /// Read off disk rather than through `PRESETS`, so this compares
    /// what mirage ships against the source of truth rather than the
    /// embedding against itself.
    #[test]
    fn every_builtin_matches_its_rocjitsu_config() {
        for preset in PRESETS {
            let path = format!(
                "{}/../../rocjitsu/configs/{}.json",
                env!("CARGO_MANIFEST_DIR"),
                preset.stem
            );
            let source = std::fs::read_to_string(&path).unwrap();
            let config: serde_json::Value = serde_json::from_str(&source).unwrap();
            let document = serde_json::json!({
                "vm": config["vm"], "topology": config["topology"]
            });
            let expected: AgentDef = serde_json::from_value(document.clone()).unwrap();
            let agent = agent(preset.name).unwrap();

            assert_eq!(agent, expected, "{}", preset.name);
            // Round-trips to the config's own bytes: a field the config
            // omits has to stay omitted, or rocjitsu's schema default
            // for it is replaced by Rust's. See `AgentDef`.
            assert_eq!(
                serde_json::to_value(&agent).unwrap(),
                document,
                "{}",
                preset.name
            );
            assert_eq!(
                serde_json::from_value::<AgentDef>(serde_json::to_value(&agent).unwrap()).unwrap(),
                agent,
                "{}",
                preset.name
            );
            // rocjitsu refuses to load a device with SDMA engines and
            // nowhere to queue to, so a config mirage ships as an agent
            // must not be one.
            let device = &agent.vm.gpu.device;
            assert!(
                device.num_sdma_engines == 0 || device.num_sdma_queues_per_engine > 0,
                "{}",
                preset.name
            );
        }
    }

    /// The three names that were mirage's whole builtin set are still
    /// there: `--profile` defaults to `mi350x`, the shipped topologies
    /// name `MI350X` and `MI300X`, and users have all three on disk.
    #[test]
    fn the_established_names_survive_reading_the_whole_directory() {
        let names: Vec<&str> = agents().into_iter().map(|(name, _)| name).collect();
        for established in ["mi300x", "mi350x", "mi450x"] {
            assert!(names.contains(&established), "{established} in {names:?}");
        }
    }

    /// One agent per GPU, not one per config file.
    #[test]
    fn no_two_agents_are_the_same_device() {
        let mut seen = std::collections::BTreeMap::new();
        for (name, agent) in agents() {
            let version = agent.vm.gpu.device.gfx_target_version;
            if let Some(other) = seen.insert(version, name) {
                panic!("agents {other} and {name} are both gfx_target_version {version}");
            }
        }
    }

    #[test]
    fn agents_are_addressed_case_insensitively() {
        assert_eq!(agent("MI350X"), agent("mi350x"));
        assert!(agent("mi350x").is_some());
        assert!(agent("no-such-gpu").is_none());
    }
}
