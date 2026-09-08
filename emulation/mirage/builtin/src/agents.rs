use mirage_core::agent::AgentDef;

use crate::presets::{MI300X, MI350X, MI450X};

pub fn agents() -> Vec<(&'static str, AgentDef)> {
    vec![
        ("mi300x", mi300x()),
        ("mi350x", mi350x()),
        ("mi450x", mi450x()),
        // \NPI new GPU: add its preset to `PRESETS` in `build.rs` and a
        // builtin agent mirroring `configs/<gpu>.json` here.
    ]
}

pub fn mi300x() -> AgentDef {
    from_preset(MI300X)
}

pub fn mi350x() -> AgentDef {
    from_preset(MI350X)
}

pub fn mi450x() -> AgentDef {
    from_preset(MI450X)
}

/// One builtin, parsed from the preset `build.rs` embedded.
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn every_builtin_matches_its_rocjitsu_config() {
        for (agent, source) in [
            (
                mi300x(),
                include_str!("../../../rocjitsu/configs/gfx942_cdna3.json"),
            ),
            (
                mi350x(),
                include_str!("../../../rocjitsu/configs/gfx950_mi355x.json"),
            ),
            (
                mi450x(),
                include_str!("../../../rocjitsu/configs/gfx1250_mi455x.json"),
            ),
        ] {
            let config: serde_json::Value = serde_json::from_str(source).unwrap();
            let expected: AgentDef = serde_json::from_value(serde_json::json!({
                "vm": config["vm"], "topology": config["topology"]
            }))
            .unwrap();
            assert_eq!(agent, expected);
            assert_eq!(
                serde_json::to_value(&agent).unwrap(),
                serde_json::json!({
                    "vm": config["vm"], "topology": config["topology"]
                })
            );
            assert!(agent.vm.gpu.device.num_sdma_queues_per_engine > 0);
            assert_eq!(
                serde_json::from_value::<AgentDef>(serde_json::to_value(&agent).unwrap()).unwrap(),
                agent
            );
        }
    }

    #[test]
    fn agents_have_expected_keys() {
        assert_eq!(
            agents()
                .into_iter()
                .map(|(name, _)| name)
                .collect::<Vec<_>>(),
            vec!["mi300x", "mi350x", "mi450x"]
        );
    }
}
