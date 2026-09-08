use mirage_core::agent::AgentDef;

use crate::presets::{MI300X, MI350X, MI450X};

pub fn agents() -> Vec<(&'static str, AgentDef)> {
    vec![
        ("mi300x", mi300x()),
        ("mi350x", mi350x()),
        ("mi450x", mi450x()),
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
