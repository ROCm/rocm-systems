//! Integration smoke test: verifies that a runtime-discovered KMD
//! library and a synthesised sim config are usable at runtime.
//!
//! The `kmd_config` round-trip is skipped when no KMD library is
//! discoverable on this machine (rocjitsu not installed).

use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDef, EmulatorKind, ExecMode};
use mirage_rocjitsu::{kmd_config, kmd_preload};

#[test]
fn sim_config_round_trip() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();

    // The full sim-config round-trip needs the KMD library, which is
    // discovered at runtime; skip when rocjitsu isn't installed.
    if kmd_preload().is_some() {
        let agent_name = agent_report
            .iter()
            .map(|(n, _)| n.clone())
            .next()
            .expect("at least one builtin agent");
        let def = EmulatorDef {
            emulator: EmulatorKind::Rocjitsu,
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
                num_nodes: 1,
                gpus_per_node: 1,
                agent: MaybeRef::Ref(agent_name),
            }),
        };
        let cfg = kmd_config(&def, None).expect("sim config should materialise");
        assert!(cfg.exists());
    }
}
