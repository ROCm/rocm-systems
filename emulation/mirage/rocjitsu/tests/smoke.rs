//! Integration smoke test: verifies that the embedded rocjitsu schema
//! and runtime-discovered KMD library are usable at runtime.
//!
//! The `kmd_config` round-trip is skipped when no KMD library is
//! discoverable on this machine (rocjitsu not installed).

use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDef, EmulatorKind, ExecMode};
use mirage_rocjitsu::{SCHEMA_FBS_BYTES, ensure_assets, kmd_config, kmd_preload, schema_fbs_path};

#[test]
fn embedded_assets_extract_round_trip() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let asset_report = ensure_assets(false).unwrap();
    let agent_report = mirage_builtin::ensure_agents(false).unwrap();

    // The schema is embedded, so it must always extract to disk.
    let schema_on_disk = schema_fbs_path();
    assert!(schema_on_disk.exists(), "schema should have been extracted");
    let bytes = std::fs::read(&schema_on_disk).unwrap();
    assert_eq!(bytes.len(), SCHEMA_FBS_BYTES.len());

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
        let (cfg, schema) = kmd_config(&def, None).expect("sim config should materialise");
        assert!(cfg.exists());
        assert!(schema.exists());
    }
    assert!(!asset_report.is_empty());
}
