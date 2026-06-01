//! Integration smoke test: verifies that the rocjitsu artifacts the
//! build script embedded are usable at runtime.
//!
//! Skipped when no rocjitsu source tree was visible at build time
//! (the embedded assets are then empty placeholders).

use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDef, ExecMode};
use mirage_rocjitsu::{
    KMD_LIB_BYTES, SCHEMA_FBS_BYTES, ensure_assets, kmd_config, kmd_lib_path, kmd_preload,
};

#[test]
fn embedded_assets_extract_round_trip() {
    if KMD_LIB_BYTES.is_empty() {
        eprintln!("skipping: no rocjitsu artifacts were embedded at build time");
        return;
    }

    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let asset_report = ensure_assets(false).unwrap();
    let agent_report = mirage_core::agent::store::ensure_builtins(false).unwrap();

    let on_disk = kmd_lib_path();
    assert!(on_disk.exists(), "kmd lib should have been extracted");
    let bytes = std::fs::read(&on_disk).unwrap();
    assert_eq!(bytes.len(), KMD_LIB_BYTES.len());
    assert_eq!(kmd_preload(), Some(on_disk));

    if !SCHEMA_FBS_BYTES.is_empty() {
        let agent_name = agent_report
            .iter()
            .map(|(n, _)| n.clone())
            .next()
            .expect("at least one builtin agent");
        let def = EmulatorDef {
            emulator: "rocjitsu".to_string(),
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
                racks: 1,
                nodes_per_rack: 1,
                gpus_per_node: 1,
                agent: MaybeRef::Ref(agent_name),
            }),
        };
        let (cfg, schema) = kmd_config(&def).expect("sim config should materialise");
        assert!(cfg.exists());
        assert!(schema.exists());
    }
    assert!(!asset_report.is_empty());
}
