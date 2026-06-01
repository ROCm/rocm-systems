//! Integration smoke test: verifies that the rocjitsu artifacts the
//! build script embedded are usable at runtime.
//!
//! Skipped when no rocjitsu source tree was visible at build time
//! (the embedded assets are then empty placeholders).

use mirage_rocjitsu::{
    CDNA3_KMD_BYTES, CDNA4_KMD_BYTES, KMD_LIB_BYTES, SCHEMA_FBS_BYTES, ensure_agents,
    ensure_assets, kmd_config, kmd_lib_path, kmd_preload,
};

#[test]
fn embedded_assets_extract_round_trip() {
    if KMD_LIB_BYTES.is_empty() && CDNA4_KMD_BYTES.is_empty() {
        eprintln!("skipping: no rocjitsu artifacts were embedded at build time");
        return;
    }

    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let asset_report = ensure_assets(false).unwrap();
    let agent_report = ensure_agents(false).unwrap();

    if !KMD_LIB_BYTES.is_empty() {
        let on_disk = kmd_lib_path();
        assert!(on_disk.exists(), "kmd lib should have been extracted");
        let bytes = std::fs::read(&on_disk).unwrap();
        assert_eq!(bytes.len(), KMD_LIB_BYTES.len());
        assert_eq!(kmd_preload(), Some(on_disk));
    }
    if !SCHEMA_FBS_BYTES.is_empty() && !CDNA4_KMD_BYTES.is_empty() {
        let (cfg, schema) = kmd_config("cdna4").expect("cdna4 config + schema should resolve");
        assert!(cfg.exists());
        assert!(schema.exists());
    }
    if !CDNA3_KMD_BYTES.is_empty() {
        assert!(agent_report.iter().any(|(n, w)| n == "cdna3" && *w));
    }
    assert!(!asset_report.is_empty());
}
