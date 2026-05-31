//! Integration smoke test: build a [`RocjitsuEmulator`] against the
//! real FFI. Skipped when `librocjitsu.so` is not on the system.
//! When prerequisites *are* met, a VM construction failure is a
//! real test failure, not a silent skip.
//!
//! Note: rocjitsu maintains process-global state inside its FFI
//! and does not support concurrent VM construction from multiple
//! threads. We intentionally pack both assertions into a single
//! `#[test]` to make the (de-facto required) serial execution
//! obvious from the source rather than rely on `--test-threads=1`.

use mirage_core::emulator::Emulator;
use mirage_core::profile::ProfileDef;
use mirage_core::registry;
use mirage_rocjitsu::{RocjitsuEmulator, is_installed, kmd_config, kmd_preload};

#[test]
fn rocjitsu_emulator_boots_and_injects_kmd_env() {
    if !is_installed() {
        eprintln!("skipping: rocjitsu library not found");
        return;
    }
    if kmd_config("cdna4").is_none() {
        eprintln!("skipping: bundled cdna4 kmd config not present");
        return;
    }
    let profile = ProfileDef {
        name: "rj-smoke".to_string(),
        description: None,
        emulator: registry::make_def(&registry::ROCJITSU, 1, 1),
    };
    // Prerequisites met -> any failure here is a real bug, not a skip.
    let emu = RocjitsuEmulator::try_new(profile).expect(
        "VM construction must succeed when rocjitsu is installed and the cdna4 config is present",
    );
    assert_eq!(emu.def().emulator, "rocjitsu");
    assert!(emu.health().healthy);

    let inj = Emulator::injection_def(&emu);
    assert!(inj.env.contains_key("RJ_CONFIG"), "RJ_CONFIG must be set");
    assert!(inj.env.contains_key("RJ_SCHEMA"), "RJ_SCHEMA must be set");
    // If the kmd .so is on disk, it MUST be advertised. Asserting in
    // both branches ensures this test can't trivially pass on a box
    // that happens to lack the interposer build.
    match kmd_preload() {
        Some(path) => assert_eq!(
            inj.ld_preload.as_deref(),
            Some(path.to_string_lossy().as_ref())
        ),
        None => assert!(
            inj.ld_preload.is_none(),
            "ld_preload must be None when kmd is absent"
        ),
    }
}
