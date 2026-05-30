//! Integration smoke test: build a [`RocjitsuEmulator`] against the
//! real FFI. Skipped when `librocjitsu.so` is not on the system.
//! When prerequisites *are* met, a VM construction failure is a
//! real test failure, not a silent skip.

use mirage_core::emulator::Emulator;
use mirage_core::profile::ProfileDef;
use mirage_core::registry;
use mirage_rocjitsu::{RocjitsuEmulator, is_installed, kmd_preload, root};

#[test]
fn try_new_boots_a_vm_when_installed() {
    if !is_installed() {
        eprintln!("skipping: rocjitsu library not found");
        return;
    }
    let cfg = root().join("configs/amdgpu_cdna4.json");
    if !cfg.exists() {
        eprintln!("skipping: bundled cdna4 config not present at {}", cfg.display());
        return;
    }
    let profile = ProfileDef {
        name: "rj-smoke".to_string(),
        description: None,
        emulator: registry::make_def(&registry::ROCJITSU, 1, 1),
    };
    // Prerequisites met -> any failure here is a real bug, not a skip.
    let emu = RocjitsuEmulator::try_new(profile).expect("VM construction must succeed when rocjitsu is installed and the cdna4 config is present");
    assert_eq!(emu.def().emulator, "rocjitsu");
    assert!(emu.health().healthy);
}

#[test]
fn injection_def_advertises_kmd_when_present() {
    if !is_installed() || !root().join("configs/amdgpu_cdna4.json").exists() {
        eprintln!("skipping: rocjitsu prerequisites missing");
        return;
    }
    let profile = ProfileDef {
        name: "rj-inj".to_string(),
        description: None,
        emulator: registry::make_def(&registry::ROCJITSU, 1, 1),
    };
    let emu = RocjitsuEmulator::try_new(profile).expect("VM construction");
    let inj = mirage_core::emulator::Emulator::injection_def(&emu);
    assert!(inj.env.contains_key("RJ_CONFIG"), "RJ_CONFIG must be set");
    assert!(inj.env.contains_key("RJ_SCHEMA"), "RJ_SCHEMA must be set");
    // If the kmd .so is on disk, it MUST be advertised. Asserting in
    // both branches ensures this test can't trivially pass on a box
    // that happens to lack the interposer build.
    match kmd_preload() {
        Some(path) => assert_eq!(inj.ld_preload.as_deref(), Some(path.to_string_lossy().as_ref())),
        None => assert!(inj.ld_preload.is_none(), "ld_preload must be None when kmd is absent"),
    }
}
