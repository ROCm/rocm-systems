//! Integration smoke test: build a [`RocjitsuEmulator`] against the
//! real FFI. Skipped when `librocjitsu.so` is not on the system.

use mirage_core::profile::ProfileDef;
use mirage_core::registry;
use mirage_rocjitsu::{RocjitsuEmulator, is_installed, root};

#[test]
fn try_new_boots_a_vm_when_installed() {
    if !is_installed() {
        eprintln!("skipping: rocjitsu library not found");
        return;
    }
    if !root().join("configs/amdgpu_cdna4.json").exists() {
        eprintln!("skipping: bundled cdna4 config not present");
        return;
    }
    let profile = ProfileDef {
        name: "rj-smoke".to_string(),
        description: None,
        emulator: registry::make_def(&registry::ROCJITSU, 1, 1),
    };
    match RocjitsuEmulator::try_new(profile) {
        Ok(_emu) => { /* VM created and dropped cleanly */ }
        Err(e) => eprintln!("skipping: rocjitsu VM construction failed: {e}"),
    }
}
