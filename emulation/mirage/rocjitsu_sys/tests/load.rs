//! Optional runtime-load test for `rocjitsu_sys`.
//!
//! Resolving the `rj_vm_*` symbols requires a real `librocjitsu_kmd.so`.
//! Point `ROCJITSU_KMD_LIB` at one to exercise this; otherwise the test
//! skips so the suite stays green on machines without rocjitsu.

use rocjitsu_sys::Lib;

#[test]
fn loads_and_resolves_symbols() {
    let Some(path) = std::env::var_os("ROCJITSU_KMD_LIB") else {
        eprintln!("ROCJITSU_KMD_LIB unset; skipping rocjitsu_sys load test");
        return;
    };
    // Loading succeeds only if every `rj_vm_*` symbol resolves.
    let lib = unsafe { Lib::open(&path) };
    assert!(
        lib.is_ok(),
        "failed to load rocjitsu library at {path:?}: {:?}",
        lib.err()
    );
}
