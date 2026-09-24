//! Optional runtime-load test for `rocjitsu_sys`.
//!
//! Resolving the `rj_vm_*` symbols requires a real `librocjitsu.so`.
//! The test probes the conventional ROCm install locations
//! (`$ROCM_HOME/lib`, `/opt/rocm/lib`) for one; otherwise it skips so
//! the suite stays green on machines without rocjitsu.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use rocjitsu_sys::{Lib, version_string};

/// Whether a sanitizer build is present that this process cannot load.
///
/// Both cases below exist to `dlopen` the library, and against a
/// sanitizer build that cannot succeed: the runtime has to be in the
/// process's initial library list, and putting it there means preloading
/// it into `cargo test` and so into `rustc`. Left to find out the hard
/// way they fail on "cannot allocate memory in static TLS block", which
/// reads as a fault in the library rather than in how it was asked for.
fn skip_for_sanitizer_build() -> bool {
    match rj_core::discovery::sanitizer_preload_missing() {
        Some(why) => {
            eprintln!("skipping: {why}");
            true
        }
        None => false,
    }
}

/// Locate the rocjitsu library to load (the combined `librocjitsu.so`).
fn locate_lib() -> Option<PathBuf> {
    const LIBS: &[&str] = &["librocjitsu.so"];

    if let Some(path) = std::env::var_os("ROCJITSU_LIB").filter(|v| !v.is_empty()) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(root) = std::env::var_os("ROCM_HOME").filter(|v| !v.is_empty()) {
        dirs.push(PathBuf::from(root).join("lib"));
    }
    dirs.push(PathBuf::from("/opt/rocm/lib"));
    dirs.into_iter()
        .flat_map(|dir| LIBS.iter().map(move |lib| dir.join(lib)))
        .find(|p| p.is_file())
}

#[test]
fn loads_shared_version_string() {
    let Some(path) = locate_lib() else {
        eprintln!("no rocjitsu library found; skipping rocjitsu_sys version test");
        return;
    };
    if skip_for_sanitizer_build() {
        return;
    }
    let version = version_string(&path)
        .unwrap_or_else(|error| panic!("failed to read version from {path:?}: {error}"));
    let mut lines = version.lines();
    assert!(
        lines
            .next()
            .is_some_and(|line| line.starts_with("rocjitsu "))
    );
    assert!(
        lines
            .next()
            .is_some_and(|line| line.starts_with("git revision: "))
    );
    assert!(
        lines
            .next()
            .is_some_and(|line| line.starts_with("git commit: "))
    );
    assert_eq!(lines.next(), None, "{version}");
}

#[test]
fn loads_and_resolves_symbols() {
    let Some(path) = locate_lib() else {
        eprintln!("no rocjitsu library found; skipping rocjitsu_sys load test");
        return;
    };
    if skip_for_sanitizer_build() {
        return;
    }
    // Loading succeeds only if every `rj_vm_*` symbol resolves.
    let lib = unsafe { Lib::open(&path) };
    assert!(
        lib.is_ok(),
        "failed to load rocjitsu library at {path:?}: {:?}",
        lib.err()
    );
}
