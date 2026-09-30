//! Optional runtime-load test for `rocjitsu_sys`.
//!
//! Resolving the `rj_vm_*` symbols requires a real `librocjitsu.so`.
//! The test probes the conventional ROCm install locations
//! (`$ROCM_HOME/lib`, `/opt/rocm/lib`) for one; otherwise it skips so
//! the suite stays green on machines without rocjitsu.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use rocjitsu_sys::{Lib, version_string};

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
    // Loading succeeds only if every `rj_vm_*` symbol resolves.
    let lib = unsafe { Lib::open(&path) };
    assert!(
        lib.is_ok(),
        "failed to load rocjitsu library at {path:?}: {:?}",
        lib.err()
    );
}
