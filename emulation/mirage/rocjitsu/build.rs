//! Build script: propagate an rpath into this crate's test binaries
//! so they can find `librocjitsu.so` at runtime.
//!
//! `rocjitsu_sys` already emits `rustc-link-arg-tests=-Wl,-rpath,...`
//! but cargo only applies that directive to *its own* package's
//! tests. Downstream crates have to re-emit the directive
//! themselves; this file does that for `mirage_rocjitsu`.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=ROCJITSU_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ROCJITSU_ROOT");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(dir) = env::var_os("ROCJITSU_LIB_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    if let Some(root) = env::var_os("ROCJITSU_ROOT") {
        let root = PathBuf::from(root);
        candidates.push(root.join("build"));
        candidates.push(root.join("build").join("lib"));
        candidates.push(root.join("build-clean"));
        candidates.push(root.join("build-clean").join("lib"));
        candidates.push(root.join("artifacts").join("lib"));
    }
    // Sibling layout used inside the rocm-systems monorepo.
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sibling_root = here.join("..").join("..").join("rocjitsu");
    for sub in [
        "build",
        "build/lib",
        "build-clean",
        "build-clean/lib",
        "artifacts/lib",
    ] {
        candidates.push(sibling_root.join(sub));
    }

    for cand in candidates {
        if cand.join("librocjitsu.so").exists() {
            println!(
                "cargo:rustc-link-arg-tests=-Wl,-rpath,{}",
                cand.display()
            );
        }
    }
}
