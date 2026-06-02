//! Build script for `mirage_rocjitsu`.
//!
//! Stages the rocjitsu assets that the mirage binary embeds via
//! `include_bytes!`:
//!
//! * `librocjitsu_kmd.so` — the LD_PRELOAD KFD interposer.
//! * `librocjitsu.so` — the rocjitsu host-side runtime.
//! * `simulation_config.fbs` — the flatbuffer schema for the kmd config.
//! * `amdgpu_cdna3_kmd.json`, `amdgpu_cdna4_kmd.json` — bundled
//!   simulation configs used as default rocjitsu topologies.
//!
//! Discovery order for each asset:
//!
//! 1. An explicit absolute path in the corresponding `ROCJITSU_*`
//!    env var (`ROCJITSU_KMD_LIB`, `ROCJITSU_LIB`,
//!    `ROCJITSU_SCHEMA_FBS`, `ROCJITSU_CDNA3_KMD`, `ROCJITSU_CDNA4_KMD`).
//! 2. The rocjitsu source tree at `$ROCJITSU_ROOT` or the sibling
//!    `../../rocjitsu` checkout. JSON configs are read from
//!    `<root>/configs/` and the schema from `<root>/schemas/`.
//! 3. For the `.so` libraries: a pre-built artifact under `<root>/build/`.
//!
//! mirage does **not** build emulators itself. If a `.so` is not
//! found as a prebuilt artifact, an empty placeholder is embedded and
//! the library is discovered at runtime from the installed system
//! (see `mirage_rocjitsu::kmd_preload` and `mirage_core::discovery`).
//!
//! If an asset cannot be located, an empty placeholder is staged in
//! `$OUT_DIR` so the crate still compiles. Runtime helpers
//! (`mirage_rocjitsu::ensure_assets`) skip writing empty assets.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=ROCJITSU_ROOT");
    println!("cargo:rerun-if-env-changed=ROCJITSU_KMD_LIB");
    println!("cargo:rerun-if-env-changed=ROCJITSU_LIB");
    println!("cargo:rerun-if-env-changed=ROCJITSU_SCHEMA_FBS");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let root = rocjitsu_root();

    stage_asset(
        "ROCJITSU_KMD_LIB_BYTES_PATH",
        &out_dir.join("librocjitsu_kmd.so"),
        find_shared_lib(
            &root,
            "ROCJITSU_KMD_LIB",
            &[
                "build/lib/rocjitsu/src/rocjitsu/kmd/librocjitsu_kmd.so",
                "build-clean/lib/rocjitsu/src/rocjitsu/kmd/librocjitsu_kmd.so",
                "build/lib/librocjitsu_kmd.so",
                "artifacts/lib/librocjitsu_kmd.so",
            ],
        ),
    );
    stage_asset(
        "ROCJITSU_LIB_BYTES_PATH",
        &out_dir.join("librocjitsu.so"),
        find_shared_lib(
            &root,
            "ROCJITSU_LIB",
            &[
                "build/librocjitsu.so",
                "build-clean/librocjitsu.so",
                "build/lib/librocjitsu.so",
                "artifacts/lib/librocjitsu.so",
            ],
        ),
    );
    stage_asset(
        "ROCJITSU_SCHEMA_FBS_PATH",
        &out_dir.join("simulation_config.fbs"),
        find_in_root(
            "ROCJITSU_SCHEMA_FBS",
            root.as_deref(),
            "schemas/simulation_config.fbs",
        ),
    );
}

/// Stage `src` (if any) at `dst`, emit a `cargo:rustc-env=KEY=<dst>`
/// directive, and a `cargo:rerun-if-changed=<src>` watch. If `src` is
/// `None`, write an empty placeholder so `include_bytes!` still
/// resolves. Runtime helpers check for the empty case and skip.
fn stage_asset(env_key: &str, dst: &Path, src: Option<PathBuf>) {
    match src {
        Some(src) => {
            println!("cargo:rerun-if-changed={}", src.display());
            if let Err(e) = fs::copy(&src, dst) {
                println!(
                    "cargo:warning=mirage_rocjitsu: failed to copy {} -> {}: {e}",
                    src.display(),
                    dst.display()
                );
                let _ = fs::write(dst, b"");
            }
        }
        None => {
            println!(
                "cargo:warning=mirage_rocjitsu: asset for {env_key} not found; embedding empty placeholder"
            );
            let _ = fs::write(dst, b"");
        }
    }
    println!("cargo:rustc-env={}={}", env_key, dst.display());
}

fn find_in_root(env_key: &str, root: Option<&Path>, rel: &str) -> Option<PathBuf> {
    if let Some(p) = env::var_os(env_key) {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let p = root?.join(rel);
    if p.exists() { Some(p) } else { None }
}

fn find_shared_lib(root: &Option<PathBuf>, env_key: &str, candidates: &[&str]) -> Option<PathBuf> {
    // 1. Explicit path supplied by the user.
    if let Some(p) = env::var_os(env_key) {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // 2. A prebuilt artifact inside the rocjitsu source tree. mirage
    //    does NOT build emulators itself — if a prebuilt copy isn't
    //    present, an empty placeholder is embedded and the library is
    //    discovered at runtime (see `mirage_rocjitsu::kmd_preload`).
    let root = root.as_deref()?;
    for cand in candidates {
        let p = root.join(cand);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn rocjitsu_root() -> Option<PathBuf> {
    if let Some(p) = env::var_os("ROCJITSU_ROOT") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sibling = here.join("..").join("..").join("rocjitsu");
    if sibling.exists() {
        return Some(sibling);
    }
    None
}
