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
//! 3. For the `.so` libraries: a pre-built artifact under `<root>/build/`,
//!    with a fallback to invoking `cmake` to build them on demand
//!    (only when `cmake` is on `$PATH` and `MIRAGE_ROCJITSU_BUILD!=0`).
//!
//! If an asset cannot be located, an empty placeholder is staged in
//! `$OUT_DIR` so the crate still compiles. Runtime helpers
//! (`mirage_rocjitsu::ensure_assets`) skip writing empty assets.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=ROCJITSU_ROOT");
    println!("cargo:rerun-if-env-changed=ROCJITSU_KMD_LIB");
    println!("cargo:rerun-if-env-changed=ROCJITSU_LIB");
    println!("cargo:rerun-if-env-changed=ROCJITSU_SCHEMA_FBS");
    println!("cargo:rerun-if-env-changed=MIRAGE_ROCJITSU_BUILD");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let root = rocjitsu_root();

    stage_asset(
        "ROCJITSU_KMD_LIB_BYTES_PATH",
        &out_dir.join("librocjitsu_kmd.so"),
        find_shared_lib(
            &root,
            &out_dir,
            "ROCJITSU_KMD_LIB",
            "librocjitsu_kmd.so",
            "rocjitsu_kmd_shim",
            &[
                "build/lib/rocjitsu/src/rocjitsu/kmd/librocjitsu_kmd.so",
                "build-clean/lib/rocjitsu/src/rocjitsu/kmd/librocjitsu_kmd.so",
                "build/lib/librocjitsu_kmd.so",
                "artifacts/lib/librocjitsu_kmd.so",
            ],
            "lib/rocjitsu/src/rocjitsu/kmd/librocjitsu_kmd.so",
        ),
    );
    stage_asset(
        "ROCJITSU_LIB_BYTES_PATH",
        &out_dir.join("librocjitsu.so"),
        find_shared_lib(
            &root,
            &out_dir,
            "ROCJITSU_LIB",
            "librocjitsu.so",
            "rocjitsu_shared",
            &[
                "build/librocjitsu.so",
                "build-clean/librocjitsu.so",
                "build/lib/librocjitsu.so",
                "artifacts/lib/librocjitsu.so",
            ],
            "librocjitsu.so",
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

fn find_shared_lib(
    root: &Option<PathBuf>,
    out_dir: &Path,
    env_key: &str,
    label: &str,
    cmake_target: &str,
    candidates: &[&str],
    built_rel: &str,
) -> Option<PathBuf> {
    if let Some(p) = env::var_os(env_key) {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let root = root.as_deref()?;
    for cand in candidates {
        let p = root.join(cand);
        if p.exists() {
            return Some(p);
        }
    }
    if env::var_os("MIRAGE_ROCJITSU_BUILD") == Some("0".into()) {
        return None;
    }
    if !has_cmd("cmake") {
        println!(
            "cargo:warning=mirage_rocjitsu: {label} not found and cmake unavailable; skipping build"
        );
        return None;
    }
    build_target(root, out_dir, cmake_target, built_rel, label)
}

fn build_target(
    root: &Path,
    out_dir: &Path,
    cmake_target: &str,
    built_rel: &str,
    label: &str,
) -> Option<PathBuf> {
    let build_dir = out_dir.join("rocjitsu-build");
    fs::create_dir_all(&build_dir).ok()?;

    println!(
        "cargo:warning=mirage_rocjitsu: building {label} via cmake (this may take a while)"
    );
    let generator = if has_cmd("ninja") {
        "Ninja"
    } else {
        "Unix Makefiles"
    };
    let configure = Command::new("cmake")
        .arg("-S")
        .arg(root)
        .arg("-B")
        .arg(&build_dir)
        .arg("-G")
        .arg(generator)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .status();
    match configure {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=mirage_rocjitsu: cmake configure failed ({s})");
            return None;
        }
        Err(e) => {
            println!("cargo:warning=mirage_rocjitsu: cmake configure failed: {e}");
            return None;
        }
    }
    let build = Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg(cmake_target)
        .status();
    match build {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=mirage_rocjitsu: cmake build failed ({s})");
            return None;
        }
        Err(e) => {
            println!("cargo:warning=mirage_rocjitsu: cmake build failed: {e}");
            return None;
        }
    }
    let built = build_dir.join(built_rel);
    if built.exists() { Some(built) } else { None }
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

fn has_cmd(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
