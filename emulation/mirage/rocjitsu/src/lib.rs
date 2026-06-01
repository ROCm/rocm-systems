//! `mirage_rocjitsu` — rocjitsu integration for the mirage binary.
//!
//! This crate embeds the rocjitsu artifacts that the mirage binary
//! needs at runtime and exposes helpers to materialise them on disk:
//!
//! * [`KMD_LIB_BYTES`] — `librocjitsu_kmd.so`, the LD_PRELOAD KFD
//!   interposer that routes real HIP/HSA syscalls into the simulator.
//! * [`SCHEMA_FBS_BYTES`] — `simulation_config.fbs`, the flatbuffer
//!   schema the kmd config is validated against.
//! * [`CDNA3_KMD_BYTES`], [`CDNA4_KMD_BYTES`] — bundled simulation
//!   configs, exposed as default rocjitsu topologies.
//!
//! See `build.rs` for how these assets are located/built at compile
//! time and the embedding fallback when the rocjitsu source tree is
//! unavailable.
//!
//! Runtime entry points:
//!
//! * [`ensure_assets`] extracts the kmd library + schema into
//!   `<MIRAGE_STATE>/rocjitsu/` so they can be referenced via
//!   filesystem paths (e.g. as `LD_PRELOAD`).
//! * [`ensure_topologies`] writes the cdna3/cdna4 simulation configs
//!   into `<MIRAGE_CONFIG>/topology/` so they appear alongside the
//!   generic mirage builtin topologies.

use std::path::{Path, PathBuf};

use mirage_core::error::Result;

/// `librocjitsu_kmd.so` bytes. Empty when the build script could not
/// locate or build the artifact.
pub static KMD_LIB_BYTES: &[u8] = include_bytes!(env!("ROCJITSU_KMD_LIB_BYTES_PATH"));

/// `librocjitsu.so` bytes. Empty when the build script could not
/// locate or build the artifact.
pub static LIB_BYTES: &[u8] = include_bytes!(env!("ROCJITSU_LIB_BYTES_PATH"));

/// `simulation_config.fbs` schema bytes. Empty when not available at
/// build time.
pub static SCHEMA_FBS_BYTES: &[u8] = include_bytes!(env!("ROCJITSU_SCHEMA_FBS_PATH"));

/// Bundled `amdgpu_cdna3_kmd.json` config. Empty when not available
/// at build time.
pub static CDNA3_KMD_BYTES: &[u8] = include_bytes!(env!("ROCJITSU_CDNA3_KMD_PATH"));

/// Bundled `amdgpu_cdna4_kmd.json` config. Empty when not available
/// at build time.
pub static CDNA4_KMD_BYTES: &[u8] = include_bytes!(env!("ROCJITSU_CDNA4_KMD_PATH"));

/// Subdirectory under `<MIRAGE_CACHE>/emulator/` where the extracted
/// runtime assets (`librocjitsu_kmd.so`, `librocjitsu.so`,
/// `simulation_config.fbs`) live.
pub const ASSET_SUBDIR: &str = "rocjitsu";

/// Name used for the extracted KMD library on disk.
pub const KMD_LIB_NAME: &str = "librocjitsu_kmd.so";

/// Name used for the extracted host-side library on disk.
pub const LIB_NAME: &str = "librocjitsu.so";

/// Name used for the extracted schema on disk.
pub const SCHEMA_FBS_NAME: &str = "simulation_config.fbs";

/// Name (without `.json` suffix) of the cdna3 builtin topology.
pub const CDNA3_TOPOLOGY_NAME: &str = "rocjitsu-cdna3";

/// Name (without `.json` suffix) of the cdna4 builtin topology.
pub const CDNA4_TOPOLOGY_NAME: &str = "rocjitsu-cdna4";

/// Directory where extracted runtime assets are stored
/// (`<MIRAGE_CACHE>/emulator/rocjitsu/`).
pub fn asset_dir() -> PathBuf {
    mirage_core::paths::mirage_cache_dir()
        .join("emulator")
        .join(ASSET_SUBDIR)
}

/// On-disk path of the extracted KMD interposer library.
pub fn kmd_lib_path() -> PathBuf {
    asset_dir().join(KMD_LIB_NAME)
}

/// On-disk path of the extracted host-side rocjitsu library.
pub fn lib_path() -> PathBuf {
    asset_dir().join(LIB_NAME)
}

/// On-disk path of the extracted flatbuffer schema.
pub fn schema_fbs_path() -> PathBuf {
    asset_dir().join(SCHEMA_FBS_NAME)
}

/// Write the embedded rocjitsu libraries + schema into
/// `<MIRAGE_CACHE>/emulator/rocjitsu/`.
///
/// If `force` is true, existing files are overwritten. Otherwise
/// only missing files are written. Empty embedded assets (i.e. the
/// build script could not find the source artifact) are skipped.
///
/// Returns the list of `(name, written)` entries.
pub fn ensure_assets(force: bool) -> Result<Vec<(String, bool)>> {
    let mut report = Vec::new();
    for (name, bytes, path) in [
        (KMD_LIB_NAME, KMD_LIB_BYTES, kmd_lib_path()),
        (LIB_NAME, LIB_BYTES, lib_path()),
        (SCHEMA_FBS_NAME, SCHEMA_FBS_BYTES, schema_fbs_path()),
    ] {
        if bytes.is_empty() {
            report.push((name.to_string(), false));
            continue;
        }
        if path.exists() && !force {
            report.push((name.to_string(), false));
            continue;
        }
        mirage_core::state::write_bytes(&path, bytes)?;
        // Mark `.so` files executable; LD_PRELOAD doesn't require it
        // but it makes the file usable from a shell as well.
        if name.ends_with(".so") {
            let _ = make_executable(&path);
        }
        report.push((name.to_string(), true));
    }
    Ok(report)
}

/// Write the bundled rocjitsu kmd configs into the mirage topology
/// directory (`<MIRAGE_CONFIG>/topology/`).
///
/// If `force` is true, existing files are overwritten. Otherwise
/// only missing files are written. Empty embedded configs are skipped.
pub fn ensure_topologies(force: bool) -> Result<Vec<(String, bool)>> {
    let mut report = Vec::new();
    for (name, bytes) in builtin_topologies() {
        if bytes.is_empty() {
            report.push((name.to_string(), false));
            continue;
        }
        let path = mirage_core::paths::topology_path(name);
        if path.exists() && !force {
            report.push((name.to_string(), false));
            continue;
        }
        mirage_core::state::write_bytes(&path, bytes)?;
        report.push((name.to_string(), true));
    }
    Ok(report)
}

/// The `(name, bytes)` pairs that [`ensure_topologies`] writes.
pub fn builtin_topologies() -> &'static [(&'static str, &'static [u8])] {
    static ENTRIES: [(&str, &[u8]); 2] = [
        (CDNA3_TOPOLOGY_NAME, CDNA3_KMD_BYTES),
        (CDNA4_TOPOLOGY_NAME, CDNA4_KMD_BYTES),
    ];
    &ENTRIES
}

/// Returns the path mirage should pass as `LD_PRELOAD` to an
/// rocjitsu-emulated workload.
///
/// Prefers the extracted on-disk copy under `<MIRAGE_STATE>/rocjitsu/`
/// (after [`ensure_assets`] has run); falls back to probing the
/// rocjitsu source/build/install layout for backwards compatibility
/// with workspaces that don't use the embedded copy.
pub fn kmd_preload() -> Option<PathBuf> {
    let extracted = kmd_lib_path();
    if extracted.exists() {
        return Some(extracted);
    }
    let root = root();
    let candidates = [
        root.join("build/lib/rocjitsu/src/rocjitsu/kmd")
            .join(KMD_LIB_NAME),
        root.join("build-clean/lib/rocjitsu/src/rocjitsu/kmd")
            .join(KMD_LIB_NAME),
        root.join("build/lib").join(KMD_LIB_NAME),
        root.join("artifacts/lib").join(KMD_LIB_NAME),
        PathBuf::from("/usr/local/lib").join(KMD_LIB_NAME),
        PathBuf::from("/usr/lib").join(KMD_LIB_NAME),
        PathBuf::from("/usr/lib/x86_64-linux-gnu").join(KMD_LIB_NAME),
        PathBuf::from("/opt/rocm/lib").join(KMD_LIB_NAME),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Returns the `(config, schema)` pair the LD_PRELOAD'd workload
/// should advertise as `RJ_CONFIG` / `RJ_SCHEMA` for `arch` (`"cdna3"`
/// or `"cdna4"`).
///
/// Prefers the extracted on-disk copies (kmd json under
/// `<MIRAGE_CONFIG>/topology/` and schema under
/// `<MIRAGE_STATE>/rocjitsu/`). Falls back to the rocjitsu source
/// tree for environments that haven't extracted the embedded assets.
pub fn kmd_config(arch: &str) -> Option<(PathBuf, PathBuf)> {
    let (topology_name, fallback_cfg_name) = match arch {
        "cdna3" => (CDNA3_TOPOLOGY_NAME, "amdgpu_cdna3_kmd.json"),
        "cdna4" => (CDNA4_TOPOLOGY_NAME, "amdgpu_cdna4_kmd.json"),
        _ => return None,
    };
    let extracted_cfg = mirage_core::paths::topology_path(topology_name);
    let extracted_schema = schema_fbs_path();
    if extracted_cfg.exists() && extracted_schema.exists() {
        return Some((extracted_cfg, extracted_schema));
    }
    let root = root();
    let cfg = root.join("configs").join(fallback_cfg_name);
    let schema = root.join("schemas").join(SCHEMA_FBS_NAME);
    if cfg.exists() && schema.exists() {
        Some((cfg, schema))
    } else {
        None
    }
}

/// The hipcc `--offload-arch` value that matches the `arch` preset
/// used by [`kmd_config`].
pub fn hipcc_offload_arch(arch: &str) -> Option<&'static str> {
    match arch {
        "cdna3" => Some("gfx942"),
        "cdna4" => Some("gfx950"),
        _ => None,
    }
}

/// Best-effort discovery of the rocjitsu source/install root. Used
/// only as a fallback when the embedded assets are not yet extracted.
pub fn root() -> PathBuf {
    if let Some(root) = std::env::var_os("ROCJITSU_ROOT") {
        return PathBuf::from(root);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("rocjitsu")
}

/// Returns true if rocjitsu is reachable in any form on this machine
/// — either the embedded KMD library was non-empty at build time, or
/// a system install / sibling build has been detected.
pub fn is_installed() -> bool {
    if !KMD_LIB_BYTES.is_empty() {
        return true;
    }
    if std::env::var_os("ROCJITSU_LIB_DIR").is_some()
        || std::env::var_os("ROCJITSU_ROOT").is_some()
    {
        return true;
    }
    for candidate in [
        "/usr/local/lib/librocjitsu.so",
        "/usr/lib/librocjitsu.so",
        "/usr/lib/x86_64-linux-gnu/librocjitsu.so",
        "/opt/rocm/lib/librocjitsu.so",
    ] {
        if Path::new(candidate).exists() {
            return true;
        }
    }
    kmd_preload().is_some()
}

#[cfg(unix)]
fn make_executable(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perm = std::fs::metadata(path)?.permissions();
    perm.set_mode(perm.mode() | 0o111);
    std::fs::set_permissions(path, perm)
}
#[cfg(not(unix))]
fn make_executable(_: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_names_are_stable() {
        let names: Vec<&str> = builtin_topologies().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["rocjitsu-cdna3", "rocjitsu-cdna4"]);
    }

    #[test]
    fn hipcc_offload_arch_known() {
        assert_eq!(hipcc_offload_arch("cdna3"), Some("gfx942"));
        assert_eq!(hipcc_offload_arch("cdna4"), Some("gfx950"));
        assert_eq!(hipcc_offload_arch("rdna99"), None);
    }

    #[test]
    fn ensure_assets_writes_or_skips() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());
        let report = ensure_assets(false).unwrap();
        assert_eq!(report.len(), 3);
        for (name, written) in &report {
            let (path, bytes_empty) = match name.as_str() {
                KMD_LIB_NAME => (kmd_lib_path(), KMD_LIB_BYTES.is_empty()),
                LIB_NAME => (lib_path(), LIB_BYTES.is_empty()),
                SCHEMA_FBS_NAME => (schema_fbs_path(), SCHEMA_FBS_BYTES.is_empty()),
                other => panic!("unexpected asset {other}"),
            };
            if bytes_empty {
                assert!(!written, "empty asset {name} should not be written");
                assert!(!path.exists());
            } else {
                assert!(written, "{name} should have been written on first run");
                assert!(path.exists());
            }
        }
    }

    #[test]
    fn ensure_topologies_writes_then_skips() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());
        let first = ensure_topologies(false).unwrap();
        for (name, written) in &first {
            let bytes_empty = match name.as_str() {
                CDNA3_TOPOLOGY_NAME => CDNA3_KMD_BYTES.is_empty(),
                CDNA4_TOPOLOGY_NAME => CDNA4_KMD_BYTES.is_empty(),
                _ => unreachable!(),
            };
            if bytes_empty {
                assert!(!written);
            } else {
                assert!(written);
            }
        }
        let second = ensure_topologies(false).unwrap();
        for (_, w) in &second {
            assert!(!w, "second run should not overwrite existing topologies");
        }
    }
}
