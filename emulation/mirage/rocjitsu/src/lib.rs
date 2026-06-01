//! `mirage_rocjitsu` — rocjitsu integration for the mirage binary.
//!
//! This crate embeds the rocjitsu artifacts that the mirage binary
//! needs at runtime and exposes helpers to materialise them on disk:
//!
//! * [`KMD_LIB_BYTES`] — `librocjitsu_kmd.so`, the LD_PRELOAD KFD
//!   interposer that routes real HIP/HSA syscalls into the simulator.
//! * [`SCHEMA_FBS_BYTES`] — `simulation_config.fbs`, the flatbuffer
//!   schema the kmd config is validated against.
//!
//! See `build.rs` for how these assets are located/built at compile
//! time and the embedding fallback when the rocjitsu source tree is
//! unavailable.
//!
//! Runtime entry points:
//!
//! * [`ensure_assets`] extracts the kmd library + schema into
//!   `<MIRAGE_CACHE>/emulator/rocjitsu/`.
//! * [`kmd_config`] synthesises a runtime `SimulationConfig` JSON
//!   from an [`mirage_core::emulator::EmulatorDef`] by resolving its
//!   topology + agent references and wrapping them with rocjitsu's
//!   required runtime fields.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use mirage_core::agent::AgentDef;
use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDef, ExecMode};
use mirage_core::error::{MirageError, Result};
use mirage_core::topology::TopologyDef;

/// `librocjitsu_kmd.so` bytes. Empty when the build script could not
/// locate or build the artifact.
pub static KMD_LIB_BYTES: &[u8] = include_bytes!(env!("ROCJITSU_KMD_LIB_BYTES_PATH"));

/// `librocjitsu.so` bytes. Empty when the build script could not
/// locate or build the artifact.
pub static LIB_BYTES: &[u8] = include_bytes!(env!("ROCJITSU_LIB_BYTES_PATH"));

/// `simulation_config.fbs` schema bytes. Empty when not available at
/// build time.
pub static SCHEMA_FBS_BYTES: &[u8] = include_bytes!(env!("ROCJITSU_SCHEMA_FBS_PATH"));

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

/// Returns the path mirage should pass as `LD_PRELOAD` to an
/// rocjitsu-emulated workload.
///
/// Prefers the extracted on-disk copy under
/// `<MIRAGE_CACHE>/emulator/rocjitsu/` (after [`ensure_assets`] has
/// run); falls back to probing the rocjitsu source/build/install
/// layout for backwards compatibility with workspaces that don't
/// use the embedded copy.
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

/// Synthesise a rocjitsu `SimulationConfig` JSON file from the given
/// [`EmulatorDef`] and return `(config_path, schema_path)` ready to
/// be passed to the LD_PRELOAD'd workload as `RJ_CONFIG` /
/// `RJ_SCHEMA`.
///
/// The agent JSON under `<MIRAGE_CONFIG>/agent/` only stores the
/// `vm` + `topology` subset that mirage owns. rocjitsu's KMD shim
/// expects a full `SimulationConfig` (max_ticks, num_threads,
/// exec_mode, vm, topology). This function:
///
/// 1. Resolves `def.topology` (and its inner `agent`), following
///    [`MaybeRef`] references against the on-disk
///    `<MIRAGE_CONFIG>/{topology,agent}/` stores.
/// 2. Wraps the agent's `vm` + `topology` with rocjitsu runtime
///    fields (`exec_mode` is taken from `def.exec_mode`; the other
///    fields use sane defaults).
/// 3. Writes the result to
///    `<MIRAGE_CACHE>/emulator/rocjitsu/sim_<hash>.json`, keyed on
///    the JSON content so identical configs share a file and stale
///    files are never overwritten in-place.
pub fn kmd_config(def: &EmulatorDef) -> Result<(PathBuf, PathBuf)> {
    let topology: TopologyDef = match &def.topology {
        MaybeRef::Owned(t) => t.clone(),
        MaybeRef::Ref(name) => mirage_core::topology::store::get(name)?,
    };
    let agent: AgentDef = match &topology.agent {
        MaybeRef::Owned(a) => a.clone(),
        MaybeRef::Ref(name) => mirage_core::agent::store::get(name)?,
    };
    let exec_mode = match def.exec_mode {
        ExecMode::Functional => "functional",
        ExecMode::Clocked => "clocked",
    };
    let sim = serde_json::json!({
        "max_ticks": 100000u64,
        "num_threads": 1u32,
        "exec_mode": exec_mode,
        "vm": agent.vm,
        "topology": agent.topology,
    });
    let bytes = serde_json::to_vec_pretty(&sim).map_err(|e| {
        MirageError::Other(format!("rocjitsu kmd_config: serialize sim config: {e}"))
    })?;
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    let key = format!("{:016x}", hasher.finish());
    let cfg = asset_dir().join(format!("sim_{key}.json"));
    if !cfg.exists() {
        mirage_core::state::write_bytes(&cfg, &bytes)?;
    }
    let schema = schema_fbs_path();
    if !schema.exists() {
        return Err(MirageError::Other(format!(
            "rocjitsu schema not extracted: {} missing (run `mirage state builtins`)",
            schema.display()
        )));
    }
    Ok((cfg, schema))
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
}
