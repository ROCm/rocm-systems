//! `mirage_rocjitsu` — a real, FFI-backed [`Emulator`] implementation
//! that drives a [`rocjitsu_sys::Vm`].
//!
//! This crate is intentionally thin: it owns a rocjitsu VM, forwards
//! lifecycle hooks (`new` / `shutdown`) to the underlying handle, and
//! reports installation status by probing the loader's search path
//! for `librocjitsu.so` (with env-var overrides). Topology is taken
//! from rocjitsu's bundled simulation configs (e.g. cdna4) rather
//! than reconstructed from [`EmulatorDef::topology`], because the
//! rocjitsu schema is far richer than the generic mirage topology.
//!
//! ## Configuration
//!
//! `EmulatorDef.options` keys understood by this implementation:
//!
//! | key       | type   | meaning                                                      |
//! |-----------|--------|--------------------------------------------------------------|
//! | `config`  | String | absolute path to a rocjitsu JSON simulation config           |
//! | `arch`    | String | shortcut: `cdna3` or `cdna4` resolves to the bundled config  |
//!
//! If neither is set, the implementation defaults to `cdna4`. When a
//! relative `config` is given it is resolved against
//! `$ROCJITSU_ROOT/configs`.

use std::path::{Path, PathBuf};

use mirage_core::common::{SimpleMap, SimpleValue};
use mirage_core::config::OptionDef;
use mirage_core::emulator::{Emulator, EmulatorDef, EmulatorDescription};
use mirage_core::exec::InjectionDef;
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::ProfileDef;
use mirage_core::session::SessionHealth;
use rocjitsu_sys::Vm;

/// Errors that can occur when constructing a [`RocjitsuEmulator`].
#[derive(Debug, thiserror::Error)]
pub enum RocjitsuError {
    /// The selected JSON config could not be located on disk.
    #[error("rocjitsu config not found: {0}")]
    ConfigMissing(PathBuf),
    /// `arch` option referenced an unsupported architecture.
    #[error("unknown rocjitsu arch: {0} (known: cdna3, cdna4)")]
    UnknownArch(String),
    /// The underlying FFI call failed.
    #[error("rocjitsu FFI error: {0}")]
    Ffi(#[from] rocjitsu_sys::Error),
}

/// A rocjitsu-backed emulator instance.
///
/// Holds the [`Vm`] handle for its lifetime; drop the value (or call
/// [`Emulator::shutdown`]) to release rocjitsu resources.
pub struct RocjitsuEmulator {
    def: EmulatorDef,
    vm: Vm,
}

impl RocjitsuEmulator {
    /// Build a new emulator without panicking on FFI failures. The
    /// [`Emulator::new`] trait method delegates to this and unwraps.
    pub fn try_new(profile: ProfileDef) -> Result<Self, RocjitsuError> {
        let def = profile.emulator.clone();
        let config = resolve_config(&def.options)?;
        let vm = Vm::create_with_default_schema(&config)?;
        Ok(Self { def, vm })
    }

    /// Returns the absolute path of the loaded rocjitsu JSON config.
    pub fn config_path(opts: &SimpleMap) -> Result<PathBuf, RocjitsuError> {
        resolve_config(opts)
    }

    /// Direct access to the underlying VM (for advanced callers).
    pub fn vm_mut(&mut self) -> &mut Vm {
        &mut self.vm
    }
}

impl Emulator for RocjitsuEmulator {
    fn description() -> EmulatorDescription {
        EmulatorDescription {
            name: "rocjitsu".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "ROCm GPU simulator: decodes AMDGPU/RISC-V ISA, event-driven PDES core."
                .to_string(),
        }
    }

    fn new(def: ProfileDef) -> Self {
        Self::try_new(def).expect("rocjitsu emulator construction failed")
    }

    fn options() -> OptionDef {
        OptionDef {
            name: "arch".to_string(),
            dtype: mirage_core::common::SimpleType::String,
            description: "GPU architecture preset: cdna3 or cdna4".to_string(),
            default_value: SimpleValue::String("cdna4".to_string()),
        }
    }

    fn shutdown(self) {
        // Drop -> drops `Vm` -> rocjitsu releases its handle.
    }

    fn validate_profile(def: &ProfileDef) -> Result<(), String> {
        if def.emulator.emulator != "rocjitsu" {
            return Err(format!(
                "profile.emulator.emulator is `{}`, expected `rocjitsu`",
                def.emulator.emulator
            ));
        }
        if def.emulator.nodes == 0 || def.emulator.gpus_per_node == 0 {
            return Err("rocjitsu requires nodes >= 1 and gpus_per_node >= 1".into());
        }
        resolve_config(&def.emulator.options).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn def(&self) -> &EmulatorDef {
        &self.def
    }

    fn installed() -> bool {
        installed_check()
    }

    fn discover_plugins() -> Vec<PluginsDef> {
        Vec::new()
    }

    fn health(&self) -> SessionHealth {
        SessionHealth {
            timestamp: chrono::Utc::now(),
            healthy: true,
            state: Some("ready".to_string()),
            terminal: false,
            message: None,
        }
    }

    fn injection_def(&self) -> InjectionDef {
        let mut env = std::collections::BTreeMap::new();
        let root = rocjitsu_root();
        if root.exists() {
            env.insert(
                "ROCJITSU_ROOT".to_string(),
                root.to_string_lossy().into_owned(),
            );
        }
        // Always advertise the JSON config + schema we actually
        // loaded; the kmd interposer reads these at process start.
        if let Ok(cfg) = resolve_config(&self.def.options) {
            env.insert("RJ_CONFIG".to_string(), cfg.to_string_lossy().into_owned());
        }
        let schema = std::path::PathBuf::from(rocjitsu_sys::SCHEMA_DIR)
            .join("simulation_config.fbs");
        if schema.exists() {
            env.insert(
                "RJ_SCHEMA".to_string(),
                schema.to_string_lossy().into_owned(),
            );
        }
        // Prefer an explicit override; otherwise probe known build /
        // install locations for the KMD interposer.
        let ld_preload = std::env::var("ROCJITSU_LD_PRELOAD")
            .ok()
            .or_else(|| discover_kmd_preload(&root).map(|p| p.to_string_lossy().into_owned()));
        InjectionDef {
            wrapper: None,
            ld_preload,
            files: Default::default(),
            env,
        }
    }
}

/// Discover `librocjitsu_kmd.so` (the KFD interposer) on disk. The
/// kmd library makes real HIP / torch / rocminfo / hsa apps talk to
/// the rocjitsu VM instead of `/dev/kfd`.
fn discover_kmd_preload(root: &Path) -> Option<std::path::PathBuf> {
    let lib_name = "librocjitsu_kmd.so";
    let candidates = [
        root.join("build/lib/rocjitsu/src/rocjitsu/kmd").join(lib_name),
        root.join("build-clean/lib/rocjitsu/src/rocjitsu/kmd").join(lib_name),
        root.join("build/lib").join(lib_name),
        root.join("artifacts/lib").join(lib_name),
        std::path::PathBuf::from("/usr/local/lib").join(lib_name),
        std::path::PathBuf::from("/usr/lib").join(lib_name),
        std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu").join(lib_name),
        std::path::PathBuf::from("/opt/rocm/lib").join(lib_name),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn resolve_config(opts: &SimpleMap) -> Result<PathBuf, RocjitsuError> {
    if let Some(SimpleValue::String(p)) = opts.get("config") {
        let path = PathBuf::from(p);
        let resolved = if path.is_absolute() {
            path
        } else {
            rocjitsu_root().join("configs").join(path)
        };
        if !resolved.exists() {
            return Err(RocjitsuError::ConfigMissing(resolved));
        }
        return Ok(resolved);
    }
    let arch = match opts.get("arch") {
        Some(SimpleValue::String(s)) => s.as_str(),
        _ => "cdna4",
    };
    // Use the _kmd config variants: they are the topologies
    // rocjitsu ships specifically for LD_PRELOAD'd ROCR apps (matches
    // rocjitsu/tests/CMakeLists.txt RJ_KMD_PRELOAD_ENV), and they
    // also work for the embedded Vm path.
    let filename = match arch {
        "cdna3" => "amdgpu_cdna3_kmd.json",
        "cdna4" => "amdgpu_cdna4_kmd.json",
        other => return Err(RocjitsuError::UnknownArch(other.to_string())),
    };
    let path = rocjitsu_root().join("configs").join(filename);
    if !path.exists() {
        return Err(RocjitsuError::ConfigMissing(path));
    }
    Ok(path)
}

fn rocjitsu_root() -> PathBuf {
    if let Some(root) = std::env::var_os("ROCJITSU_ROOT") {
        return PathBuf::from(root);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("rocjitsu")
}

fn installed_check() -> bool {
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
    false
}

/// Returns true iff `librocjitsu.so` is reachable on this machine.
/// Public so the integration test in `tests/` can share the probe.
pub fn is_installed() -> bool {
    installed_check()
}

/// Returns the discovered rocjitsu source/install root.
pub fn root() -> PathBuf {
    rocjitsu_root()
}

/// Returns the discovered path to `librocjitsu_kmd.so`, the LD_PRELOAD
/// interposer that routes real HIP/HSA syscalls into the rocjitsu
/// simulator. `None` if the kmd library has not been built/installed.
pub fn kmd_preload() -> Option<PathBuf> {
    discover_kmd_preload(&rocjitsu_root())
}

/// Returns the bundled simulation config + schema flatbuffer paths
/// for `arch` (`"cdna3"` or `"cdna4"`), if both files exist. These are
/// the values an LD_PRELOAD'd workload sets as `RJ_CONFIG` / `RJ_SCHEMA`.
pub fn kmd_config(arch: &str) -> Option<(PathBuf, PathBuf)> {
    let filename = match arch {
        "cdna3" => "amdgpu_cdna3_kmd.json",
        "cdna4" => "amdgpu_cdna4_kmd.json",
        _ => return None,
    };
    let root = rocjitsu_root();
    let cfg = root.join("configs").join(filename);
    let schema = root.join("schemas").join("simulation_config.fbs");
    if cfg.exists() && schema.exists() {
        Some((cfg, schema))
    } else {
        None
    }
}

/// The hipcc `--offload-arch` value that matches the `arch` preset
/// used by [`kmd_config`]. Mirrors the value rocjitsu's own
/// `tests/CMakeLists.txt` passes to hipcc.
pub fn hipcc_offload_arch(arch: &str) -> Option<&'static str> {
    match arch {
        "cdna3" => Some("gfx942"),
        "cdna4" => Some("gfx950"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_core::registry;

    fn rocjitsu_profile() -> ProfileDef {
        ProfileDef {
            name: "rj-test".to_string(),
            description: None,
            emulator: registry::make_def(&registry::ROCJITSU, 1, 1),
        }
    }

    #[test]
    fn description_is_stable() {
        let d = RocjitsuEmulator::description();
        assert_eq!(d.name, "rocjitsu");
        assert!(!d.version.is_empty());
    }

    #[test]
    fn validate_rejects_wrong_emulator_name() {
        let mut p = rocjitsu_profile();
        p.emulator.emulator = "noop".to_string();
        assert!(RocjitsuEmulator::validate_profile(&p).is_err());
    }

    #[test]
    fn validate_rejects_zero_nodes() {
        let mut p = rocjitsu_profile();
        p.emulator.nodes = 0;
        assert!(RocjitsuEmulator::validate_profile(&p).is_err());
    }

    #[test]
    fn validate_rejects_unknown_arch() {
        let mut p = rocjitsu_profile();
        p.emulator
            .options
            .insert("arch".to_string(), SimpleValue::String("rdna99".into()));
        assert!(RocjitsuEmulator::validate_profile(&p).is_err());
    }
}
