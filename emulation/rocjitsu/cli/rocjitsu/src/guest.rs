//! DBT guest mode: run an application built for one GPU against a
//! synthetic guest agent while its kernels execute somewhere else.
//!
//! This is a mode of the [`crate::Rocjitsu`] backend rather than a
//! backend of its own, because it is a property of the simulation config
//! the user supplied: a config carrying an enabled `dbt_guest` block
//! selects it, and `rocjitsu run --config guest.json -- ./app` is the whole
//! interface.
//!
//! # How it differs from the `rocjitsu-dbt` backend
//!
//! [`crate::dbt`] translates code objects and runs them on a real GPU the
//! application already sees. Guest mode adds a GPU the machine does not
//! have. The KMD interposer appends one synthetic node to the KFD
//! topology, so an unmodified application discovers, say, a gfx950 and
//! builds for it; the HSA hook then translates each of its code objects
//! and forwards execution to a *host* GPU underneath, which may be real
//! hardware or a RocJITsu-simulated GPU of a different architecture
//! entirely. Both halves are required — the hook alone would have no
//! guest agent to shadow, and the interposer alone would advertise a GPU
//! that cannot run anything.
//!
//! # Why the launcher resolves the host GPU
//!
//! `host_gpu_id: 0` means "the first visible GPU matching `host_isa`".
//! Three runtime layers need that answer and each of them could compute
//! it — but the interposer and the hook run inside the application, after
//! ROCm visibility settings have already reshaped what they can see, and
//! nothing guarantees they would agree with each other. So the launcher
//! resolves it once, before any of them start, and publishes the result
//! in the runtime config handoff.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rj_core::error::{Result, RocJITsuError};
use rj_core::hardware::{self, gfx_name};
use rj_core::visibility::{
    self, HostSelection, VisibleGpu, effective_visible_gpus, expanded_rocr_visible_devices,
    normalized_client_visible_devices, select_host_gpu,
};

use crate::dbt::{EnvLookup, HOOKS_LIB_NAME};

/// Where the guest's kernels actually execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    /// A real GPU in this machine's KFD topology.
    Hardware,
    /// A RocJITsu-simulated GPU described by another config file, so the
    /// whole thing runs with no AMD hardware present at all. This is the
    /// mode the CI DBT guest suites use.
    Simulator,
}

/// The launcher-relevant subset of a config's `dbt_guest` block.
///
/// Deliberately not the whole block: `guest_device`, `log_level`,
/// `signal_backtrace` and the silicon revisions are read from the same
/// file by the interposer and the hook, which have the schema. Parsing
/// them here would be a second reader to keep in step for no gain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestConfig {
    /// ISA the synthetic guest agent advertises.
    pub guest_isa: String,
    /// ISA the guest's code objects are translated to, and which the
    /// host GPU must report.
    pub host_isa: String,
    /// Preferred host KFD `gpu_id`; zero selects the first ISA match.
    pub host_gpu_id: u32,
    /// Hardware or simulated execution.
    pub backend: ExecutionBackend,
    /// External config describing the simulated host, resolved beside
    /// the guest config when relative. Empty selects the guest config
    /// itself.
    pub simulator_config: String,
}

/// Everything guest mode contributes to a workload's launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestLaunch {
    /// The host GPU the launcher resolved, published in the handoff so
    /// every runtime layer agrees on it.
    pub host_gpu_id: u32,
    /// Environment additions: the rewritten visibility selectors and the
    /// HSA tools hook wiring.
    pub env: BTreeMap<String, String>,
    /// Whether the workload needs the machine's real GPUs.
    pub host_gpus: bool,
}

/// Locate the HSA tools hook library guest mode translates through.
///
/// # Errors
///
/// Returns an error naming where the search looked when the library is
/// not installed.
pub fn hooks_lib() -> Result<PathBuf> {
    crate::dbt::hooks_preload().ok_or_else(|| {
        let detail = crate::dbt::runtime_location()
            .explain_missing()
            .unwrap_or_else(|| format!("{HOOKS_LIB_NAME} was not found"));
        RocJITsuError::Other(format!(
            "rocjitsu: dbt_guest needs the HSA tools hook library ({HOOKS_LIB_NAME}) to \
             translate guest code objects, and it was not found — {detail}"
        ))
    })
}

/// Read the `dbt_guest` block out of a rocjitsu simulation config.
///
/// Returns `None` for any config without an enabled `dbt_guest`, which
/// is every ordinary emulation config, so callers can treat this as the
/// mode switch it is.
///
/// # Errors
///
/// Returns an error when the file cannot be read or parsed, or when an
/// enabled block omits an ISA.
pub fn load(config: &Path) -> Result<Option<GuestConfig>> {
    let text = std::fs::read_to_string(config).map_err(|e| {
        RocJITsuError::Other(format!(
            "rocjitsu: cannot read config {}: {e}",
            config.display()
        ))
    })?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        RocJITsuError::Other(format!(
            "rocjitsu: cannot parse config {}: {e}",
            config.display()
        ))
    })?;
    let Some(guest) = json.get("dbt_guest") else {
        return Ok(None);
    };
    if !guest
        .get("enabled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Ok(None);
    }

    let string = |key: &str| -> String {
        guest
            .get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let guest_isa = string("guest_isa");
    let host_isa = string("host_isa");
    if guest_isa.is_empty() || host_isa.is_empty() {
        return Err(RocJITsuError::Other(
            "rocjitsu: dbt_guest requires guest_isa and host_isa".to_string(),
        ));
    }
    let backend = match guest
        .get("execution_backend")
        .and_then(serde_json::Value::as_str)
    {
        None | Some("hardware") => ExecutionBackend::Hardware,
        Some("simulator") => ExecutionBackend::Simulator,
        Some(other) => {
            return Err(RocJITsuError::Other(format!(
                "rocjitsu: dbt_guest.execution_backend '{other}' is invalid; \
                 expected 'hardware' or 'simulator'"
            )));
        }
    };

    Ok(Some(GuestConfig {
        guest_isa,
        host_isa,
        host_gpu_id: guest
            .get("host_gpu_id")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
        backend,
        simulator_config: string("simulator_config"),
    }))
}

/// Resolve the config describing the simulated host.
///
/// An empty `simulator_config` selects the guest config itself, so a
/// single file can carry both the guest block and the host VM. A
/// relative path is resolved beside the guest config rather than against
/// the working directory, so a config remains movable as a unit.
fn simulator_config_path(config: &Path, simulator_config: &str) -> PathBuf {
    if simulator_config.is_empty() {
        return config.to_path_buf();
    }
    let referenced = Path::new(simulator_config);
    if referenced.is_absolute() {
        return referenced.to_path_buf();
    }
    config.parent().unwrap_or(Path::new(".")).join(referenced)
}

/// The GPUs the guest's kernels can execute on, in the order the ROCm
/// runtimes enumerate them.
///
/// For the hardware backend this is the machine's KFD topology. For the
/// simulator backend it is the device list the referenced config
/// describes, expanded exactly as RocJITsu's own config loader expands
/// `num_gpus`: each additional GPU takes the template's identity with its
/// `gpu_id` and `unique_id` stepped by one.
///
/// # Errors
///
/// Returns an error when the simulator config cannot be read or parsed.
pub fn execution_gpus(config: &Path, guest: &GuestConfig) -> Result<Vec<VisibleGpu>> {
    if guest.backend == ExecutionBackend::Hardware {
        return Ok(hardware::kfd_gpus());
    }

    let host_config = simulator_config_path(config, &guest.simulator_config);
    let text = std::fs::read_to_string(&host_config).map_err(|e| {
        RocJITsuError::Other(format!(
            "rocjitsu: cannot read dbt_guest.simulator_config {}: {e}",
            host_config.display()
        ))
    })?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        RocJITsuError::Other(format!(
            "rocjitsu: cannot parse dbt_guest.simulator_config {}: {e}",
            host_config.display()
        ))
    })?;

    let Some(gpu) = json.get("vm").and_then(|vm| vm.get("gpu")) else {
        return Ok(Vec::new());
    };
    let Some(device) = gpu.get("device") else {
        return Ok(Vec::new());
    };
    let field = |key: &str| device.get(key).and_then(serde_json::Value::as_u64);
    let base = VisibleGpu {
        ordinal: 0,
        gpu_id: field("gpu_id")
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
        gfx_target_version: field("gfx_target_version")
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
        unique_id: field("unique_id").unwrap_or(0),
    };
    let num_gpus = gpu
        .get("num_gpus")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1)
        .max(1);
    let devices: Vec<VisibleGpu> = (0..num_gpus)
        .map(|i| VisibleGpu {
            gpu_id: base.gpu_id.saturating_add(u32::try_from(i).unwrap_or(0)),
            unique_id: base.unique_id.wrapping_add(i),
            ..base
        })
        .collect();
    Ok(visibility::enumerate_kfd_gpus(&devices))
}

/// Build everything guest mode contributes to a workload's launch.
///
/// `env` supplies the ROCm visibility settings in effect. They are read
/// rather than assumed because the rewritten selectors have to be
/// expressed in terms of what the application would otherwise have seen:
/// appending the guest ordinal to a selection the user narrowed means
/// nothing unless the narrowing is known.
///
/// `hooks_in_workload` is where the workload will find the HSA tools
/// hook — the host path for a plain run, the in-container path for a
/// containerised one — which is why it is passed in rather than
/// discovered here. See [`hooks_lib`].
///
/// # Errors
///
/// Returns an error when the host ISA is unrecognized or no visible GPU
/// can serve as the host.
pub fn plan(
    config: &Path,
    guest: &GuestConfig,
    hooks_in_workload: &str,
    env: &impl EnvLookup,
) -> Result<GuestLaunch> {
    let topology = execution_gpus(config, guest)?;
    let host_version =
        hardware::gfx_target_version_from_name(&guest.host_isa).ok_or_else(|| {
            RocJITsuError::Other(format!(
                "rocjitsu: unrecognized dbt_guest.host_isa '{}'",
                guest.host_isa
            ))
        })?;

    let rocr = env.get("ROCR_VISIBLE_DEVICES");
    let hip = env.get("HIP_VISIBLE_DEVICES");
    let cuda = env.get("CUDA_VISIBLE_DEVICES");
    let visible =
        effective_visible_gpus(&topology, rocr.as_deref(), hip.as_deref(), cuda.as_deref());
    let host_gpu_id = match select_host_gpu(&visible, guest.host_gpu_id, host_version) {
        HostSelection::Selected(gpu_id) => gpu_id,
        HostSelection::ExplicitGpuHidden(gpu_id) => {
            return Err(RocJITsuError::Other(format!(
                "rocjitsu: dbt_guest.host_gpu_id {gpu_id} is hidden by ROCm device \
                 visibility settings"
            )));
        }
        HostSelection::ExplicitGpuIsaMismatch(gpu_id) => {
            return Err(RocJITsuError::Other(format!(
                "rocjitsu: dbt_guest.host_gpu_id {gpu_id} does not match host_isa '{}'",
                guest.host_isa
            )));
        }
        HostSelection::NoIsaMatch => {
            return Err(RocJITsuError::Other(format!(
                "rocjitsu: no host GPU matches dbt_guest.host_isa '{}' ({}); {}",
                guest.host_isa,
                gfx_name(host_version),
                describe_topology(&topology)
            )));
        }
    };

    // The guest node is appended to the topology, so it is invisible to
    // an application that narrowed ROCR to a subset. Re-add it, and
    // recompute the client selector against the widened order.
    let mut launch_env = BTreeMap::new();
    let mut child_rocr = rocr.clone();
    if let Some(expanded) = expanded_rocr_visible_devices(&topology, rocr.as_deref()) {
        launch_env.insert("ROCR_VISIBLE_DEVICES".to_string(), expanded.clone());
        child_rocr = Some(expanded);
    }
    if let Some(client) = normalized_client_visible_devices(
        &topology,
        child_rocr.as_deref(),
        hip.as_deref(),
        cuda.as_deref(),
        Some(host_gpu_id),
    ) {
        launch_env.insert(client.name, client.value);
    }
    // The hook still uses the legacy tools callback path. Disabling only
    // the rocprofiler-register table-delivery path stops it validating an
    // unshadowed dispatch table before the guest-agent wrappers are
    // installed.
    launch_env.insert("HSA_TOOLS_DISABLE_REGISTER".to_string(), "1".to_string());
    launch_env.insert("HSA_TOOLS_LIB".to_string(), hooks_in_workload.to_string());

    Ok(GuestLaunch {
        host_gpu_id,
        env: launch_env,
        host_gpus: guest.backend == ExecutionBackend::Hardware,
    })
}

/// Name what the launcher did see, so a failed host match says why.
fn describe_topology(topology: &[VisibleGpu]) -> String {
    if topology.is_empty() {
        return "no GPU is visible to this launcher".to_string();
    }
    let seen: Vec<String> = topology
        .iter()
        .map(|gpu| {
            format!(
                "{} (gpu_id {})",
                gfx_name(gpu.gfx_target_version),
                gpu.gpu_id
            )
        })
        .collect();
    format!("visible: {}", seen.join(", "))
}

/// Publish the runtime config handoff for a guest invocation.
///
/// Written by the library's own writer rather than by rocjitsu, because
/// its second line is a contract with two runtime layers that read it
/// inside the application. See [`rocjitsu_sys::write_dbt_handoff`].
///
/// # Errors
///
/// Returns an error when the handoff cannot be published, which includes
/// a `librocjitsu.so` too old to have DBT guest support at all.
pub fn write_handoff(
    library: &Path,
    runtime_dir: &Path,
    config: &Path,
    host_gpu_id: u32,
) -> Result<()> {
    rocjitsu_sys::write_dbt_handoff(library, runtime_dir, config, host_gpu_id)
        .map_err(|e| RocJITsuError::Other(format!("rocjitsu: dbt_guest handoff: {e}")))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn env_of(pairs: &[(&str, &str)]) -> impl EnvLookup + use<> {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |key: &str| map.get(key).cloned()
    }

    fn write(dir: &Path, name: &str, json: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, json).unwrap();
        path
    }

    const SIMULATED_GUEST: &str = r#"{
      "dbt_guest": {
        "enabled": true,
        "guest_isa": "gfx950",
        "host_isa": "gfx942",
        "host_gpu_id": 0,
        "execution_backend": "simulator",
        "simulator_config": "host.json"
      }
    }"#;

    const SIMULATED_HOST: &str = r#"{
      "vm": { "gpu": { "num_gpus": 2, "device": {
        "gpu_id": 50148, "gfx_target_version": 90402, "unique_id": 4096
      } } }
    }"#;

    #[test]
    fn a_config_without_dbt_guest_is_not_guest_mode() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(dir.path(), "plain.json", r#"{"vm":{"gpu":{"num_gpus":1}}}"#);
        assert_eq!(load(&config).unwrap(), None);
    }

    #[test]
    fn a_disabled_dbt_guest_block_is_not_guest_mode() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            "off.json",
            r#"{"dbt_guest":{"enabled":false,"guest_isa":"gfx950","host_isa":"gfx942"}}"#,
        );
        assert_eq!(load(&config).unwrap(), None);
    }

    #[test]
    fn parses_the_launcher_relevant_fields() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(dir.path(), "guest.json", SIMULATED_GUEST);
        let guest = load(&config).unwrap().unwrap();

        assert_eq!(guest.guest_isa, "gfx950");
        assert_eq!(guest.host_isa, "gfx942");
        assert_eq!(guest.host_gpu_id, 0);
        assert_eq!(guest.backend, ExecutionBackend::Simulator);
        assert_eq!(guest.simulator_config, "host.json");
    }

    #[test]
    fn execution_backend_defaults_to_hardware() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            "hw.json",
            r#"{"dbt_guest":{"enabled":true,"guest_isa":"gfx950","host_isa":"gfx942"}}"#,
        );
        assert_eq!(
            load(&config).unwrap().unwrap().backend,
            ExecutionBackend::Hardware
        );
    }

    #[test]
    fn an_enabled_block_without_isas_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(dir.path(), "bad.json", r#"{"dbt_guest":{"enabled":true}}"#);
        let err = load(&config).unwrap_err().to_string();
        assert!(err.contains("guest_isa and host_isa"), "unexpected: {err}");
    }

    #[test]
    fn a_misspelled_execution_backend_is_refused_rather_than_defaulted() {
        // Silently falling back to hardware would run the workload on a
        // GPU the config never asked for, or fail with an unrelated
        // "no host GPU" message on a machine that has none.
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            "typo.json",
            r#"{"dbt_guest":{"enabled":true,"guest_isa":"gfx950","host_isa":"gfx942",
                "execution_backend":"simulater"}}"#,
        );
        let err = load(&config).unwrap_err().to_string();
        assert!(err.contains("simulater"), "unexpected: {err}");
    }

    #[test]
    fn simulator_topology_expands_num_gpus_like_the_config_loader() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(dir.path(), "guest.json", SIMULATED_GUEST);
        write(dir.path(), "host.json", SIMULATED_HOST);
        let guest = load(&config).unwrap().unwrap();

        let gpus = execution_gpus(&config, &guest).unwrap();
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].ordinal, 0);
        assert_eq!(gpus[0].gpu_id, 50148);
        assert_eq!(gpus[0].unique_id, 4096);
        assert_eq!(gpus[1].ordinal, 1);
        assert_eq!(gpus[1].gpu_id, 50149);
        assert_eq!(gpus[1].unique_id, 4097);
    }

    #[test]
    fn a_relative_simulator_config_resolves_beside_the_guest_config() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("configs");
        std::fs::create_dir_all(&nested).unwrap();
        let config = write(&nested, "guest.json", SIMULATED_GUEST);
        write(&nested, "host.json", SIMULATED_HOST);
        let guest = load(&config).unwrap().unwrap();

        // Resolved against the config, not the working directory, which
        // is what makes a config directory movable as a unit.
        assert_eq!(execution_gpus(&config, &guest).unwrap().len(), 2);
    }

    #[test]
    fn an_empty_simulator_config_selects_the_guest_config_itself() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            "self.json",
            r#"{
              "dbt_guest": {"enabled":true,"guest_isa":"gfx950","host_isa":"gfx942",
                            "execution_backend":"simulator"},
              "vm": {"gpu": {"num_gpus": 1, "device": {"gpu_id": 7, "gfx_target_version": 90402}}}
            }"#,
        );
        let guest = load(&config).unwrap().unwrap();
        let gpus = execution_gpus(&config, &guest).unwrap();

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].gpu_id, 7);
    }

    /// The whole point of the simulator backend: a plan on a machine
    /// with no AMD GPU at all, which is what CI runs.
    #[test]
    fn plans_a_simulated_host_without_any_real_gpu() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(dir.path(), "guest.json", SIMULATED_GUEST);
        write(dir.path(), "host.json", SIMULATED_HOST);
        let guest = load(&config).unwrap().unwrap();

        let launch = plan(&config, &guest, "/lib/hooks.so", &env_of(&[])).unwrap();

        assert_eq!(launch.host_gpu_id, 50148);
        assert!(!launch.host_gpus, "a simulated host needs no real GPU");
        assert_eq!(
            launch.env.get("HSA_TOOLS_LIB").map(String::as_str),
            Some("/lib/hooks.so")
        );
        assert_eq!(
            launch
                .env
                .get("HSA_TOOLS_DISABLE_REGISTER")
                .map(String::as_str),
            Some("1")
        );
        // No ROCR selector was set, so there is nothing to widen; the
        // client selector is synthesized to put the resolved host first.
        assert!(!launch.env.contains_key("ROCR_VISIBLE_DEVICES"));
        assert_eq!(
            launch.env.get("HIP_VISIBLE_DEVICES").map(String::as_str),
            Some("0,1")
        );
    }

    #[test]
    fn a_narrowed_rocr_selection_is_widened_to_admit_the_guest() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(dir.path(), "guest.json", SIMULATED_GUEST);
        write(dir.path(), "host.json", SIMULATED_HOST);
        let guest = load(&config).unwrap().unwrap();

        let env = env_of(&[("ROCR_VISIBLE_DEVICES", "1")]);
        let launch = plan(&config, &guest, "/lib/hooks.so", &env).unwrap();

        // Ordinal 2 is the appended guest node; without it the
        // application would discover no guest GPU at all.
        assert_eq!(
            launch.env.get("ROCR_VISIBLE_DEVICES").map(String::as_str),
            Some("1,2")
        );
        assert_eq!(launch.host_gpu_id, 50149);
    }

    #[test]
    fn an_unrecognized_host_isa_is_named_in_the_error() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            "guest.json",
            r#"{"dbt_guest":{"enabled":true,"guest_isa":"gfx950","host_isa":"mi300",
                "execution_backend":"simulator"}}"#,
        );
        let guest = load(&config).unwrap().unwrap();
        let err = plan(&config, &guest, "/lib/hooks.so", &env_of(&[]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("mi300"), "unexpected: {err}");
    }

    #[test]
    fn a_host_gpu_the_visibility_settings_hide_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            "guest.json",
            r#"{"dbt_guest":{"enabled":true,"guest_isa":"gfx950","host_isa":"gfx942",
                "host_gpu_id":50149,"execution_backend":"simulator",
                "simulator_config":"host.json"}}"#,
        );
        write(dir.path(), "host.json", SIMULATED_HOST);
        let guest = load(&config).unwrap().unwrap();

        // ROCR leaves only ordinal 0 (gpu_id 50148) visible, so the
        // pinned 50149 cannot be honoured. Falling back to the other GPU
        // would run the workload somewhere the config did not ask for.
        let env = env_of(&[("ROCR_VISIBLE_DEVICES", "0")]);
        let err = plan(&config, &guest, "/lib/hooks.so", &env)
            .unwrap_err()
            .to_string();
        assert!(err.contains("hidden"), "unexpected: {err}");
    }

    #[test]
    fn a_host_isa_no_visible_gpu_provides_is_refused_with_what_was_seen() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            "guest.json",
            r#"{"dbt_guest":{"enabled":true,"guest_isa":"gfx950","host_isa":"gfx1201",
                "execution_backend":"simulator","simulator_config":"host.json"}}"#,
        );
        write(dir.path(), "host.json", SIMULATED_HOST);
        let guest = load(&config).unwrap().unwrap();

        let err = plan(&config, &guest, "/lib/hooks.so", &env_of(&[]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("gfx1201"), "unexpected: {err}");
        assert!(
            err.contains("gfx942"),
            "the error must say what was seen: {err}"
        );
    }
}
