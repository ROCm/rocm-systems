//! Integration smoke test: verifies that a runtime-discovered KMD
//! library and a synthesised sim config are usable at runtime.
//!
//! The `kmd_config` round-trip is skipped when no KMD library is
//! discoverable on this machine (rocjitsu not installed).

use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDef, ExecMode};
use mirage_rocjitsu::{kmd_config, kmd_preload};

#[test]
fn sim_config_round_trip() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();

    // The full sim-config round-trip needs the KMD library, which is
    // discovered at runtime; skip when rocjitsu isn't installed.
    if kmd_preload().is_some() {
        let agent_name = agent_report
            .iter()
            .map(|(n, _)| n.clone())
            .next()
            .expect("at least one builtin agent");
        let def = EmulatorDef {
            emulator: "rocjitsu".to_string(),
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
                num_nodes: 1,
                gpus_per_node: 1,
                agent: MaybeRef::Ref(agent_name),
            }),
        };
        let cfg = kmd_config(&def, None).expect("sim config should materialise");
        assert!(cfg.exists());
    }
}

/// The profile's per-node GPU count must flow into the synthesised
/// rocjitsu config as `vm.gpu.num_gpus`, so `--gpus-per-node N` exposes
/// N devices to the workload. Does not require the KMD library: the
/// `session: None` path writes a content-addressed config file.
#[test]
fn gpus_per_node_drives_num_gpus() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report
        .iter()
        .map(|(n, _)| n.clone())
        .next()
        .expect("at least one builtin agent");

    let def = EmulatorDef {
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Functional,
        options: Default::default(),
        topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
            num_nodes: 1,
            gpus_per_node: 3,
            agent: MaybeRef::Ref(agent_name),
        }),
    };

    let cfg = kmd_config(&def, None).expect("sim config should materialise");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&cfg).unwrap()).unwrap();
    assert_eq!(json["vm"]["gpu"]["num_gpus"], 3);
}

/// When rocjitsu is installed, the injected workload environment carries
/// the overridable RCCL/HSA defaults the upstream RCCL collective tests
/// rely on. Skipped when the KMD library is not discoverable.
#[test]
fn injection_emits_rccl_env_defaults() {
    use mirage_core::emulator::get_emulator_backend;
    use mirage_core::profile::ProfileDef;
    use mirage_core::session::{SessionDef, SessionId};

    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    if kmd_preload().is_none() {
        return; // rocjitsu not installed on this host
    }

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report
        .iter()
        .map(|(n, _)| n.clone())
        .next()
        .expect("at least one builtin agent");

    let emulator = EmulatorDef {
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Functional,
        options: Default::default(),
        topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref(agent_name),
        }),
    };
    let id = SessionId::new("rccl-env-test").unwrap();
    let profile = ProfileDef {
        name: "rccl-env-test".to_string(),
        description: None,
        emulator,
        containerize: None,
    };
    let def = SessionDef {
        id: id.clone(),
        profile: MaybeRef::Owned(profile),
        workdir: ".".to_string(),
        daemon: false,
        created_at: chrono::Utc::now(),
    };
    let layout = mirage_core::paths::SessionLayout::for_id(&id);
    mirage_core::state::write_json(&layout.def(), &def).unwrap();

    let backend = get_emulator_backend("rocjitsu").expect("rocjitsu backend registered");
    let injection = backend
        .injection_def(&id)
        .expect("injection should succeed with a discoverable KMD library");

    for (key, value) in [
        ("HSA_ENABLE_SDMA", "1"),
        ("ROCPROFILER_REGISTER_ENABLED", "0"),
        ("HSA_NO_SCRATCH_RECLAIM", "1"),
        ("NCCL_P2P_DISABLE", "1"),
        ("NCCL_SHM_DISABLE", "1"),
        ("NCCL_SOCKET_NTHREADS", "1"),
        ("NCCL_NSOCKS_PERTHREAD", "1"),
        ("NCCL_SOCKET_IFNAME", "lo"),
    ] {
        assert_eq!(
            injection.env.get(key).map(String::as_str),
            Some(value),
            "expected {key}={value} in rocjitsu injection env"
        );
    }
}
