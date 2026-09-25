//! Integration smoke test: verifies that a runtime-discovered KMD
//! library and a synthesised sim config are usable at runtime.
//!
//! The `kmd_config` round-trip is skipped when no KMD library is
//! discoverable on this machine (rocjitsu not installed).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
            extra: Default::default(),
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
        let session = tmp.path().join("session");
        let cfg = kmd_config(&def, &session).expect("sim config should materialise");
        assert!(cfg.exists());
    }
}

/// The profile's per-node GPU count must flow into the synthesised
/// rocjitsu config as `vm.gpu.num_gpus`, so `--gpus-per-node N` exposes
/// N devices to the workload. Does not require the KMD library; the
/// config is written into a scratch directory standing in for a
/// session's.
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
        extra: Default::default(),
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

    let session = tmp.path().join("session");
    let cfg = kmd_config(&def, &session).expect("sim config should materialise");
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
    use mirage_core::session::{SessionContext, SessionId};

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
        extra: Default::default(),
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
    let runtime_dir = tmp.path().join("scratch");
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let ctx = SessionContext {
        id: SessionId::new("rccl-env-test").unwrap(),
        profile: ProfileDef {
            name: "rccl-env-test".to_string(),
            description: None,
            emulator,
            containerize: None,
        },
        runtime_dir,
        daemon: false,
    };

    let backend = get_emulator_backend("rocjitsu").expect("rocjitsu backend registered");
    let injection = backend
        .injection_def(&ctx)
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

/// The injection names the process hosting the daemon, and only in
/// daemon mode.
///
/// Without it the interposer has nothing to check the peer answering its
/// socket against, so it trusts whoever answered and says so on the
/// workload's stderr — at every launch, in front of the workload's own
/// output:
///
/// ```text
/// [rj warn] daemon: authorizing pid N to read this address space; no
/// launcher named a daemon for this client, ...
/// ```
///
/// The value is load-bearing rather than cosmetic in both directions. A
/// *wrong* PID is worse than none: the interposer refuses the peer
/// outright (`RefusedPeerMismatch`) and the pageable transfers that need
/// `process_vm_readv` stop working. So this asserts the exact identity —
/// this process, which is the one `start_daemon` calls
/// `rj_daemon_start` on — and not merely that the variable is set.
#[test]
fn injection_names_the_daemon_hosting_process_in_daemon_mode() {
    use mirage_core::emulator::get_emulator_backend;
    use mirage_core::profile::ProfileDef;
    use mirage_core::session::{SessionContext, SessionId};

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

    let backend = get_emulator_backend("rocjitsu").expect("rocjitsu backend registered");
    let injection_for = |daemon: bool| {
        let emulator = EmulatorDef {
            extra: Default::default(),
            emulator: "rocjitsu".to_string(),
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
                num_nodes: 1,
                gpus_per_node: 1,
                agent: MaybeRef::Ref(agent_name.clone()),
            }),
        };
        let runtime_dir = tmp.path().join(if daemon { "daemon" } else { "local" });
        std::fs::create_dir_all(&runtime_dir).unwrap();
        let ctx = SessionContext {
            id: SessionId::new("daemon-pid-test").unwrap(),
            profile: ProfileDef {
                name: "daemon-pid-test".to_string(),
                description: None,
                emulator,
                containerize: None,
            },
            runtime_dir,
            daemon,
        };
        backend
            .injection_def(&ctx)
            .expect("injection should succeed with a discoverable KMD library")
    };

    assert_eq!(
        injection_for(true).env.get("ROCJITSU_DAEMON_PID"),
        Some(&std::process::id().to_string()),
        "daemon mode must name this process, the one that hosts the daemon"
    );

    // `--in-process` connects to no socket. Naming a daemon there would
    // describe one that was never started.
    assert_eq!(
        injection_for(false).env.get("ROCJITSU_DAEMON_PID"),
        None,
        "in-process mode has no daemon to name"
    );
}

/// Drop-in `--config <path>` mode: the user's config file is used
/// verbatim, but the rocjitsu runtime directory it implies belongs to
/// the *session*, not to whatever directory the user keeps their config
/// in. Two runs sharing a config file must not share a runtime
/// directory, and neither may write beside a file mirage does not own
/// and will not clean up. Skipped when the KMD library is not
/// discoverable.
#[test]
fn dropin_config_keeps_its_runtime_dir_inside_the_session() {
    use mirage_core::common::SimpleValue;
    use mirage_core::emulator::get_emulator_backend;
    use mirage_core::profile::ProfileDef;
    use mirage_core::session::{SessionContext, SessionId};

    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    if kmd_preload().is_none() {
        return; // rocjitsu not installed on this host
    }

    // A config file of the user's, in a directory of the user's.
    let user_dir = tmp.path().join("home/work");
    std::fs::create_dir_all(&user_dir).unwrap();
    let user_config = user_dir.join("cfg.json");
    std::fs::write(&user_config, b"{}\n").unwrap();

    let mut emulator = EmulatorDef {
        extra: Default::default(),
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Functional,
        options: Default::default(),
        topology: MaybeRef::Ref("unused-in-dropin-mode".to_string()),
    };
    emulator.options.insert(
        "config".to_string(),
        SimpleValue::String(user_config.display().to_string()),
    );

    let session_dir = tmp.path().join("session");
    std::fs::create_dir_all(&session_dir).unwrap();
    let ctx = SessionContext {
        id: SessionId::new("dropin-test").unwrap(),
        profile: ProfileDef {
            name: "dropin-test".to_string(),
            description: None,
            emulator,
            containerize: None,
        },
        runtime_dir: session_dir.clone(),
        daemon: false,
    };

    let backend = get_emulator_backend("rocjitsu").expect("rocjitsu backend registered");
    let injection = backend
        .injection_def(&ctx)
        .expect("injection should succeed with a discoverable KMD library");

    let runtime_dir = injection
        .env
        .get("ROCJITSU_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .expect("the injection must name a rocjitsu runtime directory");
    assert!(
        runtime_dir.starts_with(&session_dir),
        "runtime dir {} must live inside the session dir {}",
        runtime_dir.display(),
        session_dir.display()
    );
    // It is still wired up: the discovery file names the user's config.
    assert_eq!(
        std::fs::read_to_string(runtime_dir.join("config_path")).unwrap(),
        format!("{}\n", user_config.display())
    );
    // And the user's directory is exactly as they left it.
    let left_behind: Vec<_> = std::fs::read_dir(&user_dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert_eq!(
        left_behind,
        vec![std::ffi::OsString::from("cfg.json")],
        "nothing may be written beside the user's own config file"
    );
}
