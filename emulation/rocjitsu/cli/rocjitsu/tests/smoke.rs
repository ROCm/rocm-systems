//! Integration smoke test: verifies that a runtime-discovered KMD
//! library and a synthesised sim config are usable at runtime.
//!
//! The `kmd_config` round-trip is skipped when no KMD library is
//! discoverable on this machine (rocjitsu not installed).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rj_backend_rocjitsu::{kmd_config, kmd_preload};
use rj_core::common::MaybeRef;
use rj_core::emulator::{EmulatorDef, ExecMode};

#[test]
fn sim_config_round_trip() {
    let _g = rj_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    rj_core::paths::set_test_root(tmp.path());

    let agent_report = rj_builtin::ensure_agents(false).unwrap();

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
            topology: MaybeRef::Owned(rj_core::topology::TopologyDef {
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
    let _g = rj_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    rj_core::paths::set_test_root(tmp.path());

    let agent_report = rj_builtin::ensure_agents(false).unwrap();
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
        topology: MaybeRef::Owned(rj_core::topology::TopologyDef {
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

/// Every builtin profile generates a config the emulator will accept.
///
/// The field-by-field tests beside this one cannot answer this. The
/// loader enforces relationships *between* fields — the rule that
/// `num_sdma_queues_per_engine` must be nonzero when `num_sdma_engines`
/// is shipped in all three profiles, failing every session at daemon
/// start, while each field on its own looked reasonable. So this asks
/// the loader rather than the document, which is the only thing that
/// settles it.
///
/// Every builtin, not a representative one: they are hand-written, and
/// what is wrong with one is exactly what the others can be right about.
/// Skipped when rocjitsu is not installed, since there is no loader to
/// ask.
#[test]
fn every_builtin_generates_a_config_the_emulator_accepts() {
    let _g = rj_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    rj_core::paths::set_test_root(tmp.path());

    let Some(lib) = kmd_preload() else {
        return;
    };
    // This case answers its question by loading the library, and against
    // a sanitizer build it cannot: the runtime has to be in the process's
    // initial library list, and putting it there means preloading it into
    // `cargo test` and so into `rustc`. Without this the `dlopen` fails
    // and the failure is reported as "the builtin generates a config
    // rocjitsu refuses", which is a claim about the builtin and is false.
    if let Some(why) = rj_core::discovery::sanitizer_preload_missing() {
        eprintln!("SKIP: the builtins cannot be put through the real loader here: {why}");
        return;
    }
    let agents = rj_builtin::ensure_agents(false).unwrap();
    assert!(
        !agents.documents.is_empty(),
        "there are builtin agents to check"
    );

    for (name, _) in &agents.documents {
        let def = EmulatorDef {
            emulator: "rocjitsu".to_string(),
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Owned(rj_core::topology::TopologyDef {
                num_nodes: 1,
                gpus_per_node: 1,
                agent: MaybeRef::Ref(name.clone()),
            }),
        };
        let session = tmp.path().join(format!("session-{name}"));
        let cfg = kmd_config(&def, &session).expect("sim config should materialise");
        let json = std::fs::read(&cfg).unwrap();
        let json = std::ffi::CString::new(json).expect("a config has no interior NUL");
        rocjitsu_sys::config_is_loadable(&lib, &json).unwrap_or_else(|e| {
            panic!(
                "the `{name}` builtin generates a config rocjitsu \
                                        refuses, so every session on it fails at start: {e}"
            )
        });

        // And the check is a check. Take the same config, put back the
        // exact mistake all three shipped with — SDMA engines and no
        // queue depth — and the loader must refuse it. Without this the
        // assertion above would still pass against a loader that had
        // stopped validating anything.
        let mut doc: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&cfg).unwrap()).unwrap();
        let device = &mut doc["vm"]["gpu"]["device"];
        assert_ne!(
            device["num_sdma_engines"], 0,
            "`{name}` must declare SDMA engines for this to mean anything"
        );
        device["num_sdma_queues_per_engine"] = serde_json::json!(0);
        let broken = std::ffi::CString::new(serde_json::to_vec(&doc).unwrap()).unwrap();
        rocjitsu_sys::config_is_loadable(&lib, &broken).expect_err(
            "a config with SDMA engines and no queue depth is the one rocjitsu refuses",
        );
    }
}

/// When rocjitsu is installed, the injected workload environment carries
/// the overridable RCCL/HSA defaults the upstream RCCL collective tests
/// rely on. Skipped when the KMD library is not discoverable.
#[test]
fn injection_emits_rccl_env_defaults() {
    use rj_core::emulator::get_emulator_backend;
    use rj_core::profile::ProfileDef;
    use rj_core::session::{SessionContext, SessionId};

    let _g = rj_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    rj_core::paths::set_test_root(tmp.path());

    if kmd_preload().is_none() {
        return; // rocjitsu not installed on this host
    }

    let agent_report = rj_builtin::ensure_agents(false).unwrap();
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
        topology: MaybeRef::Owned(rj_core::topology::TopologyDef {
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

/// Drop-in `--config <path>` mode: the user's config file is used
/// verbatim, but the rocjitsu runtime directory it implies belongs to
/// the *session*, not to whatever directory the user keeps their config
/// in. Two runs sharing a config file must not share a runtime
/// directory, and neither may write beside a file rocjitsu does not own
/// and will not clean up. Skipped when the KMD library is not
/// discoverable.
#[test]
fn dropin_config_keeps_its_runtime_dir_inside_the_session() {
    use rj_core::common::SimpleValue;
    use rj_core::emulator::get_emulator_backend;
    use rj_core::profile::ProfileDef;
    use rj_core::session::{SessionContext, SessionId};

    let _g = rj_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    rj_core::paths::set_test_root(tmp.path());

    if kmd_preload().is_none() {
        return; // rocjitsu not installed on this host
    }

    // A config file of the user's, in a directory of the user's.
    let user_dir = tmp.path().join("home/work");
    std::fs::create_dir_all(&user_dir).unwrap();
    let user_config = user_dir.join("cfg.json");
    std::fs::write(&user_config, b"{}\n").unwrap();

    let mut emulator = EmulatorDef {
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
