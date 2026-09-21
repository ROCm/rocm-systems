//! Integration tests for the `rocjitsu-dbt` (Dynamic Binary Translation)
//! emulator backend.
//!
//! These cover the parts that do not need a real GPU or a built
//! rocjitsu: registry wiring, ISA resolution, and the `HSA_TOOLS_LIB`
//! env contract `injection_def` produces (exercised with a stand-in
//! hook library so the contract is checked even when rocjitsu is not
//! installed).
//!
//! Environment overrides are supplied through
//! [`RocjitsuDbt::injection_def_with`] rather than by mutating the
//! process environment. `std::env::set_var` is `unsafe` in Rust 2024
//! because it races every other thread reading the environment, and the
//! test binary is multi-threaded; passing the lookup in keeps these tests
//! sound and independent of each other.
//!
//! The final test attempts a *real* end-to-end translation with the
//! `rj_dbt_translate` CLI when both the tool and a sample code object
//! are discoverable on this host; it skips cleanly otherwise.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use mirage_core::common::{MaybeRef, SimpleValue};
use mirage_core::emulator::{EmulatorDef, ExecMode, get_emulator_backend};
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionContext, SessionId};
use mirage_core::topology::TopologyDef;
use mirage_rocjitsu::dbt::{self, EnvLookup, RocjitsuDbt};

/// Build a single-GPU emulator def pinning a `gfx_target_version` guest.
fn def_for_guest(gfx_target_version: u32) -> EmulatorDef {
    let mut agent = mirage_core::agent::AgentDef::default();
    agent.vm.gpu.device.gfx_target_version = gfx_target_version;
    EmulatorDef {
        emulator: dbt::NAME.to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Functional,
        options: Default::default(),
        topology: MaybeRef::Owned(TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Owned(agent),
        }),
    }
}

/// Build the resolved session context a backend is handed.
///
/// Backends used to be given only a `SessionId` and had to read the
/// session's definition back off disk; they are now handed everything
/// they need, so a test can construct one directly with no filesystem
/// state at all.
fn context(emulator: EmulatorDef, runtime_dir: &std::path::Path) -> SessionContext {
    SessionContext {
        id: SessionId::new("dbt-test").unwrap(),
        profile: ProfileDef {
            name: "dbt-test".to_string(),
            description: None,
            emulator,
            containerize: None,
        },
        runtime_dir: runtime_dir.to_path_buf(),
        daemon: false,
    }
}

/// An environment containing exactly the given pairs.
fn env_of(pairs: &[(&str, &str)]) -> impl EnvLookup + use<> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    move |key: &str| map.get(key).cloned()
}

/// Write a stand-in hook library so discovery succeeds without a built
/// rocjitsu, and return its path.
fn stub_hook(dir: &std::path::Path) -> PathBuf {
    let path = dir.join("librocjitsu_hooks.so");
    std::fs::write(&path, b"\x7fELF stub").unwrap();
    path
}

#[test]
fn backend_is_registered() {
    let backend =
        get_emulator_backend(dbt::NAME).expect("rocjitsu-dbt backend must be in the registry");
    let desc = backend.description();
    assert_eq!(desc.name, dbt::NAME);
    // The backend advertises the target/source ISA options.
    let opt_names: Vec<&str> = desc
        .options_schema
        .iter()
        .map(|o| o.name.as_str())
        .collect();
    assert!(opt_names.contains(&dbt::OPT_TARGET_ISA));
    assert!(opt_names.contains(&dbt::OPT_SOURCE_ISA));
}

#[test]
fn injection_emits_hsa_tools_env_contract() {
    let tmp = tempfile::tempdir().unwrap();
    let fake_hook = stub_hook(tmp.path());
    // Pin both ISAs explicitly so the contract is deterministic and
    // independent of whatever GPU (if any) this host has.
    let env = env_of(&[
        ("ROCJITSU_HOOKS_LIB", fake_hook.to_str().unwrap()),
        ("RJ_DBT_TARGET_ISA", "gfx1201"),
        ("RJ_DBT_LOG", "2"),
    ]);

    let ctx = context(def_for_guest(90500), tmp.path()); // gfx950 guest
    let injection = RocjitsuDbt
        .injection_def_with(&ctx, &env)
        .expect("injection should succeed with a discoverable hook");

    // The hook is loaded via HSA_TOOLS_LIB, never LD_PRELOAD.
    assert_eq!(injection.ld_preload, None);
    assert_eq!(
        injection.env.get("HSA_TOOLS_LIB").map(String::as_str),
        Some(fake_hook.to_str().unwrap())
    );
    assert_eq!(
        injection.env.get("RJ_DBT_TARGET_ISA").map(String::as_str),
        Some("gfx1201")
    );
    // Source ISA is derived from the gfx950 guest agent.
    assert_eq!(
        injection.env.get("RJ_DBT_SOURCE_ISA").map(String::as_str),
        Some("gfx950")
    );
    // The hook log level is forwarded from the environment.
    assert_eq!(
        injection.env.get("RJ_DBT_LOG").map(String::as_str),
        Some("2")
    );
    // DBT runs translated code on the host's real GPU.
    assert!(injection.host_gpus);
}

#[test]
fn injection_fails_without_a_target_isa() {
    let tmp = tempfile::tempdir().unwrap();
    let fake_hook = stub_hook(tmp.path());
    let env = env_of(&[("ROCJITSU_HOOKS_LIB", fake_hook.to_str().unwrap())]);

    // A gfx950 guest with no target option: on a host with no supported
    // GPU there is no target to translate to, so injection must fail
    // loudly rather than run unemulated. (On a host that *does* have a
    // supported GPU, a target is auto-selected and injection succeeds —
    // accept either, but never a silent wrong env.)
    let ctx = context(def_for_guest(90500), tmp.path());
    match RocjitsuDbt.injection_def_with(&ctx, &env) {
        Err(e) => assert!(
            e.to_string().contains("no target ISA"),
            "unexpected error: {e}"
        ),
        Ok(injection) => assert!(
            injection.env.contains_key("RJ_DBT_TARGET_ISA"),
            "a successful injection must carry a target ISA"
        ),
    }
}

#[test]
fn injection_fails_without_a_hook_library() {
    // The hook is what performs the translation. Without it the workload
    // would load its untranslated code objects and either fail obscurely
    // or run native code, so this must be a loud error.
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("nope.so");
    let env = env_of(&[
        ("ROCJITSU_HOOKS_LIB", missing.to_str().unwrap()),
        ("RJ_DBT_TARGET_ISA", "gfx1201"),
    ]);
    let ctx = context(def_for_guest(90500), tmp.path());

    // Only assert the failure when the host has no real hook library to
    // fall back on; a developer machine with rocjitsu built legitimately
    // finds one.
    if dbt::hooks_preload().is_none() {
        let err = RocjitsuDbt.injection_def_with(&ctx, &env).unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }
}

#[test]
fn target_isa_option_drives_injection() {
    let tmp = tempfile::tempdir().unwrap();
    let fake_hook = stub_hook(tmp.path());
    let env = env_of(&[("ROCJITSU_HOOKS_LIB", fake_hook.to_str().unwrap())]);

    let mut emulator = def_for_guest(90402); // gfx942 guest
    emulator.options.insert(
        dbt::OPT_TARGET_ISA.to_string(),
        SimpleValue::String("gfx950".to_string()),
    );
    let ctx = context(emulator, tmp.path());
    let injection = RocjitsuDbt.injection_def_with(&ctx, &env).unwrap();
    assert_eq!(
        injection.env.get("RJ_DBT_TARGET_ISA").map(String::as_str),
        Some("gfx950")
    );
    assert_eq!(
        injection.env.get("RJ_DBT_SOURCE_ISA").map(String::as_str),
        Some("gfx942")
    );
}

#[test]
fn a_containerised_session_points_the_hook_at_its_in_container_path() {
    // The host path does not exist inside the container; ROCR would fail
    // to load the hook and the workload would run untranslated.
    let tmp = tempfile::tempdir().unwrap();
    let fake_hook = stub_hook(tmp.path());
    let env = env_of(&[
        ("ROCJITSU_HOOKS_LIB", fake_hook.to_str().unwrap()),
        ("RJ_DBT_TARGET_ISA", "gfx1201"),
    ]);

    let mut ctx = context(def_for_guest(90500), tmp.path());
    ctx.profile.containerize = Some(mirage_core::profile::ContainerizedDef {
        provider: None,
        image: "img:latest".to_string(),
        mounts: Vec::new(),
        ports: Vec::new(),
        devices: Vec::new(),
        groups: Vec::new(),
        hacks: Vec::new(),
    });

    let injection = RocjitsuDbt.injection_def_with(&ctx, &env).unwrap();
    let tools_lib = injection.env.get("HSA_TOOLS_LIB").unwrap();
    assert!(
        tools_lib.starts_with("/mnt/mirage/lib/"),
        "expected an in-container path, got {tools_lib}"
    );
    // And the host library must actually be mounted there.
    assert!(
        injection
            .mounts
            .iter()
            .any(|m| &m.container_path == tools_lib
                && m.host_path == fake_hook.display().to_string()),
        "the hook must be bind-mounted at the path HSA_TOOLS_LIB names: {:?}",
        injection.mounts
    );
}

/// Locate the `rj_dbt_translate` CLI: an explicit `RJ_DBT_TRANSLATE`
/// override, then `$PATH`, then the in-tree rocjitsu build output.
fn find_dbt_translate() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RJ_DBT_TRANSLATE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(out) = Command::new("sh")
        .arg("-c")
        .arg("command -v rj_dbt_translate")
        .output()
        && out.status.success()
    {
        let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    // In-tree build, relative to this crate. The rocjitsu CMake build
    // links the CLI into `build/tools/`.
    let candidate = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rocjitsu/build/tools/rj_dbt_translate");
    candidate.is_file().then_some(candidate)
}

/// Find a sample gfx950 HSA code object to translate. Probes the
/// ROCm-SDK wheel assets a developer venv ships.
fn find_gfx950_code_object() -> Option<PathBuf> {
    if let Ok(out) = Command::new("sh")
        .arg("-c")
        .arg(
            "find $HOME/rocm-systems/emulation -path '*hsa-amd-aqlprofile*' \
             -name 'gfx950_*.hsaco' 2>/dev/null | head -1",
        )
        .output()
        && out.status.success()
    {
        let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !path.is_empty() {
            let p = PathBuf::from(path);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    // Fall back to a raw gfx950 code object emitted by the rocjitsu
    // `device_kernels` build (a standalone AMDGPU code object, not a HIP
    // fatbin), so the test is self-sufficient from just a rocjitsu build.
    let in_tree = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rocjitsu/build/kernels/triton_cdna4_matmul_dynamic_32x32x64.hsaco");
    in_tree.is_file().then_some(in_tree)
}

/// Real end-to-end translation: drive the actual `rj_dbt_translate`
/// pipeline to retarget a real gfx950 code object to gfx1201. Skipped
/// unless both the tool and a sample code object are available on this
/// host (the rocjitsu CMake `rj_dbt_translate` target builds the CLI).
///
/// This validates what is in mirage's scope — that the integration
/// invokes the translator correctly on a real input and the pipeline
/// runs to a deterministic, structured result. The rocjitsu DBT
/// translator is an MVP whose per-instruction coverage is still
/// growing, so a translatable-source kernel may legitimately stop at a
/// structured "expansion rule not implemented" diagnostic; that means
/// the pipeline ran, which is the contract under test. The test fails
/// only on outcomes that would indicate a wiring bug: the tool failing
/// to run, a usage error, or a failure to *parse* the input.
#[test]
fn real_dbt_translation_when_tool_available() {
    let Some(tool) = find_dbt_translate() else {
        eprintln!("rj_dbt_translate not found; skipping real translation test");
        return;
    };
    let Some(code_object) = find_gfx950_code_object() else {
        eprintln!("no gfx950 sample code object found; skipping real translation test");
        return;
    };
    eprintln!(
        "translating {} (gfx950 -> gfx1201) with {}",
        code_object.display(),
        tool.display()
    );

    let output = Command::new(&tool)
        .arg(&code_object)
        .arg("--input-target")
        .arg("gfx950")
        .arg("--output-target")
        .arg("gfx1201")
        .arg("--output-mode")
        .arg("code-object")
        .output()
        .expect("rj_dbt_translate should be executable");
    let stderr = String::from_utf8_lossy(&output.stderr);

    // A parse failure means the input was not a real AMDGPU code object
    // — i.e. the test fed the tool the wrong thing. That is a wiring bug.
    assert!(
        !stderr.contains("failed to parse input"),
        "the sample was not a valid code object: {stderr}"
    );

    if output.status.success() {
        // Happy path: a fully translated code object was emitted.
        assert!(
            !output.stdout.is_empty(),
            "a successful translation must emit a code object on stdout"
        );
        return;
    }

    // Otherwise the pipeline must have run far enough to decode the
    // input and reach a *structured* translation diagnostic (a known
    // translator-coverage gap), not crashed or printed a usage error.
    let ran_pipeline = stderr.contains("translation failed")
        || stderr.contains("legalization")
        || stderr.contains("expand")
        || stderr.contains("resource");
    assert!(
        ran_pipeline,
        "rj_dbt_translate did not run the translation pipeline; \
         exit={:?} stderr={stderr}",
        output.status.code()
    );
    eprintln!(
        "translation pipeline ran; stopped at a translator-coverage gap (expected for the MVP):\n{}",
        stderr.lines().next().unwrap_or_default()
    );
}
