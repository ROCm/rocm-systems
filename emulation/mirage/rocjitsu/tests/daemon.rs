//! Integration tests for the shared in-process RocJitsu daemon API.
//!
//! The safe RAII wrapper lives in `rocjitsu_sys`, the one crate allowed
//! to write `unsafe`, so these tests drive it without any of their own.
//!
//! Stands up the `librocjitsu` daemon through the Rust lifecycle wrapper,
//! connects a client speaking the daemon RPC protocol, performs the
//! handshake the KMD interposer would, and verifies the daemon serves a
//! live simulated device. The whole test is skipped when no rocjitsu KMD
//! library is discoverable on this machine (install rocjitsu under
//! `$ROCM_HOME` or a sibling monorepo build to exercise it).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDef, ExecMode};
use mirage_rocjitsu::{kmd_config, kmd_preload};
use rocjitsu_sys::RjDaemonStatus;
use rocjitsu_sys::daemon::Daemon;

/// Build the 16-byte RPC header the wire protocol uses.
fn header(opcode: u16, request_id: u32, payload_bytes: u32, result: i32) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0..2].copy_from_slice(&opcode.to_ne_bytes());
    h[4..8].copy_from_slice(&request_id.to_ne_bytes());
    h[8..12].copy_from_slice(&payload_bytes.to_ne_bytes());
    h[12..16].copy_from_slice(&result.to_ne_bytes());
    h
}

fn read_exact(stream: &mut UnixStream, buf: &mut [u8]) {
    stream.read_exact(buf).expect("read response");
}

/// Start a daemon, or `None` when this machine's `librocjitsu.so` cannot
/// provide one.
///
/// A library that is *present but does not export the daemon API* — a
/// stale sibling build, or one configured without the daemon — is
/// indistinguishable, for these tests, from one that is not installed at
/// all, and must skip for the same reason. Only a load failure counts as
/// "not available"; anything else is a real failure and still panics.
fn start_or_skip(
    lib: &std::path::Path,
    config: &std::path::Path,
    runtime_dir: &std::path::Path,
) -> Option<Daemon> {
    match Daemon::start(lib, config, runtime_dir) {
        Ok(daemon) => Some(daemon),
        Err(e @ rocjitsu_sys::daemon::DaemonError::Load { .. }) => {
            eprintln!("rocjitsu KMD library cannot host a daemon ({e}); skipping");
            None
        }
        Err(e) => panic!("daemon should start: {e}"),
    }
}

#[test]
fn daemon_serves_handshake() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    // The daemon needs the KMD library (it exports the rj_vm_* API);
    // skip cleanly when rocjitsu is not installed.
    let Some(lib) = kmd_preload() else {
        eprintln!("rocjitsu KMD library not found; skipping daemon test");
        return;
    };

    // Synthesise a real sim config from a builtin agent.
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
            gpus_per_node: 1,
            agent: MaybeRef::Ref(agent_name),
        }),
    };
    let config = kmd_config(&def, tmp.path()).expect("sim config should materialise");

    let runtime_dir = tmp.path().join("rt");
    let Some(daemon) = start_or_skip(&lib, &config, &runtime_dir) else {
        return;
    };
    assert_eq!(daemon.status(), RjDaemonStatus::Running);
    assert!(
        daemon.socket_path().exists(),
        "daemon socket should be bound"
    );

    // Connect and perform the handshake the interposer would.
    let mut stream = UnixStream::connect(daemon.socket_path()).expect("connect to daemon");
    stream
        .write_all(&header(0 /* HANDSHAKE */, 1, 0, 0))
        .expect("send handshake");

    let mut resp = [0u8; 16];
    read_exact(&mut stream, &mut resp);
    let payload_bytes = u32::from_ne_bytes(resp[8..12].try_into().unwrap()) as usize;
    let result = i32::from_ne_bytes(resp[12..16].try_into().unwrap());
    assert_eq!(result, 0, "handshake should succeed");
    assert!(
        payload_bytes >= 16,
        "handshake payload should carry the response struct"
    );

    let mut payload = vec![0u8; payload_bytes];
    read_exact(&mut stream, &mut payload);
    let version = u32::from_ne_bytes(payload[0..4].try_into().unwrap());
    let topo_len = u32::from_ne_bytes(payload[8..12].try_into().unwrap()) as usize;
    let drm_len = u32::from_ne_bytes(payload[12..16].try_into().unwrap()) as usize;
    assert_eq!(version, 3, "protocol version should match the interposer");
    // RpcHandshakeResponse is 328 bytes (16 fixed fields + 312-byte
    // RpcGpuInfo), then the topology and DRM path strings.
    assert_eq!(
        payload_bytes,
        328 + topo_len + drm_len,
        "handshake payload framing should be self-consistent"
    );
    assert!(topo_len > 0, "daemon should report a sysfs topology path");

    // Cleanly close the client session.
    stream
        .write_all(&header(2 /* CLOSE */, 2, 0, 0))
        .expect("send close");
    let mut close_resp = [0u8; 16];
    read_exact(&mut stream, &mut close_resp);

    // Shutting the daemon down removes its socket.
    let socket_path = daemon.socket_path().to_path_buf();
    daemon.stop();
    assert!(
        !socket_path.exists(),
        "daemon socket should be removed on shutdown"
    );
}

/// Dropping the handle must tear the daemon down just as `stop` does, so
/// an unwind past the owner cannot leak the simulated device or leave a
/// stale socket behind.
#[test]
fn dropping_the_handle_stops_the_daemon() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let Some(lib) = kmd_preload() else {
        eprintln!("rocjitsu KMD library not found; skipping daemon test");
        return;
    };
    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report.iter().map(|(n, _)| n.clone()).next().unwrap();
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
    let config = kmd_config(&def, tmp.path()).unwrap();
    let runtime_dir = tmp.path().join("rt");

    let socket_path = {
        let Some(daemon) = start_or_skip(&lib, &config, &runtime_dir) else {
            return;
        };
        let path = daemon.socket_path().to_path_buf();
        assert!(path.exists());
        path
    };
    assert!(
        !socket_path.exists(),
        "dropping the daemon handle must remove its socket"
    );
}

/// A second handshake on a fresh connection must also succeed: the
/// daemon serves many clients over its lifetime (one per workload
/// open of `/dev/kfd`).
#[test]
fn daemon_serves_multiple_clients() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let Some(lib) = kmd_preload() else {
        eprintln!("rocjitsu KMD library not found; skipping daemon test");
        return;
    };

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report.iter().map(|(n, _)| n.clone()).next().unwrap();
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
    let config = kmd_config(&def, tmp.path()).unwrap();
    let runtime_dir = tmp.path().join("rt");
    let Some(daemon) = start_or_skip(&lib, &config, &runtime_dir) else {
        return;
    };
    assert_eq!(daemon.status(), RjDaemonStatus::Running);

    for req_id in 0..3u32 {
        let mut stream = UnixStream::connect(daemon.socket_path()).expect("connect");
        stream.write_all(&header(0, req_id, 0, 0)).unwrap();
        let mut resp = [0u8; 16];
        read_exact(&mut stream, &mut resp);
        let payload_bytes = u32::from_ne_bytes(resp[8..12].try_into().unwrap()) as usize;
        assert!(payload_bytes >= 16, "client {req_id} handshake framed");
        let mut payload = vec![0u8; payload_bytes];
        read_exact(&mut stream, &mut payload);
        stream.write_all(&header(2, req_id, 0, 0)).unwrap();
        let mut close_resp = [0u8; 16];
        read_exact(&mut stream, &mut close_resp);
    }
}
