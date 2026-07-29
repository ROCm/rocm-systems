//! End-to-end tests for the daemon's HTTP/WebSocket API.
//!
//! Each test spawns a real `mirage daemon --addr` subprocess against a
//! fresh temp XDG root and exercises the REST + WS surface with `reqwest`
//! and `tokio-tungstenite`. The daemon owns its sessions in-process, so
//! these cover the same manager the CLI reaches over the Unix socket —
//! and, by construction, prove the two surfaces agree.
//!
//! Gated on the `webui` feature: the HTTP API and the served SPA only
//! exist when the web UI is compiled in.
#![cfg(feature = "webui")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio_tungstenite::tungstenite::Message;

mod harness;
use harness::{TEST_EMULATOR, skip_without_emulator};

/// Pick an unused localhost port. Subject to a race but acceptable
/// for tests.
fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

struct Daemon {
    _dir: TempDir,
    child: Child,
    base: String,
    #[allow(dead_code)]
    config: PathBuf,
    #[allow(dead_code)]
    runtime: PathBuf,
    socket: PathBuf,
}

impl Daemon {
    fn spawn() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config");
        let runtime = dir.path().join("runtime");
        let state = dir.path().join("state");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::create_dir_all(&runtime).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let mirage_bin = PathBuf::from(env!("CARGO_BIN_EXE_mirage"));
        let port = free_port();
        let addr = format!("127.0.0.1:{port}");
        // A short socket path: `sun_path` is a fixed 108-byte array, and
        // a socket nested under a long tempdir silently fails to bind.
        let socket = std::env::temp_dir().join(format!("mrgd-{}-{port}.sock", std::process::id()));
        let child = Command::new(&mirage_bin)
            .args([
                "daemon",
                "--addr",
                &addr,
                "--socket",
                socket.to_str().unwrap(),
                // Never exit mid-test.
                "--idle-timeout",
                "0",
            ])
            .env("XDG_CONFIG_HOME", &config)
            .env("XDG_RUNTIME_DIR", &runtime)
            .env("XDG_STATE_HOME", &state)
            .env("MIRAGE_BIN", &mirage_bin)
            .env_remove("MIRAGE_LOG")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn mirage daemon");
        let base = format!("http://{addr}");
        let d = Daemon {
            _dir: dir,
            child,
            base,
            config,
            runtime,
            socket,
        };
        d.wait_ready();
        d
    }

    fn wait_ready(&self) {
        let deadline = Instant::now() + Duration::from_secs(15);
        let url = format!("{}/api/paths", self.base);
        while Instant::now() < deadline {
            if let Ok(resp) = reqwest::blocking::get(&url)
                && resp.status().is_success()
            {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("daemon never became reachable on {}", self.base);
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn ws_url(&self, path: &str) -> String {
        format!("ws://{}{path}", self.base.trim_start_matches("http://"))
    }

    fn get_json(&self, path: &str) -> (reqwest::StatusCode, Value) {
        let r = reqwest::blocking::get(self.url(path)).unwrap();
        let status = r.status();
        let v: Value = r.json().unwrap_or(Value::Null);
        (status, v)
    }

    fn post_json(&self, path: &str, body: &Value) -> (reqwest::StatusCode, Value) {
        let c = reqwest::blocking::Client::new();
        let r = c.post(self.url(path)).json(body).send().unwrap();
        let status = r.status();
        let v: Value = r.json().unwrap_or(Value::Null);
        (status, v)
    }

    fn put_json(&self, path: &str, body: &Value) -> (reqwest::StatusCode, Value) {
        let c = reqwest::blocking::Client::new();
        let r = c.put(self.url(path)).json(body).send().unwrap();
        let status = r.status();
        let v: Value = r.json().unwrap_or(Value::Null);
        (status, v)
    }

    fn delete(&self, path: &str) -> (reqwest::StatusCode, Value) {
        let c = reqwest::blocking::Client::new();
        let r = c.delete(self.url(path)).send().unwrap();
        let status = r.status();
        let v: Value = r.json().unwrap_or(Value::Null);
        (status, v)
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        // SIGTERM, then wait: the daemon tears every session down on the
        // way out, so waiting for it is what guarantees no workload
        // survives the test. There is no separate sweep for stray child
        // pids any more — with one owner that always reaps, there is
        // nothing left to sweep.
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(self.child.id() as i32),
            nix::sys::signal::Signal::SIGTERM,
        );
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
        let _ = std::fs::remove_file(self.socket.with_extension("lock"));
    }
}

fn create_profile(d: &Daemon, name: &str) {
    let prof = json!({
        "name": name,
        "description": "test profile",
        "emulator": {
            "emulator": TEST_EMULATOR,
            "plugins": {},
            "exec_mode": "Functional",
            "options": {},
            "topology": {"num_nodes": 1, "gpus_per_node": 1, "agent": "MI350X"}
        }
    });
    let (s, b) = d.put_json(&format!("/api/profiles/{name}"), &prof);
    assert!(s.is_success(), "put profile failed: {s} {b}");
}

#[test]
fn paths_endpoint_reports_overridden_dirs() {
    let d = Daemon::spawn();
    let (s, v) = d.get_json("/api/paths");
    assert!(s.is_success());
    assert!(
        v["config"]
            .as_str()
            .unwrap()
            .starts_with(d.config.to_str().unwrap()),
        "expected config under tempdir, got {v}"
    );
    assert!(
        v["runtime"]
            .as_str()
            .unwrap()
            .starts_with(d.runtime.to_str().unwrap())
    );
}

#[test]
fn profile_crud_via_http() {
    let d = Daemon::spawn();

    // Preserve the auto-seeded builtin profiles as the CRUD baseline.
    let (_, initial_profiles) = d.get_json("/api/profiles");
    let initial_count = initial_profiles
        .as_array()
        .expect("profile list should be an array")
        .len();

    // Create.
    create_profile(&d, "p1");

    // List.
    let (_, v) = d.get_json("/api/profiles");
    let names = v.as_array().expect("profile list should be an array");
    assert_eq!(names.len(), initial_count + 1);
    assert!(names.contains(&json!("p1")));

    // Get.
    let (_, v) = d.get_json("/api/profiles/p1");
    assert_eq!(v["name"], "p1");

    // Delete.
    let (s, _) = d.delete("/api/profiles/p1");
    assert!(s.is_success());
    let (_, v) = d.get_json("/api/profiles");
    assert_eq!(v, initial_profiles);
}

#[test]
fn missing_profile_returns_404() {
    let d = Daemon::spawn();
    let (s, _) = d.get_json("/api/profiles/nope");
    assert_eq!(s, reqwest::StatusCode::NOT_FOUND);
}

#[test]
fn invalid_session_id_returns_400() {
    let d = Daemon::spawn();
    let (s, _) = d.get_json("/api/sessions/..bad..");
    assert_eq!(s, reqwest::StatusCode::BAD_REQUEST);
}

#[test]
fn session_lifecycle_via_http() {
    if skip_without_emulator() {
        return;
    }
    let d = Daemon::spawn();
    create_profile(&d, "demo");

    // Create session (daemon spawns the host).
    let (s, v) = d.post_json(
        "/api/sessions",
        &json!({"profile": "demo", "ready_timeout": 10}),
    );
    assert!(s.is_success(), "create_session: {s} {v}");
    let sid = v["id"].as_str().unwrap().to_string();

    // List shows it.
    let (_, list) = d.get_json("/api/sessions");
    assert_eq!(list.as_array().unwrap().len(), 1);

    // Health endpoint reports healthy.
    let (_, state) = d.get_json(&format!("/api/sessions/{sid}"));
    assert!(state["health"]["healthy"].as_bool().unwrap_or(false));

    // Tear down.
    let (s, _) = d.delete(&format!("/api/sessions/{sid}"));
    assert!(s.is_success());

    // Now empty.
    let (_, list) = d.get_json("/api/sessions");
    assert_eq!(list.as_array().unwrap().len(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_attach_streams_output_via_websocket() {
    if skip_without_emulator() {
        return;
    }
    // Spawn the daemon on a blocking thread so its Drop (which waits
    // on the subprocess) doesn't run inside the tokio reactor.
    let d = tokio::task::spawn_blocking(Daemon::spawn).await.unwrap();
    let client = reqwest::Client::new();

    create_profile_async(&client, &d, "demo").await;

    // Create session.
    let v: Value = client
        .post(d.url("/api/sessions"))
        .json(&json!({"profile": "demo", "ready_timeout": 10}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let sid = v["id"].as_str().unwrap().to_string();

    // Create exec running a short command.
    let v: Value = client
        .post(d.url(&format!("/api/sessions/{sid}/execs")))
        .json(&json!({
            "command": "/bin/sh",
            "args": ["-c", "echo hello-from-exec; echo err 1>&2; exit 7"],
            "keep": true
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let exec_id = v["id"].as_str().unwrap().to_string();

    // Attach via websocket and collect packets until ExecExit.
    let url = d.ws_url(&format!("/api/sessions/{sid}/execs/{exec_id}/attach"));
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    let mut stdout = String::new();
    let mut stderr = String::new();
    let mut exit_code: Option<i32> = None;
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        let frame = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .ok()
            .flatten();
        let Some(Ok(msg)) = frame else { break };
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        if let Some(out) = v.get("Output") {
            let data: Vec<u8> = serde_json::from_value(out["data"].clone()).unwrap();
            let s = String::from_utf8_lossy(&data).to_string();
            match out["stream"].as_str().unwrap_or("") {
                "Stdout" => stdout.push_str(&s),
                "Stderr" => stderr.push_str(&s),
                _ => {}
            }
        } else if let Some(exit) = v.get("ExecExit") {
            exit_code = Some(exit["exit_code"].as_i64().unwrap() as i32);
            break;
        }
    }
    let _ = ws.send(Message::Close(None)).await;

    // The exec runs under a PTY, so stdout and stderr are merged onto the
    // terminal (the stdout stream), exactly as in a real terminal.
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("hello-from-exec"),
        "got output: {combined:?}"
    );
    assert!(combined.contains("err"), "got output: {combined:?}");
    assert_eq!(exit_code, Some(7));

    // Status now reflects ended.
    let status: Value = client
        .get(d.url(&format!("/api/sessions/{sid}/execs/{exec_id}")))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["ended"], true);
    assert_eq!(status["exit_code"], 7);

    // Cleanup the session.
    let _ = client
        .delete(d.url(&format!("/api/sessions/{sid}")))
        .send()
        .await;
    // Drop the daemon on a blocking thread for the same reason as spawn.
    tokio::task::spawn_blocking(move || drop(d)).await.unwrap();
}

async fn create_profile_async(client: &reqwest::Client, d: &Daemon, name: &str) {
    let prof = json!({
        "name": name,
        "description": "test profile",
        "emulator": {
            "emulator": TEST_EMULATOR,
            "plugins": {},
            "exec_mode": "Functional",
            "options": {},
            "topology": {"num_nodes": 1, "gpus_per_node": 1, "agent": "MI350X"}
        }
    });
    let r = client
        .put(d.url(&format!("/api/profiles/{name}")))
        .json(&prof)
        .send()
        .await
        .unwrap();
    assert!(
        r.status().is_success(),
        "put profile failed: {}",
        r.status()
    );
}

#[test]
fn signal_terminates_long_running_exec() {
    if skip_without_emulator() {
        return;
    }
    let d = Daemon::spawn();
    create_profile(&d, "demo");

    let (_, v) = d.post_json(
        "/api/sessions",
        &json!({"profile": "demo", "ready_timeout": 10}),
    );
    let sid = v["id"].as_str().unwrap().to_string();

    // Long-running sleep.
    let (_, v) = d.post_json(
        &format!("/api/sessions/{sid}/execs"),
        &json!({
            "command": "/bin/sh",
            "args": ["-c", "sleep 30"],
            "keep": true
        }),
    );
    let exec_id = v["id"].as_str().unwrap().to_string();

    // Wait for it to actually start (status.started==true).
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        let (_, st) = d.get_json(&format!("/api/sessions/{sid}/execs/{exec_id}"));
        if st["started"].as_bool().unwrap_or(false) {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // SIGKILL it.
    let (s, _) = d.post_json(
        &format!("/api/sessions/{sid}/execs/{exec_id}/signal"),
        &json!({"signal": libc::SIGKILL}),
    );
    assert!(s.is_success());

    // Wait for ended.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut ended = false;
    while Instant::now() < deadline {
        let (_, st) = d.get_json(&format!("/api/sessions/{sid}/execs/{exec_id}"));
        if st["ended"].as_bool().unwrap_or(false) {
            ended = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(ended, "exec did not end after signal");

    let _ = d.delete(&format!("/api/sessions/{sid}"));
}

#[test]
fn dashboard_spa_is_served() {
    let d = Daemon::spawn();
    let r = reqwest::blocking::get(d.url("/")).unwrap();
    assert!(r.status().is_success());
    let ct = r
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.starts_with("text/html"), "got content-type: {ct}");
    let body = r.text().unwrap();
    assert!(body.to_ascii_lowercase().contains("<html"));
}

#[test]
fn unknown_spa_route_falls_back_to_index() {
    let d = Daemon::spawn();
    let r = reqwest::blocking::get(d.url("/sessions/something")).unwrap();
    assert!(r.status().is_success());
    let body = r.text().unwrap();
    assert!(body.to_ascii_lowercase().contains("<html"));
}
