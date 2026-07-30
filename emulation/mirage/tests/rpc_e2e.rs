//! Tests for the Unix-socket control plane itself.
//!
//! The other suites go through the CLI and so exercise the protocol
//! incidentally. These target it directly: daemon startup and ownership,
//! the version handshake, malformed input, connection lifetime, and the
//! attach stream's duplex behaviour. Those are the paths where a bug
//! shows up as a hang or a confusing error rather than a failed command.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod harness;

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use harness::{Env, marker, pid_alive, skip_without_emulator, tagged_sleep, wait_for};
use mirage_core::proto::{PROTOCOL_VERSION, Request, Response, codec};
use tokio::net::UnixStream;
use tokio_util::codec::Framed;

/// Open a framed connection to an environment's daemon.
async fn connect(env: &Env) -> Framed<UnixStream, tokio_util::codec::LengthDelimitedCodec> {
    let stream = UnixStream::connect(env.socket()).await.unwrap();
    Framed::new(stream, codec())
}

/// Send one request and read one response.
async fn exchange(env: &Env, request: &Request) -> Response {
    let mut framed = connect(env).await;
    framed
        .send(serde_json::to_vec(request).unwrap().into())
        .await
        .unwrap();
    let frame = framed.next().await.expect("a response").unwrap();
    serde_json::from_slice(&frame).unwrap()
}

/// Make sure a daemon is running for this environment.
fn ensure_daemon(env: &Env) {
    env.ok(&["paths"]);
    // `paths` is answered locally, so force a round trip.
    env.ok(&["session", "list"]);
}

#[tokio::test]
async fn the_cli_starts_a_daemon_on_demand() {
    let env = Env::new();
    assert!(!env.daemon_running(), "no daemon should exist yet");

    // Any command needing the control plane brings one up. Requiring the
    // user to start it by hand would make the common case a two-step.
    env.ok(&["session", "list"]);
    assert!(env.daemon_running());
    assert!(env.socket().exists());
}

#[tokio::test]
async fn configuration_commands_do_not_start_a_daemon() {
    // A background process appearing because you ran `mirage profile
    // list` is a surprise the user did not ask for. These are answered
    // in-process from the config store and the link-time registry.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    for args in [
        vec!["paths"],
        vec!["emulators"],
        vec!["profile", "list"],
        vec!["agent", "list"],
        vec!["topology", "list"],
    ] {
        env.ok(&args);
        assert!(
            !env.socket().exists(),
            "`mirage {}` started a daemon",
            args.join(" ")
        );
    }

    // Writing configuration is equally daemon-free; the daemon reads
    // profiles off disk when it needs them, so there is no cached copy to
    // keep coherent.
    env.create_profile("written-offline");
    assert!(!env.socket().exists());
    assert!(env.ok(&["profile", "list"]).contains("written-offline"));

    // But anything touching a session does start one.
    env.ok(&["session", "list"]);
    assert!(env.socket().exists(), "a session command must reach a daemon");
}

#[tokio::test]
async fn a_profile_written_offline_is_visible_to_the_daemon() {
    // The two paths write and read the same files, so there is no window
    // in which the CLI and the daemon disagree.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("shared");
    let id = env.start_session("shared", "uses-it");
    assert_eq!(id, "uses-it");
    env.ok(&["session", "stop", "uses-it"]);
}

#[tokio::test]
async fn autostart_can_be_disabled() {
    let env = Env::new();
    let out = env
        .mirage()
        .env("MIRAGE_AUTOSTART", "0")
        .args(["session", "list"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no mirage daemon is running"), "{err}");
    // And nothing was started behind the user's back.
    assert!(!env.socket().exists());
}

#[tokio::test]
async fn only_one_daemon_owns_the_socket() {
    let env = Env::new();
    ensure_daemon(&env);
    let first = env.daemon_pid().unwrap();

    // A second daemon on the same socket must defer rather than steal it
    // or fail confusingly. Two daemons would mean two disjoint views of
    // "every session", which nothing downstream could reconcile.
    let out = env
        .mirage()
        .args(["daemon", "--idle-timeout", "1"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", out);

    assert_eq!(
        env.daemon_pid().unwrap(),
        first,
        "the original daemon must still own the socket"
    );
}

#[tokio::test]
async fn racing_clients_produce_exactly_one_daemon() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // Several CLIs starting at once all want a daemon. The lock is what
    // makes exactly one win; without it they would race to bind and the
    // losers would fail with EADDRINUSE.
    let handles: Vec<_> = (0..6)
        .map(|_| {
            let mut cmd = env.mirage();
            cmd.args(["session", "list"]);
            std::thread::spawn(move || cmd.output().unwrap())
        })
        .collect();
    for h in handles {
        let out = h.join().unwrap();
        assert!(
            out.status.success(),
            "a racing client failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert!(env.daemon_running());
}

#[tokio::test]
async fn the_version_handshake_accepts_a_matching_client() {
    let env = Env::new();
    ensure_daemon(&env);
    let response = exchange(
        &env,
        &Request::Hello {
            version: PROTOCOL_VERSION,
        },
    )
    .await;
    match response {
        Response::Hello { version, .. } => assert_eq!(version, PROTOCOL_VERSION),
        other => panic!("expected a Hello, got {other:?}"),
    }
}

#[tokio::test]
async fn the_version_handshake_rejects_a_mismatched_client() {
    let env = Env::new();
    ensure_daemon(&env);

    // A CLI and a daemon from different builds meet routinely: the daemon
    // is long-lived and auto-started, so upgrading mirage leaves the old
    // one running. Misreading its frames would be far worse than a clear
    // refusal, and the message has to say how to fix it.
    let response = exchange(
        &env,
        &Request::Hello {
            version: PROTOCOL_VERSION + 99,
        },
    )
    .await;
    match response {
        Response::Error { message, .. } => {
            assert!(message.contains("protocol version mismatch"), "{message}");
            assert!(message.contains("daemon stop"), "must say how to recover: {message}");
        }
        other => panic!("expected an error, got {other:?}"),
    }
}

#[tokio::test]
async fn a_malformed_frame_gets_an_error_not_a_dropped_connection() {
    let env = Env::new();
    ensure_daemon(&env);

    let mut framed = connect(&env).await;
    framed.send(b"this is not json".to_vec().into()).await.unwrap();
    let frame = framed
        .next()
        .await
        .expect("the daemon must answer rather than hang up")
        .unwrap();
    let response: Response = serde_json::from_slice(&frame).unwrap();
    match response {
        Response::Error { message, .. } => assert!(message.contains("malformed"), "{message}"),
        other => panic!("expected an error, got {other:?}"),
    }

    // And the daemon is still healthy afterwards.
    assert!(env.daemon_running());
}

#[tokio::test]
async fn a_client_that_disconnects_mid_request_does_not_disturb_the_daemon() {
    let env = Env::new();
    ensure_daemon(&env);

    for _ in 0..20 {
        let mut framed = connect(&env).await;
        framed
            .send(serde_json::to_vec(&Request::SessionList).unwrap().into())
            .await
            .unwrap();
        // Drop without reading the reply.
        drop(framed);
    }

    assert!(env.daemon_running(), "the daemon died on a rude client");
    env.ok(&["session", "list"]);
}

#[tokio::test]
async fn errors_keep_their_kind_across_the_wire() {
    let env = Env::new();
    ensure_daemon(&env);

    // The CLI branches on the kind — "already gone" is success for an
    // idempotent cleanup, a real failure otherwise — so flattening
    // everything to a string would lose information the caller needs.
    let response = exchange(
        &env,
        &Request::SessionState {
            id: mirage_core::session::SessionId::new("ghost").unwrap(),
        },
    )
    .await;
    match response {
        Response::Error { kind, message } => {
            assert_eq!(kind, mirage_core::proto::ErrorKind::SessionNotFound);
            assert!(message.contains("ghost"), "{message}");
        }
        other => panic!("expected a not-found error, got {other:?}"),
    }
}

#[tokio::test]
async fn daemon_status_reports_what_it_owns() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "counted");

    let response = exchange(&env, &Request::DaemonStatus).await;
    match response {
        Response::DaemonStatus { pid, sessions, .. } => {
            assert!(pid_alive(pid));
            assert_eq!(sessions, 1);
        }
        other => panic!("expected a status, got {other:?}"),
    }

    env.ok(&["session", "stop", "counted"]);
}

#[tokio::test]
async fn attach_streams_output_and_ends_with_an_exit() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "att");

    let exec_ref = mirage_core::exec::ExecRef {
        session: mirage_core::session::SessionId::new("att").unwrap(),
        exec: mirage_core::exec::ExecId::from_counter(0),
    };
    env.ok(&[
        "exec",
        "start",
        "att",
        "--detach",
        "--keep",
        "--",
        "/bin/sh",
        "-c",
        "echo streamed; exit 4",
    ]);

    let mut framed = connect(&env).await;
    framed
        .send(
            serde_json::to_vec(&Request::Attach {
                exec: exec_ref.clone(),
            })
            .unwrap()
            .into(),
        )
        .await
        .unwrap();

    let mut output = Vec::new();
    let mut exit = None;
    let mut saw_end = false;
    while let Ok(Some(frame)) = tokio::time::timeout(Duration::from_secs(30), framed.next())
        .await
        .map(|f| f.transpose().unwrap())
    {
        match serde_json::from_slice::<Response>(&frame).unwrap() {
            Response::Stream(mirage_core::ctl::StreamPacket::Output { data, .. }) => {
                output.extend_from_slice(&data);
            }
            Response::Stream(mirage_core::ctl::StreamPacket::ExecExit { exit_code }) => {
                exit = Some(exit_code);
            }
            Response::StreamEnd => {
                saw_end = true;
                break;
            }
            _ => {}
        }
    }
    assert_eq!(exit, Some(4));
    assert!(saw_end, "the server must mark the end of the stream");
    assert!(String::from_utf8_lossy(&output).contains("streamed"));

    env.ok(&["session", "stop", "att"]);
}

#[tokio::test]
async fn stdin_and_signals_travel_on_the_attach_connection() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "duplex");
    let tag = marker("duplex");

    env.ok(&[
        "exec",
        "start",
        "duplex",
        "--detach",
        "--keep",
        "--",
        "/bin/sh",
        "-c",
        &tagged_sleep(&tag),
    ]);
    wait_for("the workload to start", Duration::from_secs(15), || {
        harness::count_processes(&tag) > 0
    });

    let exec_ref = mirage_core::exec::ExecRef {
        session: mirage_core::session::SessionId::new("duplex").unwrap(),
        exec: mirage_core::exec::ExecId::from_counter(0),
    };

    let mut framed = connect(&env).await;
    framed
        .send(
            serde_json::to_vec(&Request::Attach {
                exec: exec_ref.clone(),
            })
            .unwrap()
            .into(),
        )
        .await
        .unwrap();

    // Signalling on the *same* connection is what keeps a Ctrl-C ordered
    // against the output the user is reacting to.
    framed
        .send(
            serde_json::to_vec(&Request::ExecSignal {
                exec: exec_ref,
                sig: libc::SIGTERM,
            })
            .unwrap()
            .into(),
        )
        .await
        .unwrap();

    // The stream must terminate as a result.
    let mut ended = false;
    while let Ok(Some(frame)) = tokio::time::timeout(Duration::from_secs(30), framed.next())
        .await
        .map(|f| f.transpose().unwrap())
    {
        if matches!(
            serde_json::from_slice::<Response>(&frame).unwrap(),
            Response::StreamEnd
        ) {
            ended = true;
            break;
        }
    }
    assert!(ended, "signalling over the attach connection did not end the exec");

    env.ok(&["session", "stop", "duplex"]);
}

#[tokio::test]
async fn attaching_to_a_missing_exec_reports_it_rather_than_hanging() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "missing");

    let mut framed = connect(&env).await;
    framed
        .send(
            serde_json::to_vec(&Request::Attach {
                exec: mirage_core::exec::ExecRef {
                    session: mirage_core::session::SessionId::new("missing").unwrap(),
                    exec: mirage_core::exec::ExecId::from_counter(99),
                },
            })
            .unwrap()
            .into(),
        )
        .await
        .unwrap();

    let frame = tokio::time::timeout(Duration::from_secs(10), framed.next())
        .await
        .expect("attaching to a missing exec must answer, not hang")
        .expect("a response")
        .unwrap();
    match serde_json::from_slice::<Response>(&frame).unwrap() {
        Response::Error { kind, .. } => {
            assert_eq!(kind, mirage_core::proto::ErrorKind::ExecNotFound);
        }
        other => panic!("expected a not-found error, got {other:?}"),
    }

    env.ok(&["session", "stop", "missing"]);
}

#[tokio::test]
async fn attaching_to_a_silent_exec_is_acknowledged_immediately() {
    // Whether an attach was accepted has to be answerable before the
    // workload says anything, and a long-running job may say nothing for
    // hours. Without an acknowledgement frame a client cannot distinguish
    // "attached to a silent `sleep`" from "rejected, no such exec" — so
    // it either blocks or, as the CLI did, treats a refusal as an empty
    // stream and reports success for an exec that does not exist.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "silent");
    let tag = marker("silent");

    env.ok(&[
        "exec", "start", "silent", "--detach", "--keep", "--", "/bin/sh", "-c",
        &tagged_sleep(&tag),
    ]);
    wait_for("the workload to start", Duration::from_secs(15), || {
        harness::count_processes(&tag) > 0
    });

    let mut framed = connect(&env).await;
    framed
        .send(
            serde_json::to_vec(&Request::Attach {
                exec: mirage_core::exec::ExecRef {
                    session: mirage_core::session::SessionId::new("silent").unwrap(),
                    exec: mirage_core::exec::ExecId::from_counter(0),
                },
            })
            .unwrap()
            .into(),
        )
        .await
        .unwrap();

    let frame = tokio::time::timeout(Duration::from_secs(10), framed.next())
        .await
        .expect("a successful attach must be acknowledged, not wait for output")
        .expect("a response")
        .unwrap();
    assert!(
        matches!(
            serde_json::from_slice::<Response>(&frame).unwrap(),
            Response::Ok
        ),
        "the daemon must acknowledge an accepted attach before any output"
    );

    env.ok(&["session", "stop", "silent"]);
}

#[tokio::test]
async fn the_daemon_exits_when_idle() {
    let env = Env::new();
    // A background process with no exit condition is the thing users
    // resent about the previous design's stray hosts.
    let mut child = env
        .mirage()
        .args(["daemon", "--idle-timeout", "1"])
        .spawn()
        .unwrap();

    wait_for("the daemon to bind", Duration::from_secs(15), || {
        env.socket().exists()
    });

    let status = wait_for_exit(&mut child, Duration::from_secs(30));
    assert!(
        status.is_some(),
        "an idle daemon must exit rather than linger forever"
    );
}

#[tokio::test]
async fn the_idle_timeout_does_not_fire_while_a_session_exists() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let mut child = env
        .mirage()
        .args(["daemon", "--idle-timeout", "1"])
        .spawn()
        .unwrap();
    wait_for("the daemon to bind", Duration::from_secs(15), || {
        env.socket().exists()
    });

    env.create_profile("p");
    env.start_session("p", "busy");

    // A session that is merely idle is still a session someone expects to
    // find later; only an empty daemon may exit.
    std::thread::sleep(Duration::from_secs(4));
    assert!(
        wait_for_exit(&mut child, Duration::from_millis(200)).is_none(),
        "the daemon exited while it still owned a session"
    );
    assert!(env.ok(&["session", "list"]).contains("busy"));

    env.ok(&["session", "stop", "busy"]);
    let _ = child.kill();
    let _ = child.wait();
}

/// Wait up to `timeout` for `child` to exit.
fn wait_for_exit(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Option<std::process::ExitStatus> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    return None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return None,
        }
    }
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the rpc suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
