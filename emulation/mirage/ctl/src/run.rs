//! `mirage run` and `mirage exec`: the two commands that start processes.
//!
//! They look symmetrical and are not, and the asymmetry is the whole
//! design:
//!
//! * `mirage run` **owns** a session. It brings one up in its own
//!   process, serves a socket so other terminals can find it, runs the
//!   command, and tears everything down on the way out. Nothing survives
//!   it.
//! * `mirage exec` **borrows** one. It asks a live run to describe its
//!   session, then starts the processes itself, in its own terminal, as
//!   its own children.
//!
//! The reason `exec` spawns locally rather than asking the run to do it
//! is terminals. A child inherits the standard streams of whoever forked
//! it, so a process started by the run process would talk to the run's
//! terminal — not to the terminal the user typed `mirage exec` into.
//! Having the client spawn is what makes `mirage exec <run> -- bash` an
//! interactive shell in the window you ran it from, with no
//! pseudo-terminal, no output forwarding and no stdin relay.
//!
//! Both build their process grid with the same
//! [`mirage_supervisor::build_specs`], from the same description, so a
//! command behaves identically whichever way it was started.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use mirage_core::exec::{ExecArgs, ExecDef, ExecId};
use mirage_core::proto::SessionDescription;
use mirage_core::session::{CreateSessionRequest, SessionId};
use mirage_supervisor::{Exec, Run, rpc::ControlSocket};
use tokio::sync::mpsc;

use crate::{ExecArgsCli, RunArgs, apply_profile_overrides, parse_envs, split_argv};

/// How long a session gets to become healthy before `run` gives up.
///
/// Time spent pulling or building an image is not counted against it;
/// see [`Run::wait_ready`].
const READY_TIMEOUT: Duration = Duration::from_secs(60);

/// `mirage run`: own a session for the lifetime of one command.
pub async fn run_cmd(a: RunArgs) -> anyhow::Result<ExitCode> {
    let mut profile = mirage_core::store::profile_get(&a.profile)?;
    let profile_ref = apply_profile_overrides(
        &mut profile,
        a.image.clone(),
        &a.mounts,
        &a.ports,
        a.container_provider.clone(),
        a.emulator.clone(),
        a.exec_mode,
        &a.options,
        &a.plugins,
        a.config.clone(),
        a.num_nodes,
        a.gpus_per_node,
        &a.hacks,
        &a.profile,
    )?;

    let workdir = a.workdir.clone().unwrap_or_else(|| {
        std::env::current_dir().map_or_else(|_| "/".to_string(), |p| p.display().to_string())
    });

    let run = Arc::new(Run::start(CreateSessionRequest {
        id: None,
        profile: profile_ref,
        workdir: workdir.clone(),
        daemon: !a.in_process,
    })?);
    let session = run.id().clone();
    eprintln!("mirage: session {session}");

    // From here on every exit path must go through teardown, including
    // the error ones: a session whose bring-up half-succeeded still has
    // containers to remove.
    let outcome = run_owned(&run, &a, &session, &workdir).await;
    run.destroy().await;
    outcome
}

/// The body of `mirage run`, with teardown guaranteed by the caller.
async fn run_owned(
    run: &Arc<Run>,
    a: &RunArgs,
    session: &SessionId,
    workdir: &str,
) -> anyhow::Result<ExitCode> {
    let health = run.wait_ready(READY_TIMEOUT).await?;
    if !health.healthy {
        let state = health.state.as_deref().unwrap_or("unknown");
        match health.message {
            Some(msg) => anyhow::bail!("session failed to start ({state}): {msg}"),
            None => anyhow::bail!("session failed to start ({state})"),
        }
    }

    // Serve the description socket only once the session is actually
    // ready. A client that connects earlier would get a description with
    // no containers and no emulator environment in it, and would happily
    // start a workload straight onto the real host.
    let socket = ControlSocket::bind(&mirage_core::paths::run_socket_path(session)).await?;

    let (cmd, args) = split_argv(&a.argv);
    let def = ExecDef {
        timestamp: chrono::Utc::now(),
        session: session.clone(),
        exec: ExecArgs {
            command: cmd,
            args,
            env: parse_envs(&a.envs)?,
            workdir: a.workdir.clone(),
        },
        worker_exec: None,
        nproc_per_node: a.nproc_per_node.unwrap_or(1).max(1),
        capture_all: a.capture_all,
    };
    let _ = workdir;

    let (exec, output) = run.exec(&def).await?;

    // Serve the socket for as long as the workload runs. `select!` rather
    // than a spawned task so the server stops when the workload does,
    // without a second thing to cancel.
    let serving = socket.serve(run.clone());
    tokio::pin!(serving);

    tokio::select! {
        code = supervise_locally(exec, output, !def.capture_all) => code,
        () = &mut serving => unreachable!("the control socket serves until dropped"),
    }
}

/// `mirage exec`: run a command inside a session someone else owns.
pub async fn exec_cmd(a: ExecArgsCli) -> anyhow::Result<ExitCode> {
    let session = match a.session.clone() {
        Some(id) => id,
        None => sole_live_run()?,
    };
    let desc = describe(&session).await?;

    let (cmd, args) = split_argv(&a.argv);
    let def = ExecDef {
        timestamp: chrono::Utc::now(),
        session: session.clone(),
        exec: ExecArgs {
            command: cmd,
            args,
            env: parse_envs(&a.envs)?,
            workdir: a.workdir.clone(),
        },
        worker_exec: None,
        nproc_per_node: a.nproc_per_node.unwrap_or(1).max(1),
        capture_all: a.capture_all,
    };

    // A client-side exec id, distinct from anything the run process is
    // using. It only names this command's pid files, and two execs in
    // different processes must not collide on them.
    let id = ExecId::new(format!("x-{}", std::process::id()))
        .map_err(|e| anyhow::anyhow!("could not build an exec id: {e}"))?;
    let specs = mirage_supervisor::build_specs(&desc, &def, &id)?;
    let capture_all = def.capture_all;
    let (exec, output) = Exec::start(id, def, specs);

    supervise_locally(exec, output, !capture_all).await
}

/// Run an exec to completion in this terminal, printing captured output
/// and stopping it cleanly if we are interrupted.
///
/// Returns the exit code this process should use.
async fn supervise_locally(
    exec: Arc<Exec>,
    output: mpsc::Receiver<mirage_supervisor::OutputChunk>,
    owns_terminal: bool,
) -> anyhow::Result<ExitCode> {
    // Always drain the channel. Under `--capture-all` this is what prints
    // the labelled output; otherwise it is empty and finishes at once,
    // because nothing was piped.
    let printer = tokio::spawn(mirage_supervisor::output::print_labelled(output));

    // Put rank 0 in the foreground, so an interactive program can read
    // the terminal at all. Dropped — and so given back — on every exit
    // path below, including the interrupted one.
    let _terminal = owns_terminal
        .then(|| exec.rank_zero_pid().map(TerminalHandoff::give_to))
        .flatten()
        .flatten();

    // Ctrl-C reaches us, not the workload: children lead their own
    // process groups, so the terminal's foreground group is this process
    // alone. Forwarding it deliberately — and then falling through to the
    // normal wait — is what makes a workload get a chance to clean up,
    // and what makes the caller's teardown run rather than being skipped
    // by an abrupt exit.
    let interrupted = async {
        let sig = interrupt().await;
        exec.signal(sig).await.ok();
        // A second interrupt means the user is not waiting any longer.
        interrupt().await;
    };

    tokio::select! {
        () = exec.wait_finished() => {}
        () = interrupted => {
            exec.terminate().await;
        }
    }

    // Wait for the printer so the last lines are on screen before we
    // return and the caller starts tearing the session down.
    let _ = printer.await;

    let code = exec.status().exit_code.unwrap_or(0);
    // Exit codes are a byte. Masking preserves the shell's `128 + signal`
    // convention for a signal-killed workload rather than saturating it.
    Ok(ExitCode::from((code & 0xff) as u8))
}

/// Resolve when this process is asked to stop, yielding the signal number.
///
/// `SIGTERM` as well as `SIGINT`: a CI runner cancelling a job, or a
/// shell script's `kill`, sends the former, and both have to reach the
/// cleanup path rather than killing mirage outright and stranding a
/// container.
async fn interrupt() -> i32 {
    use tokio::signal::unix::{SignalKind, signal};
    let (Ok(mut sigint), Ok(mut sigterm)) = (
        signal(SignalKind::interrupt()),
        signal(SignalKind::terminate()),
    ) else {
        // Without handlers the default disposition applies and this
        // future must never win a race.
        std::future::pending::<()>().await;
        unreachable!()
    };
    tokio::select! {
        _ = sigint.recv() => libc::SIGINT,
        _ = sigterm.recv() => libc::SIGTERM,
    }
}

/// Ask the run that owns `session` to describe it.
async fn describe(session: &SessionId) -> anyhow::Result<SessionDescription> {
    use futures::{SinkExt as _, StreamExt as _};
    use mirage_core::proto::{Request, Response, codec};

    let path = mirage_core::paths::run_socket_path(session);
    let stream = tokio::net::UnixStream::connect(&path).await.map_err(|e| {
        anyhow::anyhow!(
            "no `mirage run` is serving session {session} ({e}). \
             A session exists only while the `mirage run` that created \
             it is alive."
        )
    })?;
    let mut framed = tokio_util::codec::Framed::new(stream, codec());

    let request = serde_json::to_vec(&Request::Describe)?;
    framed.send(request.into()).await?;

    let Some(frame) = framed.next().await else {
        anyhow::bail!("the run serving session {session} closed the connection without answering");
    };
    match serde_json::from_slice::<Response>(&frame?)? {
        Response::Description(desc) => Ok(*desc),
        Response::Error(message) => anyhow::bail!("{message}"),
    }
}

/// The session id of the only live run, when there is exactly one.
///
/// Making the argument optional is not a shortcut: the overwhelmingly
/// common case is one run in one terminal and an exec in another, and
/// requiring the user to copy an id for it would be friction with no
/// purpose. When the guess would be ambiguous the error lists the
/// candidates rather than picking one.
fn sole_live_run() -> anyhow::Result<SessionId> {
    let live = live_runs();
    match live.len() {
        1 => Ok(live[0].clone()),
        0 => anyhow::bail!(
            "no `mirage run` is running. Start one in another terminal, \
             or name a session explicitly."
        ),
        _ => {
            let names: Vec<&str> = live.iter().map(SessionId::as_str).collect();
            anyhow::bail!(
                "several runs are live ({}); name the one you mean, \
                 e.g. `mirage exec {} -- <command>`",
                names.join(", "),
                names[0]
            )
        }
    }
}

/// How many runs are currently serving a socket.
#[must_use]
pub fn live_run_count() -> usize {
    live_runs().len()
}

/// Every session with a socket in the runtime directory.
///
/// A socket may be stale — left by a run that was `SIGKILL`ed — and this
/// does not filter those out; connecting to one fails with a clear
/// message, which is a better outcome than silently ignoring a session
/// the user believes is alive.
fn live_runs() -> Vec<SessionId> {
    let Ok(entries) = std::fs::read_dir(mirage_core::paths::run_socket_root()) else {
        return Vec::new();
    };
    let mut ids: Vec<SessionId> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension()? != "sock" {
                return None;
            }
            SessionId::new(path.file_stem()?.to_str()?).ok()
        })
        .collect();
    ids.sort();
    ids
}

/// Hands the controlling terminal to a workload, and takes it back.
///
/// # Why this is necessary
///
/// Every workload process leads its own process group, so that mirage can
/// signal a forking workload's whole tree as a unit. The kernel's job
/// control rules then apply: a process in a *background* process group
/// that reads the controlling terminal is sent `SIGTTIN` and stopped. An
/// interactive `bash` started by `mirage run -- bash` would hang on the
/// first keystroke — not slowly, not intermittently, but every time.
///
/// The fix is the one a shell uses to put a job in the foreground:
/// `tcsetpgrp` the workload's process group onto the terminal, and take
/// it back when the workload is done. With the terminal theirs, reads
/// work, `Ctrl-C` and `Ctrl-Z` are delivered to the workload rather than
/// to mirage, and a shell's own job control works inside it.
///
/// # SIGTTOU
///
/// Giving the terminal *back* is itself a background write to it, which
/// earns `SIGTTOU` and would stop mirage exactly when it is trying to
/// clean up. `SIGTTOU` is therefore blocked for the duration. Blocking —
/// rather than installing a handler — is what makes `tcsetpgrp` simply
/// succeed, and it is available as a safe API.
struct TerminalHandoff {
    /// The process group to restore, i.e. mirage's own.
    restore: nix::unistd::Pid,
    /// The mask to put back when we are done.
    previous_mask: nix::sys::signal::SigSet,
}

impl TerminalHandoff {
    /// Give the terminal to the process group led by `pid`.
    ///
    /// Returns `None` when there is nothing to do: stdin is not a
    /// terminal (a pipe, a CI runner, `< /dev/null`), or mirage is not
    /// itself in the foreground. Neither is an error — a workload reading
    /// a pipe needs no handoff, and job control does not apply.
    fn give_to(pid: u32) -> Option<Self> {
        use std::io::IsTerminal as _;
        use std::os::fd::AsFd as _;

        let stdin = std::io::stdin();
        if !stdin.is_terminal() {
            return None;
        }
        let restore = nix::unistd::getpgrp();
        // Only the foreground group may hand the terminal on. If mirage
        // is already in the background, the terminal is somebody else's
        // and taking it would be a hijack.
        if nix::unistd::tcgetpgrp(stdin.as_fd()).ok()? != restore {
            return None;
        }

        let mut ttou = nix::sys::signal::SigSet::empty();
        ttou.add(nix::sys::signal::Signal::SIGTTOU);
        let previous_mask = nix::sys::signal::pthread_sigmask(
            nix::sys::signal::SigmaskHow::SIG_BLOCK,
            Some(&ttou),
            None,
        )
        .ok()
        .and_then(|()| nix::sys::signal::SigSet::thread_get_mask().ok())
        .unwrap_or_else(nix::sys::signal::SigSet::empty);

        let target = nix::unistd::Pid::from_raw(i32::try_from(pid).ok()?);
        if nix::unistd::tcsetpgrp(stdin.as_fd(), target).is_err() {
            return None;
        }
        Some(Self {
            restore,
            previous_mask,
        })
    }
}

impl Drop for TerminalHandoff {
    fn drop(&mut self) {
        use std::os::fd::AsFd as _;
        let stdin = std::io::stdin();
        // Best effort: if this fails the terminal is left with a
        // foreground group that has exited, which the kernel resolves on
        // the next read. Nothing here may panic — it runs on the failure
        // path too.
        let _ = nix::unistd::tcsetpgrp(stdin.as_fd(), self.restore);
        let _ = nix::sys::signal::pthread_sigmask(
            nix::sys::signal::SigmaskHow::SIG_SETMASK,
            Some(&self.previous_mask),
            None,
        );
    }
}
