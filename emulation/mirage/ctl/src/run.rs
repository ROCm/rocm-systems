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
//! Having the client spawn is what makes `mirage exec -- bash` an
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
    // Before anything that could need cleaning up. See [`Interrupts`].
    let mut interrupts = Interrupts::install()?;

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

    // Claim the session id and its socket *before* bring-up starts.
    //
    // The socket is what makes this run visible to everything else —
    // `mirage exec` finds it, and `mirage state purge` uses it to decide
    // which containers are orphans. Binding it only once the session was
    // ready left a window, the whole length of an image pull, in which
    // this run did not exist as far as purge was concerned: it would
    // force-remove the containers being created here and delete the
    // scratch directory out from under them. Binding first closes it.
    //
    // Bound now, served later: nothing answers until the session is
    // healthy, so a client still cannot be handed a description with no
    // containers and no emulator environment in it.
    let session = SessionId::generate();
    let socket = ControlSocket::bind(&mirage_core::paths::run_socket_path(&session)).await?;

    let run = Arc::new(Run::start(CreateSessionRequest {
        id: Some(session.clone()),
        profile: profile_ref,
        workdir: workdir.clone(),
        daemon: !a.in_process,
    })?);
    eprintln!("mirage: session {session}");

    // From here on every exit path must go through teardown, including
    // the error ones: a session whose bring-up half-succeeded still has
    // containers to remove.
    let outcome = run_owned(&run, &a, &socket, &mut interrupts).await;
    run.destroy().await;
    outcome
}

/// The body of `mirage run`, with teardown guaranteed by the caller.
async fn run_owned(
    run: &Arc<Run>,
    a: &RunArgs,
    socket: &ControlSocket,
    interrupts: &mut Interrupts,
) -> anyhow::Result<ExitCode> {
    // Answer clients from the first instant, bring-up included.
    //
    // The socket is bound before this function is called, so a connect
    // succeeds immediately — and if nothing is accepting yet, the client
    // sits in the backlog with no timeout and no output for however long
    // an image pull takes. Serving from here turns that silent hang into
    // `session … is not ready (pulling)`, because `Session::describe`
    // refuses until the session is healthy.
    let serving = socket.serve(run.clone());
    tokio::pin!(serving);

    // Race bring-up against an interrupt. Pulling an image can take
    // minutes, and a user who changes their mind in the middle of it
    // should get their prompt back — with the half-built session removed
    // by the caller's teardown, not left for `mirage state purge`.
    let health = tokio::select! {
        health = run.wait_ready(READY_TIMEOUT) => health?,
        sig = interrupts.next() => {
            return Ok(ExitCode::from(u8::try_from(128 + sig).unwrap_or(130)));
        }
        () = &mut serving => unreachable!("the control socket serves until dropped"),
    };
    if !health.healthy {
        let state = health.state.as_deref().unwrap_or("unknown");
        match health.message {
            Some(msg) => anyhow::bail!("session failed to start ({state}): {msg}"),
            None => anyhow::bail!("session failed to start ({state})"),
        }
    }

    let session = run.id();
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
        clear_env: a.clear_env_vars,
    };

    let (exec, output) = run.exec(&def).await?;

    // Keep serving for as long as the workload runs. `select!` rather
    // than a spawned task so the server stops when the workload does,
    // without a second thing to cancel.
    tokio::select! {
        code = supervise_locally(exec, output, !def.capture_all, interrupts) => code,
        () = &mut serving => unreachable!("the control socket serves until dropped"),
    }
}

/// `mirage exec`: run a command inside a session someone else owns.
pub async fn exec_cmd(a: ExecArgsCli) -> anyhow::Result<ExitCode> {
    let mut interrupts = Interrupts::install()?;

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
        clear_env: a.clear_env_vars,
    };

    // A client-side exec id, distinct from anything the run process is
    // using. It only names this command's pid files, and two execs in
    // different processes must not collide on them.
    //
    // The pid alone is not enough: pids are recycled, the pid files live
    // in the run's scratch directory for as long as the *run* lasts, and
    // a long-lived run outlives many `mirage exec` invocations. A start
    // timestamp makes the id unique per invocation rather than per pid.
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let id = ExecId::new(format!("x-{}-{started}", std::process::id()))
        .map_err(|e| anyhow::anyhow!("could not build an exec id: {e}"))?;
    let specs = mirage_supervisor::build_specs(&desc, &def, &id)?;
    let capture_all = def.capture_all;
    let (exec, output) = Exec::start(id, def, specs);

    supervise_locally(exec, output, !capture_all, &mut interrupts).await
}

/// Run an exec to completion in this terminal, printing captured output
/// and stopping it cleanly if we are interrupted.
///
/// Returns the exit code this process should use.
async fn supervise_locally(
    exec: Arc<Exec>,
    output: mpsc::Receiver<mirage_supervisor::OutputChunk>,
    owns_terminal: bool,
    interrupts: &mut Interrupts,
) -> anyhow::Result<ExitCode> {
    // Always drain the channel. Under `--capture-all` this is what prints
    // the labelled output; otherwise it is empty and finishes at once,
    // because nothing was piped.
    let printer = tokio::spawn(mirage_supervisor::output::print_labelled(output));

    // Put rank 0 in the foreground, so an interactive program can read
    // the terminal at all. Dropped — and so given back — on every exit
    // path below, including the interrupted one.
    //
    // Only for a single-process job. Handing the terminal over makes rank
    // 0's process group the terminal's *foreground* group, and the tty
    // driver then delivers Ctrl-C to that group alone: not to mirage, and
    // not to the other ranks. On a grid that means Ctrl-C kills rank 0,
    // leaves ranks 1..N running, and never wakes the interrupt handling
    // below — so `wait_finished` never resolves and a second Ctrl-C goes
    // to a group that no longer exists. Keeping the terminal for a
    // multi-process job costs nothing (a grid is not an interactive
    // program) and makes Ctrl-C reach mirage, which forwards it to every
    // rank.
    let single_process = exec.live_pids().len() <= 1;
    let _terminal = (owns_terminal && single_process)
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
        let sig = interrupts.next().await;
        exec.signal(sig).await.ok();
        // A second interrupt means the user is not waiting any longer.
        interrupts.next().await;
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

/// The signals that must never kill mirage outright, armed once.
///
/// `SIGTERM` as well as `SIGINT`: a CI runner cancelling a job, or a
/// shell script's `kill`, sends the former, and both have to reach the
/// cleanup path.
///
/// # Why this is installed before anything else happens
///
/// A tokio signal handler is registered the first time a `Signal` stream
/// is *created*, and until then the signal keeps its default
/// disposition — which for both of these is "terminate immediately". Arm
/// them lazily, at the point the workload is awaited, and everything
/// before that point is unprotected: a Ctrl-C during a multi-minute
/// image pull killed mirage outright, with no teardown, leaving the
/// containers and the network it had already created behind. That is the
/// exact moment a user is most likely to press Ctrl-C.
///
/// Installing them first makes the guarantee unconditional: from the
/// first line of the command to the last, an interrupt is something
/// mirage handles rather than something that happens to it.
struct Interrupts {
    sigint: tokio::signal::unix::Signal,
    sigterm: tokio::signal::unix::Signal,
}

impl Interrupts {
    /// Arm the handlers.
    ///
    /// # Errors
    ///
    /// Returns an error if the handlers cannot be registered, which is
    /// worth failing on rather than continuing unprotected.
    fn install() -> anyhow::Result<Self> {
        use tokio::signal::unix::{SignalKind, signal};
        Ok(Self {
            sigint: signal(SignalKind::interrupt())?,
            sigterm: signal(SignalKind::terminate())?,
        })
    }

    /// Resolve on the next interrupt, yielding its signal number.
    ///
    /// A signal that arrived earlier is not lost: tokio buffers one per
    /// kind, so an interrupt during bring-up is delivered the moment
    /// anything waits for it.
    async fn next(&mut self) -> i32 {
        tokio::select! {
            _ = self.sigint.recv() => libc::SIGINT,
            _ = self.sigterm.recv() => libc::SIGTERM,
        }
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
                 e.g. `mirage exec --session {} -- <command>`",
                names.join(", "),
                names[0]
            )
        }
    }
}

/// The sessions whose sockets actually answer, removing the corpses.
///
/// [`live_runs`] lists socket *files*, and a file outlives a run that was
/// `SIGKILL`ed. That is the right answer for `mirage exec`, which wants to
/// report "that session is gone" rather than silently skip it, and the
/// wrong one for cleanup: a corpse socket would make `purge` refuse to run
/// precisely when the crash it exists to clean up after has happened.
///
/// Connecting is the same liveness test
/// [`ControlSocket::bind`](mirage_supervisor::rpc::ControlSocket::bind)
/// uses, and a socket nothing answers on is unlinked here so the next
/// caller does not have to re-test it.
pub async fn answering_runs() -> Vec<SessionId> {
    let mut answering = Vec::new();
    for id in live_runs() {
        let path = mirage_core::paths::run_socket_path(&id);
        match tokio::net::UnixStream::connect(&path).await {
            Ok(_) => answering.push(id),
            // Only these two say anything about the *run*: nothing is
            // listening, or the file is gone.
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
                ) =>
            {
                let _ = std::fs::remove_file(&path);
            }
            // Everything else is about *us* — a full accept backlog on a
            // live run that has not started serving yet, this process out
            // of file descriptors — and must be read as "still alive".
            // Reading it as death unlinks a live run's socket, which it
            // never rebinds, and then `purge` sees no live runs and
            // force-removes that run's containers and scratch directory
            // out from under it.
            Err(e) => {
                tracing::warn!(
                    session = %id,
                    path = %path.display(),
                    "could not probe a run's socket ({e}); assuming it is alive"
                );
                answering.push(id);
            }
        }
    }
    answering
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
/// clean up. `SIGTTOU` is therefore blocked around every `tcsetpgrp`.
/// Blocking — rather than installing a handler — is what makes
/// `tcsetpgrp` simply succeed, and it is available as a safe API.
///
/// The block is taken and released *around each call* rather than held
/// for the workload's lifetime, because `pthread_sigmask` is per-thread
/// and this value is dropped on whichever tokio worker the supervising
/// task happens to resume on. A mask blocked on the thread that called
/// [`TerminalHandoff::give_to`] says nothing about the thread that runs
/// `Drop`, so a long-lived block would protect the wrong thread and
/// leave mirage to be stopped by `SIGTTOU` in the middle of teardown.
struct TerminalHandoff {
    /// The process group to restore, i.e. mirage's own.
    restore: nix::unistd::Pid,
}

/// Run `f` with `SIGTTOU` blocked on the current thread, restoring the
/// thread's previous mask afterwards.
///
/// The previous mask is captured by `pthread_sigmask`'s `oldset`
/// argument. Reading it back with `thread_get_mask` after the block would
/// return the mask *including* `SIGTTOU`, so restoring it would leave the
/// signal blocked for good — and, since masks survive `fork`/`exec`, that
/// leak would be inherited by every process spawned afterwards.
fn with_sigttou_blocked<T>(f: impl FnOnce() -> T) -> T {
    use nix::sys::signal::{SigSet, Signal, SigmaskHow, pthread_sigmask};

    let mut ttou = SigSet::empty();
    ttou.add(Signal::SIGTTOU);
    let mut previous = SigSet::empty();
    let blocked = pthread_sigmask(SigmaskHow::SIG_BLOCK, Some(&ttou), Some(&mut previous)).is_ok();

    let out = f();

    if blocked {
        let _ = pthread_sigmask(SigmaskHow::SIG_SETMASK, Some(&previous), None);
    }
    out
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

        let target = nix::unistd::Pid::from_raw(i32::try_from(pid).ok()?);
        let handed = with_sigttou_blocked(|| nix::unistd::tcsetpgrp(stdin.as_fd(), target));
        if handed.is_err() {
            return None;
        }
        Some(Self { restore })
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
        //
        // The block is re-taken here rather than inherited from
        // `give_to`: this runs on whichever worker thread the task was
        // resumed on, and signal masks are per-thread.
        with_sigttou_blocked(|| {
            let _ = nix::unistd::tcsetpgrp(stdin.as_fd(), self.restore);
        });
    }
}
