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
    let (mut interrupts, mut relays) = install_signals()?;

    let mut profile = mirage_core::store::profile_get(&a.profile).map_err(profile_error)?;
    let profile_ref = apply_profile_overrides(&mut profile, &a)?;
    check_run_args(&a, &profile)?;

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
    // Declare the job's shape before anything can ask about the session.
    //
    // `--nproc-per-node` is a property of the job this run *is*, not of
    // the one command below, and every rank variable is numbered in that
    // grid — so a `mirage exec` from another terminal has to be told it
    // or it invents a different world and joins this one's rendezvous
    // with it. Said here, before `run_owned` serves the socket, so there
    // is no instant in which a borrower could be answered with the
    // default.
    run.set_job_shape(a.nproc_per_node.unwrap_or(1));
    eprintln!("mirage: session {session}");

    // Answer clients from the first instant, bring-up and teardown
    // included.
    //
    // Owned here rather than inside `run_owned` so that it outlives it.
    // Teardown is the second-longest thing a run does — a container
    // removal per node and then a network — and while it ran there was
    // nothing calling `accept`. The socket file was still there and the
    // listener was still bound, so a `mirage exec` in another terminal
    // connected, sat in the kernel backlog, and was told thirty seconds
    // later that the run "is either still starting up, or shutting
    // down". It was shutting down, and the session had known that since
    // before the client dialled.
    let serving = socket.serve(run.clone());
    tokio::pin!(serving);

    // From here on every exit path must go through teardown, including
    // the error ones: a session whose bring-up half-succeeded still has
    // containers to remove.
    let outcome = run_owned(&run, &a, &mut interrupts, &mut relays, &mut serving).await;

    // Still answering. `Session::attach` refuses once `tearing_down` is
    // set and `describe` refuses once the session is not healthy, so a
    // borrower arriving now is told so in one round trip instead of
    // waiting out `DESCRIBE_TIMEOUT` to be guessed at.
    tokio::select! {
        () = run.destroy() => {}
        () = &mut serving => unreachable!("the control socket serves until dropped"),
    }
    outcome
}

/// Blame a `--profile` the store could not hand over.
///
/// Only one thing is added, and only to the not-found case: the command
/// that lists what *is* there. Where mirage looked used to be added here
/// too, and is not any more — [`MirageError::ProfileNotFound`] carries
/// the directory itself now, so saying it again printed the same path
/// twice in one sentence.
///
/// [`MirageError::ProfileNotFound`]: mirage_core::error::MirageError::ProfileNotFound
fn profile_error(e: mirage_core::error::MirageError) -> anyhow::Error {
    if e.is_not_found() {
        anyhow::anyhow!("{e}. `mirage profile list` shows what is there.")
    } else {
        anyhow::Error::new(e)
    }
}

/// Refuse a `mirage run` that cannot work, before it costs a session.
///
/// Everything here is a filesystem stat, a string split or a
/// multiplication, and every one of them used to be checked after
/// bring-up: `--env NOPE` was reported by the exec, a grid too large by
/// [`mirage_supervisor::build_specs`], and a missing `--workdir` by the
/// spawned child's `chdir`. A run that is going to be refused had
/// therefore already created a session, containers, a network and an
/// emulator daemon, and torn all of it down again — several seconds to
/// say something knowable before the first one.
fn check_run_args(a: &RunArgs, profile: &mirage_core::profile::ProfileDef) -> anyhow::Result<()> {
    // Parsed here for the error; `exec_def` parses it again to build the
    // map it actually passes, which is cheap and keeps that function
    // usable on its own.
    parse_envs(&a.envs)?;
    if let Some(nodes) = crate::profile_node_count(profile) {
        let nproc = a.nproc_per_node.unwrap_or(1);
        crate::check_grid(nodes, nproc)?;
        // Before bring-up, for the same reason as the grid check above:
        // a job that cannot open a pipe for every rank should not first
        // create containers, a network and an emulator daemon.
        crate::ensure_descriptors_for(u64::from(nodes) * u64::from(nproc))?;
    }
    // A containerised session's workdir is a path inside the container,
    // which this filesystem knows nothing about.
    if profile.containerize.is_none()
        && let Some(workdir) = &a.workdir
    {
        crate::check_host_workdir(workdir)?;
    }
    Ok(())
}

/// The body of `mirage run`, with teardown guaranteed by the caller.
async fn run_owned(
    run: &Arc<Run>,
    a: &RunArgs,
    interrupts: &mut Interrupts,
    relays: &mut Relays,
    serving: &mut std::pin::Pin<&mut impl Future<Output = ()>>,
) -> anyhow::Result<ExitCode> {
    // Race bring-up against an interrupt. Pulling an image can take
    // minutes, and a user who changes their mind in the middle of it
    // should get their prompt back — with the half-built session removed
    // by the caller's teardown, not left for `mirage state purge`.
    //
    // And say what is happening while it happens. Bring-up already
    // describes every phase it enters, but until now only to `tracing`,
    // which is off unless the user passed `-v` — so the common case of a
    // cold image was several silent minutes with no way to tell a slow
    // pull from a hung one.
    let health = tokio::select! {
        health = run.wait_ready(READY_TIMEOUT) => health?,
        sig = interrupts.next() => {
            // Said, like every other interrupt path says it. This one
            // printed nothing at all: the prompt came back, and the only
            // output was whatever internal `tracing` line the abandoned
            // bring-up happened to emit on its way out — which describes
            // mirage's own bookkeeping and reads like a fault. What the
            // user needs to know is that the interrupt arrived and that
            // the half-built session is being removed, which is what the
            // caller does next.
            eprintln!(
                "mirage: {}: session {} was still starting; removing what it had created",
                signal_name(sig),
                run.id()
            );
            return Ok(ExitCode::from(u8::try_from(128 + sig).unwrap_or(130)));
        }
        () = report_progress(run.watch_health()) => {
            unreachable!("progress reporting ends only with the session")
        }
        () = &mut *serving => unreachable!("the control socket serves until dropped"),
    };
    if !health.healthy {
        let state = health.state.as_deref().unwrap_or("unknown");
        match health.message {
            Some(msg) => anyhow::bail!("session failed to start ({state}): {msg}"),
            None => anyhow::bail!("session failed to start ({state})"),
        }
    }

    let session = run.id();

    // From here on the session is healthy, which means it is borrowable
    // and may already have been borrowed: the socket has been answering
    // since before bring-up finished. So every exit below has to go
    // through the same borrower wait, not just the one that reaches the
    // bottom of the function.
    //
    // The `?`s inside used to return straight past it. A run whose own
    // command was fine waited for a borrower; a run whose own command had
    // a bad `--workdir`, an unstartable program or a grid the session
    // cannot fit returned immediately, and the caller's `Run::destroy`
    // then tore the session down under a borrower that had done nothing
    // wrong. Same session, same borrower, opposite treatment, decided by
    // something the borrower has no part in.
    let outcome: anyhow::Result<ExitCode> = async {
        // `--gdb` / `--gdb-ex` wrap the workload under ROCgdb for
        // interactive GPU kernel debugging; `--gdb-ex` implies `--gdb`.
        // Applied here rather than inside `exec_def` because it is a
        // `run` flag: `mirage exec` reaches an existing job's node and
        // has no `--gdb` to honour, and wrapping there would put a second
        // debugger around a workload the run already wrapped.
        let argv = if a.gdb || !a.gdb_ex.is_empty() {
            crate::gdb_wrap_argv(&a.argv, &a.gdb_ex)
        } else {
            a.argv.clone()
        };
        // `run` starts the whole job; `exec --node N` is how you reach
        // one node of it, so the node is unset here.
        let def = exec_def(
            session.clone(),
            &argv,
            &a.envs,
            a.workdir.clone(),
            a.nproc_per_node,
            None,
            a.clear_env_vars,
        )?;

        // Raced against the socket, not awaited alone. Starting an exec
        // reaches a container `--workdir` probe -- a provider round trip
        // -- and `serving` is polled by the bring-up `select!` and by the
        // workload `select!` and by nothing in between, so without this
        // arm the run answers nobody for the length of that probe: bound,
        // accepting into the backlog, and silent. A `mirage exec` in
        // another terminal waited it out and was then told this run "is
        // either still starting up, or shutting down", which is neither.
        let (exec, output) = tokio::select! {
            started = run.exec(&def) => started?,
            () = &mut *serving => unreachable!("the control socket serves until dropped"),
        };

        // Keep serving for as long as the workload runs. `select!` rather
        // than a spawned task so the server stops when the workload does,
        // without a second thing to cancel.
        tokio::select! {
            code = supervise_locally(exec, output, interrupts, relays) => code,
            () = &mut *serving => unreachable!("the control socket serves until dropped"),
        }
    }
    .await;

    // This run's own command is finished; the session it owns may not be.
    //
    // A `mirage exec` in another terminal borrows this session, and its
    // workload is that process's child, not ours — we cannot see it, wait
    // on it, or signal it. What we can do is not pull the session out
    // from under it: returning here takes us into `Run::destroy`, which
    // stops the emulator daemon, runs the backend's shutdown hook,
    // removes the containers and deletes the scratch directory holding
    // the emulator's config and socket. A borrower mid-job gets an I/O
    // error, or a SIGKILL with no grace period, depending on where it is.
    //
    // So the run stays up while anyone is still borrowing it. Not
    // silently — a `mirage run -- sleep 5` that does not return after
    // five seconds needs to say why — and not unconditionally: the wait
    // is unbounded, because no timeout would be right for somebody else's
    // job, and an interrupt is how the user says they have waited enough.
    wait_for_borrowers(run, interrupts, serving).await;

    outcome
}

/// Print each new bring-up phase to stderr as the session reports it.
///
/// Never resolves on its own: the caller races it against readiness, and
/// dropping it stops the reporting. Written that way rather than as a
/// spawned task so there is nothing left to cancel once bring-up is over.
///
/// Only *changed* messages are printed. Health is republished on every
/// phase, and several phases share a message shape ("node 1/4 started",
/// "node 2/4…"), so echoing every notification would repeat lines that
/// say nothing new.
///
/// A **terminal failure** is not printed here at all, and that omission
/// is the point. The reason bring-up gives is published as health *and*
/// returned to [`run_owned`], which turns it into the fatal error `main`
/// prints — so echoing it here printed the same text twice, or three
/// times with the log line the supervisor used to emit beside it, and
/// nondeterministically, because whether this loop woke before teardown
/// replaced the value was a race. For a `--hack` build failure, whose
/// message carries the provider's own output, that was hundreds of
/// duplicated lines. Progress is this function's job; the answer is the
/// caller's.
///
/// stderr, not stdout: this is mirage talking about itself, and it must
/// not land in the middle of a workload's piped output.
async fn report_progress(
    mut health: tokio::sync::watch::Receiver<mirage_core::session::SessionHealth>,
) {
    let mut last: Option<String> = None;
    let mut entered = tokio::time::Instant::now();
    loop {
        {
            // Scoped so the watch guard is released before the await
            // below; holding one across a yield point is denied by the
            // workspace lints, and for good reason — it would block every
            // publisher, which here is bring-up itself.
            let current = health.borrow_and_update();
            if let Some(line) = progress_line(&current, last.as_deref()) {
                eprintln!("mirage: {line}");
                last = Some(line);
                entered = tokio::time::Instant::now();
            }
        }
        match tokio::time::timeout(STALL_NOTICE, health.changed()).await {
            // Health changed; loop and report whatever is new.
            Ok(Ok(())) => {}
            Ok(Err(_)) => {
                // The session is gone; there is nothing further to report
                // and the caller is about to finish anyway. Park rather
                // than return, so this stays the "never resolves" arm of
                // a `select!` and cannot be mistaken for readiness.
                std::future::pending::<()>().await;
            }
            Err(_elapsed) => {
                let stalled = {
                    let current = health.borrow();
                    stall_notice(&current, entered.elapsed())
                };
                if let Some(notice) = stalled {
                    eprintln!("mirage: {notice}");
                }
            }
        }
    }
}

/// How long a phase may go without saying anything before mirage does.
const STALL_NOTICE: Duration = Duration::from_secs(30);

/// What to say about a phase that has gone quiet, if anything.
///
/// Only the *externally bounded* phases — pulling an image, building a
/// derived one — because they are the ones that can legitimately go quiet
/// for a long time and are therefore the ones whose readiness clock
/// [`Run::wait_ready`](mirage_supervisor::Run::wait_ready) deliberately
/// suspends. Every other phase is already covered: if it stops moving,
/// the deadline expires and the run fails with a message that says so.
///
/// Which left exactly one hole, and it was the whole of what a stalled
/// bring-up looked like from outside: `podman pull` reports itself once
/// and then goes silent, so a registry that never answers and a
/// three-gigabyte image that is downloading fine produce the same
/// output — one line, and then nothing, for as long as it takes. Saying
/// how long it has been going is what tells the two apart, and naming
/// the way out is what stops a user who has decided it is stuck from
/// having to guess whether mirage will notice a Ctrl-C.
fn stall_notice(health: &mirage_core::session::SessionHealth, elapsed: Duration) -> Option<String> {
    use mirage_core::session::state;
    let phase = health.state.as_deref()?;
    if !state::is_externally_bounded(Some(phase)) {
        return None;
    }
    Some(format!(
        "still {phase} after {}; Ctrl-C to stop waiting for it",
        humanised(elapsed)
    ))
}

/// A duration as a person would say it: `45s`, `2m 30s`.
fn humanised(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    match (seconds / 60, seconds % 60) {
        (0, s) => format!("{s}s"),
        (m, s) => format!("{m}m {s}s"),
    }
}

/// What [`report_progress`] should print for a health snapshot, given
/// the last line it printed.
///
/// Split out from the loop so the rule can be stated and tested rather
/// than inferred from a nest of conditions. Two things are refused:
///
/// * a message identical to the last one, because health is republished
///   on every phase and several phases share a message shape; and
/// * a **terminal failure**, which is the reason bring-up gave and which
///   [`run_owned`] returns as the process's fatal error. Printing it here
///   too is how the same paragraph came to be shown twice — three times
///   with the supervisor's log line beside it — and nondeterministically,
///   since whether this loop woke before teardown replaced the value was
///   a race.
fn progress_line(
    health: &mirage_core::session::SessionHealth,
    last: Option<&str>,
) -> Option<String> {
    let terminal_failure =
        health.terminal && health.state.as_deref() == Some(mirage_core::session::state::FAILED);
    health
        .message
        .as_deref()
        .filter(|_| !terminal_failure)
        .filter(|message| last != Some(message))
        .map(str::to_string)
}

/// Hold the session open while other terminals are still using it.
async fn wait_for_borrowers(
    run: &Arc<Run>,
    interrupts: &mut Interrupts,
    serving: &mut std::pin::Pin<&mut impl Future<Output = ()>>,
) {
    let borrowers = run.borrowers();
    if borrowers == 0 {
        return;
    }
    eprintln!(
        "mirage: this command has finished, but {borrowers} `mirage exec` \
         borrower(s) are still using session {}",
        run.id()
    );
    eprintln!("mirage: waiting for them; Ctrl-C to tear the session down anyway");

    tokio::select! {
        () = run.wait_for_borrowers() => {}
        _ = interrupts.next() => {
            // Teardown publishes the closing signal, which drops every
            // lease connection, so the borrowers are told rather than
            // simply having their session removed. They still have to
            // stop their own processes; those are their children.
            eprintln!(
                "mirage: tearing down session {} with {} borrower(s) still attached",
                run.id(),
                run.borrowers()
            );
        }
        () = &mut *serving => unreachable!("the control socket serves until dropped"),
    }
}

/// `mirage exec`: run a command inside a session someone else owns.
pub async fn exec_cmd(a: ExecArgsCli) -> anyhow::Result<ExitCode> {
    let (mut interrupts, mut relays) = install_signals()?;

    let session = match a.session.clone() {
        Some(id) => id,
        None => sole_live_run().await?,
    };
    // A client-side exec id, distinct from anything the run process is
    // using. It names this command's pid files, and two execs in
    // different processes must not collide on them.
    //
    // The pid alone is not enough: pids are recycled, the pid files live
    // in the run's scratch directory for as long as the *run* lasts, and
    // a long-lived run outlives many `mirage exec` invocations. A start
    // timestamp makes the id unique per invocation rather than per pid.
    //
    // Minted before the attach, because the attach carries it: the run
    // has no other way to learn it — the run does not start these
    // processes — and without it a workload of ours that reparents stops
    // being attributable to this live lease. See `Session::attach`.
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let id = ExecId::new(format!("x-{}-{started}", std::process::id()))
        .map_err(|e| anyhow::anyhow!("could not build an exec id: {e}"))?;

    // Attach rather than merely describe: the lease is held for the whole
    // exec, so the run that owns this session waits for us instead of
    // tearing the emulator and the scratch directory down underneath a
    // live workload.
    let (desc, mut lease) = attach(&session, Some(id.clone())).await?;

    // Now that the session has described itself, we know whether the
    // workdir names a directory here or one inside a container.
    if desc.containers.is_none()
        && let Some(workdir) = &a.workdir
    {
        crate::check_host_workdir(workdir)?;
    }

    let def = exec_def(
        session.clone(),
        &a.argv,
        &a.envs,
        a.workdir.clone(),
        a.nproc_per_node,
        a.node,
        a.clear_env_vars,
    )?;

    // `mirage exec` spawns its ranks as its own children in its own
    // terminal, so the streams to ask about are this process's; see
    // [`mirage_supervisor::CallerStreams`].
    let specs = mirage_supervisor::build_specs(
        &desc,
        &def,
        &id,
        mirage_supervisor::CallerStreams::probe(),
    )?;
    // The same headroom, in the process that will actually hold the
    // pipes: an exec spawns its ranks as its own children. Asked after
    // `build_specs` because that is what settles how many there are —
    // `--node` narrows the grid, and asking before would reserve for
    // ranks this command is not going to start.
    crate::ensure_descriptors_for(specs.len() as u64)?;
    let (exec, output) = Exec::start(id, def, specs);

    // Normally the run waits for us and the lease outlives the workload.
    // The other case is the user declining to wait: the run closes the
    // lease, and the workload has to be stopped here rather than left to
    // find out when its container is removed or its emulator socket
    // vanishes. Racing the two is what turns that into a clean SIGTERM.
    let closing = async {
        lease.closed().await;
        eprintln!("mirage: the run owning session {session} is shutting down");
    };
    let supervised = supervise_locally(exec.clone(), output, &mut interrupts, &mut relays);
    tokio::pin!(supervised);

    tokio::select! {
        code = &mut supervised => code,
        () = closing => {
            exec.terminate().await;
            // Still through `supervise_locally`, so the printer is
            // drained and the exit code is the workload's own rather
            // than an invented one.
            supervised.await
        }
    }
}

/// Build the [`ExecDef`] for one command invocation.
///
/// Shared by `run` and `exec` so the two cannot drift. They differ in
/// exactly one field — `run` starts the whole job and passes `node:
/// None`, `exec` passes whatever `--node` named — and that difference is
/// a parameter here rather than a second copy of the literal. It was a
/// second copy, and the copy is how `--node` came to be parsed and then
/// dropped on the floor: the comment explaining why `None` is right for
/// `run` was carried into `exec` along with the value it justified.
///
/// # Errors
///
/// Returns an error if an `--env` argument is not in `KEY=VALUE` form.
fn exec_def(
    session: SessionId,
    argv: &[String],
    envs: &[String],
    workdir: Option<String>,
    nproc_per_node: Option<u32>,
    node: Option<u32>,
    clear_env: bool,
) -> anyhow::Result<ExecDef> {
    let (command, args) = split_argv(argv);
    let env = parse_envs(envs)?;
    // Say so when a `--env` will lose. The precedence is deliberate — a
    // rank that disagrees with the grid deadlocks its own collectives —
    // but silence made `--env RANK=3` look like it had been applied.
    //
    // A `mirage:` line on stderr, not a `tracing` record: this is mirage
    // talking to the person who typed the flag, and every other thing
    // mirage says to them looks the same. As a log line it arrived with a
    // timestamp, a level and this module's path in front of it — the
    // shape of an internal fault rather than of an answer — and only when
    // the user had passed `-v`, which is to say not when it mattered.
    for key in crate::mirage_owned_env(env.keys()) {
        eprintln!(
            "mirage: --env {key}=… is ignored: mirage sets {key} itself on every process, \
             so that every rank agrees on the shape of the job"
        );
    }
    Ok(ExecDef {
        timestamp: chrono::Utc::now(),
        session,
        exec: ExecArgs {
            command,
            args,
            env,
            workdir,
        },
        worker_exec: None,
        nproc_per_node: nproc_per_node.unwrap_or(1).max(1),
        node,
        clear_env,
    })
}

/// Run an exec to completion in this terminal, printing captured output
/// and stopping it cleanly if we are interrupted.
///
/// Returns the exit code this process should use.
async fn supervise_locally(
    exec: Arc<Exec>,
    output: mpsc::Receiver<mirage_supervisor::OutputChunk>,
    interrupts: &mut Interrupts,
    relays: &mut Relays,
) -> anyhow::Result<ExitCode> {
    // Always drain the channel. For a multi-process exec this is what
    // prints the labelled output; for a single-process one it is empty
    // and finishes at once, because nothing was piped.
    let printer = tokio::spawn(mirage_supervisor::output::print_labelled(output));

    warn_if_stdin_is_dropped(&exec);

    // Lend the terminal to a single-process exec, at the moment it asks
    // for it. Given back when this future is dropped, which is every
    // exit path below including the interrupted one.
    let lending = lend_terminal(exec.terminal_pid());
    tokio::pin!(lending);

    // Ctrl-C reaches us for as long as we still hold the terminal, and
    // reaches the workload directly once we have lent it out. Forwarding
    // it deliberately in the first case — and then falling through to the
    // normal wait — is what makes a workload get a chance to clean up,
    // and what makes the caller's teardown run rather than being skipped
    // by an abrupt exit.
    //
    // Narrated, because an interrupt that visibly does nothing is
    // indistinguishable from an interrupt that was not delivered. A
    // workload that ignores `SIGINT` is not rare — a training loop
    // catching it to checkpoint, a shell with `trap ''` — and mirage
    // sitting silently through two of them, waiting on something it never
    // named, is the whole of what a user sees go wrong.
    let interrupted = async {
        let sig = interrupts.next().await;
        eprintln!(
            "mirage: {}: asked the workload to stop; \
             interrupt again to stop waiting for it",
            signal_name(sig)
        );
        exec.signal(sig).await.ok();
        // A second interrupt means the user is not waiting any longer.
        interrupts.next().await;
        eprintln!("mirage: not waiting any longer; stopping the workload");
    };

    // Forward the application-defined signals, forever, drawing no
    // conclusion from any of them. A scheduler sending `SIGUSR1` every
    // half hour to say "checkpoint now" must be able to do that all day
    // without the run deciding it has been asked to stop; see
    // [`RELAY_SIGNALS`].
    let relaying = async {
        loop {
            let sig = relays.next().await;
            eprintln!("mirage: {}: forwarded to the workload", signal_name(sig));
            exec.signal(sig).await.ok();
        }
    };
    tokio::pin!(relaying);

    tokio::select! {
        () = exec.wait_finished() => {}
        () = &mut relaying => unreachable!("relayed signals are forwarded for the whole exec"),
        () = interrupted => {
            // `terminate` escalates SIGTERM to SIGKILL on its own, so it
            // ends — but for a containerised grid it is a provider round
            // trip per rank and can take a while, and a user who has
            // already said twice that they are done waiting must not have
            // to guess whether the third interrupt did anything either.
            tokio::select! {
                () = exec.terminate() => {}
                _ = interrupts.next() => {
                    eprintln!("mirage: killing the workload outright");
                    exec.kill_now();
                    exec.wait_finished().await;
                }
            }
        }
        () = &mut lending => unreachable!("the terminal is lent until the exec is over"),
    }

    // Wait for the printer so the last lines are on screen before we
    // return and the caller starts tearing the session down.
    let _ = printer.await;

    let code = exec.status().exit_code.unwrap_or(0);
    // Exit codes are a byte. Masking preserves the shell's `128 + signal`
    // convention for a signal-killed workload rather than saturating it.
    Ok(ExitCode::from((code & 0xff) as u8))
}

/// The name a user would recognise for `sig`.
///
/// The first four are named by what happened rather than by the constant,
/// because that is how the line reads: `mirage: interrupted: …`. The two
/// a scheduler sends have no such word — nothing about `SIGUSR1` says
/// what the sender meant by it — so they are named as themselves, which
/// is also the string the user will find in their scheduler's manual.
fn signal_name(sig: i32) -> &'static str {
    match sig {
        libc::SIGINT => "interrupted",
        libc::SIGQUIT => "quit",
        libc::SIGHUP => "hangup",
        libc::SIGTERM => "terminated",
        libc::SIGUSR1 => "SIGUSR1",
        libc::SIGUSR2 => "SIGUSR2",
        _ => "signalled",
    }
}

/// Say, once, that nothing is going to read the input being piped in.
///
/// A job of several processes connects nobody's stdin, and the README
/// explains why: one terminal cannot be shared between readers, and
/// handing it to rank 0 would mean keystrokes going somewhere the user
/// cannot see. Documented is not the same as visible, though —
/// `mirage run --nproc-per-node 2 -- ./job < input` reads an immediate
/// EOF in every rank, which from the outside is indistinguishable from a
/// broken pipe, a bad path, or mirage dropping the data.
///
/// Only when the caller actually supplied input, because the sentence is
/// about *their* bytes: an interactive caller has piped nothing in, and
/// neither has one who wrote `< /dev/null` — telling either of them that
/// what they piped in is being discarded is mirage describing something
/// that did not happen.
fn warn_if_stdin_is_dropped(exec: &Arc<Exec>) {
    use std::os::fd::AsRawFd as _;
    if !stdin_is_being_discarded(
        exec.owns_terminal(),
        stdin_source(std::io::stdin().as_raw_fd()),
    ) {
        return;
    }
    eprintln!(
        "mirage: this job runs several processes, so none of them is given stdin — \
         one terminal cannot be shared between readers. What you piped in is being \
         discarded; pass it as a file the workload opens, or run one process \
         (`mirage exec --node N`) if it has to be read."
    );
}

/// What is on the other end of stdin, as far as the notice cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StdinSource {
    /// A terminal. The caller is typing; nothing was piped in.
    Terminal,
    /// A pipe, a socket or a file — bytes somebody redirected here
    /// meaning a process to read them.
    Redirected,
    /// Nothing that carries input: `< /dev/null`, a closed descriptor, a
    /// descriptor that cannot be described at all.
    Nothing,
}

/// Classify the descriptor `fd`, which is stdin everywhere but in tests.
///
/// `isatty` first and then the file *type*, because "not a terminal" was
/// the whole test before and it is not the same question. `< /dev/null`
/// is not a terminal; neither is a closed stdin, nor stdin inherited from
/// a service manager that opened it on the null device. All three used to
/// count as input the user had supplied and was losing.
///
/// A character device is read as carrying nothing. That is exactly right
/// for `/dev/null`, which is what redirections in this position
/// overwhelmingly are, and wrong for the rare `< /dev/zero` — an endless
/// stream that really is being discarded and will not be mentioned. The
/// two are indistinguishable without hard-coding device numbers, and
/// silence about an infinite source of nothing is the cheaper mistake.
fn stdin_source(fd: std::os::fd::RawFd) -> StdinSource {
    use nix::sys::stat::SFlag;
    if nix::unistd::isatty(fd).unwrap_or(false) {
        return StdinSource::Terminal;
    }
    let Ok(stat) = nix::sys::stat::fstat(fd) else {
        // No such descriptor: stdin was closed before mirage started.
        return StdinSource::Nothing;
    };
    let kind = SFlag::from_bits_truncate(stat.st_mode) & SFlag::S_IFMT;
    if kind == SFlag::S_IFIFO || kind == SFlag::S_IFSOCK || kind == SFlag::S_IFREG {
        StdinSource::Redirected
    } else {
        StdinSource::Nothing
    }
}

/// Whether the caller piped something in that no rank will ever read.
///
/// The two conditions are independent and both load-bearing: a
/// single-process exec is handed the caller's stdin whatever it is, so it
/// loses nothing, and a caller who supplied no input has nothing to lose
/// either.
fn stdin_is_being_discarded(owns_terminal: bool, stdin: StdinSource) -> bool {
    !owns_terminal && stdin == StdinSource::Redirected
}

/// How often a lent terminal's borrower is checked for having asked.
///
/// Fast at first and slow afterwards, because the distribution is not
/// uniform: a program that wants the terminal wants it at once — an
/// interactive `bash` stops itself before it prints a prompt — while a
/// program that asks for it an hour in is asking for a password or a
/// confirmation, where half a second is nothing. The slow rate is what
/// keeps a day-long non-interactive run from polling `/proc` fifty times
/// a second for no reason.
const TERMINAL_POLL_EAGER: Duration = Duration::from_millis(20);
const TERMINAL_POLL_SETTLED: Duration = Duration::from_millis(500);

/// How long the eager rate lasts.
const TERMINAL_EAGER_FOR: Duration = Duration::from_secs(3);

/// Hold the terminal for the workload and hand it over when it asks.
///
/// Never resolves: the caller races it against the workload finishing,
/// and dropping it takes the terminal back.
///
/// # Why this is not done up front
///
/// Handing the terminal over is what makes an interactive workload
/// possible at all — see [`TerminalHandoff`] — and it is also what makes
/// mirage unreachable, because the tty delivers `SIGINT` to the
/// foreground process group and to nothing else. Doing it at spawn time
/// therefore gave the terminal away to every single-process workload,
/// interactive or not, and a `mirage run -- ./job` whose job ignores
/// `SIGINT` became a run that could not be interrupted from its own
/// terminal: Ctrl-C went to the job, the job ignored it, mirage never
/// woke, and nothing was printed. The second interrupt the help promises
/// "stops waiting for it" never arrived, because mirage was not listening
/// to that terminal any more.
///
/// Lending it on demand splits the two cases the way they actually
/// differ. A workload that never reads the terminal never takes it, so
/// mirage stays the foreground group and the documented two-interrupt
/// behaviour holds. A workload that does read it is stopped by the kernel
/// with `SIGTTIN` the first time it tries — that is precisely what a
/// background process group reading its controlling terminal earns — and
/// being stopped is the ask. Mirage hands the terminal over, continues
/// it, and from then on Ctrl-C belongs to the workload, which for an
/// interactive program is exactly right and is what `mirage run -- bash`
/// needs.
async fn lend_terminal(pid: Option<u32>) {
    let _handoff = match pid {
        Some(pid) => hand_over_when_asked(pid).await,
        None => None,
    };
    // Held, not returned, so the terminal goes back when the caller drops
    // this future rather than at some point it has to remember.
    std::future::pending::<()>().await;
}

/// Wait for `pid` to be stopped for want of the terminal, then give it.
async fn hand_over_when_asked(pid: u32) -> Option<TerminalHandoff> {
    if !TerminalHandoff::ours_to_lend() {
        // Not a terminal, or not ours: there is nothing to lend and
        // nothing to watch for.
        return None;
    }
    let eager_until = tokio::time::Instant::now() + TERMINAL_EAGER_FOR;
    loop {
        match process_state(pid) {
            // Stopped. For a process in a background process group with
            // a controlling terminal that means `SIGTTIN` or `SIGTTOU`,
            // i.e. it tried to use the terminal — and if it stopped for
            // some other reason, handing it the terminal it is about to
            // be resumed onto costs nothing.
            Some(b'T' | b't') => {
                // A handoff that does not take is retried rather than
                // given up on. Giving up would leave a workload stopped
                // for want of a terminal that nothing will ever hand it,
                // which is a hang with no way out; mirage may have lost
                // the foreground group only momentarily, and the poll
                // below costs nothing while it has.
                if let Some(handoff) = TerminalHandoff::give_to(pid) {
                    // Only now: continued before the handoff, it would
                    // stop again on its very next read.
                    resume(pid);
                    return Some(handoff);
                }
            }
            Some(_) => {}
            // Gone, or never readable. Either way nobody is going to ask.
            None => return None,
        }
        let interval = if tokio::time::Instant::now() < eager_until {
            TERMINAL_POLL_EAGER
        } else {
            TERMINAL_POLL_SETTLED
        };
        tokio::time::sleep(interval).await;
    }
}

/// The single-letter state the kernel reports for `pid`, if it is alive.
///
/// Read from `/proc` rather than waited for: the workload's `Child` is
/// owned by the supervisor task that will reap it, and a second waiter
/// racing it for the same status is how a wait becomes a lost exit code.
/// `waitid(WNOWAIT)` would not reap, but it would still be a second party
/// to the child, whereas reading its state is an observation and nothing
/// more.
///
/// The comm field is parenthesised and may itself contain spaces and
/// brackets, so the state is taken from after the *last* `)`, which is
/// the parse `proc(5)` documents.
fn process_state(pid: u32) -> Option<u8> {
    let stat = std::fs::read(format!("/proc/{pid}/stat")).ok()?;
    let close = stat.iter().rposition(|b| *b == b')')?;
    stat.get(close + 1..)?
        .iter()
        .copied()
        .find(|b| !b.is_ascii_whitespace())
}

/// Let a stopped workload carry on, now that it has the terminal.
///
/// The group, because that is what the kernel stopped: `SIGTTIN` is
/// delivered to the whole foreground-denied process group, so a workload
/// that had already forked has children stopped alongside it.
fn resume(pid: u32) {
    let Ok(raw) = i32::try_from(pid) else {
        return;
    };
    if raw <= 0 {
        return;
    }
    let _ = nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(-raw),
        nix::sys::signal::Signal::SIGCONT,
    );
}

/// Every signal that means "stop this", and the rule that picks them.
///
/// The set used to be extended one bug report at a time — `SIGINT`, then
/// `SIGTERM`, then `SIGHUP` — and each addition was the same discovery
/// made again: a signal whose default disposition kills mirage outright
/// runs no teardown, so the workload survives, the socket stays in `run/`
/// claiming a session that no longer exists, and the scratch directory
/// stays on disk. So the membership rule is written down here instead,
/// and the list is everything that satisfies it:
///
/// **a signal whose default disposition ends the process, and which a
/// person or a scheduler sends to say they want this command to stop.**
///
/// * `SIGINT` — Ctrl-C, the one everybody means.
/// * `SIGQUIT` — Ctrl-\, the *same keyboard*, and the same intent said
///   more forcefully. Its default action also writes a core file, which
///   handling it gives up: a core dump of mirage tells nobody anything,
///   and the exchange is a stranded session for a file that gets deleted.
/// * `SIGTERM` — `kill`, a CI runner cancelling a job, a service manager
///   stopping a unit.
/// * `SIGHUP` — the terminal window closed, which is the commonest way a
///   run is abandoned.
///
/// What is deliberately *not* here is as much of the rule as what is.
/// `SIGUSR1` and `SIGUSR2` satisfy the first half of it — unhandled,
/// they end mirage where it stands — and fail the second: nobody sends
/// them to say "stop", they say whatever the workload decided they say.
/// They were in this list once, and being in it meant a scheduler's
/// checkpoint warning aborted the bring-up it was warning about and a
/// job checkpointing on a timer terminated itself on its second one. So
/// they are caught, forwarded, and counted for nothing; see
/// [`RELAY_SIGNALS`]. `SIGTSTP`, `SIGTTIN` and `SIGTTOU` are job control
/// rather than stopping — suspending a run must keep working, and
/// `SIGTTOU` in particular is blocked around the terminal handoff rather
/// than caught (see [`TerminalHandoff`]). `SIGPIPE` is already ignored
/// process-wide by the Rust runtime and would be a lie about intent
/// anyway. `SIGKILL` and `SIGSTOP` cannot be caught at all, which is
/// precisely why `mirage cleanup` exists.
///
/// # Why this is installed before anything else happens
///
/// A tokio signal handler is registered the first time a `Signal` stream
/// is *created*, and until then the signal keeps its default
/// disposition — which for every one of these is "terminate
/// immediately". Arm them lazily, at the point the workload is awaited,
/// and everything before that point is unprotected: a Ctrl-C during a
/// multi-minute image pull killed mirage outright, with no teardown,
/// leaving the containers and the network it had already created behind.
/// That is the exact moment a user is most likely to press Ctrl-C.
///
/// Installing them first makes the guarantee unconditional: from the
/// first line of the command to the last, an interrupt is something
/// mirage handles rather than something that happens to it.
struct Interrupts {
    /// One armed stream per signal, paired with the number that armed it
    /// so [`Interrupts::next`] can report which one arrived.
    signals: Vec<(i32, tokio::signal::unix::Signal)>,
}

/// The signals that mean "stop", and nothing else.
///
/// Every one of them is a request to end the job: a Ctrl-C, a `kill`, a
/// `SIGHUP` from a closing terminal, a scheduler's `SIGTERM` before it
/// takes the node back. That shared meaning is what makes the escalation
/// ladder in [`supervise_locally`] coherent — a second one means the user
/// has stopped waiting, a third means stop now — and it is why the ladder
/// may count them interchangeably.
const STOP_SIGNALS: [i32; 4] = [libc::SIGINT, libc::SIGQUIT, libc::SIGTERM, libc::SIGHUP];

/// The signals that mean whatever the workload decided they mean.
///
/// `SIGUSR1` and `SIGUSR2` are application-defined. Schedulers use them
/// to say "checkpoint now" -- Slurm's `--signal=USR1@60` is the common
/// one -- and a training loop that catches `SIGUSR1` to write a
/// checkpoint and carry on is doing exactly what its sender intended.
///
/// They used to sit in the same ladder as Ctrl-C, and being in it cost
/// two things. During bring-up any of the six aborted the session, so a
/// scheduler's checkpoint warning destroyed the job it was warning. After
/// startup they counted toward escalation, so a job signalled twice --
/// which for a periodic checkpoint is a matter of course -- reached "not
/// waiting any longer" and had its workload terminated.
///
/// So they are forwarded and nothing else: the workload hears them,
/// mirage draws no conclusion from them, and no number of them ends a
/// run. See [`Relays`].
const RELAY_SIGNALS: [i32; 2] = [libc::SIGUSR1, libc::SIGUSR2];

/// The armed handlers for [`RELAY_SIGNALS`].
///
/// Separate from [`Interrupts`] rather than a flag on it, because the two
/// are awaited concurrently by [`supervise_locally`] — one arm counting
/// interrupts, one arm forwarding relays — and a single `&mut` could not
/// be in both.
struct Relays {
    signals: Vec<(i32, tokio::signal::unix::Signal)>,
}

/// Arm every signal mirage handles, in both groups.
///
/// # Errors
///
/// Returns an error if the handlers cannot be registered, which is worth
/// failing on rather than continuing unprotected.
fn install_signals() -> anyhow::Result<(Interrupts, Relays)> {
    use tokio::signal::unix::{SignalKind, signal};
    let arm = |numbers: &[i32]| -> anyhow::Result<Vec<(i32, tokio::signal::unix::Signal)>> {
        numbers
            .iter()
            .map(|number| Ok((*number, signal(SignalKind::from_raw(*number))?)))
            .collect()
    };
    Ok((
        Interrupts {
            signals: arm(&STOP_SIGNALS)?,
        },
        Relays {
            signals: arm(&RELAY_SIGNALS)?,
        },
    ))
}

/// Resolve on the next signal in `signals`, yielding its number.
///
/// Shared by both groups: the waiting is identical and only the meaning
/// of the answer differs.
async fn next_signal(signals: &mut [(i32, tokio::signal::unix::Signal)]) -> i32 {
    // `select_all` rather than a `select!` arm per signal: the arms were
    // identical and the list is long enough that adding one by hand is
    // how a signal comes to be armed and then never waited for. Each
    // future is boxed because `select_all` needs `Unpin`, and all of them
    // are dropped when one wins — `Signal::recv` is cancel-safe, so a
    // signal that arrives in that instant is still buffered for the next
    // call.
    let waits = signals.iter_mut().map(|(number, signal)| {
        let number = *number;
        Box::pin(async move {
            signal.recv().await;
            number
        })
    });
    futures::future::select_all(waits).await.0
}

impl Relays {
    /// Resolve on the next relayed signal, yielding its number.
    ///
    /// A signal that arrived before anything was waiting is not lost:
    /// tokio buffers one per kind. So a `SIGUSR1` sent during bring-up,
    /// when there is no workload to forward it to, reaches the workload
    /// as soon as there is one — late, but delivered, which is the better
    /// of the two failures for a checkpoint request.
    async fn next(&mut self) -> i32 {
        next_signal(&mut self.signals).await
    }
}

impl Interrupts {
    /// Resolve on the next interrupt, yielding its signal number.
    ///
    /// A signal that arrived earlier is not lost: tokio buffers one per
    /// kind, so an interrupt during bring-up is delivered the moment
    /// anything waits for it.
    ///
    /// The number is the one that actually arrived, so the exit status
    /// mirage reports and the signal it forwards to the workload are
    /// both the truth — a hangup is forwarded as a hangup rather than as
    /// a generic stop.
    ///
    /// Only [`STOP_SIGNALS`] resolve here. `SIGUSR1` and `SIGUSR2` reach
    /// the workload through [`Relays`] instead, and deliberately count
    /// for nothing on the way.
    async fn next(&mut self) -> i32 {
        next_signal(&mut self.signals).await
    }
}

/// How long to wait for a run to answer.
///
/// Bounded because connecting proves less than it looks like it does: a
/// bound listener accepts into the kernel backlog whether or not anything
/// is calling `accept`, so a connection succeeding says only that the
/// socket file belongs to a process that once bound it.
///
/// It used to be load-bearing rather than a backstop. A run served its
/// socket only between "the session is healthy" and "the workload has
/// ended", and the two windows outside that were long: an image pull on
/// one side, and on the other `Run::destroy` removing a container per
/// node and then a network. A `mirage exec` arriving in either sat in
/// the backlog until this deadline and was then told the run was "either
/// still starting up, or shutting down" — a guess, offered thirty
/// seconds late, about something the run had known all along.
///
/// A run answers throughout its whole life now, so both windows give a
/// real answer in one round trip: `session … is not ready (pulling)`
/// during bring-up, and `session … is shutting down and cannot be
/// attached to` during teardown. What is left for the deadline is the
/// case it was always for — a run that is wedged, or a socket file left
/// behind by one that was `SIGKILL`ed — where nothing is going to answer
/// at all.
///
/// It bounds the request and its answer only. An [`Attached`] lease is
/// held for as long as the borrowed workload runs, which is unbounded by
/// construction.
const DESCRIBE_TIMEOUT: Duration = Duration::from_secs(30);

/// A live borrow of somebody else's session.
///
/// The value is the open socket, and holding it is the claim: the run
/// counts this connection as a borrower and will not tear the session
/// down while it is open. Dropping it releases the claim, and so does
/// this process dying, which is the reason the lease is a connection
/// rather than a message.
///
/// [`Attached::closed`] resolves when the run has decided to tear down
/// anyway — the user pressed Ctrl-C rather than waiting — so the borrowed
/// workload can be stopped rather than left to discover it when its
/// container is removed.
struct Attached {
    framed:
        tokio_util::codec::Framed<tokio::net::UnixStream, tokio_util::codec::LengthDelimitedCodec>,
}

impl Attached {
    /// Resolve when the run closes the lease.
    async fn closed(&mut self) {
        use futures::StreamExt as _;
        // Nothing is expected on this stream: the protocol has no message
        // after the description. Reading it is how the end of the
        // connection is observed.
        while self.framed.next().await.is_some() {}
    }
}

/// Ask the run that owns `session` to describe it, and hold a lease on it.
///
/// The lease is what stops the owning run from tearing the session down
/// while this process is still using it. `mirage exec` starts its
/// processes itself, in its own terminal, so the run has no other way to
/// know they exist — and teardown stops the emulator daemon, runs the
/// backend's shutdown hook and deletes the scratch directory those
/// processes are actively reading.
async fn attach(
    session: &SessionId,
    exec: Option<ExecId>,
) -> anyhow::Result<(SessionDescription, Attached)> {
    tokio::time::timeout(DESCRIBE_TIMEOUT, attach_inner(session, exec))
        .await
        .unwrap_or_else(|_| {
            anyhow::bail!(
                "the run serving session {session} did not answer within {DESCRIBE_TIMEOUT:?}. \
                 It is either still starting up, or shutting down."
            )
        })
}

async fn attach_inner(
    session: &SessionId,
    exec: Option<ExecId>,
) -> anyhow::Result<(SessionDescription, Attached)> {
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

    let request = serde_json::to_vec(&Request::Attach { exec })?;
    framed.send(request.into()).await?;

    let Some(frame) = framed.next().await else {
        anyhow::bail!("the run serving session {session} closed the connection without answering");
    };
    match serde_json::from_slice::<Response>(&frame?)? {
        Response::Description(desc) => Ok((*desc, Attached { framed })),
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
///
/// Candidates are runs that *answer*, not socket files. A `SIGKILL`ed run
/// leaves its socket behind — the documented, expected leak `mirage
/// cleanup` exists for — and counting files meant one `kill -9` broke
/// auto-selection completely: with a corpse beside a live run this said
/// "several runs are live" and listed a dead session as a candidate, and
/// with a corpse alone it picked the corpse and then failed to connect to
/// it. [`answering_runs`] is the same probe cleanup uses, and it unlinks
/// the corpses on the way past.
async fn sole_live_run() -> anyhow::Result<SessionId> {
    let live = answering_runs().await;
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
/// `live_runs` lists socket *files*, and a file outlives a run that was
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
/// The candidate set, not the answer: a socket file outlives the run that
/// bound it, so this includes the corpses a `SIGKILL` leaves behind.
/// [`answering_runs`] is what separates the two, and is the only caller —
/// a session named explicitly on the command line is not looked up here
/// at all, because `mirage exec --session <id>` should say that *that*
/// session is gone rather than that it does not exist.
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
/// # Why it is lent rather than given
///
/// "Delivered to the workload rather than to mirage" is the cost as well
/// as the point, so the handoff is made when the workload asks for the
/// terminal and not before — see [`lend_terminal`], which is the only
/// caller. A workload that never reads the terminal never takes it, and
/// mirage stays interruptible from the window it was started in.
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
    use nix::sys::signal::{SigSet, SigmaskHow, Signal, pthread_sigmask};

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
    /// Whether mirage has a terminal that is its to lend.
    ///
    /// The same two conditions [`TerminalHandoff::give_to`] checks, asked
    /// before anything is watched rather than after: with no terminal —
    /// a pipe, a CI runner, `< /dev/null` — or with mirage already in the
    /// background, there is nothing to hand over and no reason to poll
    /// for somebody asking.
    fn ours_to_lend() -> bool {
        use std::io::IsTerminal as _;
        use std::os::fd::AsFd as _;

        let stdin = std::io::stdin();
        stdin.is_terminal()
            && nix::unistd::tcgetpgrp(stdin.as_fd()).ok() == Some(nix::unistd::getpgrp())
    }

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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn session() -> SessionId {
        SessionId::new("s").unwrap()
    }

    fn argv() -> Vec<String> {
        vec!["/bin/true".to_string()]
    }

    #[test]
    fn exec_carries_the_node_it_was_given() {
        // The regression that made `mirage exec --node 2` a no-op: the
        // flag was parsed and then dropped, because `exec_cmd` built its
        // own `ExecDef` literal with `node: None` copied from `run`. The
        // exec then fanned out to every node, which — several processes
        // sharing one terminal — takes the captured branch and connects
        // nobody's stdin, i.e. the exact opposite of what `--node` is for.
        let def = exec_def(session(), &argv(), &[], None, None, Some(2), false).unwrap();
        assert_eq!(def.node, Some(2));
    }

    #[test]
    fn run_starts_the_whole_job() {
        // `run` has no `--node`: it brings the session up and runs on all
        // of it. This is the one field the two commands must differ on.
        let def = exec_def(session(), &argv(), &[], None, None, None, false).unwrap();
        assert_eq!(def.node, None);
    }

    #[test]
    fn the_rest_of_the_definition_is_built_the_same_way_for_both() {
        let def = exec_def(
            session(),
            &["/bin/echo".to_string(), "hi".to_string()],
            &["K=V".to_string()],
            Some("/w".to_string()),
            Some(4),
            None,
            true,
        )
        .unwrap();
        assert_eq!(def.exec.command, "/bin/echo");
        assert_eq!(def.exec.args, vec!["hi".to_string()]);
        assert_eq!(def.exec.env.get("K").map(String::as_str), Some("V"));
        assert_eq!(def.exec.workdir.as_deref(), Some("/w"));
        assert_eq!(def.nproc_per_node, 4);
        assert!(def.clear_env);
    }

    #[test]
    fn a_zero_process_count_is_clamped_rather_than_starting_no_processes() {
        // Unreachable from the CLI, which rejects `--nproc-per-node 0` at
        // parse time, and kept as the backstop for any other caller: a
        // grid of zero processes is a session with nothing in it.
        let def = exec_def(session(), &argv(), &[], None, Some(0), None, false).unwrap();
        assert_eq!(def.nproc_per_node, 1);
    }

    #[test]
    fn a_missing_profile_says_where_mirage_looked_exactly_once() {
        // The store's error names the directory, and this call site used
        // to name it a second time in the very next clause: "profile not
        // found: ghost (mirage looked in /…/profile). mirage looked in
        // /…/profile; `mirage profile list` shows what is there."
        use mirage_core::store::DocKind;
        // The config directory is process-wide state another test can be
        // moving, and this asserts on the path inside the message.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());
        let dir = DocKind::Profile.root().display().to_string();
        let e = format!(
            "{:#}",
            profile_error(mirage_core::error::MirageError::not_found(
                DocKind::Profile,
                "ghost"
            ))
        );
        mirage_core::paths::clear_test_root();
        assert_eq!(
            e.matches(&dir).count(),
            1,
            "the directory belongs in the error that knows it, and only there: {e}"
        );
        // The half the error does not carry is still said.
        assert!(e.contains("mirage profile list"), "{e}");

        // Anything that is not a missing profile is passed through
        // untouched; there is no list to point at.
        let other = format!(
            "{:#}",
            profile_error(mirage_core::error::MirageError::other("disk on fire"))
        );
        assert_eq!(other, "disk on fire");
    }

    /// A single-threaded runtime for the async helpers below.
    ///
    /// `#[tokio::test]` would make the test body itself async, and these
    /// hold [`mirage_core::paths::test_env_lock`] — a `MutexGuard` across
    /// an await point, which the workspace denies for good reason.
    /// Blocking on the future instead keeps the guard on one thread and
    /// off the await path.
    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    /// A health snapshot in a named phase, with a message.
    fn phase(state: &str, message: &str) -> mirage_core::session::SessionHealth {
        mirage_core::session::SessionHealth::phase(false, state, Some(message.to_string()))
    }

    #[test]
    fn a_fatal_reason_is_reported_once_and_by_the_error_path() {
        // The same paragraph used to be printed two or three times, from
        // identical input, because bring-up's failure is *both*
        // published as health and returned to the caller — and progress
        // reporting echoed the health while `main` printed the error. For
        // a `--hack` build failure, whose message carries the provider's
        // whole build log, that was hundreds of duplicated lines.
        let reason = "pulling image quay.io/nope:1 failed: unauthorized";
        let failed = mirage_core::session::SessionHealth::failed(reason);
        assert!(
            failed.terminal && failed.state.as_deref() == Some(mirage_core::session::state::FAILED),
            "the fixture must be the shape bring-up publishes: {failed:?}"
        );
        assert_eq!(
            progress_line(&failed, None),
            None,
            "the fatal reason is the caller's to report, and it does"
        );

        // Progress itself is untouched: this is a filter on one kind of
        // message, not a mute.
        assert_eq!(
            progress_line(&phase("pulling", "pulling image big:latest"), None).as_deref(),
            Some("pulling image big:latest")
        );
        // And an unchanged message is still only said once, which is the
        // rule that was already there.
        assert_eq!(
            progress_line(&phase("starting", "node 1/4"), Some("node 1/4")),
            None
        );
    }

    #[test]
    fn a_pull_that_has_gone_quiet_says_how_long_it_has_been_quiet_for() {
        // A stalled bring-up used to look exactly like a healthy slow
        // one: `podman pull` reports itself once and then says nothing,
        // and mirage suspends the readiness clock for it on purpose — so
        // a registry that never answers produced one line of output and
        // then silence, indefinitely, with no hint that a Ctrl-C would
        // even be noticed.
        let pulling = phase("pulling", "pulling image big:latest");
        let notice = stall_notice(&pulling, Duration::from_secs(150))
            .expect("a suspended phase that has gone quiet must say so");
        assert!(notice.contains("still pulling"), "{notice}");
        assert!(notice.contains("2m 30s"), "{notice}");
        assert!(
            notice.contains("Ctrl-C"),
            "the way out must be named: {notice}"
        );

        // Not for the phases whose clock is running: those already fail
        // with "made no progress", and a second voice saying the same
        // thing first would be noise.
        assert_eq!(
            stall_notice(&phase("starting", "node 1/4"), STALL_NOTICE),
            None
        );
        assert_eq!(
            stall_notice(
                &mirage_core::session::SessionHealth::phase(true, "ready", None),
                STALL_NOTICE
            ),
            None
        );
    }

    #[test]
    fn a_piped_stdin_that_no_rank_will_read_is_reported() {
        // Documented behaviour that is indistinguishable from a broken
        // pipe: `mirage run --nproc-per-node 2 -- ./job < input` gives
        // every rank an immediate EOF, silently.
        assert!(
            stdin_is_being_discarded(false, StdinSource::Redirected),
            "a grid started with a redirected stdin loses what was piped in"
        );
        // Not a warning anyone else needs. One process is handed the
        // caller's stdin whatever it is...
        assert!(!stdin_is_being_discarded(true, StdinSource::Redirected));
        assert!(!stdin_is_being_discarded(true, StdinSource::Terminal));
        // ...and a caller who supplied no input has nothing to lose.
        assert!(!stdin_is_being_discarded(false, StdinSource::Terminal));
        assert!(!stdin_is_being_discarded(false, StdinSource::Nothing));
    }

    #[test]
    fn nothing_is_being_discarded_when_nothing_was_piped_in() {
        // The notice says "What you piped in is being discarded", and it
        // said it to people who had piped nothing in: the test was "stdin
        // is not a terminal", which `< /dev/null`, a closed stdin and a
        // service manager's null stdin all satisfy. Every one of those is
        // a multi-process run being told it has lost data that never
        // existed.
        use std::os::fd::AsRawFd as _;

        let null = std::fs::File::open("/dev/null").unwrap();
        assert_eq!(stdin_source(null.as_raw_fd()), StdinSource::Nothing);

        // A descriptor that is not open at all — `mirage run … 0<&-`.
        assert_eq!(stdin_source(-1), StdinSource::Nothing);

        // What the notice is actually for: a pipe, and a file.
        let (read, _write) = nix::unistd::pipe().unwrap();
        assert_eq!(stdin_source(read.as_raw_fd()), StdinSource::Redirected);
        let file = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(
            stdin_source(file.as_file().as_raw_fd()),
            StdinSource::Redirected
        );
    }

    /// Serialises the tests that raise real signals at this process.
    ///
    /// A signal is delivered to the process, not to a test: every armed
    /// [`Interrupts`] in this binary is woken by it, so two of these
    /// running at once would each be answered with the other's signal.
    /// A tokio mutex rather than a `std` one because the guard is held
    /// across the awaits below, which the workspace denies for the
    /// blocking kind and for good reason.
    static RAISING_SIGNALS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn every_signal_that_means_stop_is_handled_rather_than_fatal() {
        let _serialised = RAISING_SIGNALS.lock().await;
        // `SIGQUIT` is Ctrl-\ — the same keyboard as Ctrl-C, and the same
        // intent. Unhandled, the default disposition ends mirage where it
        // stands: no teardown, the workload still running, the socket
        // still claiming a live session and the scratch directory still
        // on disk. Like the hangup test below, this does not merely fail
        // before the fix — it takes the test binary down with it, which
        // is the same thing happening for the same reason.
        let (mut interrupts, _relays) = install_signals().unwrap();
        for signal in [
            nix::sys::signal::Signal::SIGQUIT,
            nix::sys::signal::Signal::SIGTERM,
            nix::sys::signal::Signal::SIGINT,
        ] {
            nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(std::process::id() as i32),
                signal,
            )
            .unwrap();
            let sig = tokio::time::timeout(Duration::from_secs(10), interrupts.next())
                .await
                .unwrap_or_else(|_| panic!("{signal:?} must reach the interrupt handling"));
            assert_eq!(sig, signal as i32);
            // And it is named for the user rather than reported as an
            // anonymous "signalled".
            assert_ne!(signal_name(sig), "signalled", "{signal:?}");
        }
    }

    #[tokio::test]
    async fn the_application_defined_signals_are_handled_but_not_interrupts() {
        let _serialised = RAISING_SIGNALS.lock().await;
        // Both halves matter, and they used to be one.
        //
        // Handled: unhandled, `SIGUSR1`'s default disposition ends mirage
        // where it stands — no teardown, workload still running — and a
        // scheduler sending it to warn of preemption would be the thing
        // that stranded the job.
        //
        // But *not* an interrupt: it is application-defined, and what
        // sends it usually means "checkpoint now". In the same ladder as
        // Ctrl-C it aborted bring-up, and after startup a second one
        // reached "not waiting any longer" and terminated the workload —
        // so a job checkpointing on a timer killed itself on the second
        // checkpoint.
        let (mut interrupts, mut relays) = install_signals().unwrap();
        for signal in [
            nix::sys::signal::Signal::SIGUSR1,
            nix::sys::signal::Signal::SIGUSR2,
        ] {
            nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(std::process::id() as i32),
                signal,
            )
            .unwrap();
            let sig = tokio::time::timeout(Duration::from_secs(10), relays.next())
                .await
                .unwrap_or_else(|_| panic!("{signal:?} must be caught and forwarded"));
            assert_eq!(sig, signal as i32);
            assert_ne!(signal_name(sig), "signalled", "{signal:?}");
        }

        // And none of that reached the ladder. Checked after both, so a
        // relay that leaked into the stop set has had every chance to
        // arrive.
        assert!(
            tokio::time::timeout(Duration::from_millis(250), interrupts.next())
                .await
                .is_err(),
            "an application-defined signal must not count as a request to stop"
        );
    }

    #[tokio::test]
    async fn closing_the_terminal_is_an_interrupt_and_not_a_death() {
        // `SIGHUP` is what closing a terminal window sends, and closing
        // the window is how people end a run they have lost interest in.
        // Unhandled, its default disposition ends mirage outright: exit
        // 129, no teardown, the workload still running, the socket still
        // in `run/` claiming a live session and the scratch directory
        // still on disk.
        //
        // This test would not merely fail before the fix — it would take
        // the test binary down with it, which is the same thing happening
        // for the same reason.
        let _serialised = RAISING_SIGNALS.lock().await;
        let (mut interrupts, _relays) = install_signals().unwrap();
        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(std::process::id() as i32),
            nix::sys::signal::Signal::SIGHUP,
        )
        .unwrap();

        let sig = tokio::time::timeout(Duration::from_secs(10), interrupts.next())
            .await
            .expect("a hangup must reach the interrupt handling, not the default action");
        assert_eq!(sig, libc::SIGHUP);
    }

    #[test]
    fn a_stopped_workload_is_recognised_as_asking_for_the_terminal() {
        // How the terminal is lent on demand: a process in a background
        // process group that reads its controlling terminal is stopped by
        // `SIGTTIN`, and being stopped is the ask. Reading the state is
        // what makes that observable without becoming a second party to
        // somebody else's child.
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "while true; do sleep 1; done"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id();

        assert!(
            matches!(process_state(pid), Some(b'S' | b'R' | b'D')),
            "a running process must not read as stopped: {:?}",
            process_state(pid).map(char::from)
        );

        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGSTOP,
        )
        .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !matches!(process_state(pid), Some(b'T' | b't')) {
            assert!(
                std::time::Instant::now() < deadline,
                "a stopped process was never seen as stopped: {:?}",
                process_state(pid).map(char::from)
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn a_run_that_was_killed_is_not_a_candidate_for_exec() {
        // The socket file outlives a `SIGKILL`ed run: that leak is
        // documented, expected, and what `mirage cleanup` exists for. So
        // counting files rather than answers meant a single `kill -9`
        // broke `mirage exec` with no `--session`: with a corpse beside a
        // live run it reported "several runs are live" and offered the
        // dead one as a candidate.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());
        std::fs::create_dir_all(mirage_core::paths::run_socket_root()).unwrap();

        let dead = SessionId::new("s-dead").unwrap();
        let dead_path = mirage_core::paths::run_socket_path(&dead);
        // Bound and dropped: the file stays, nothing answers on it. The
        // listener is not unlinked on drop, which is precisely why a
        // killed run leaves one behind.
        drop(std::os::unix::net::UnixListener::bind(&dead_path).unwrap());

        let alive = SessionId::new("s-live").unwrap();
        let _listener =
            std::os::unix::net::UnixListener::bind(mirage_core::paths::run_socket_path(&alive))
                .unwrap();

        let picked = runtime().block_on(sole_live_run());
        mirage_core::paths::clear_test_root();

        assert_eq!(picked.unwrap(), alive);
        assert!(
            !dead_path.exists(),
            "the corpse socket should have been unlinked on the way past"
        );
    }

    #[test]
    fn a_corpse_alone_is_not_a_run_to_exec_into() {
        // The other half: with only a dead socket present, auto-selection
        // used to pick the corpse and then fail to connect to it. There
        // is no run here at all, and that is what it should say.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());
        std::fs::create_dir_all(mirage_core::paths::run_socket_root()).unwrap();
        let dead = SessionId::new("s-dead").unwrap();
        drop(
            std::os::unix::net::UnixListener::bind(mirage_core::paths::run_socket_path(&dead))
                .unwrap(),
        );

        let picked = runtime().block_on(sole_live_run());
        mirage_core::paths::clear_test_root();

        let e = picked.unwrap_err().to_string();
        assert!(e.contains("no `mirage run` is running"), "{e}");
    }
}
