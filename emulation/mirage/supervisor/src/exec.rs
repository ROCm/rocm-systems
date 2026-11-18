//! A running exec: the process grid for one command invocation.
//!
//! An exec owns `num_nodes * nproc_per_node` workload processes, their
//! output fan-out, and the task that supervises them. Its whole life is
//! driven by one background task, which is what makes the lifecycle
//! statable: the exec is running exactly while that task is running, and
//! everything it owns is released before the task returns.
//!
//! Cancellation flows through a [`CancellationToken`] rather than a flag
//! that has to be polled. Destroying a session, removing an exec, and
//! shutting the daemon down all cancel the same token, and each process's
//! supervising task responds by escalating that process to termination
//! and reaping it.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecStatus, NodeStatus};
use nix::sys::signal::Signal;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::process::{
    ContainerProc, Exit, OutputChunk, SpawnSpec, Spawned, signal_process_group_only, spawn,
};

/// Depth of the channel carrying output from the pump tasks to whoever
/// is printing it.
///
/// Back-pressure is intentional: if the printer cannot keep up, the pumps
/// slow down and eventually the workload blocks on a full pipe, which is
/// exactly how a pipe is supposed to behave. Dropping output instead
/// would make `--capture-all` quietly lossy.
///
/// Only reached under `--capture-all`; otherwise nothing is piped and the
/// channel stays empty for the exec's whole life.
const OUTPUT_CHANNEL_DEPTH: usize = 512;

/// A running or finished exec.
#[derive(Debug)]
pub struct Exec {
    /// This exec's id within its session.
    pub id: ExecId,
    /// The definition it was started from.
    pub def: ExecDef,
    /// Aggregate status, updated as processes start and exit.
    status: Mutex<ExecStatus>,
    /// Live pids by global rank, for signal delivery. Entries are removed
    /// as processes are reaped so a signal can never reach a recycled pid.
    pids: Mutex<BTreeMap<u32, u32>>,
    /// For a containerised exec, where each rank's workload really runs.
    ///
    /// The pids above belong to the provider's *client*; the workload is
    /// in the container's own PID namespace, so a signal has to be
    /// delivered through the provider instead. Fixed at start, because
    /// the mapping does not change over the exec's life.
    containers: BTreeMap<u32, ContainerProc>,
    /// Whether this exec is the single-process kind that takes the
    /// caller's terminal whole.
    ///
    /// Decided from the specs, not from how many processes happen to be
    /// alive later. Those are different answers: a two-rank exec whose
    /// second rank fails to spawn has one live pid but was built with
    /// [`StdioMode::Capture`], so its surviving rank has `/dev/null` on
    /// stdin and must *not* be handed the terminal — doing so makes
    /// mirage a background process group and diverts Ctrl-C away from
    /// the interrupt handling that drives teardown.
    owns_terminal: bool,
    /// Cancels every process supervisor belonging to this exec.
    cancel: CancellationToken,
    /// Set once the exec has fully finished and been cleaned up.
    ///
    /// A `watch` channel rather than a `Notify`. `Notify` is
    /// edge-triggered and only registers a waiter when its future is
    /// first *polled*, so a `notify_waiters()` landing between a caller's
    /// "is it done yet?" check and its `await` is lost — and the caller
    /// waits forever on an exec that already finished. That would be a
    /// hang in `session destroy`, which is exactly the path that must be
    /// reliable. `watch` is level-triggered: `wait_for` inspects the
    /// current value before suspending, so there is no such window.
    finished: watch::Sender<bool>,
}

impl Exec {
    /// Spawn `specs` as one exec and start supervising them.
    ///
    /// Returns as soon as the processes exist. Spawn failures are not
    /// fatal to the exec: the failing rank records exit code 127 and its
    /// reason is published on the exec's output, so a caller attached to
    /// the exec sees *why* rather than an exec that started and never
    /// ended. That distinction matters — an exec with no terminal state
    /// is one a client waits on forever.
    pub fn start(
        id: ExecId,
        def: ExecDef,
        specs: Vec<SpawnSpec>,
    ) -> (Arc<Self>, mpsc::Receiver<OutputChunk>) {
        let (tx, rx) = mpsc::channel(OUTPUT_CHANNEL_DEPTH);

        let mut status = ExecStatus {
            started: false,
            ended: false,
            exit_code: None,
            started_at: Some(Utc::now()),
            ended_at: None,
            nodes: BTreeMap::new(),
        };

        // Keep each process paired with the rank it serves. Deriving the
        // rank from the pid later would be ambiguous the moment a pid is
        // retired, and it is information we already have here.
        let mut spawned: Vec<(u32, Spawned)> = Vec::with_capacity(specs.len());
        let mut failures: Vec<(u32, String)> = Vec::new();
        let mut pids = BTreeMap::new();
        let mut containers = BTreeMap::new();

        for spec in &specs {
            match spawn(spec, tx.clone()) {
                Ok(child) => {
                    status.started = true;
                    pids.insert(spec.node, child.pid());
                    if let Some(container) = child.container() {
                        containers.insert(spec.node, container.clone());
                    }
                    status.nodes.insert(
                        spec.node,
                        NodeStatus {
                            pid: Some(child.pid()),
                            exit_code: None,
                        },
                    );
                    spawned.push((spec.node, child));
                }
                Err(reason) => {
                    status.started = true;
                    status.nodes.insert(
                        spec.node,
                        NodeStatus {
                            pid: None,
                            exit_code: Some(Exit::NOT_FOUND),
                        },
                    );
                    failures.push((spec.node, reason));
                }
            }
        }
        // Drop our sender so the forwarder ends once every pump is done.
        drop(tx);

        // One process, on a mode that takes the caller's stdin: this is
        // the interactive shape. `build_specs` gives every rank of one
        // exec the same mode, so any spec answers for all of them.
        let owns_terminal = specs.len() == 1 && specs.iter().all(|s| s.stdio.owns_terminal());

        let exec = Arc::new(Self {
            id,
            def,
            status: Mutex::new(status),
            pids: Mutex::new(pids),
            containers,
            owns_terminal,
            cancel: CancellationToken::new(),
            finished: watch::channel(false).0,
        });

        // Report spawn failures on this process's own stderr. A rank that
        // never started still has to say why: an exec that silently runs
        // three of its four ranks is far worse than one that fails.
        for (node, reason) in failures {
            eprintln!("mirage: node {node}: {reason}");
        }

        // Supervise the processes to completion.
        tokio::spawn(supervise(exec.clone(), spawned));

        (exec, rx)
    }

    /// A snapshot of the exec's aggregate status.
    #[must_use]
    pub fn status(&self) -> ExecStatus {
        self.lock_status().clone()
    }

    /// The process group to hand the caller's terminal to, if any.
    ///
    /// `Some` only for a single-process exec on an inheriting stdio mode
    /// — the interactive shape, where the workload's stdin *is* the
    /// caller's terminal and it therefore has to become the terminal's
    /// foreground group or be stopped by `SIGTTIN` on its first read.
    ///
    /// Keyed off the exec's shape rather than a rank number on purpose.
    /// The pid map is keyed by *global* rank, so `mirage exec --node 2`
    /// — the whole reason single-node execs exist — registers its one
    /// process under rank 2 and a lookup of rank 0 finds nothing: the
    /// shell would inherit the terminal, sit in a background process
    /// group, and stop on the first keystroke.
    #[must_use]
    pub fn terminal_pid(&self) -> Option<u32> {
        if !self.owns_terminal {
            return None;
        }
        let pids = self.lock_pids();
        match pids.len() {
            1 => pids.values().next().copied(),
            _ => None,
        }
    }

    /// Whether this exec is the single-process kind that takes the
    /// caller's terminal whole.
    ///
    /// Answered from the exec's *shape*, so it is stable for the exec's
    /// whole life — unlike [`Exec::terminal_pid`], which is also `None`
    /// once the process has been reaped. A caller that needs to say
    /// something about the shape ("nothing is reading your stdin") wants
    /// this one.
    #[must_use]
    pub fn owns_terminal(&self) -> bool {
        self.owns_terminal
    }

    /// Pids of the processes still running in this exec.
    #[must_use]
    pub fn live_pids(&self) -> Vec<u32> {
        self.lock_pids().values().copied().collect()
    }

    /// Synchronously `SIGKILL` every live process group.
    ///
    /// The last-resort path: no grace period, no awaiting, and it can be
    /// called from a `Drop` or a panic handler where there is no runtime
    /// to await on. It does not reap — the supervising tasks do that if
    /// they get to run — so [`Exec::terminate`] remains the correct way to
    /// stop an exec. This exists so that "the process is definitely gone"
    /// is reachable even when "await teardown properly" is not.
    pub fn kill_now(&self) {
        // The pid list is copied out and the lock released before any
        // process is signalled. Signalling a containerised rank forks a
        // provider client, and holding the pid mutex across one fork per
        // rank would block every other reader of it — in a path whose
        // whole purpose is to work from a `Drop` or a panic handler.
        for (node, pid) in self.lock_pids_for_kill() {
            // Containerised: the pid is the provider's client, in our
            // namespace, and killing it leaves the workload running
            // inside the container. Push the signal through the provider
            // too, without waiting — there is no runtime to wait on here.
            if let Some(container) = self.containers.get(&node) {
                container.signal_now(Signal::SIGKILL);
            }
            signal_process_group_only(pid, Signal::SIGKILL);
        }
    }

    /// Whether every process has exited.
    #[must_use]
    pub fn is_ended(&self) -> bool {
        self.lock_status().ended
    }

    /// Send `sig` to every live process group in this exec.
    ///
    /// # Errors
    ///
    /// Returns an error if `sig` is not a valid signal number, so a typo
    /// is reported rather than silently dropped.
    pub async fn signal(&self, sig: i32) -> Result<()> {
        let signal = Signal::try_from(sig)
            .map_err(|_| MirageError::other(format!("invalid signal: {sig}")))?;
        let live: Vec<(u32, u32)> = self
            .lock_pids()
            .iter()
            .map(|(node, pid)| (*node, *pid))
            .collect();

        // Containerised ranks first, and concurrently: each one is a
        // `provider exec` round trip, and a job with many nodes would
        // otherwise deliver a Ctrl-C to rank 15 seconds after rank 0.
        let (nodes, forwarded): (Vec<u32>, Vec<_>) = live
            .iter()
            .filter_map(|(node, _)| self.containers.get(node).map(|c| (*node, c.signal(signal))))
            .unzip();
        // Which ranks the provider actually reached. The result is not
        // decoration: `ContainerProc::signal` reports `false` when the
        // rank has not recorded its in-container pid yet, or when the
        // provider itself failed, and a rank skipped below on the
        // assumption that it was signalled would receive nothing at all —
        // a Ctrl-C that visibly does nothing.
        let delivered: std::collections::BTreeSet<u32> = futures::future::join_all(forwarded)
            .await
            .into_iter()
            .zip(nodes)
            .filter_map(|(reached, node)| reached.then_some(node))
            .collect();

        for (node, pid) in live {
            // A containerised rank the provider reached has been
            // signalled already. Signalling the client's group as well
            // would kill the proxy out from under a workload that is
            // handling the signal it was just sent — but if the forward
            // failed there is no such workload to protect, and the
            // client's group is the only thing left to signal.
            if delivered.contains(&node) {
                continue;
            }
            // The *group*, never the bare pid. A pid leaves this map only
            // after its process has been reaped, and the removal happens
            // a scheduling quantum after the reap — so a signal arriving
            // in that window sees a number the kernel may already have
            // reissued. `kill(-pid)` can only reach a process group that
            // still has a living member, which narrows the window to the
            // recycled pid *also* having become a group leader;
            // `signal_group`'s bare-pid fallback would have no such
            // protection and could signal any unrelated process.
            signal_process_group_only(pid, signal);
        }
        Ok(())
    }

    /// Terminate every process and wait until the exec is fully finished.
    ///
    /// Returns only once every process has been reaped, so a caller that
    /// awaits it can then truthfully report that nothing is left running.
    /// Idempotent, and safe to call concurrently from any number of tasks.
    pub async fn terminate(&self) {
        self.cancel.cancel();
        self.wait_finished().await;
    }

    /// Wait until the exec finishes.
    pub async fn wait_finished(&self) {
        let mut done = self.finished.subscribe();
        // `wait_for` checks the current value before suspending, so an
        // exec that finished before this call returns immediately.
        if done.wait_for(|finished| *finished).await.is_err() {
            // The sender was dropped, which only happens when the exec
            // itself is being dropped. Nothing is left to wait for.
        }
    }

    fn lock_pids_for_kill(&self) -> Vec<(u32, u32)> {
        self.lock_pids().iter().map(|(n, p)| (*n, *p)).collect()
    }

    fn lock_status(&self) -> std::sync::MutexGuard<'_, ExecStatus> {
        self.status.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_pids(&self) -> std::sync::MutexGuard<'_, BTreeMap<u32, u32>> {
        self.pids.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for Exec {
    /// Last resort: nothing that owns an `Exec` may drop it with
    /// processes still running.
    ///
    /// [`Run`](crate::Run) has had this guarantee since it was written,
    /// but `mirage exec` never builds a `Run` — it holds an `Exec`
    /// directly — so on that path a panic or an early `?` return had
    /// nothing enforcing it. For a native workload tokio's
    /// `kill_on_drop` eventually catches it; for a containerised one it
    /// does not, because killing the provider's client leaves the
    /// workload alive inside the container. Attaching the rule to the
    /// type that owns the processes covers both owners at once.
    ///
    /// A no-op in the normal case: every pid has already been retired by
    /// its supervising task.
    fn drop(&mut self) {
        self.kill_now();
    }
}

/// Drive every process of an exec to completion.
async fn supervise(exec: Arc<Exec>, spawned: Vec<(u32, Spawned)>) {
    let mut tasks = Vec::with_capacity(spawned.len());
    for (node, mut child) in spawned {
        let exec = exec.clone();
        let cancel = exec.cancel.clone();
        let pid = child.pid();
        tasks.push(tokio::spawn(async move {
            // Race the process against cancellation. Both arms end with
            // the child reaped: `wait` reaps it naturally, `terminate`
            // reaps it after escalating SIGTERM to SIGKILL.
            //
            // Nothing that has to happen once a process is finished
            // belongs in one of these arms. The cancellation arm is the
            // rare one — a workload normally just exits — so anything
            // attached to it silently does not happen for most runs,
            // which is exactly how the process-group sweep came to be
            // skipped on every ordinary exit. It now hangs off the reap
            // itself, in `Spawned::reaped`, and so does everything else
            // of that kind.
            let natural = tokio::select! {
                exit = child.wait() => Some(exit),
                () = cancel.cancelled() => None,
            };
            let exit = match natural {
                Some(exit) => exit,
                None => child.terminate().await,
            };

            // Stop advertising this pid before announcing the exit: once
            // reaped, the number can be reused by an unrelated process,
            // and a signal racing in must not reach it.
            exec.lock_pids().remove(&node);
            {
                let mut status = exec.lock_status();
                status.nodes.insert(
                    node,
                    NodeStatus {
                        pid: Some(pid),
                        exit_code: Some(exit.code),
                    },
                );
            }
            exit
        }));
    }

    for task in tasks {
        // A panicking supervisor task must not strand the exec in a
        // never-ending state; the join error is recorded like any other
        // abnormal exit.
        if let Err(e) = task.await {
            tracing::error!("exec process supervisor task failed: {e}");
        }
    }

    let exit_code = {
        let mut status = exec.lock_status();
        let code = aggregate_exit_code(&status);
        status.ended = true;
        status.ended_at = Some(Utc::now());
        status.exit_code = Some(code);
        code
    };

    // Publish completion. `send_replace` rather than `send`: the value
    // must be recorded even when nothing is currently waiting, so a
    // caller that asks later still sees it.
    exec.finished.send_replace(true);
    tracing::debug!(exec = %exec.id, exit_code, "exec finished");
}

/// The exec's overall exit code: the exit furthest from zero across every
/// process.
///
/// Taking the worst rather than rank 0's means a job where one worker
/// crashed and the head exited cleanly is reported as a failure, which is
/// what a caller scripting against the exit code needs.
fn aggregate_exit_code(status: &ExecStatus) -> i32 {
    status
        .nodes
        .values()
        .filter_map(|n| n.exit_code)
        .fold(0, |worst, code| {
            if code.abs() > worst.abs() {
                code
            } else {
                worst
            }
        })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use mirage_core::exec::{ExecArgs, StdStream};
    use mirage_core::session::SessionId;
    use std::time::Duration;

    fn def() -> ExecDef {
        ExecDef {
            timestamp: Utc::now(),
            session: SessionId::new("t").unwrap(),
            exec: ExecArgs {
                command: "/bin/true".to_string(),
                args: vec![],
                env: BTreeMap::new(),
                workdir: None,
            },
            worker_exec: None,
            nproc_per_node: 1,
            node: None,
            clear_env: false,
        }
    }

    fn spec(node: u32, script: &str) -> SpawnSpec {
        SpawnSpec {
            node,
            command: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: BTreeMap::new(),
            workdir: None,
            // Captured, so the tests can read what the processes wrote.
            // The default — inheriting the test runner's own streams —
            // would print the output instead of returning it.
            stdio: crate::process::StdioMode::Capture,
            inherit_env: false,
            container: None,
        }
    }

    /// Start an exec and collect its output into a shared buffer.
    ///
    /// The collector runs for the exec's whole life, exactly as the CLI's
    /// printer does, so these tests exercise the real path by which
    /// output leaves a captured exec.
    fn start(specs: Vec<SpawnSpec>) -> (Arc<Exec>, Collected) {
        let (exec, rx) = Exec::start(ExecId::from_counter(0), def(), specs);
        (exec, Collected::draining(rx))
    }

    /// Output gathered from a running exec, readable at any point.
    #[derive(Clone)]
    struct Collected(Arc<Mutex<Vec<OutputChunk>>>);

    impl Collected {
        fn draining(mut rx: mpsc::Receiver<OutputChunk>) -> Self {
            let chunks = Arc::new(Mutex::new(Vec::new()));
            let sink = chunks.clone();
            tokio::spawn(async move {
                while let Some(chunk) = rx.recv().await {
                    sink.lock().unwrap_or_else(|e| e.into_inner()).push(chunk);
                }
            });
            Self(chunks)
        }

        fn text(&self, stream: StdStream) -> String {
            let guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let mut buf = Vec::new();
            for chunk in guard.iter().filter(|c| c.stream == stream) {
                buf.extend_from_slice(&chunk.data);
            }
            String::from_utf8_lossy(&buf).into_owned()
        }
    }

    async fn finish(exec: &Arc<Exec>) -> ExecStatus {
        tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
            .await
            .expect("exec must finish");
        exec.status()
    }

    #[tokio::test]
    async fn a_single_process_exec_reports_its_exit_code() {
        let (exec, _out) = start(vec![spec(0, "exit 4")]);
        let status = finish(&exec).await;
        assert!(status.started);
        assert!(status.ended);
        assert_eq!(status.exit_code, Some(4));
        assert_eq!(status.nodes[&0].exit_code, Some(4));
    }

    #[tokio::test]
    async fn output_is_captured_per_stream() {
        let (exec, out) = start(vec![spec(0, "echo out; echo err 1>&2")]);
        finish(&exec).await;
        assert_eq!(out.text(StdStream::Stdout).trim(), "out");
        assert_eq!(out.text(StdStream::Stderr).trim(), "err");
    }

    #[tokio::test]
    async fn the_worst_exit_across_ranks_wins() {
        // Rank 0 succeeds, a worker fails: the exec must report failure.
        let (exec, _out) = start(vec![
            spec(0, "exit 0"),
            spec(1, "exit 9"),
            spec(2, "exit 0"),
        ]);
        let status = finish(&exec).await;
        assert_eq!(status.exit_code, Some(9));
        assert_eq!(status.nodes.len(), 3);
    }

    #[tokio::test]
    async fn a_failed_spawn_still_produces_a_terminal_exec() {
        // The critical property: an exec that could not start must still
        // end, or an attached client waits forever.
        let mut bad = spec(0, "");
        bad.command = "definitely-not-a-real-binary".to_string();
        let (exec, _out) = start(vec![bad]);
        let status = finish(&exec).await;
        assert!(status.ended, "an exec that cannot start must still end");
        assert_eq!(status.exit_code, Some(127));
        // The reason itself goes to mirage's own stderr — it is a mirage
        // diagnostic, not workload output, and the workload produced
        // none.
        assert_eq!(status.nodes[&0].pid, None);
    }

    #[tokio::test]
    async fn a_partial_spawn_failure_still_runs_the_others() {
        let mut bad = spec(1, "");
        bad.command = "definitely-not-a-real-binary".to_string();
        let (exec, out) = start(vec![spec(0, "echo alive"), bad]);
        let status = finish(&exec).await;
        assert_eq!(status.nodes[&0].exit_code, Some(0));
        assert_eq!(status.nodes[&1].exit_code, Some(127));
        assert_eq!(status.exit_code, Some(127));
        assert!(out.text(StdStream::Stdout).contains("alive"));
    }

    #[tokio::test]
    async fn terminate_ends_a_long_running_exec_and_reaps_it() {
        let (exec, _out) = start(vec![spec(0, "sleep 300"), spec(1, "sleep 300")]);
        // Let the processes actually start.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let pids: Vec<u32> = exec.status().nodes.values().filter_map(|n| n.pid).collect();
        assert_eq!(pids.len(), 2);

        tokio::time::timeout(Duration::from_secs(30), exec.terminate())
            .await
            .expect("terminate must complete");

        assert!(exec.is_ended());
        for pid in pids {
            assert!(
                crate::process::wait_gone(pid, Duration::from_secs(5)).await,
                "pid {pid} survived exec termination"
            );
        }
    }

    #[tokio::test]
    async fn terminate_is_idempotent_and_safe_after_natural_exit() {
        let (exec, _out) = start(vec![spec(0, "exit 0")]);
        finish(&exec).await;
        // Terminating a finished exec must return immediately, not hang
        // waiting for a notification that will never come.
        tokio::time::timeout(Duration::from_secs(5), exec.terminate())
            .await
            .expect("terminate on a finished exec must return immediately");
        tokio::time::timeout(Duration::from_secs(5), exec.terminate())
            .await
            .expect("terminate must be idempotent");
    }

    #[tokio::test]
    async fn terminate_returns_even_when_the_exec_finishes_in_the_race_window() {
        // Regression: completion used to be signalled with a `Notify`,
        // which is edge-triggered and registers a waiter only when its
        // future is first polled. An exec finishing between a caller's
        // "already done?" check and its `await` lost the wakeup and hung
        // the caller — a hang in `session destroy`, of all places.
        //
        // Hammering a very short exec puts the finish inside that window
        // often enough to catch a regression.
        for _ in 0..200 {
            let (exec, _out) = start(vec![spec(0, "exit 0")]);
            tokio::time::timeout(Duration::from_secs(10), exec.terminate())
                .await
                .expect("terminate must never hang");
            assert!(exec.is_ended());
        }
    }

    #[tokio::test]
    async fn wait_finished_returns_for_an_already_finished_exec() {
        let (exec, _out) = start(vec![spec(0, "exit 0")]);
        finish(&exec).await;
        // Level-triggered: asking after the fact must answer immediately
        // rather than waiting for an edge that has already passed.
        for _ in 0..5 {
            tokio::time::timeout(Duration::from_secs(5), exec.wait_finished())
                .await
                .expect("waiting on a finished exec must return at once");
        }
    }

    #[tokio::test]
    async fn concurrent_terminates_all_return() {
        let (exec, _out) = start(vec![spec(0, "sleep 300")]);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let waiters: Vec<_> = (0..8)
            .map(|_| {
                let exec = exec.clone();
                tokio::spawn(async move { exec.terminate().await })
            })
            .collect();
        for w in waiters {
            tokio::time::timeout(Duration::from_secs(30), w)
                .await
                .expect("every concurrent terminate must return")
                .unwrap();
        }
        assert!(exec.is_ended());
    }

    #[tokio::test]
    async fn signal_reaches_the_workload() {
        let (exec, _out) = start(vec![spec(0, "sleep 300")]);
        tokio::time::sleep(Duration::from_millis(100)).await;
        exec.signal(libc::SIGTERM).await.unwrap();
        let status = finish(&exec).await;
        assert_eq!(status.exit_code, Some(128 + libc::SIGTERM));
    }

    #[tokio::test]
    async fn an_invalid_signal_number_is_rejected() {
        let (exec, _out) = start(vec![spec(0, "exit 0")]);
        let err = exec.signal(9999).await.unwrap_err();
        assert!(err.to_string().contains("invalid signal"), "{err}");
        finish(&exec).await;
    }

    #[tokio::test]
    async fn a_single_node_exec_offers_its_terminal_whatever_rank_it_is() {
        // `mirage exec --node 2 -- bash`: one process, on the caller's
        // own streams. Its global rank is 2, not 0, and the terminal
        // handoff used to look the pid up under rank 0 — so the shell
        // inherited the terminal, sat in a background process group, and
        // stopped with SIGTTIN on the first keystroke.
        let mut s = spec(2, "sleep 300");
        s.stdio = crate::process::StdioMode::Inherit { stdin: true };
        let (exec, _out) = start(vec![s]);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let pid = exec.terminal_pid();
        assert!(
            pid.is_some(),
            "a one-process exec on node 2 must still be handed the terminal"
        );
        assert_eq!(pid, exec.status().nodes[&2].pid);
        exec.terminate().await;
    }

    #[tokio::test]
    async fn a_grid_never_offers_its_terminal_even_when_only_one_rank_survives() {
        // The shape is fixed at spawn: this exec was built captured, so
        // every rank has `/dev/null` on stdin and none of them may become
        // the terminal's foreground group. Deciding from the *live* pid
        // count read a partly-failed grid as the interactive case and
        // handed the terminal away, which diverts Ctrl-C from mirage and
        // so from the teardown it drives.
        let mut bad = spec(1, "");
        bad.command = "definitely-not-a-real-binary".to_string();
        let (exec, _out) = start(vec![spec(0, "sleep 300"), bad]);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            exec.live_pids().len(),
            1,
            "the setup must leave exactly one rank running"
        );
        assert_eq!(
            exec.terminal_pid(),
            None,
            "a captured exec must not be handed the terminal"
        );
        exec.terminate().await;
    }

    #[tokio::test]
    async fn signalling_a_finished_exec_is_harmless() {
        let (exec, _out) = start(vec![spec(0, "exit 0")]);
        finish(&exec).await;
        // Every pid has been retired, so this must reach nothing at all
        // rather than an unrelated process that reused the number.
        exec.signal(libc::SIGKILL).await.unwrap();
    }
}
