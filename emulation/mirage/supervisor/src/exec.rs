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
use mirage_core::ctl::StreamPacket;
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecStatus, NodeStatus};
use nix::sys::signal::Signal;
use tokio::sync::{Mutex as AsyncMutex, mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::output::OutputHub;
use crate::process::{ContainerProc, Exit, SpawnSpec, Spawned, signal_process_group_only, spawn};

/// Depth of the channel carrying output from the pump tasks to the hub.
///
/// Back-pressure is intentional: if the hub cannot keep up, the pumps
/// slow down and eventually the workload blocks on a full pipe, which is
/// exactly how a pipe is supposed to behave. Dropping output instead
/// would make `mirage logs` quietly lossy.
const OUTPUT_CHANNEL_DEPTH: usize = 512;

/// A running or finished exec.
#[derive(Debug)]
pub struct Exec {
    /// This exec's id within its session.
    pub id: ExecId,
    /// The definition it was started from.
    pub def: ExecDef,
    /// Output fan-out for attach and logs.
    pub hub: Arc<OutputHub>,
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
    /// Rank 0's input, whether a pipe or a terminal. Async because
    /// writing can block when the reader is slow, and blocking the
    /// runtime there would stall every other session the daemon owns.
    stdin: AsyncMutex<Option<crate::process::ProcessInput>>,
    /// Cancels every process supervisor belonging to this exec.
    cancel: CancellationToken,
    /// The task moving pumped output into the hub. Awaited before the
    /// exec is marked finished, so every byte a process wrote is
    /// published before its `ExecExit`.
    forwarder: Mutex<Option<tokio::task::JoinHandle<()>>>,
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
    pub fn start(id: ExecId, def: ExecDef, specs: Vec<SpawnSpec>, replay_bytes: usize) -> Arc<Self> {
        let hub = Arc::new(OutputHub::new(replay_bytes));
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

        // Only rank 0 gets an attached stdin: it is the process a user
        // interacts with, and fanning one input stream out to several
        // readers has no sensible semantics.
        //
        // Every other rank's input is taken and dropped immediately,
        // which closes the pipe and gives that process EOF on stdin.
        // Merely *not attaching* it is not the same thing: the handle
        // would stay inside its `Spawned` for the process's whole life,
        // holding the write end open, and a worker that reads to EOF —
        // `cat`, `sort`, a launcher reading a manifest — would block
        // forever on input that can never arrive, so the exec would never
        // finish either.
        let mut stdin = None;
        for (node, child) in &mut spawned {
            let input = child.stdin.take();
            if *node == 0 {
                stdin = input;
            }
        }

        let exec = Arc::new(Self {
            id,
            def,
            hub: hub.clone(),
            status: Mutex::new(status),
            pids: Mutex::new(pids),
            containers,
            stdin: AsyncMutex::new(stdin),
            cancel: CancellationToken::new(),
            forwarder: Mutex::new(None),
            finished: watch::channel(false).0,
        });

        // Report spawn failures on the exec's own output stream, where an
        // attached client will actually see them.
        for (node, reason) in failures {
            hub.publish(StreamPacket::Output {
                node,
                stream: mirage_core::ctl::StdStream::Stderr,
                data: format!("mirage: {reason}\n").into_bytes(),
            });
            hub.publish(StreamPacket::NodeExit {
                node,
                exit_code: Exit::NOT_FOUND,
            });
        }

        // Forward pumped output into the hub, and remember the task so
        // teardown can wait for it to drain.
        *exec.forwarder.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(tokio::spawn(forward_output(rx, hub)));

        // Supervise the processes to completion.
        tokio::spawn(supervise(exec.clone(), spawned));

        exec
    }

    /// A snapshot of the exec's aggregate status.
    #[must_use]
    pub fn status(&self) -> ExecStatus {
        self.lock_status().clone()
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
        for (node, pid) in self.lock_pids().iter() {
            // Containerised: the pid is the provider's client, in our
            // namespace, and killing it leaves the workload running
            // inside the container. Push the signal through the provider
            // too, without waiting — there is no runtime to wait on here.
            if let Some(container) = self.containers.get(node) {
                container.signal_now(Signal::SIGKILL);
            }
            signal_process_group_only(*pid, Signal::SIGKILL);
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
        let forwarded = live
            .iter()
            .filter_map(|(node, _)| self.containers.get(node))
            .map(|container| container.signal(signal));
        futures::future::join_all(forwarded).await;

        for (node, pid) in live {
            // A containerised rank has already been signalled through the
            // provider. Signalling the client's group as well would kill
            // the proxy out from under a workload that is handling the
            // signal it was just sent.
            if self.containers.contains_key(&node) {
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

    /// Write to rank 0's stdin.
    ///
    /// # Errors
    ///
    /// Returns an error if the exec has no stdin (its rank 0 failed to
    /// spawn) or the pipe has been closed by the process exiting.
    pub async fn write_stdin(&self, data: &[u8]) -> Result<()> {
        let mut guard = self.stdin.lock().await;
        let Some(stdin) = guard.as_mut() else {
            return Err(MirageError::other(
                "exec has no open stdin (the process has exited or never started)",
            ));
        };
        match stdin.write_all(data).await {
            Ok(()) => Ok(()),
            Err(e) => {
                // A broken pipe means the process is gone. Drop the handle
                // so the next write reports the clearer "no open stdin"
                // rather than repeating an OS error.
                *guard = None;
                Err(MirageError::other(format!(
                    "writing to exec stdin failed: {e}"
                )))
            }
        }
    }

    /// Signal end-of-input to rank 0.
    ///
    /// For a pipe this closes it. For a terminal there is nothing to
    /// close, so the EOF character is written and the line discipline
    /// turns it into an end-of-file — which is what makes `cat` on a
    /// terminal finish, exactly as pressing Ctrl-D would.
    pub async fn close_stdin(&self) {
        let mut guard = self.stdin.lock().await;
        if let Some(input) = guard.as_mut()
            && let Err(e) = input.send_eof().await
        {
            tracing::debug!("could not signal end-of-input: {e}");
        }
        guard.take();
    }

    /// Resize rank 0's terminal. A no-op on pipes.
    ///
    /// # Errors
    ///
    /// Returns an error if the exec has no open input, or the resize
    /// fails.
    pub async fn resize(&self, rows: u16, cols: u16) -> Result<()> {
        let guard = self.stdin.lock().await;
        let Some(input) = guard.as_ref() else {
            // Nothing to resize; the process has exited. Not worth an
            // error: a resize is advisory and racing an exit is normal.
            return Ok(());
        };
        input
            .resize(rows, cols)
            .map_err(|e| MirageError::other(format!("resizing the terminal failed: {e}")))
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

    fn lock_status(&self) -> std::sync::MutexGuard<'_, ExecStatus> {
        self.status.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_pids(&self) -> std::sync::MutexGuard<'_, BTreeMap<u32, u32>> {
        self.pids.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Move pumped output into the hub until every pump has finished.
async fn forward_output(mut rx: mpsc::Receiver<crate::process::OutputChunk>, hub: Arc<OutputHub>) {
    while let Some(chunk) = rx.recv().await {
        hub.publish(StreamPacket::Output {
            node: chunk.node,
            stream: chunk.stream,
            data: chunk.data,
        });
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
            exec.hub.publish(StreamPacket::NodeExit {
                node,
                exit_code: exit.code,
            });
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

    // Wait for the forwarder to drain before publishing the exit.
    //
    // Output travels pump task -> channel -> hub, which is two hops, and
    // waiting on the child only guarantees the first. Without this, a
    // process that writes and exits immediately can have its final bytes
    // reach the hub *after* the `ExecExit` packet — and since a client is
    // documented to stop reading at the exit, that output is simply lost.
    // It shows up as flakiness under load, where the forwarder is less
    // likely to be scheduled promptly.
    //
    // The wait is bounded: if a descendant outlived the exec and is still
    // holding the write end of a pipe, its pump never ends and neither
    // would this.
    let forwarder = exec
        .forwarder
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    if let Some(forwarder) = forwarder {
        const DRAIN: std::time::Duration = std::time::Duration::from_secs(5);
        if tokio::time::timeout(DRAIN, forwarder).await.is_err() {
            tracing::debug!(
                exec = %exec.id,
                "output forwarder did not drain; a descendant is holding a pipe"
            );
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

    // Close stdin so nothing holds the write end of a pipe to a dead
    // process, and so a later `session_stdin` gets a clear error.
    //
    // Bounded, like the two waits above and for the same reason. This
    // takes the stdin lock, and on a terminal it also *writes* the EOF
    // character — so a concurrent `session_stdin` blocked on a full pipe
    // that a surviving grandchild is not reading would hold it forever.
    // Everything downstream of here is what makes an exec finish, and
    // `session_destroy` waits on that with no timeout of its own: a hang
    // in this line is a permanently unkillable session.
    const CLOSE_STDIN: std::time::Duration = std::time::Duration::from_secs(5);
    if tokio::time::timeout(CLOSE_STDIN, exec.close_stdin())
        .await
        .is_err()
    {
        tracing::debug!(
            exec = %exec.id,
            "closing stdin did not complete; a descendant is holding the pipe"
        );
    }

    exec.hub.finish(exit_code);
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
            if code.abs() > worst.abs() { code } else { worst }
        })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::output::DEFAULT_REPLAY_BYTES;
    use mirage_core::ctl::StdStream;
    use mirage_core::exec::ExecArgs;
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
            tty: false,
            tty_rows: 0,
            tty_cols: 0,
            keep: true,
        }
    }

    fn spec(node: u32, script: &str) -> SpawnSpec {
        SpawnSpec {
            node,
            command: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: BTreeMap::new(),
            workdir: None,
            stdio: crate::process::StdioMode::Pipes,
            inherit_env: false,
            container: None,
        }
    }

    fn start(specs: Vec<SpawnSpec>) -> Arc<Exec> {
        Exec::start(ExecId::from_counter(0), def(), specs, DEFAULT_REPLAY_BYTES)
    }

    async fn finish(exec: &Arc<Exec>) -> ExecStatus {
        tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
            .await
            .expect("exec must finish");
        exec.status()
    }

    fn output_text(exec: &Arc<Exec>, stream: StdStream) -> String {
        let mut buf = Vec::new();
        for pkt in exec.hub.snapshot() {
            if let StreamPacket::Output { stream: s, data, .. } = pkt
                && s == stream
            {
                buf.extend_from_slice(&data);
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    #[tokio::test]
    async fn a_single_process_exec_reports_its_exit_code() {
        let exec = start(vec![spec(0, "exit 4")]);
        let status = finish(&exec).await;
        assert!(status.started);
        assert!(status.ended);
        assert_eq!(status.exit_code, Some(4));
        assert_eq!(status.nodes[&0].exit_code, Some(4));
    }

    #[tokio::test]
    async fn output_is_captured_per_stream() {
        let exec = start(vec![spec(0, "echo out; echo err 1>&2")]);
        finish(&exec).await;
        assert_eq!(output_text(&exec, StdStream::Stdout).trim(), "out");
        assert_eq!(output_text(&exec, StdStream::Stderr).trim(), "err");
    }

    #[tokio::test]
    async fn the_worst_exit_across_ranks_wins() {
        // Rank 0 succeeds, a worker fails: the exec must report failure.
        let exec = start(vec![spec(0, "exit 0"), spec(1, "exit 9"), spec(2, "exit 0")]);
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
        let exec = start(vec![bad]);
        let status = finish(&exec).await;
        assert!(status.ended);
        assert_eq!(status.exit_code, Some(127));
        assert!(
            output_text(&exec, StdStream::Stderr).contains("command not found"),
            "the reason must reach the client: {:?}",
            output_text(&exec, StdStream::Stderr)
        );
    }

    #[tokio::test]
    async fn a_partial_spawn_failure_still_runs_the_others() {
        let mut bad = spec(1, "");
        bad.command = "definitely-not-a-real-binary".to_string();
        let exec = start(vec![spec(0, "echo alive"), bad]);
        let status = finish(&exec).await;
        assert_eq!(status.nodes[&0].exit_code, Some(0));
        assert_eq!(status.nodes[&1].exit_code, Some(127));
        assert_eq!(status.exit_code, Some(127));
        assert!(output_text(&exec, StdStream::Stdout).contains("alive"));
    }

    #[tokio::test]
    async fn terminate_ends_a_long_running_exec_and_reaps_it() {
        let exec = start(vec![spec(0, "sleep 300"), spec(1, "sleep 300")]);
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
        let exec = start(vec![spec(0, "exit 0")]);
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
            let exec = start(vec![spec(0, "exit 0")]);
            tokio::time::timeout(Duration::from_secs(10), exec.terminate())
                .await
                .expect("terminate must never hang");
            assert!(exec.is_ended());
        }
    }

    #[tokio::test]
    async fn wait_finished_returns_for_an_already_finished_exec() {
        let exec = start(vec![spec(0, "exit 0")]);
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
        let exec = start(vec![spec(0, "sleep 300")]);
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
        let exec = start(vec![spec(0, "sleep 300")]);
        tokio::time::sleep(Duration::from_millis(100)).await;
        exec.signal(libc::SIGTERM).await.unwrap();
        let status = finish(&exec).await;
        assert_eq!(status.exit_code, Some(128 + libc::SIGTERM));
    }

    #[tokio::test]
    async fn an_invalid_signal_number_is_rejected() {
        let exec = start(vec![spec(0, "exit 0")]);
        let err = exec.signal(9999).await.unwrap_err();
        assert!(err.to_string().contains("invalid signal"), "{err}");
        finish(&exec).await;
    }

    #[tokio::test]
    async fn signalling_a_finished_exec_is_harmless() {
        let exec = start(vec![spec(0, "exit 0")]);
        finish(&exec).await;
        // Every pid has been retired, so this must reach nothing at all
        // rather than an unrelated process that reused the number.
        exec.signal(libc::SIGKILL).await.unwrap();
    }

    #[tokio::test]
    async fn stdin_is_forwarded_to_rank_zero() {
        let exec = Exec::start(
            ExecId::from_counter(0),
            def(),
            vec![SpawnSpec {
                node: 0,
                command: "/bin/cat".to_string(),
                args: vec![],
                env: BTreeMap::new(),
                workdir: None,
                stdio: crate::process::StdioMode::Pipes,
                inherit_env: false,
                container: None,
            }],
            DEFAULT_REPLAY_BYTES,
        );
        exec.write_stdin(b"hello stdin\n").await.unwrap();
        exec.close_stdin().await;
        finish(&exec).await;
        assert_eq!(output_text(&exec, StdStream::Stdout), "hello stdin\n");
    }

    #[tokio::test]
    async fn workers_get_eof_on_stdin_rather_than_an_open_pipe() {
        // Only rank 0 has an attached stdin, but "not attached" has to
        // mean *closed*, not "held open by the daemon". A worker that
        // reads to EOF — `cat`, `sort`, a launcher reading a manifest —
        // otherwise blocks forever on input that can never arrive, and
        // because the exec ends only when every rank does, the whole exec
        // hangs and `session destroy` hangs with it.
        let exec = Exec::start(
            ExecId::from_counter(0),
            def(),
            vec![
                spec(0, "exit 0"),
                SpawnSpec {
                    node: 1,
                    command: "/bin/cat".to_string(),
                    args: vec![],
                    env: BTreeMap::new(),
                    workdir: None,
                    stdio: crate::process::StdioMode::Pipes,
                    inherit_env: false,
                    container: None,
                },
            ],
            DEFAULT_REPLAY_BYTES,
        );
        let status = tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
            .await
            .map(|()| exec.status())
            .expect("a worker reading stdin must see EOF, not hang");
        assert_eq!(status.nodes[&1].exit_code, Some(0));
    }

    #[tokio::test]
    async fn writing_stdin_after_exit_reports_an_error() {
        let exec = start(vec![spec(0, "exit 0")]);
        finish(&exec).await;
        let err = exec.write_stdin(b"too late").await.unwrap_err();
        assert!(err.to_string().contains("stdin"), "{err}");
    }

    #[tokio::test]
    async fn attach_after_completion_replays_everything() {
        let exec = start(vec![spec(0, "echo replayed; exit 2")]);
        finish(&exec).await;
        let sub = exec.hub.subscribe();
        assert_eq!(sub.finished, Some(2));
        let text: String = sub
            .replay
            .iter()
            .filter_map(|p| match p {
                StreamPacket::Output { data, .. } => {
                    Some(String::from_utf8_lossy(data).into_owned())
                }
                _ => None,
            })
            .collect();
        assert!(text.contains("replayed"), "{text:?}");
    }

    #[tokio::test]
    async fn all_output_is_published_before_the_exit_packet() {
        // A client is documented to stop reading at `ExecExit`, so any
        // output published after it is lost. Output travels
        // pump -> channel -> hub, and waiting on the child only covers
        // the first hop; without waiting for the forwarder too, a process
        // that writes and exits immediately loses its final bytes. It
        // reproduces under load, so the loop matters.
        for i in 0..40 {
            let exec = start(vec![spec(0, &format!("echo marker-{i}"))]);
            finish(&exec).await;

            let packets = exec.hub.snapshot();
            let exit_at = packets
                .iter()
                .position(|p| matches!(p, StreamPacket::ExecExit { .. }))
                .expect("an exit must be published");
            let text: String = packets[..exit_at]
                .iter()
                .filter_map(|p| match p {
                    StreamPacket::Output { data, .. } => {
                        Some(String::from_utf8_lossy(data).into_owned())
                    }
                    _ => None,
                })
                .collect();
            assert!(
                text.contains(&format!("marker-{i}")),
                "round {i}: output was published after the exit packet, \
                 so a client would never see it. Packets: {packets:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_late_attacher_sees_the_full_output_of_a_finished_exec() {
        for i in 0..25 {
            let exec = start(vec![spec(0, &format!("echo late-{i}"))]);
            finish(&exec).await;

            // Subscribing only after completion is the `mirage logs` path.
            let sub = exec.hub.subscribe();
            let text: String = sub
                .replay
                .iter()
                .filter_map(|p| match p {
                    StreamPacket::Output { data, .. } => {
                        Some(String::from_utf8_lossy(data).into_owned())
                    }
                    _ => None,
                })
                .collect();
            assert!(text.contains(&format!("late-{i}")), "round {i}: {text:?}");
            assert!(sub.finished.is_some());
        }
    }

    #[tokio::test]
    async fn exactly_one_exec_exit_is_published() {
        let exec = start(vec![spec(0, "exit 0"), spec(1, "exit 0")]);
        finish(&exec).await;
        let exits = exec
            .hub
            .snapshot()
            .into_iter()
            .filter(|p| matches!(p, StreamPacket::ExecExit { .. }))
            .count();
        assert_eq!(exits, 1);
    }

    #[tokio::test]
    async fn every_rank_reports_a_node_exit() {
        let exec = start(vec![spec(0, "exit 0"), spec(1, "exit 1"), spec(2, "exit 2")]);
        finish(&exec).await;
        let mut seen: Vec<(u32, i32)> = exec
            .hub
            .snapshot()
            .into_iter()
            .filter_map(|p| match p {
                StreamPacket::NodeExit { node, exit_code } => Some((node, exit_code)),
                _ => None,
            })
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![(0, 0), (1, 1), (2, 2)]);
    }
}
