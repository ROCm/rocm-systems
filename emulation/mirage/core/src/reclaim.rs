//! Finding and removing the workload processes a dead run left behind.
//!
//! This is the process-side counterpart of
//! [`container::reclaim_orphans`](crate::container::reclaim_orphans), and
//! it exists for the same reason. A session lives in the memory of the
//! `mirage run` that owns it, so a run that was `SIGKILL`ed — by an
//! impatient `kill -9`, by the OOM killer during a large emulated job, by
//! a machine losing power — takes its record of every process with it.
//! The workloads are reparented to init and are then owned by nobody:
//! still holding the emulated device, still burning cores, and invisible
//! to every later mirage.
//!
//! Mirage used to prevent that with `PR_SET_PDEATHSIG`, which asked the
//! kernel to kill each workload when its parent died. That was the
//! workspace's only `unsafe`, and it could not be applied to the one
//! spawn path that leaks the most — node containers, launched from a
//! `spawn_blocking` thread where pdeathsig tracks a pool thread that may
//! retire while the run is perfectly healthy. The leak is accepted
//! instead, and made recoverable here.
//!
//! # The marker
//!
//! Recovery needs something that survives the death of the only process
//! that knew about the session, which rules out anything mirage would
//! have had to *write*: a pid file is written before the crash and is
//! stale after it, and a stale pid is a pid the kernel may have reissued
//! to something unrelated.
//!
//! So the marker lives on the process itself:
//! [`ENV_SESSION`](crate::container::ENV_SESSION) is set in every
//! workload's environment. It cannot go stale, because it is gone the
//! moment the process is; there is no recycling window, because the pid
//! and the evidence are read from the same `/proc` entry; and it is
//! inherited, so a workload's forked grandchildren carry it too — which a
//! pid file recording only the ranks mirage spawned would have missed.
//!
//! The cost of inheritance is the same property in reverse: anything
//! started from an interactive `mirage exec -- bash` inherits the tag and
//! is reclaimed with that session. That is the intended reading — it is
//! part of the session's process tree — but it is worth knowing before
//! running a build from inside one.

use std::collections::HashSet;

use crate::container::ENV_SESSION;
use crate::session::SessionId;

/// A workload process belonging to a session that is no longer live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stranded {
    /// The process id.
    pub pid: u32,
    /// The session it was started for.
    pub session: SessionId,
}

/// Every process tagged with a session that is not in `live`.
///
/// Only processes this user owns are considered: `/proc/<pid>/environ` is
/// readable by its owner alone, so another user's process is skipped
/// rather than reported as unreclaimable.
///
/// The caller's own process is never returned, and neither is anything
/// sharing the caller's session. Running `mirage cleanup` from inside a
/// shell started by `mirage exec` is not exotic — it is exactly what a
/// user does when that session has gone wrong — and reclaiming the
/// session you are standing in would kill your own shell mid-command.
#[must_use]
pub fn stranded_workloads(live: &[SessionId]) -> Vec<Stranded> {
    let mut excluded: HashSet<String> = live.iter().map(|s| s.as_str().to_string()).collect();
    if let Ok(own) = std::env::var(ENV_SESSION) {
        excluded.insert(own);
    }
    let me = std::process::id();

    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut found: Vec<Stranded> = entries
        .flatten()
        .filter_map(|entry| {
            let pid: u32 = entry.file_name().to_str()?.parse().ok()?;
            if pid == me {
                return None;
            }
            let session = session_of(pid)?;
            if excluded.contains(session.as_str()) {
                return None;
            }
            Some(Stranded { pid, session })
        })
        .collect();
    // Deterministic order, so the summary a user reads is stable and two
    // runs of `--dry-run` against the same machine agree.
    found.sort_by(|a, b| (a.session.as_str(), a.pid).cmp(&(b.session.as_str(), b.pid)));
    found
}

/// `SIGKILL` each of `stranded`.
///
/// Takes the list rather than re-deriving it so a caller can act on
/// exactly what it reported. `mirage cleanup --dry-run` prints the scan
/// and a real run kills it, and re-scanning in between would let the two
/// disagree about what was found.
///
/// `SIGKILL` and not an escalation: these processes are, by construction,
/// ones nothing is supervising and nothing is waiting on, so there is no
/// exit status for a grace period to preserve and nobody to report it to.
/// The escalation in `mirage_supervisor::process::terminate` is for
/// teardown, where the run is alive and the workload's own cleanup still
/// means something.
///
/// The pid alone is signalled, not `kill(-pid)`. Every workload leads its
/// own process group, so the group form would also be correct — but the
/// scan already returns every descendant that kept the tag, and a group
/// signal would reach processes that had *dropped* it by joining a
/// mirage-led group, which is a wider claim than the marker supports.
pub fn reap(stranded: &[Stranded]) {
    for s in stranded {
        let Ok(raw) = i32::try_from(s.pid) else {
            continue;
        };
        // Guarded for the same reason as `process::signal_group`: a
        // non-positive pid would address every process the user can
        // signal, or the caller's own group. `read_dir` on `/proc` cannot
        // produce one, which is exactly why it is cheap to assert.
        if raw <= 0 {
            continue;
        }
        if let Err(e) = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(raw),
            nix::sys::signal::Signal::SIGKILL,
        ) {
            tracing::debug!(pid = s.pid, session = %s.session, "could not reclaim: {e}");
        }
    }
}

/// Scan for stranded workloads and kill them, returning what was killed.
pub fn reap_stranded(live: &[SessionId]) -> Vec<Stranded> {
    let stranded = stranded_workloads(live);
    reap(&stranded);
    stranded
}

/// The session named in a process's environment, if it has one.
fn session_of(pid: u32) -> Option<SessionId> {
    // Not `read_to_string`: an environment is arbitrary bytes and need not
    // be UTF-8, and one invalid byte in an unrelated variable must not
    // hide the tag.
    let environ = std::fs::read(format!("/proc/{pid}/environ")).ok()?;
    let prefix = format!("{ENV_SESSION}=");
    environ
        .split(|b| *b == 0)
        .find_map(|entry| {
            std::str::from_utf8(entry)
                .ok()?
                .strip_prefix(&prefix)
                .map(str::to_string)
        })
        .and_then(|value| SessionId::new(value).ok())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// A tagged child, killed when the guard is dropped so a failing
    /// assertion cannot leave a `sleep` behind for the rest of the suite.
    struct Tagged(std::process::Child);

    impl Tagged {
        /// One tagged process, and exactly one.
        ///
        /// `exec` rather than a `while` loop: a shell looping over `sleep`
        /// has a child for most of its life, that child inherits the tag,
        /// and a test counting processes then races the loop.
        fn spawn(session: &str) -> Self {
            Self::script(session, "exec sleep 300")
        }

        fn script(session: &str, script: &str) -> Self {
            let child = std::process::Command::new("/bin/sh")
                .args(["-c", script])
                .env(ENV_SESSION, session)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap();
            Self(child)
        }

        fn pid(&self) -> u32 {
            self.0.id()
        }
    }

    impl Drop for Tagged {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    /// A session id no other test or machine will collide with.
    fn unique(name: &str) -> SessionId {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        SessionId::new(format!(
            "reclaimtest-{name}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ))
        .unwrap()
    }

    #[test]
    fn a_tagged_process_is_found_by_its_session() {
        let session = unique("found");
        let child = Tagged::spawn(session.as_str());
        let found = stranded_workloads(&[]);
        assert!(
            found.contains(&Stranded {
                pid: child.pid(),
                session: session.clone(),
            }),
            "a process tagged {session} was not found: {found:?}"
        );
    }

    #[test]
    fn a_live_session_is_left_alone() {
        // The whole safety property: `mirage cleanup` runs while other
        // runs are healthy, and must reclaim only what no live run
        // accounts for.
        let session = unique("live");
        let child = Tagged::spawn(session.as_str());
        let found = stranded_workloads(std::slice::from_ref(&session));
        assert!(
            !found.iter().any(|s| s.pid == child.pid()),
            "a live session's workload was reported as stranded: {found:?}"
        );
    }

    #[test]
    fn reaping_kills_the_process() {
        let session = unique("reap");
        let mut child = Tagged::spawn(session.as_str());
        let pid = child.pid();

        // Only this test's own process. `reap_stranded(&[])` would be
        // machine-wide, and the rest of this module's tests hold tagged
        // children of their own while running in parallel with it.
        let mine = mine(&session);
        assert_eq!(
            mine.len(),
            1,
            "expected exactly this test's child: {mine:?}"
        );
        assert_eq!(mine[0].pid, pid);
        reap(&mine);

        // The child is ours, so it becomes a zombie rather than
        // disappearing outright; the exit status is what says the signal
        // landed.
        let status = child.0.wait().unwrap();
        assert!(!status.success(), "the process should have been killed");
    }

    #[test]
    fn a_grandchild_that_inherited_the_tag_is_reclaimed_too() {
        // The reason the marker is an environment variable and not a pid
        // file: mirage only knows the pids of the ranks it spawned, and a
        // workload that forks — a shell script, `torchrun`, an MPI
        // launcher — leaves descendants that no such file would name.
        // They inherit the environment, so they inherit the tag.
        let session = unique("grandchild");
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("grandchild.pid");
        let child = Tagged::script(
            session.as_str(),
            &format!(
                "sleep 300 & echo $! > {marker}; wait",
                marker = marker.display()
            ),
        );

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let grandchild = loop {
            if let Ok(text) = std::fs::read_to_string(&marker)
                && let Ok(pid) = text.trim().parse::<u32>()
            {
                break pid;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "grandchild never started"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        };

        let found = mine(&session);
        assert!(
            found.iter().any(|s| s.pid == child.pid()),
            "the workload itself was not found: {found:?}"
        );
        assert!(
            found.iter().any(|s| s.pid == grandchild),
            "a forked grandchild was not found: {found:?}"
        );

        reap(&found);
        let gone = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while std::path::Path::new(&format!("/proc/{grandchild}")).exists() {
            assert!(
                std::time::Instant::now() < gone,
                "grandchild {grandchild} survived the reap"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// The stranded processes belonging to one test's session.
    ///
    /// These tests run alongside each other and alongside whatever else
    /// is on the machine, so nothing here may act on an unfiltered scan.
    fn mine(session: &SessionId) -> Vec<Stranded> {
        stranded_workloads(&[])
            .into_iter()
            .filter(|s| &s.session == session)
            .collect()
    }

    #[test]
    fn an_untagged_process_is_never_a_candidate() {
        // Every process on the machine is scanned, so the tag is the only
        // thing standing between this and killing unrelated work.
        let mut plain = std::process::Command::new("/bin/sh")
            .args(["-c", "while true; do sleep 1; done"])
            .env_remove(ENV_SESSION)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let pid = plain.id();
        let found = stranded_workloads(&[]);
        let _ = plain.kill();
        let _ = plain.wait();
        assert!(
            !found.iter().any(|s| s.pid == pid),
            "an untagged process was reported as stranded"
        );
    }

    #[test]
    fn the_scan_never_reports_the_caller() {
        // `mirage cleanup` run from inside a `mirage exec -- bash` of the
        // very session being cleaned would otherwise kill itself.
        let found = stranded_workloads(&[]);
        assert!(!found.iter().any(|s| s.pid == std::process::id()));
    }
}
