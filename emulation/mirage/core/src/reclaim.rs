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
//! [`crate::container::ENV_SESSION`] is set in every
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
//!
//! ## Which mirage
//!
//! A session name is not enough on its own. The question this module
//! answers is "does any live run account for this process?", and the list
//! of live runs it is given comes from the sockets in *the caller's*
//! runtime directory. Two mirages can be running under two different
//! `MIRAGE_RUNTIME` directories — a test suite beside an interactive
//! session, two CI jobs on one machine — and neither one's registry
//! mentions the other's sessions. Session name alone therefore reports a
//! perfectly healthy workload of the other runtime as stranded, and
//! reclaiming it means `SIGKILL`ing a running job that nothing is wrong
//! with.
//!
//! So the marker is a pair. Every workload also carries
//! [`crate::container::ENV_RUNTIME`], holding the resolved
//! runtime directory of the run that started it, and the scan ignores
//! anything whose recorded directory is not the caller's own — the same
//! pair, and the same rule, as the [`LABEL_SESSION`] and [`LABEL_RUNTIME`]
//! labels on containers.
//!
//! A process with no recorded runtime at all is ignored too, and that is
//! the deliberate half. It cannot be attributed to any runtime directory,
//! and killing what cannot be attributed is exactly the failure this pair
//! exists to prevent. The consequence is that a workload left behind by a
//! mirage older than [`ENV_RUNTIME`] is never reclaimed by this one — a
//! leak, recoverable with `kill` by the user who can see it, and a much
//! smaller wrong than destroying somebody's running job.
//!
//! Skipped, though, is not the same as unmentioned. "`mirage cleanup`
//! found nothing" and "`mirage cleanup` found a running process it dared
//! not touch" are different facts about the machine, and only the second
//! one leaves the user with something to do. [`scan`] therefore returns
//! both lists — what it will reclaim, and what it deliberately would not
//! — so a caller can reclaim the first and report the second instead of
//! discarding it. `kill` by hand is only an option for a user who has
//! been told the process is there.
//!
//! [`LABEL_SESSION`]: crate::container::LABEL_SESSION
//! [`LABEL_RUNTIME`]: crate::container::LABEL_RUNTIME

use std::collections::HashSet;

use crate::container::{ENV_RUNTIME, ENV_SESSION, owning_runtime, same_runtime};
use crate::session::SessionId;

/// A workload process belonging to a session that is no longer live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stranded {
    /// The process id.
    pub pid: u32,
    /// The session it was started for.
    pub session: SessionId,
}

/// What one pass over `/proc` found: what may be reclaimed, and what was
/// deliberately left alone.
///
/// Two lists rather than one because the second is news. A process that
/// cannot be attributed to a runtime directory is skipped — see *Which
/// mirage* in the [module documentation](self) — and a caller that only
/// ever saw the first list reported "nothing to clean up" about a machine
/// with a stranded workload still running on it, and then deleted that
/// session's scratch directory, which was the last record anything had of
/// it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Scan {
    /// Workloads this runtime directory started whose session no live run
    /// accounts for. These are the ones [`reap`] may kill.
    pub reclaimable: Vec<Stranded>,
    /// Workloads tagged with a session but with no runtime directory
    /// recorded, so there is no way to tell whose they are. Never killed;
    /// always worth saying.
    pub unattributable: Vec<Stranded>,
}

impl Scan {
    /// The sessions that have an unattributable process still running.
    ///
    /// A session in this set is not finished with, whatever the socket
    /// registry says, so nothing of it should be thrown away — its
    /// scratch directory least of all, since with the process
    /// unattributable that directory is the only remaining evidence the
    /// session ever existed.
    #[must_use]
    pub fn unattributable_sessions(&self) -> std::collections::BTreeSet<&str> {
        self.unattributable
            .iter()
            .map(|s| s.session.as_str())
            .collect()
    }
}

/// Every process this runtime directory started whose session is not in
/// `live`.
///
/// Only processes this user owns are considered: `/proc/<pid>/environ` is
/// readable by its owner alone, so another user's process is skipped
/// rather than reported as unreclaimable.
///
/// Only processes started under the caller's own runtime directory are
/// considered either, and a process that does not say which directory
/// started it is skipped rather than claimed. See *Which mirage* in the
/// [module documentation](self) for why both halves of that are
/// deliberate.
///
/// The caller's own process is never returned, and neither is anything
/// sharing the caller's session. Running `mirage cleanup` from inside a
/// shell started by `mirage exec` is not exotic — it is exactly what a
/// user does when that session has gone wrong — and reclaiming the
/// session you are standing in would kill your own shell mid-command.
#[must_use]
pub fn stranded_workloads(live: &[SessionId]) -> Vec<Stranded> {
    scan(live).reclaimable
}

/// One pass over `/proc`, keeping both what may be reclaimed and what was
/// skipped for want of a runtime mark.
///
/// The filters are exactly those [`stranded_workloads`] describes; the
/// difference is only that a candidate rejected by the runtime test — and
/// rejected for the one reason a caller can act on, that it records no
/// runtime at all — is kept rather than dropped. A process belonging to a
/// *different* runtime directory is still dropped without a word: it is
/// somebody else's healthy job, and it is not this cleanup's business
/// that it exists.
#[must_use]
pub fn scan(live: &[SessionId]) -> Scan {
    let mut excluded: HashSet<String> = live.iter().map(|s| s.as_str().to_string()).collect();
    if let Ok(own) = std::env::var(ENV_SESSION) {
        excluded.insert(own);
    }
    let me = std::process::id();
    // Resolved once for the whole scan rather than per candidate: this
    // walks every entry in `/proc`, and canonicalising the same directory
    // for each of them is a syscall per process on the machine.
    let ours = owning_runtime();

    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Scan::default();
    };
    let mut out = Scan::default();
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == me {
            continue;
        }
        let Some((session, runtime)) = marker_of(pid) else {
            continue;
        };
        if excluded.contains(session.as_str()) {
            continue;
        }
        match runtime {
            Some(recorded) if same_runtime(&recorded, &ours) => {
                out.reclaimable.push(Stranded { pid, session });
            }
            // Someone else's runtime directory, and so none of this
            // cleanup's business either to kill or to mention.
            Some(_) => {}
            None => out.unattributable.push(Stranded { pid, session }),
        }
    }
    // Deterministic order, so the summary a user reads is stable and two
    // runs of `--dry-run` against the same machine agree.
    let by_session_then_pid =
        |a: &Stranded, b: &Stranded| (a.session.as_str(), a.pid).cmp(&(b.session.as_str(), b.pid));
    out.reclaimable.sort_by(by_session_then_pid);
    out.unattributable.sort_by(by_session_then_pid);
    out
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

/// The session a process's environment names, and the runtime directory
/// it records, if it is tagged at all.
///
/// The runtime is `None` for a process tagged by a mirage that predates
/// [`ENV_RUNTIME`]; the session is what makes a process a candidate, so
/// its absence ends the read.
fn marker_of(pid: u32) -> Option<(SessionId, Option<String>)> {
    // Not `read_to_string`: an environment is arbitrary bytes and need not
    // be UTF-8, and one invalid byte in an unrelated variable must not
    // hide the tag.
    let environ = std::fs::read(format!("/proc/{pid}/environ")).ok()?;
    let session_prefix = format!("{ENV_SESSION}=");
    let runtime_prefix = format!("{ENV_RUNTIME}=");
    let mut session = None;
    let mut runtime = None;
    for entry in environ.split(|b| *b == 0) {
        let Ok(entry) = std::str::from_utf8(entry) else {
            continue;
        };
        // First occurrence wins for each, which is what `execve` gives a
        // process that was handed a duplicate name.
        if let Some(value) = entry.strip_prefix(&session_prefix) {
            session.get_or_insert_with(|| value.to_string());
        } else if let Some(value) = entry.strip_prefix(&runtime_prefix) {
            runtime.get_or_insert_with(|| value.to_string());
        }
    }
    Some((SessionId::new(session?).ok()?, runtime))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::container::PinnedRuntime;

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

        /// A process carrying the whole marker: this session, and this
        /// runtime directory as its owner.
        fn script(session: &str, script: &str) -> Self {
            Self::owned_by(session, Some(&owning_runtime()), script)
        }

        /// A process tagged with `session`, and with `runtime` as the
        /// directory that owns it — or with no runtime at all, as one
        /// started by a mirage older than the marker's second half would
        /// be.
        fn owned_by(session: &str, runtime: Option<&str>, script: &str) -> Self {
            Self::marked(Some(session), runtime, script)
        }

        /// A process carrying whichever halves of the marker are given.
        ///
        /// Both are optional so a test can withhold exactly one of them
        /// and leave the other in place; a candidate rejected for two
        /// reasons at once proves neither of them.
        fn marked(session: Option<&str>, runtime: Option<&str>, script: &str) -> Self {
            let mut command = std::process::Command::new("/bin/sh");
            command
                .args(["-c", script])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            match session {
                Some(session) => command.env(ENV_SESSION, session),
                // Removed for the same reason the runtime is below: the
                // suite may be run from inside a `mirage exec` shell,
                // which exports it.
                None => command.env_remove(ENV_SESSION),
            };
            match runtime {
                Some(runtime) => command.env(ENV_RUNTIME, runtime),
                // Removed rather than left unset: the suite may well be
                // run from a shell that exported it, and inheriting it
                // would make an "untagged" process silently tagged.
                None => command.env_remove(ENV_RUNTIME),
            };
            Self(command.spawn().unwrap())
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
        let _runtime = PinnedRuntime::new();
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
    fn a_dead_session_of_our_own_runtime_is_still_reported() {
        // The feature itself, asserted against an explicitly stamped
        // marker rather than the helper's default, because the cheapest
        // way to "fix" a cleanup that kills too much is to break
        // reclamation altogether and never report anything again.
        let _runtime = PinnedRuntime::new();
        let session = unique("ours");
        let child = Tagged::owned_by(session.as_str(), Some(&owning_runtime()), "exec sleep 300");
        let found = stranded_workloads(&[]);
        assert!(
            found.iter().any(|s| s.pid == child.pid()),
            "a workload of this runtime whose session is not live must be \
             reclaimable: {found:?}"
        );
    }

    #[test]
    fn a_process_owned_by_another_runtime_is_not_a_candidate() {
        // The bug this pair of markers exists for. `mirage cleanup` under
        // one `MIRAGE_RUNTIME` has never heard of a session belonging to
        // another, so by session name alone a perfectly healthy workload
        // of that other mirage is an orphan — and reclaiming it means
        // `SIGKILL`ing a running job.
        let _runtime = PinnedRuntime::new();
        let elsewhere = tempfile::tempdir().unwrap();
        let session = unique("elsewhere");
        let child = Tagged::owned_by(
            session.as_str(),
            Some(&elsewhere.path().to_string_lossy()),
            "exec sleep 300",
        );
        let found = stranded_workloads(&[]);
        assert!(
            !found.iter().any(|s| s.pid == child.pid()),
            "a live workload of another runtime directory was reported as \
             stranded: {found:?}"
        );
    }

    #[test]
    fn a_process_with_no_recorded_runtime_is_not_a_candidate() {
        // Tagged with a session but by a mirage that predates the runtime
        // half of the marker, so there is no way to tell whose it is.
        // Skipping leaks it; killing it is the cross-runtime kill in a
        // different disguise, because "no runtime recorded" is exactly
        // what somebody else's process looks like.
        let _runtime = PinnedRuntime::new();
        let session = unique("unmarked");
        let child = Tagged::owned_by(session.as_str(), None, "exec sleep 300");
        let found = stranded_workloads(&[]);
        assert!(
            !found.iter().any(|s| s.pid == child.pid()),
            "an unattributable process was reported as stranded: {found:?}"
        );
    }

    #[test]
    fn a_live_session_is_left_alone() {
        // The whole safety property: `mirage cleanup` runs while other
        // runs are healthy, and must reclaim only what no live run
        // accounts for.
        let _runtime = PinnedRuntime::new();
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
        let _runtime = PinnedRuntime::new();
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
        // disappearing outright; the exit status is what says which
        // signal landed. Which signal, not merely "it did not exit 0":
        // `SIGTERM` also fails that weaker assertion, and `SIGTERM` is
        // the wrong answer here — these processes are supervised by
        // nobody and waited on by nobody, so a grace period preserves an
        // exit status for a reader who does not exist, and a workload
        // that ignores `SIGTERM` (which is why the supervisor's own
        // teardown escalates) would simply survive the cleanup.
        use std::os::unix::process::ExitStatusExt as _;
        let status = child.0.wait().unwrap();
        assert!(!status.success(), "the process should have been killed");
        assert_eq!(
            status.signal(),
            Some(nix::sys::signal::Signal::SIGKILL as i32),
            "a reclaimed workload must be killed outright, not asked: {status:?}"
        );
    }

    #[test]
    fn a_grandchild_that_inherited_the_tag_is_reclaimed_too() {
        // The reason the marker is an environment variable and not a pid
        // file: mirage only knows the pids of the ranks it spawned, and a
        // workload that forks — a shell script, `torchrun`, an MPI
        // launcher — leaves descendants that no such file would name.
        // They inherit the environment, so they inherit the tag.
        let _runtime = PinnedRuntime::new();
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

    #[test]
    fn an_unattributable_process_is_reported_rather_than_silently_skipped() {
        // Not killing it is right: "no runtime recorded" is exactly what
        // somebody else's process looks like, and killing what cannot be
        // attributed is the bug this whole marker exists to prevent. But
        // a cleanup that skips it *and says nothing* tells the user their
        // machine is clean while a workload of theirs is still running on
        // it — and the caller then deleted that session's scratch
        // directory, the last record anything had that the session had
        // ever existed.
        let _runtime = PinnedRuntime::new();
        let session = unique("unattributable");
        let child = Tagged::owned_by(session.as_str(), None, "exec sleep 300");

        let scan = scan(&[]);
        assert!(
            !scan.reclaimable.iter().any(|s| s.pid == child.pid()),
            "an unattributable process must never be reclaimed: {scan:?}"
        );
        assert!(
            scan.unattributable.iter().any(|s| s.pid == child.pid()),
            "an unattributable process must still be reported: {scan:?}"
        );
        assert!(
            scan.unattributable_sessions().contains(session.as_str()),
            "its session has something running and is not finished with: {scan:?}"
        );
    }

    #[test]
    fn a_process_of_another_runtime_is_not_even_mentioned() {
        // The other side of the report: an unattributable process is the
        // caller's problem to hear about, and a healthy workload of a
        // different `MIRAGE_RUNTIME` is not. Listing it would invite
        // exactly the "clean it up then" reaction the runtime mark exists
        // to prevent.
        let _runtime = PinnedRuntime::new();
        let elsewhere = tempfile::tempdir().unwrap();
        let session = unique("elsewhere-report");
        let child = Tagged::owned_by(
            session.as_str(),
            Some(&elsewhere.path().to_string_lossy()),
            "exec sleep 300",
        );
        let scan = scan(&[]);
        assert!(
            !scan.reclaimable.iter().any(|s| s.pid == child.pid()),
            "{scan:?}"
        );
        assert!(
            !scan.unattributable.iter().any(|s| s.pid == child.pid()),
            "another runtime's healthy workload is not this cleanup's news: {scan:?}"
        );
    }

    #[test]
    fn a_tag_is_found_beside_an_environment_variable_that_is_not_utf8() {
        // An environment is arbitrary bytes. `read_to_string` on
        // `/proc/<pid>/environ` fails outright on the first invalid one,
        // so a single non-UTF-8 variable anywhere in the block — a path
        // in a legacy encoding, a locale-mangled value inherited from the
        // shell — hid the session tag and made the whole workload
        // invisible to `mirage cleanup`. It is read as bytes and split on
        // NUL for exactly this, and nothing was holding that in place.
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt as _;

        let _runtime = PinnedRuntime::new();
        let session = unique("nonutf8");
        let mut command = std::process::Command::new("/bin/sh");
        command
            .args(["-c", "exec sleep 300"])
            .env(ENV_SESSION, session.as_str())
            .env(ENV_RUNTIME, owning_runtime())
            // Invalid UTF-8: a lone 0xff can begin no sequence. Not a NUL
            // and not an `=`, so it stays one variable in the block and
            // the only thing wrong with it is that it cannot be decoded.
            .env("MIRAGE_TEST_NON_UTF8", OsString::from_vec(vec![0xff, 0xfe]))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let child = Tagged(command.spawn().unwrap());

        let found = mine(&session);
        assert!(
            found.iter().any(|s| s.pid == child.pid()),
            "a workload was hidden by an unrelated variable that is not \
             UTF-8: {found:?}"
        );
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
        // Every process on the machine is scanned, so the session tag is
        // the only thing standing between this and killing unrelated
        // work — and this test has to make it the only thing. The child
        // therefore carries the runtime half of the marker, which is the
        // shape of anything started from a shell that exported
        // `MIRAGE_RUNTIME` to point mirage at a directory: it passes the
        // runtime filter, so the missing session tag is the sole reason
        // it must not be reported. Withholding both halves instead — as
        // this test used to — leaves the runtime filter rejecting the
        // child on its own, and the session check can be deleted outright
        // without the assertion noticing.
        let _runtime = PinnedRuntime::new();
        let plain = Tagged::marked(None, Some(&owning_runtime()), "exec sleep 300");
        let found = stranded_workloads(&[]);
        assert!(
            !found.iter().any(|s| s.pid == plain.pid()),
            "an untagged process was reported as stranded: {found:?}"
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
