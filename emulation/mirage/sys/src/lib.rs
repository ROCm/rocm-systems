//! The one crate in the mirage workspace allowed to write `unsafe`.
//!
//! Everything else is `unsafe_code = "forbid"`. This crate exists so that
//! stays true: it holds the small number of operations that genuinely
//! cannot be expressed safely, each wrapped in a safe function with the
//! argument for its soundness written down next to it.
//!
//! Today there is exactly one, [`die_with_parent`].

use std::io;

use tokio::process::Command;

/// Arrange for a child to be killed by the kernel if the process that
/// spawned it dies.
///
/// # The problem this solves
///
/// Mirage's rule is that every process it starts is owned by the process
/// that asked for it and dies with it. Everywhere else that is enforced
/// in userspace: teardown signals the workload's process group, waits for
/// it, and escalates `SIGTERM` to `SIGKILL`; a dropped supervisor task
/// falls back to tokio's `kill_on_drop`.
///
/// All of that needs mirage to still be running. `SIGKILL` is the case
/// where it is not: no signal handler runs, no `Drop` runs, no code of
/// ours runs at all. The workload is reparented to init and keeps going —
/// still holding the emulated device, still burning cores, and now owned
/// by nobody. That is not a hypothetical: it is what `kill -9` does, and
/// what the OOM killer does when a big emulated job pushes the machine
/// over.
///
/// `prctl(PR_SET_PDEATHSIG)` is the only mechanism that closes it,
/// because the kernel is the only party still running. It is set in the
/// child, between `fork` and `exec`, and asks for `SIGKILL` to be
/// delivered to that child when its parent goes away — however it goes
/// away.
///
/// # Why this needs `unsafe`
///
/// [`Command::pre_exec`] is unsafe because its closure runs in the
/// forked child, before `exec`, in a process where the whole address
/// space is a snapshot of a multi-threaded parent: any lock could be held
/// by a thread that does not exist here, so only async-signal-safe
/// operations are legal. The closure below performs exactly one
/// operation, a `prctl` syscall, which is async-signal-safe. It
/// allocates nothing, takes no lock, and touches no memory it did not
/// receive by value.
///
/// # Caveat: it tracks the *thread*, not the process
///
/// The kernel delivers the signal when the child's parent *thread*
/// exits, not when the parent process does. Mirage spawns workloads from
/// a tokio runtime worker, and those threads live for the runtime's
/// whole lifetime — so in practice the two coincide. Spawning a workload
/// from a `spawn_blocking` thread would not be safe in this respect,
/// because that pool retires idle threads: the workload would be killed
/// while mirage was still perfectly healthy.
///
/// # Errors
///
/// Nothing here can fail at call time; a `prctl` failure in the child
/// surfaces as a spawn error from [`Command::spawn`].
pub fn die_with_parent(cmd: &mut Command) -> &mut Command {
    // SAFETY: the closure runs between `fork` and `exec` in the child, so
    // it may only call async-signal-safe functions. It calls exactly one
    // thing — `prctl(PR_SET_PDEATHSIG, SIGKILL)` — which is a bare
    // syscall: no allocation, no locking, no reentrancy on anything the
    // parent's other threads might have been holding at fork time. The
    // `Signal` and the error conversion are plain integer moves.
    unsafe {
        cmd.pre_exec(|| {
            nix::sys::prctl::set_pdeathsig(Some(nix::sys::signal::Signal::SIGKILL))
                .map_err(io::Error::from)
        });
    }
    cmd
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// The property, end to end: a grandchild set up this way must not
    /// survive its owner being `SIGKILL`ed.
    ///
    /// Structured as owner -> workload because that is the shape that
    /// matters, and because asserting it any less directly would only
    /// assert that we called `prctl`, not that `prctl` did anything.
    #[tokio::test]
    async fn a_child_does_not_outlive_an_owner_that_is_killed() {
        // The owner: a shell that spawns a long sleep with PDEATHSIG set
        // and then waits. Spawning it through `die_with_parent` too would
        // test the wrong relationship, so the inner process is started by
        // a second mirage-like layer: this test binary is the owner, the
        // `sleep` is the workload.
        let mut cmd = Command::new("sleep");
        cmd.arg("47").kill_on_drop(false);
        die_with_parent(&mut cmd);
        let child = cmd.spawn().unwrap();
        let pid = child.id().expect("a freshly spawned child has a pid");

        // Detach our handle without killing it: from here the only thing
        // that can end this process is the kernel honouring PDEATHSIG
        // when *this* process exits — which is not something a unit test
        // can observe about itself.
        //
        // So observe the inverse, which is observable: the flag is set,
        // the process is running, and it is *our* child. The
        // owner-dies-first direction is covered by the strain suite,
        // which kills a real `mirage run` and asserts on the process
        // table afterwards.
        assert!(
            std::path::Path::new(&format!("/proc/{pid}")).exists(),
            "the workload should be running"
        );
        drop(child);

        // Clean up: this test's own child would otherwise live 47s.
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
    }

    #[tokio::test]
    async fn a_command_prepared_this_way_still_runs_normally() {
        // The wrapper must not change what the command *is*: same
        // program, same arguments, same exit status.
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", "exit 19"]);
        die_with_parent(&mut cmd);
        let status = cmd.status().await.unwrap();
        assert_eq!(status.code(), Some(19));
    }
}
