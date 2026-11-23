// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Bounded core-pipe output with stable helper identity, even when the
//! application has its own process-wide child reaper.

use std::io::{self, Write};
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixDatagram;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Constructs a diagnostic error outside the pre-exec child.
fn native(message: &str) -> io::Error {
    io::Error::other(message)
}

/// Bounds each side of the bootstrap exchange independently of handler runtime.
const HANDOFF_TIMEOUT_MS: i32 = 5000;

/// Waits with a monotonic deadline using only child-safe native operations.
fn wait_readable(socket: RawFd, timeout_ms: i32) -> io::Result<()> {
    // SAFETY: clock_gettime and poll use initialized stack records and neither
    // allocate nor acquire application locks. EINTR does not restart the budget.
    unsafe {
        let mut now: libc::timespec = std::mem::zeroed();
        if libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now) != 0 {
            return Err(io::Error::last_os_error());
        }
        let deadline = i128::from(now.tv_sec) * 1_000_000_000
            + i128::from(now.tv_nsec)
            + i128::from(timeout_ms) * 1_000_000;
        loop {
            if libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now) != 0 {
                return Err(io::Error::last_os_error());
            }
            let remaining =
                deadline - (i128::from(now.tv_sec) * 1_000_000_000 + i128::from(now.tv_nsec));
            if remaining <= 0 {
                return Err(io::ErrorKind::TimedOut.into());
            }
            let millis = i32::try_from((remaining + 999_999) / 1_000_000)
                .map_err(|_| io::Error::from_raw_os_error(libc::EOVERFLOW))?;
            let mut descriptor = libc::pollfd {
                fd: socket,
                events: libc::POLLIN,
                revents: 0,
            };
            let ready = libc::poll(&raw mut descriptor, 1, millis);
            if ready > 0 {
                return Ok(());
            }
            if ready == 0 {
                continue;
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}

/// Sends approval or rejection without risking SIGPIPE on a failed exchange.
fn send_decision(socket: RawFd, approved: bool) -> io::Result<()> {
    let byte = u8::from(approved);
    // SAFETY: send synchronously reads one live byte from this private socket.
    let sent = unsafe {
        libc::send(
            socket,
            (&raw const byte).cast(),
            1,
            libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
        )
    };
    if sent == 1 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Allows exec only after the parent has retained cleanup ownership.
fn await_approval(socket: RawFd) -> io::Result<()> {
    wait_readable(socket, HANDOFF_TIMEOUT_MS)?;
    let mut byte = 0_u8;
    // SAFETY: recv writes one initialized stack byte, without allocation/locks.
    let received = unsafe { libc::recv(socket, (&raw mut byte).cast(), 1, libc::MSG_DONTWAIT) };
    if received == 1 && byte == 1 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(libc::ECANCELED))
    }
}

/// Opens identity in the child itself, before arbitrary helper code can exit.
/// Only raw Linux syscalls and initialized stack storage run after fork.
fn send_own_pidfd(socket: RawFd) -> io::Result<()> {
    // SAFETY: No allocation or application lock is used after fork. The kernel
    // synchronously copies these stack records. A successful SCM_RIGHTS send
    // retains the pidfd independently of process exit and external reaping.
    unsafe {
        let fd = libc::syscall(libc::SYS_pidfd_open, libc::getpid(), 0) as i32;
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // Require stable group signaling before exec can create descendants.
        if libc::syscall(
            libc::SYS_pidfd_send_signal,
            fd,
            0,
            std::ptr::null::<libc::siginfo_t>(),
            libc::PIDFD_SIGNAL_PROCESS_GROUP,
        ) != 0
        {
            let error = io::Error::last_os_error();
            libc::close(fd);
            return Err(error);
        }
        let mut byte = 1_u8;
        let mut vector = libc::iovec {
            iov_base: (&raw mut byte).cast(),
            iov_len: 1,
        };
        let mut control = [0_usize; 4];
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &raw mut vector;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = libc::CMSG_SPACE(size_of::<i32>() as u32) as usize;
        let cmsg = libc::CMSG_FIRSTHDR(&raw const message);
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(size_of::<i32>() as u32) as usize;
        libc::CMSG_DATA(cmsg).cast::<i32>().write(fd);
        let sent = libc::sendmsg(
            socket,
            &raw const message,
            libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
        );
        let error = io::Error::last_os_error();
        libc::close(fd);
        if sent == 1 { Ok(()) } else { Err(error) }
    }
}

/// Receives the one descriptor queued by the gated bootstrap child.
fn receive_pidfd(socket: RawFd) -> io::Result<OwnedFd> {
    // SAFETY: The socket is private to this spawn and its peer sent precisely
    // one descriptor. All receive buffers are initialized and suitably aligned.
    unsafe {
        let mut byte = 0_u8;
        let mut vector = libc::iovec {
            iov_base: (&raw mut byte).cast(),
            iov_len: 1,
        };
        let mut control = [0_usize; 4];
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &raw mut vector;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = size_of_val(&control);
        let received = libc::recvmsg(
            socket,
            &raw mut message,
            libc::MSG_DONTWAIT | libc::MSG_CMSG_CLOEXEC,
        );
        if received < 0 {
            return Err(io::Error::last_os_error());
        }
        let cmsg = libc::CMSG_FIRSTHDR(&raw const message);
        let fd = (!cmsg.is_null()
            && (*cmsg).cmsg_level == libc::SOL_SOCKET
            && (*cmsg).cmsg_type == libc::SCM_RIGHTS
            && (*cmsg).cmsg_len == libc::CMSG_LEN(size_of::<i32>() as u32) as usize)
            .then(|| OwnedFd::from_raw_fd(libc::CMSG_DATA(cmsg).cast::<i32>().read()));
        if message.msg_flags & libc::MSG_CTRUNC != 0 {
            return Err(native("truncated core handler pidfd message"));
        }
        if received != 1 || byte != 1 || message.msg_flags & libc::MSG_TRUNC != 0 {
            return Err(native("invalid core handler pidfd message"));
        }
        fd.ok_or_else(|| native("missing core handler pidfd"))
    }
}

/// Owns stable cleanup identity from receipt through successful consumption.
struct PipeHandler {
    /// Original process/group identity, independent of application child reaping.
    pidfd: OwnedFd,
    /// Successful consumption releases the group without further signaling.
    complete: bool,
}

impl Drop for PipeHandler {
    fn drop(&mut self) {
        if !self.complete {
            let _ = signal_group(&self.pidfd, libc::SIGKILL);
            loop {
                match child_exited(&self.pidfd, false) {
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    _ => break,
                }
            }
        }
    }
}

/// Retains identity before acknowledging that arbitrary handler code may run.
fn receive_handler(
    socket: &UnixDatagram,
    receive: impl FnOnce(RawFd) -> io::Result<OwnedFd>,
) -> io::Result<PipeHandler> {
    let fd = socket.as_raw_fd();
    let receipt = wait_readable(fd, HANDOFF_TIMEOUT_MS).and_then(|()| receive(fd));
    let pidfd = match receipt {
        Ok(pidfd) => pidfd,
        Err(error) => {
            // Rejection needs no identity handle. If delivery fails, the
            // child's independent deadline still prevents handler execution.
            let _ = send_decision(fd, false);
            return Err(error);
        }
    };
    let handler = PipeHandler {
        pidfd,
        complete: false,
    };
    send_decision(fd, true)?;
    Ok(handler)
}

/// Creates a handler group only after retaining its pre-exec identity.
fn spawn_pipe(args: &[String]) -> io::Result<(std::process::ChildStdin, PipeHandler)> {
    spawn_pipe_with_receiver(args, receive_pidfd)
}

/// Shares the spawn protocol with native receipt-failure injection tests.
fn spawn_pipe_with_receiver(
    args: &[String],
    receive: impl FnOnce(RawFd) -> io::Result<OwnedFd> + Send,
) -> io::Result<(std::process::ChildStdin, PipeHandler)> {
    let (parent, child_socket) = UnixDatagram::pair()?;
    let socket = child_socket.as_raw_fd();
    let mut command = Command::new(&args[0]);
    command
        .args(&args[1..])
        .stdin(Stdio::piped())
        .process_group(0);
    // SAFETY: The closure captures only a descriptor retained through spawn.
    // Its implementation uses raw syscalls/stack data and does not allocate or
    // take locks. Failed bootstrap must use _exit: returning pre_exec Err would
    // make Command reap by numeric PID and assert if an application reaper won.
    unsafe {
        command.pre_exec(move || {
            if send_own_pidfd(socket)
                .and_then(|()| await_approval(socket))
                .is_err()
            {
                libc::_exit(127);
            }
            Ok(())
        });
    }
    std::thread::scope(|scope| {
        // spawn waits for exec, so receipt and approval must run concurrently.
        let receiver = std::thread::Builder::new()
            .name("hsa-core-pidfd".into())
            .spawn_scoped(scope, move || receive_handler(&parent, receive))?;
        let child = command.spawn();
        drop(child_socket);
        // On missing identity the bootstrap has self-exited without exec.
        // It can remain a zombie until application reaping or fatal parent exit;
        // do not signal or wait through a potentially reused numeric PID.
        let handler = receiver
            .join()
            .map_err(|_| native("core handler receiver panicked"))??;
        let mut child = child?;
        let pipe = child
            .stdin
            .take()
            .ok_or_else(|| native("missing pipe stdin"))?;
        Ok((pipe, handler))
    })
}

/// Signals the original process group without a numeric PID lookup.
fn signal_group(pidfd: &OwnedFd, signal: i32) -> io::Result<()> {
    // SAFETY: The owned descriptor identifies the original process/group; the
    // null siginfo requests ordinary signal delivery with no borrowed pointers.
    if unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd.as_raw_fd(),
            signal,
            std::ptr::null::<libc::siginfo_t>(),
            libc::PIDFD_SIGNAL_PROCESS_GROUP,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Reaps only the original child; an application reaper produces ECHILD.
fn child_exited(pidfd: &OwnedFd, nonblocking: bool) -> io::Result<Option<bool>> {
    // SAFETY: waitid writes one initialized record and uses the retained pidfd,
    // so a reused numeric PID can never select another application child.
    unsafe {
        let mut info: libc::siginfo_t = std::mem::zeroed();
        if libc::waitid(
            libc::P_PIDFD,
            pidfd.as_raw_fd() as u32,
            &raw mut info,
            libc::WEXITED | if nonblocking { libc::WNOHANG } else { 0 },
        ) != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok((info.si_pid() != 0).then(|| info.si_code == libc::CLD_EXITED && info.si_status() == 0))
    }
}

/// Splits a pipe command using `ROCr`'s double-quote and backslash rules.
pub(super) fn command_args(command: &str) -> io::Result<Vec<String>> {
    let mut args = Vec::new();
    let mut word = String::new();
    let (mut quoted, mut escaped, mut started) = (false, false, false);
    for ch in command.chars() {
        if escaped {
            word.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
            started = true;
        } else if ch == '"' {
            quoted = !quoted;
            started = true;
        } else if ch.is_ascii_whitespace() && !quoted {
            if started {
                args.push(std::mem::take(&mut word));
                started = false;
            }
        } else {
            word.push(ch);
            started = true;
        }
    }
    if escaped || quoted {
        return Err(native("unterminated core pipe command"));
    }
    if started {
        args.push(word);
    }
    if args.first().is_none_or(String::is_empty) {
        return Err(native("empty core pipe command"));
    }
    Ok(args)
}

/// Keeps a broken pipe from replacing the intended SIGABRT with SIGPIPE.
struct BlockSigpipe {
    /// Calling thread's exact signal mask, restored after consuming our signal.
    previous: libc::sigset_t,
    /// Set containing only SIGPIPE.
    set: libc::sigset_t,
}

impl BlockSigpipe {
    /// Blocks SIGPIPE only on the dump-writing thread.
    fn new() -> io::Result<Self> {
        // SAFETY: Both sets are initialized before use; pthread_sigmask writes
        // the old mask and copies the new one synchronously.
        unsafe {
            let mut set = std::mem::zeroed();
            let mut previous = std::mem::zeroed();
            libc::sigemptyset(&raw mut set);
            libc::sigaddset(&raw mut set, libc::SIGPIPE);
            let status = libc::pthread_sigmask(libc::SIG_BLOCK, &raw const set, &raw mut previous);
            if status != 0 {
                return Err(io::Error::from_raw_os_error(status));
            }
            Ok(Self { previous, set })
        }
    }
}
impl Drop for BlockSigpipe {
    fn drop(&mut self) {
        // SAFETY: The saved masks are initialized. Zero-timeout sigtimedwait
        // consumes a pending signal without waiting; no global disposition changes.
        unsafe {
            if libc::sigismember(&raw const self.previous, libc::SIGPIPE) == 0 {
                let timeout = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                };
                libc::sigtimedwait(
                    &raw const self.set,
                    std::ptr::null_mut(),
                    &raw const timeout,
                );
            }
            libc::pthread_sigmask(
                libc::SIG_SETMASK,
                &raw const self.previous,
                std::ptr::null_mut(),
            );
        }
    }
}

/// A bounded nonblocking stream to a potentially unresponsive user handler.
struct PipeWriter {
    /// Child stdin owned until all bytes have been sent.
    pipe: std::process::ChildStdin,
    /// Shared write-and-exit deadline.
    deadline: Instant,
}
impl Write for PipeWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        loop {
            if Instant::now() >= self.deadline {
                return Err(io::ErrorKind::TimedOut.into());
            }
            match self.pipe.write(bytes) {
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    let mut poll = libc::pollfd {
                        fd: self.pipe.as_raw_fd(),
                        events: libc::POLLOUT,
                        revents: 0,
                    };
                    // SAFETY: One live poll record and a bounded millisecond timeout.
                    if unsafe { libc::poll(&raw mut poll, 1, 20) } < 0 {
                        let error = io::Error::last_os_error();
                        if error.kind() != io::ErrorKind::Interrupted {
                            return Err(error);
                        }
                    }
                }
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Executes an explicitly requested handler with a bounded write/exit deadline.
pub(super) fn write_pipe(
    command: &str,
    emit: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> io::Result<()> {
    write_pipe_with_timeout(command, Duration::from_secs(5), emit)
}

/// Owns a separate handler process group until consumption succeeds or cleanup.
pub(super) fn write_pipe_with_timeout(
    command: &str,
    timeout: Duration,
    emit: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> io::Result<()> {
    let args = command_args(command)?;
    let (pipe, mut handler) = spawn_pipe(&args)?;
    let result = (|| {
        // SAFETY: fcntl reads/sets descriptor flags without consuming this live fd.
        let flags = unsafe { libc::fcntl(pipe.as_raw_fd(), libc::F_GETFL) };
        if flags < 0
            || unsafe { libc::fcntl(pipe.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
        {
            return Err(io::Error::last_os_error());
        }
        let deadline = Instant::now() + timeout;
        {
            let _sigpipe = BlockSigpipe::new()?;
            let mut writer = PipeWriter { pipe, deadline };
            emit(&mut writer)?;
        }
        loop {
            match child_exited(&handler.pidfd, true) {
                Ok(Some(true)) => {
                    return Ok(());
                }
                Ok(Some(false)) => return Err(native("core pipe handler failed")),
                Ok(None) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
            if Instant::now() >= deadline {
                return Err(io::ErrorKind::TimedOut.into());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    })();
    handler.complete = result.is_ok();
    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Interrupts poll without restarting it, only in an isolated test child.
    extern "C" fn interrupt_wait(_signal: i32) {}

    /// Checks that repeated signals cannot extend the bootstrap deadline.
    fn interrupted_deadline_child() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let (socket, _peer) = UnixDatagram::pair().unwrap();
        // SAFETY: Only this isolated process installs the handler. The signal
        // is sent to this test thread; its handler has no side effects.
        let thread = unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = interrupt_wait as *const () as usize;
            libc::sigemptyset(&raw mut action.sa_mask);
            assert_eq!(
                libc::sigaction(libc::SIGUSR1, &raw const action, std::ptr::null_mut()),
                0
            );
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&raw mut set);
            libc::sigaddset(&raw mut set, libc::SIGUSR1);
            assert_eq!(
                libc::pthread_sigmask(libc::SIG_UNBLOCK, &raw const set, std::ptr::null_mut()),
                0
            );
            libc::pthread_self()
        };
        let done = Arc::new(AtomicBool::new(false));
        let stop = done.clone();
        let sender = std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                // SAFETY: The receiving thread remains live until this joins.
                assert_eq!(unsafe { libc::pthread_kill(thread, libc::SIGUSR1) }, 0);
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        let start = Instant::now();
        let result = wait_readable(socket.as_raw_fd(), 50);
        done.store(true, Ordering::Release);
        sender.join().unwrap();
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    /// Exercises genuine receipt and ACK failures after the child's send.
    fn handoff_failure_child(case: &str) {
        // SAFETY: These signal/resource settings affect only this subprocess.
        // No HSA session or unrelated child exists in the selected test.
        unsafe {
            assert_ne!(
                libc::signal(
                    libc::SIGCHLD,
                    if case == "exhausted-ignored" {
                        libc::SIG_IGN
                    } else {
                        libc::SIG_DFL
                    }
                ),
                libc::SIG_ERR
            );
        }
        let path =
            std::env::temp_dir().join(format!("libhsa-core-handoff-{}.marker", std::process::id()));
        assert!(!path.exists());
        let args = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            // Even a regressed handler and its descendant retire on their own.
            "printf started > \"$1\"; sleep 1 & wait".to_owned(),
            "core-test".to_owned(),
            path.to_str().unwrap().to_owned(),
        ];
        let start = Instant::now();
        let result = spawn_pipe_with_receiver(&args, |socket| {
            if case == "ack-error" {
                let pidfd = receive_pidfd(socket)?;
                assert!(!path.exists(), "handler ran before approval");
                // SAFETY: Disable sends on this live private socket, forcing a
                // real EPIPE after identity receipt. No descriptor is closed.
                assert_eq!(unsafe { libc::shutdown(socket, libc::SHUT_WR) }, 0);
                return Ok(pidfd);
            }
            let mut previous: libc::rlimit = unsafe { std::mem::zeroed() };
            // SAFETY: The receiver has observed a queued message, so fork/send
            // already occurred. NOFILE zero blocks descriptor installation even
            // if spawn concurrently closes its own setup fds. Restore before
            // any assertion or test output; no unrelated work runs here.
            unsafe {
                assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, &raw mut previous), 0);
                let exhausted = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: previous.rlim_max,
                };
                assert_eq!(
                    libc::setrlimit(libc::RLIMIT_NOFILE, &raw const exhausted),
                    0
                );
                let receipt = receive_pidfd(socket);
                let restored = libc::setrlimit(libc::RLIMIT_NOFILE, &raw const previous);
                assert_eq!(restored, 0);
                receipt
            }
        });
        assert!(start.elapsed() < Duration::from_secs(2));
        let error = result.err().unwrap();
        if case == "ack-error" {
            assert_eq!(error.raw_os_error(), Some(libc::EPIPE));
        } else {
            assert_eq!(error.to_string(), "truncated core handler pidfd message");
        }
        // This isolated test may reap its own failed bootstrap. Production
        // deliberately never falls back to numeric wait/signal without pidfd.
        let deadline = Instant::now() + Duration::from_secs(2);
        // SAFETY: gettid takes no pointers and has no failure mode. The raw
        // syscall also works on glibc versions without a gettid wrapper.
        let children = format!("/proc/self/task/{}/children", unsafe {
            libc::syscall(libc::SYS_gettid)
        });
        loop {
            let mut status = 0;
            // SAFETY: Only this test's bootstrap can be a child of this process.
            let reaped = unsafe { libc::waitpid(-1, &raw mut status, libc::WNOHANG) };
            if reaped > 0 {
                assert!(libc::WIFEXITED(status));
                assert_eq!(libc::WEXITSTATUS(status), 127);
            }
            if std::fs::read_to_string(&children)
                .unwrap()
                .trim()
                .is_empty()
            {
                break;
            }
            assert!(Instant::now() < deadline, "bootstrap did not retire");
            std::thread::sleep(Duration::from_millis(5));
        }
        let ran = path.exists();
        if ran {
            std::fs::remove_file(path).unwrap();
        }
        assert!(!ran, "handler executed despite failed handoff");
    }

    #[test]
    fn pidfd_handoff_failures_never_execute_the_handler() {
        if let Ok(case) = std::env::var("LIBHSA_CORE_HANDOFF_TEST_CHILD") {
            if case == "interrupted" {
                interrupted_deadline_child();
            } else {
                handoff_failure_child(&case);
            }
            return;
        }
        for case in ["exhausted", "exhausted-ignored", "ack-error", "interrupted"] {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "core_dump::pipe::tests::pidfd_handoff_failures_never_execute_the_handler",
                ])
                .env("LIBHSA_CORE_HANDOFF_TEST_CHILD", case)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(8);
            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break status;
                }
                if Instant::now() >= deadline {
                    child.kill().unwrap();
                    break child.wait().unwrap();
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            assert!(status.success(), "{case}: {status}");
        }
    }

    #[test]
    fn pidfd_group_cleanup_survives_reaping_of_the_original_leader() {
        let path =
            std::env::temp_dir().join(format!("libhsa-core-pidfd-{}.pid", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let args = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            format!("sleep 60 & echo $! > {}; exit 0", path.display()),
        ];
        let (pipe, handler) = spawn_pipe(&args).unwrap();
        drop(pipe);
        // Deliberately reap the leader before cleanup, as an application reaper
        // may do. Its descriptor still names the original descendant group.
        assert_eq!(child_exited(&handler.pidfd, false).unwrap(), Some(true));
        signal_group(&handler.pidfd, libc::SIGKILL).unwrap();
        assert_eq!(
            child_exited(&handler.pidfd, true)
                .unwrap_err()
                .raw_os_error(),
            Some(libc::ECHILD)
        );
        let pid: u32 = std::fs::read_to_string(&path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        std::fs::remove_file(path).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"));
            if stat
                .as_ref()
                .map_or(true, |stat| stat.split_whitespace().nth(2) == Some("Z"))
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "original helper group still running"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn missing_pidfd_handoff_is_nonblocking_and_never_uses_a_numeric_pid() {
        let (parent, _child) = UnixDatagram::pair().unwrap();
        assert_eq!(
            receive_pidfd(parent.as_raw_fd()).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
}
