//! Interactive-terminal tests: `mirage run -- bash` has to be a usable
//! shell.
//!
//! These drive the CLI through a real pseudo-terminal, the way a user's
//! terminal emulator does, and assert on what appears on the screen. That
//! is the only way to test this honestly — a shell decides whether to
//! print a prompt, echo input, and enable line editing by calling
//! `isatty`, so a test that pipes stdin proves nothing about the
//! interactive case.
//!
//! The local pty here comes from `pty-process`, the same crate the
//! supervisor uses for the remote one, so no test needs `unsafe` either.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod harness;

use std::io::{Read as _, Write as _};
use std::time::{Duration, Instant};

use harness::Env;

/// A `mirage` command running on a local pseudo-terminal.
struct Terminal {
    pty: pty_process::blocking::Pty,
    child: std::process::Child,
    seen: String,
}

impl Terminal {
    /// Run `args` on a terminal of the given size.
    fn spawn(env: &Env, rows: u16, cols: u16, args: &[&str]) -> Self {
        let (pty, pts) = pty_process::blocking::open().unwrap();
        pty.resize(pty_process::Size::new(rows, cols)).unwrap();

        let mut cmd = pty_process::blocking::Command::new(env.bin());
        cmd = cmd.args(args);
        for (key, value) in env.child_env() {
            cmd = cmd.env(key, value);
        }
        // A terminal type programs recognise, so line editing and cursor
        // handling behave as they would for a user.
        cmd = cmd.env("TERM", "xterm-256color");
        let child = cmd.spawn(pts).expect("spawn mirage on a terminal");

        Self {
            pty,
            child,
            seen: String::new(),
        }
    }

    /// Read whatever the terminal produces for up to `timeout`, or until
    /// `marker` appears.
    fn read_until(&mut self, marker: &str, timeout: Duration) -> bool {
        use std::os::fd::AsFd as _;

        let deadline = Instant::now() + timeout;
        let mut buf = [0u8; 8192];
        while Instant::now() < deadline {
            if self.seen.contains(marker) {
                return true;
            }
            // Poll so a quiet terminal does not block past the deadline.
            let mut fds = [nix::poll::PollFd::new(
                self.pty.as_fd(),
                nix::poll::PollFlags::POLLIN,
            )];
            let ready = nix::poll::poll(&mut fds, 200u16).unwrap_or(0);
            if ready > 0 {
                match self.pty.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => self.seen.push_str(&String::from_utf8_lossy(&buf[..n])),
                    // EIO is how a pty reports "the far end closed".
                    Err(_) => break,
                }
            }
        }
        self.seen.contains(marker)
    }

    /// Type a line, as a user would.
    fn type_line(&mut self, line: &str) {
        self.pty.write_all(line.as_bytes()).unwrap();
        self.pty.write_all(b"\n").unwrap();
        self.pty.flush().unwrap();
    }

    /// Resize the local terminal, which must propagate to the workload.
    fn resize(&self, rows: u16, cols: u16) {
        self.pty
            .resize(pty_process::Size::new(rows, cols))
            .unwrap();
        // The CLI reacts to SIGWINCH; the local pty resize does not raise
        // one in the CLI (it is not our controlling terminal), so send it
        // explicitly, exactly as a terminal emulator would.
        let pid = nix::unistd::Pid::from_raw(i32::try_from(self.child.id()).unwrap());
        let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGWINCH);
    }

    /// Everything seen so far.
    fn transcript(&self) -> &str {
        &self.seen
    }

    /// Wait for the command to exit.
    fn wait(mut self) -> std::process::ExitStatus {
        // Drain anything still buffered so the transcript is complete and
        // the child is not blocked writing into a full pty.
        self.read_until("\u{0}never-appears", Duration::from_secs(3));
        self.child.wait().unwrap()
    }
}

#[test]
fn bash_is_a_working_interactive_shell() {
    let env = Env::new();
    env.create_profile("p");

    let mut term = Terminal::spawn(
        &env,
        24,
        80,
        &["run", "--profile", "p", "--", "bash", "--norc", "-i"],
    );

    // A shell prints a prompt only when it believes it is interactive,
    // which means only when `isatty(0)` is true.
    assert!(
        term.read_until("$", Duration::from_secs(30)),
        "bash never printed a prompt; it does not think it is \
         interactive.\nTranscript:\n{}",
        term.transcript()
    );

    term.type_line("echo interactive-$((6*7))");
    assert!(
        term.read_until("interactive-42", Duration::from_secs(30)),
        "the command did not run.\nTranscript:\n{}",
        term.transcript()
    );
    // The terminal's line discipline echoes what was typed. On pipes
    // nothing would echo, and the user would type blind.
    assert!(
        term.transcript().contains("echo interactive-"),
        "input was not echoed.\nTranscript:\n{}",
        term.transcript()
    );

    // `read -p` writes its prompt to the terminal and blocks for input:
    // the round trip that a pipe-only design cannot do.
    term.type_line("read -p 'name? ' answer && echo \"got:$answer\"");
    assert!(
        term.read_until("name?", Duration::from_secs(30)),
        "read's prompt never appeared.\nTranscript:\n{}",
        term.transcript()
    );
    term.type_line("mirage-user");
    assert!(
        term.read_until("got:mirage-user", Duration::from_secs(30)),
        "the shell never received the typed answer.\nTranscript:\n{}",
        term.transcript()
    );

    term.type_line("exit");
    let status = term.wait();
    assert!(status.success(), "the shell exited with {status:?}");
}

#[test]
fn the_workload_sees_the_terminal_size() {
    let env = Env::new();
    env.create_profile("p");

    // A program that draws to the screen reads its size from the
    // terminal. If mirage does not propagate it, everything renders into
    // the wrong shape.
    let mut term = Terminal::spawn(
        &env,
        40,
        132,
        &[
            "run",
            "--profile",
            "p",
            "--",
            "bash",
            "--norc",
            "-c",
            "stty size",
        ],
    );
    assert!(
        term.read_until("40 132", Duration::from_secs(30)),
        "the workload saw the wrong terminal size.\nTranscript:\n{}",
        term.transcript()
    );
    term.wait();
}

#[test]
fn resizing_the_terminal_reaches_the_workload() {
    let env = Env::new();
    env.create_profile("p");

    // Report the size, then report it again after a resize. The second
    // reading must reflect the new geometry, which only happens if
    // SIGWINCH propagated all the way to the remote pty.
    let mut term = Terminal::spawn(
        &env,
        24,
        80,
        &[
            "run",
            "--profile",
            "p",
            "--",
            "bash",
            "--norc",
            "-c",
            "stty size; sleep 2; stty size",
        ],
    );
    assert!(
        term.read_until("24 80", Duration::from_secs(30)),
        "initial size wrong.\nTranscript:\n{}",
        term.transcript()
    );

    term.resize(50, 200);
    assert!(
        term.read_until("50 200", Duration::from_secs(30)),
        "the resize never reached the workload.\nTranscript:\n{}",
        term.transcript()
    );
    term.wait();
}

#[test]
fn a_terminal_exec_reports_its_exit_code() {
    let env = Env::new();
    env.create_profile("p");
    let term = Terminal::spawn(
        &env,
        24,
        80,
        &["run", "--profile", "p", "--", "bash", "--norc", "-c", "exit 17"],
    );
    let status = term.wait();
    assert_eq!(
        status.code(),
        Some(17),
        "a terminal exec must still report its exit code"
    );
}

#[test]
fn ctrl_c_interrupts_a_foreground_program() {
    let env = Env::new();
    env.create_profile("p");

    // Job control is the other half of "a working shell": Ctrl-C must
    // reach the foreground program as SIGINT, which requires the child to
    // be a session leader with the pty as its controlling terminal.
    let mut term = Terminal::spawn(
        &env,
        24,
        80,
        &["run", "--profile", "p", "--", "bash", "--norc", "-i"],
    );
    assert!(term.read_until("$", Duration::from_secs(30)));

    term.type_line("sleep 300; echo after-interrupt");
    // Give the sleep time to become the foreground process.
    std::thread::sleep(Duration::from_millis(750));
    // 0x03 is Ctrl-C; the line discipline turns it into SIGINT for the
    // foreground process group.
    term.pty.write_all(&[0x03]).unwrap();
    term.pty.flush().unwrap();

    assert!(
        term.read_until("after-interrupt", Duration::from_secs(30)),
        "Ctrl-C did not interrupt the foreground program; the shell never \
         came back.\nTranscript:\n{}",
        term.transcript()
    );

    term.type_line("exit");
    term.wait();
}

#[test]
fn a_piped_run_stays_on_pipes_and_keeps_streams_separate() {
    // The counterpart to the tests above: without a terminal on both
    // ends, `auto` must not allocate one. A pty would merge stderr into
    // stdout and rewrite newlines, breaking redirection.
    let env = Env::new();
    env.create_profile("p");

    let out = env.run(&[
        "run",
        "--profile",
        "p",
        "--",
        "/bin/sh",
        "-c",
        "if [ -t 1 ]; then echo TTY; else echo PIPE; fi; echo to-err 1>&2",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(stdout.contains("PIPE"), "stdout was: {stdout}");
    assert!(
        !stdout.contains("to-err"),
        "stderr leaked into stdout, so this ran on a terminal: {stdout}"
    );
    assert!(stderr.contains("to-err"), "stderr was: {stderr}");
    // And byte-exact: no `\r` inserted by a line discipline.
    assert!(
        !stdout.contains('\r'),
        "output was translated by a terminal: {stdout:?}"
    );
}
