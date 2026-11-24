// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Fatal GPU core capture. The HSA registry retains queue, memory and loader
//! owners for the whole snapshot. Diagnostic policy never turns a capture
//! failure into continued execution of an unhandled GPU fault.

/// Immutable `ROCr`-compatible output policy.
#[derive(Default)]
pub(crate) struct Config {
    /// Explicitly suppress GPU dumps on unhandled exceptions.
    disabled: bool,
    /// User file/pipe pattern; absent means the kernel pattern with GPU fallback.
    pattern: Option<String>,
}

impl Config {
    /// Samples policy before application workers can report a fatal error.
    pub(crate) fn from_environment() -> Self {
        Self {
            disabled: std::env::var_os("HSA_DISABLE_COREDUMP_ON_EXCEPTION")
                .is_some_and(|value| value == "1"),
            pattern: std::env::var("HSA_COREDUMP_PATTERN")
                .ok()
                .filter(|value| !value.is_empty()),
        }
    }
}

/// Serializes competing fatal workers through capture and process termination.
/// This function is called outside the registry and application callback locks.
pub(crate) fn fatal_abort() -> ! {
    static CAPTURE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _capture = CAPTURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    #[cfg(target_os = "linux")]
    let _ = linux::dump();
    // Application stdio locks can be held by a thread waiting for this GPU.
    // Do not acquire one to report a failed diagnostic before terminating.
    std::process::abort();
}

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
mod pipe;

#[cfg(all(test, target_os = "linux"))]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    /// A failed capture must not wait on a GPU-waiting application stdio lock.
    #[test]
    fn fatal_capture_error_aborts_with_application_stderr_locked() {
        use std::os::unix::process::ExitStatusExt;
        use std::sync::{Arc, Barrier};
        use std::time::{Duration, Instant};
        if std::env::var_os("LIBHSA_CORE_FATAL_TEST_CHILD").is_some() {
            let limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            // SAFETY: This isolated subprocess suppresses its own CPU core file.
            assert_eq!(
                unsafe { libc::setrlimit(libc::RLIMIT_CORE, &raw const limit) },
                0
            );
            let barrier = Arc::new(Barrier::new(2));
            let worker = barrier.clone();
            std::thread::spawn(move || {
                let _stderr = std::io::stderr().lock();
                worker.wait();
                loop {
                    std::thread::park();
                }
            });
            barrier.wait();
            super::fatal_abort();
        }
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "core_dump::tests::fatal_capture_error_aborts_with_application_stderr_locked",
            ])
            .env("LIBHSA_CORE_FATAL_TEST_CHILD", "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert_eq!(status.signal(), Some(libc::SIGABRT));
                break;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("fatal capture waited on the application's stderr lock");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
