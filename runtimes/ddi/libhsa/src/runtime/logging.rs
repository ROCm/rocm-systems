// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA logging configuration and retained writes.

use super::*;

thread_local! {
    static IN_LOG_WRITE: Cell<bool> = const { Cell::new(false) };
}

/// Prevents C stream callbacks from recursively entering the log writer or
/// replacing its stream while the borrowed pointer is in use.
pub(crate) struct LogWriteScope;

impl LogWriteScope {
    fn enter() -> Option<Self> {
        if IN_LOG_WRITE.with(|active| active.replace(true)) {
            return None;
        }
        Some(Self)
    }

    pub(crate) fn active() -> bool {
        IN_LOG_WRITE.with(Cell::get)
    }
}

impl Drop for LogWriteScope {
    fn drop(&mut self) {
        IN_LOG_WRITE.with(|active| active.set(false));
    }
}

/// Logging destination and flags protected during C stream use.
struct LogConfig {
    /// Enabled log categories supplied by the caller.
    flags: [u8; 8],
    /// Borrowed C stream, or stderr when absent.
    stream: Option<BorrowedCStream>,
    /// Prevents stream replacement during final shutdown.
    stopping: bool,
}

/// Borrowed C `FILE` pointer governed by the HSA logging contract.
#[derive(Clone, Copy)]
struct BorrowedCStream(NonNull<c_void>);

// SAFETY: The C stdio stream is used only while LogOutput holds its config
// mutex. The HSA caller keeps the stream open while it is configured, and C
// stdio serializes access from other threads using the same FILE pointer.
unsafe impl Send for BorrowedCStream {}

unsafe extern "C" {
    fn fwrite(buffer: *const c_void, size: usize, count: usize, stream: *mut c_void) -> usize;
    fn fflush(stream: *mut c_void) -> i32;
}

/// Shared logging sink with an atomic category fast path.
pub(crate) struct LogOutput {
    /// Enabled categories checked before locking the stream.
    enabled: AtomicU64,
    /// Serializes output and changes to the borrowed stream.
    config: Mutex<LogConfig>,
}

impl LogOutput {
    pub(super) fn new() -> Self {
        Self {
            enabled: AtomicU64::new(0),
            config: Mutex::new(LogConfig {
                flags: [0; 8],
                stream: None,
                stopping: false,
            }),
        }
    }

    pub(crate) fn set(&self, flags: [u8; 8], stream: *mut c_void) -> Status {
        let Ok(mut config) = self.config.lock() else {
            return ERROR;
        };
        if config.stopping {
            return INVALID_RUNTIME_STATE;
        }
        config.flags = flags;
        config.stream = NonNull::new(stream).map(BorrowedCStream);
        self.enabled
            .store(u64::from_le_bytes(flags), Ordering::Release);
        SUCCESS
    }

    pub(super) fn stop(&self) {
        let mut config = self
            .config
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        config.stopping = true;
        config.flags = [0; 8];
        config.stream = None;
        self.enabled.store(0, Ordering::Release);
    }

    fn write(&self, flag: u32, line: &[u8]) {
        let Some(_scope) = LogWriteScope::enter() else {
            return;
        };
        let Ok(config) = self.config.lock() else {
            return;
        };
        if config.stopping || !logging_flag_enabled(config.flags, flag) {
            return;
        }
        if let Some(stream) = config.stream {
            // SAFETY: The public logging contract keeps the borrowed FILE open
            // while enabled. This mutex prevents replacement or shutdown from
            // finishing until the C stdio calls finish.
            unsafe {
                let _ = fwrite(line.as_ptr().cast(), 1, line.len(), stream.0.as_ptr());
                let _ = fflush(stream.0.as_ptr());
            }
        } else {
            let mut stream = std::io::stderr().lock();
            let _ = stream.write_all(line);
            let _ = stream.flush();
        }
    }

    pub(super) fn prepare(
        self: &Arc<Self>,
        tracker: &Arc<InFlightTracker>,
        flag: u32,
        arguments: Arguments<'_>,
    ) -> Option<LogWork> {
        if flag >= 64 || self.enabled.load(Ordering::Acquire) & (1_u64 << flag) == 0 {
            return None;
        }
        let inflight = tracker.enter()?;
        let mut line = String::from("[***rocddi***] ");
        std::fmt::write(&mut line, arguments).ok()?;
        line.push('\n');
        Some(LogWork {
            output: self.clone(),
            _inflight: inflight,
            flag,
            line,
        })
    }
}

/// Log line that retains its runtime generation until emission completes.
pub(crate) struct LogWork {
    /// Sink retained while writing outside the registry lock.
    output: Arc<LogOutput>,
    /// Blocks shutdown from freeing state while this work is outstanding.
    _inflight: InFlightToken,
    /// Category to check before emitting the line.
    flag: u32,
    /// Complete formatted line owned by this work item.
    line: String,
}

impl LogWork {
    pub(crate) fn write(self) {
        self.output.write(self.flag, self.line.as_bytes());
    }
}

pub(super) fn logging_flag_enabled(flags: [u8; 8], flag: u32) -> bool {
    usize::try_from(flag / 8)
        .ok()
        .and_then(|index| flags.get(index))
        .is_some_and(|byte| byte & (1 << (flag % 8)) != 0)
}
