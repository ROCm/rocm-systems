// SPDX-License-Identifier: MIT

//! One bounded DRM command stream in the KFD-bound device VM.
//!
//! A private context and timeline completion object are acquired at creation.
//! The atomic slot admits one submission at a time without a userspace lock.
//! Native fences, rather than elapsed time or a terminal error, prove when the
//! caller may reuse an indirect buffer.

use super::memory::{DeviceVm, error, native_error};
use super::{drm, util};
use crate::host_storage::{Owned, Shared};
use crate::kernel_queue::{KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait};
use crate::{Error, ErrorKind};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::Instant;

const IDLE: u64 = 0;
const SUBMITTING: u64 = u64::MAX;
const TERMINAL_DRIVER: u8 = 1;
const TERMINAL_CONTRACT: u8 = 2;
const TERMINAL_LOST: u8 = 3;

fn retire_slot(slot: &AtomicU64, retired: &AtomicU64, submission: u64) {
    retired.fetch_max(submission, Ordering::AcqRel);
    let _ = slot.compare_exchange(submission, IDLE, Ordering::AcqRel, Ordering::Acquire);
}

// A mismatched DRM sequence cannot prove retirement of this submission. The
// preattached timeline point is the only independent completion evidence.
fn resolve_mismatched_sequence(
    slot: &AtomicU64,
    retired: &AtomicU64,
    submission: u64,
    result: io::Result<bool>,
) -> Result<KernelQueueWait, Error> {
    match result {
        Ok(true) => {
            retire_slot(slot, retired, submission);
            Err(error(
                ErrorKind::DriverContract,
                "DRM returned an unexpected command sequence",
            ))
        }
        Ok(false) => Ok(KernelQueueWait::TimedOut),
        Err(source) => Err(native_error("DRM command timeline wait", source)),
    }
}

fn validate_command(command: KernelCommand) -> Result<u32, Error> {
    let byte_length = u32::try_from(command.byte_length)
        .map_err(|_| error(ErrorKind::InvalidArgument, "native command is too long"))?;
    if command.device_address == 0
        || command.device_address % 4 != 0
        || command.byte_length == 0
        || command.byte_length % 4 != 0
        || command
            .device_address
            .checked_add(command.byte_length)
            .is_none()
    {
        return Err(error(
            ErrorKind::InvalidArgument,
            "invalid native command range",
        ));
    }
    Ok(byte_length)
}

/// One private native context, completion fence, and atomic submission slot.
pub(crate) struct KfdKernelQueue {
    vm: Shared<DeviceVm>,
    process: u32,
    context_id: Option<u32>,
    completion_syncobj: Option<u32>,
    ip_type: u32,
    slot: AtomicU64,
    accepted: AtomicU64,
    retired: AtomicU64,
    ambiguous_submission: AtomicBool,
    sequence_mismatch: AtomicBool,
    terminal: AtomicU8,
    context_free_ambiguous: bool,
    syncobj_destroy_ambiguous: bool,
}

impl KfdKernelQueue {
    pub(super) fn create(
        vm: Shared<DeviceVm>,
        format: KernelQueueFormat,
    ) -> Result<Owned<Self>, Error> {
        vm.check()?;
        let slot = Owned::<Self>::try_new_uninit(vm.allocator())?;
        let ip_type = match format {
            KernelQueueFormat::Pm4 => drm::HW_IP_COMPUTE,
            KernelQueueFormat::Sdma => drm::HW_IP_DMA,
        };
        // Publish the owner before the first native acquisition. Rollback and
        // Drop then share exactly the same resumable cleanup state.
        let mut queue = slot.write(Self {
            vm,
            process: std::process::id(),
            context_id: None,
            completion_syncobj: None,
            ip_type,
            slot: AtomicU64::new(IDLE),
            accepted: AtomicU64::new(0),
            retired: AtomicU64::new(0),
            ambiguous_submission: AtomicBool::new(false),
            sequence_mismatch: AtomicBool::new(false),
            terminal: AtomicU8::new(0),
            context_free_ambiguous: false,
            syncobj_destroy_ambiguous: false,
        });
        let render_vm = queue.vm.clone();
        let render = render_vm.render()?;
        queue.completion_syncobj = Some(match drm::create_syncobj(render) {
            Ok(handle) => handle,
            Err(source) => {
                if source.raw_os_error() == Some(14) {
                    // The handle may exist without a trustworthy returned ID.
                    std::mem::forget(queue.vm.clone());
                }
                return Err(native_error("DRM completion object creation", source));
            }
        });
        let context_id = match drm::create_context(render) {
            Ok(context_id) => context_id,
            Err(source) => {
                if source.raw_os_error() == Some(14) {
                    std::mem::forget(queue.vm.clone());
                }
                let failure = native_error("DRM command context creation", source);
                return match queue.destroy() {
                    Ok(()) => Err(failure),
                    Err(rollback) => Err(rollback),
                };
            }
        };
        queue.context_id = Some(context_id);
        // WAIT_CS creates the per-IP context entity. Sequence zero cannot be a
        // submitted job; observing it now avoids that lazy setup on submit.
        if !matches!(
            drm::wait_submission(render, context_id, ip_type, 0, Some(0)),
            Ok(true)
        ) {
            let failure = error(
                ErrorKind::Unsupported,
                "DRM command context cannot initialize the requested GPU IP",
            );
            return match queue.destroy() {
                Ok(()) => Err(failure),
                Err(rollback) => Err(rollback),
            };
        }
        Ok(queue)
    }

    fn observe_terminal(&self, kind: ErrorKind) {
        let code = match kind {
            ErrorKind::DeviceLost => TERMINAL_LOST,
            ErrorKind::DriverContract => TERMINAL_CONTRACT,
            _ => TERMINAL_DRIVER,
        };
        let _ = self
            .terminal
            .compare_exchange(0, code, Ordering::AcqRel, Ordering::Acquire);
    }

    fn terminal_kind(&self) -> Option<ErrorKind> {
        match self.terminal.load(Ordering::Acquire) {
            0 => None,
            TERMINAL_DRIVER => Some(ErrorKind::Driver),
            TERMINAL_CONTRACT => Some(ErrorKind::DriverContract),
            _ => Some(ErrorKind::DeviceLost),
        }
    }

    fn check_process(&self) -> Result<(), Error> {
        util::check_process(self.process)
            .map_err(|source| native_error("DRM command context process check", source))
    }

    fn retire(&self, submission: u64) {
        retire_slot(&self.slot, &self.retired, submission);
    }

    pub(super) fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        self.check_process()?;
        let byte_length = validate_command(command)?;
        if let Some(kind) = self.terminal_kind() {
            return Err(error(kind, "kernel queue has a terminal failure"));
        }
        if self
            .slot
            .compare_exchange(IDLE, SUBMITTING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(error(
                ErrorKind::Busy,
                "kernel queue submission slot is occupied",
            ));
        }
        if self.vm.has_observed_loss() {
            self.observe_terminal(ErrorKind::DeviceLost);
        }
        if let Some(kind) = self.terminal_kind() {
            self.slot.store(IDLE, Ordering::Release);
            return Err(error(kind, "kernel queue has a terminal failure"));
        }
        let Some(submission) = self
            .accepted
            .load(Ordering::Acquire)
            .checked_add(1)
            .filter(|submission| *submission != SUBMITTING)
        else {
            self.slot.store(IDLE, Ordering::Release);
            return Err(error(
                ErrorKind::ResourceExhausted,
                "kernel queue submission identity exhausted",
            ));
        };
        let Some(context_id) = self.context_id else {
            self.slot.store(IDLE, Ordering::Release);
            return Err(error(
                ErrorKind::Internal,
                "kernel queue context was released",
            ));
        };
        let Some(syncobj) = self.completion_syncobj else {
            self.slot.store(IDLE, Ordering::Release);
            return Err(error(
                ErrorKind::Internal,
                "kernel queue fence was released",
            ));
        };
        let render = match self.vm.render() {
            Ok(render) => render,
            Err(failure) => {
                self.slot.store(IDLE, Ordering::Release);
                return Err(failure);
            }
        };
        // Every caller-visible accepted identity also names a preattached
        // timeline point. A copyout EFAULT may follow native acceptance, so it
        // must be published as an accepted failed submission, not a rejection.
        match drm::submit_indirect_buffer(
            render,
            context_id,
            self.ip_type,
            command.device_address,
            byte_length,
            syncobj,
            submission,
        ) {
            Ok(native_sequence) => {
                if native_sequence != submission {
                    self.sequence_mismatch.store(true, Ordering::Release);
                    self.observe_terminal(ErrorKind::DriverContract);
                }
                self.slot.store(submission, Ordering::Release);
                self.accepted.store(submission, Ordering::Release);
                Ok(submission)
            }
            Err(source) if source.raw_os_error() == Some(14) => {
                self.observe_terminal(ErrorKind::DriverContract);
                self.ambiguous_submission.store(true, Ordering::Release);
                self.slot.store(submission, Ordering::Release);
                self.accepted.store(submission, Ordering::Release);
                Ok(submission)
            }
            Err(source) => {
                if source.raw_os_error() == Some(19) {
                    self.observe_terminal(ErrorKind::DeviceLost);
                }
                self.slot.store(IDLE, Ordering::Release);
                Err(native_error("DRM command submission", source))
            }
        }
    }

    pub(super) fn status(&self) -> KernelQueueStatus {
        KernelQueueStatus {
            retired_submission: self.retired.load(Ordering::Acquire),
            terminal: self
                .terminal_kind()
                .or_else(|| self.vm.has_observed_loss().then_some(ErrorKind::DeviceLost)),
        }
    }

    pub(super) fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        _poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        let start = Instant::now();
        self.check_process()?;
        let accepted = self.accepted.load(Ordering::Acquire);
        if submission == 0 || submission > accepted {
            return Err(error(
                ErrorKind::InvalidArgument,
                "unknown kernel submission",
            ));
        }
        if submission <= self.retired.load(Ordering::Acquire) {
            return Ok(KernelQueueWait::Retired);
        }
        let context_id = self
            .context_id
            .ok_or_else(|| error(ErrorKind::Internal, "kernel queue context was released"))?;
        let remaining = if timeout_nanoseconds == u64::MAX {
            None
        } else {
            Some(
                timeout_nanoseconds
                    .saturating_sub(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)),
            )
        };
        if self.sequence_mismatch.load(Ordering::Acquire) {
            let syncobj = self
                .completion_syncobj
                .ok_or_else(|| error(ErrorKind::Internal, "kernel queue fence was released"))?;
            return resolve_mismatched_sequence(
                &self.slot,
                &self.retired,
                submission,
                drm::wait_timeline_point(self.vm.render()?, syncobj, submission, remaining),
            );
        }
        let result = drm::wait_submission(
            self.vm.render()?,
            context_id,
            self.ip_type,
            submission,
            remaining,
        );
        match result {
            Ok(true) => {
                self.retire(submission);
                if let Some(kind) = self.terminal_kind() {
                    Err(error(kind, "kernel queue has a terminal failure"))
                } else {
                    Ok(KernelQueueWait::Retired)
                }
            }
            Ok(false) => Ok(KernelQueueWait::TimedOut),
            Err(source) => {
                if source.raw_os_error() == Some(22)
                    && self.ambiguous_submission.load(Ordering::Acquire)
                {
                    // A private, primed context cannot allocate a future
                    // sequence without this queue. EINVAL here proves the
                    // ambiguous submission never reached native acceptance.
                    self.retire(submission);
                } else if let Some(syncobj) = self.completion_syncobj {
                    // Fence errors make WAIT_CS return errno even after the
                    // fence signals. The timeline can independently prove
                    // retirement without interpreting that error as progress.
                    if matches!(
                        drm::wait_timeline_point(self.vm.render()?, syncobj, submission, Some(0)),
                        Ok(true)
                    ) {
                        self.retire(submission);
                    }
                }
                if source.raw_os_error() == Some(19) || self.vm.has_observed_loss() {
                    self.observe_terminal(ErrorKind::DeviceLost);
                } else if matches!(source.raw_os_error(), Some(5 | 22)) {
                    self.observe_terminal(ErrorKind::Driver);
                }
                Err(native_error("DRM command completion wait", source))
            }
        }
    }

    pub(super) fn destroy(&mut self) -> Result<(), Error> {
        self.check_process()?;
        if self.slot.load(Ordering::Acquire) == SUBMITTING {
            return Err(error(ErrorKind::Busy, "kernel submission is in progress"));
        }
        let accepted = self.accepted.load(Ordering::Acquire);
        if accepted > self.retired.load(Ordering::Acquire) {
            let _ = self.wait(accepted, 0, 0);
        }
        if accepted > self.retired.load(Ordering::Acquire) {
            return Err(error(ErrorKind::Busy, "kernel submission has not retired"));
        }
        if self.context_free_ambiguous || self.syncobj_destroy_ambiguous {
            return Err(error(
                ErrorKind::DriverContract,
                "DRM handle destruction outcome is uncertain",
            ));
        }
        let render = self.vm.render()?;
        if let Some(context_id) = self.context_id {
            if let Err(source) = drm::destroy_context(render, context_id) {
                self.context_free_ambiguous = source.raw_os_error() == Some(14);
                return Err(native_error("DRM command context release", source));
            }
            self.context_id = None;
        }
        if let Some(syncobj) = self.completion_syncobj {
            if let Err(source) = drm::destroy_syncobj(render, syncobj) {
                self.syncobj_destroy_ambiguous = source.raw_os_error() == Some(14);
                return Err(native_error("DRM completion object release", source));
            }
            self.completion_syncobj = None;
        }
        Ok(())
    }
}

impl Drop for KfdKernelQueue {
    fn drop(&mut self) {
        if self.destroy().is_err() {
            // A failed cleanup retains the exact render VM until process
            // teardown, even if a caller drops the owner after a failed call.
            std::mem::forget(self.vm.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{memory, sys, sysfs, uapi};
    use super::*;
    use crate::host_storage::Allocator;
    use std::fs::File;
    use std::sync::Arc;

    #[allow(clippy::unwrap_used)]
    fn scripted_vm() -> Shared<DeviceVm> {
        let allocator = Allocator::default();
        let kfd = Shared::new(
            sys::Kfd::with_hook(
                File::open("/dev/null").unwrap(),
                Arc::new(|call| {
                    if let sys::Call::Wait(args, _) = call {
                        args.result = uapi::WAIT_TIMEOUT;
                    }
                    Ok(())
                }),
            ),
            allocator,
        )
        .unwrap();
        let node = sysfs::NativeNode {
            node: 1,
            gpu_id: 42,
            render_minor: Some(128),
            unique_id: Some(123),
            identity: [0; 16],
            queues: sysfs::NativeQueueProperties {
                gfx_target: 120_001,
                xcc_count: 1,
                ..sysfs::NativeQueueProperties::default()
            },
            local_memory_bytes: 1 << 30,
            public_memory_bytes: 0,
        };
        memory::queue_fixture(kfd, File::open("/dev/null").unwrap(), node)
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn mismatched_native_sequence_cannot_release_command_storage_early() {
        let vm = scripted_vm();
        drm::with_script(
            [
                drm::TestCall::CreateSyncobj(7),
                drm::TestCall::CreateContext(5),
                drm::TestCall::WaitSubmission(Ok(true)),
                drm::TestCall::Submit(Ok(42)),
                drm::TestCall::WaitTimeline(Err(110)),
                drm::TestCall::WaitTimeline(Err(110)),
                drm::TestCall::WaitTimeline(Ok(true)),
                drm::TestCall::DestroyContext,
                drm::TestCall::DestroySyncobj,
            ],
            || {
                let mut queue = KfdKernelQueue::create(vm.clone(), KernelQueueFormat::Pm4).unwrap();
                let submission = queue
                    .submit(KernelCommand {
                        device_address: 0x1000,
                        byte_length: 4,
                    })
                    .unwrap();
                assert_eq!(submission, 1);
                assert_eq!(queue.status().terminal, Some(ErrorKind::DriverContract));
                assert_eq!(
                    queue.wait(submission, 0, 0).unwrap(),
                    KernelQueueWait::TimedOut
                );
                assert_eq!(queue.status().retired_submission, 0);
                assert_eq!(
                    queue.destroy().err().map(|e| e.kind()),
                    Some(ErrorKind::Busy)
                );
                assert_eq!(queue.status().retired_submission, 0);
                assert_eq!(
                    queue.wait(submission, 0, 0).err().map(|e| e.kind()),
                    Some(ErrorKind::DriverContract)
                );
                assert_eq!(queue.status().retired_submission, submission);
                queue.destroy().unwrap();
            },
        );
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn ambiguous_context_free_never_replays_a_recycled_id() {
        let vm = scripted_vm();
        drm::with_script(
            [
                drm::TestCall::CreateSyncobj(7),
                drm::TestCall::CreateContext(5),
                drm::TestCall::WaitSubmission(Ok(true)),
                drm::TestCall::FailDestroyContext(14),
                drm::TestCall::CreateSyncobj(8),
                drm::TestCall::CreateContext(5),
                drm::TestCall::WaitSubmission(Ok(true)),
                drm::TestCall::DestroyContext,
                drm::TestCall::DestroySyncobj,
            ],
            || {
                let mut old = KfdKernelQueue::create(vm.clone(), KernelQueueFormat::Pm4).unwrap();
                assert!(old.destroy().is_err());
                let mut next = KfdKernelQueue::create(vm.clone(), KernelQueueFormat::Pm4).unwrap();
                assert_eq!(old.destroy().unwrap_err().kind(), ErrorKind::DriverContract);
                next.destroy().unwrap();
            },
        );
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn failed_setup_uses_owned_rollback_and_reports_cleanup_failure() {
        let vm = scripted_vm();
        drm::with_script(
            [
                drm::TestCall::CreateSyncobj(7),
                drm::TestCall::FailCreateContext(22),
                drm::TestCall::FailDestroySyncobj(14),
            ],
            || {
                let error = KfdKernelQueue::create(vm.clone(), KernelQueueFormat::Pm4)
                    .err()
                    .unwrap();
                assert_eq!(error.native_error_code(), Some(14));
            },
        );
        drm::with_script(
            [
                drm::TestCall::CreateSyncobj(7),
                drm::TestCall::CreateContext(5),
                drm::TestCall::WaitSubmission(Err(22)),
                drm::TestCall::FailDestroyContext(16),
                drm::TestCall::DestroyContext,
                drm::TestCall::DestroySyncobj,
            ],
            || {
                let error = KfdKernelQueue::create(vm.clone(), KernelQueueFormat::Pm4)
                    .err()
                    .unwrap();
                assert_eq!(error.native_error_code(), Some(16));
                // Drop retries the ordinary EBUSY rollback through the owner.
            },
        );
    }

    #[test]
    fn mismatched_sequence_retains_command_until_timeline_completion() {
        let slot = AtomicU64::new(1);
        let retired = AtomicU64::new(0);
        assert!(matches!(
            resolve_mismatched_sequence(&slot, &retired, 1, Ok(false)),
            Ok(KernelQueueWait::TimedOut)
        ));
        assert_eq!(slot.load(Ordering::Acquire), 1);
        assert_eq!(retired.load(Ordering::Acquire), 0);

        let failed_wait = resolve_mismatched_sequence(
            &slot,
            &retired,
            1,
            Err(io::Error::from(io::ErrorKind::Other)),
        );
        assert!(failed_wait.is_err());
        assert_eq!(slot.load(Ordering::Acquire), 1);
        assert_eq!(retired.load(Ordering::Acquire), 0);

        let completed = resolve_mismatched_sequence(&slot, &retired, 1, Ok(true));
        assert_eq!(
            completed.err().map(|failure| failure.kind()),
            Some(ErrorKind::DriverContract)
        );
        assert_eq!(slot.load(Ordering::Acquire), IDLE);
        assert_eq!(retired.load(Ordering::Acquire), 1);
    }

    #[test]
    fn stale_waiter_cannot_release_a_new_submission_slot() {
        let slot = AtomicU64::new(1);
        let retired = AtomicU64::new(0);
        retire_slot(&slot, &retired, 1);
        assert_eq!(slot.load(Ordering::Acquire), IDLE);
        slot.store(2, Ordering::Release);
        retire_slot(&slot, &retired, 1);
        assert_eq!(slot.load(Ordering::Acquire), 2);
        assert_eq!(retired.load(Ordering::Acquire), 1);
        retire_slot(&slot, &retired, 2);
        assert_eq!(slot.load(Ordering::Acquire), IDLE);
        assert_eq!(retired.load(Ordering::Acquire), 2);
    }
}
