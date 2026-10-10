// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMDF kernel queue ownership and validation over the neutral rocddi queue.
//!
//! Public handles, family selection, command access checks, and ABI status
//! translation stay here. rocddi owns only bounded native submission state.

use crate::generated::amdf::*;
use crate::instance::{self, Device};
use crate::memory;
use crate::support::*;
use rocddi::device::gpu::queue::{
    KernelQueue as NativeQueue, KernelQueueFormat, KernelQueueStatus as NativeStatus,
    KernelQueueWait,
};
use rocddi::host_storage::Owned;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

struct KernelQueue {
    native: NativeQueue,
    device: *mut Device,
    info: amdf_kernel_queue_info_t,
    accepted: AtomicU64,
    destroying: AtomicBool,
}

impl KernelQueue {
    fn require_usable(&self) -> Result<(), u64> {
        if self.destroying.load(Ordering::Acquire) {
            Err(PRECONDITION)
        } else {
            Ok(())
        }
    }

    fn observe_error(&self, error: &rocddi::Error) -> u64 {
        if error.kind() == rocddi::ErrorKind::DeviceLost {
            // SAFETY: A registered kernel queue borrows this live device.
            unsafe { (*self.device).observe_loss(self.info.reset_epoch) };
        }
        native(error)
    }

    fn public_status(&self, sample: NativeStatus) -> amdf_kernel_queue_status_t {
        if sample.terminal == Some(rocddi::ErrorKind::DeviceLost) {
            // SAFETY: A registered kernel queue borrows this live device.
            unsafe { (*self.device).observe_loss(self.info.reset_epoch) };
        }
        let terminal_status = sample.terminal.map_or(0, |kind| {
            native(&rocddi::Error::Operation {
                kind,
                detail: "kernel queue terminal failure",
            })
        });
        amdf_kernel_queue_status_t {
            retired_submission: sample.retired_submission,
            state: match sample.terminal {
                Some(rocddi::ErrorKind::DeviceLost) => AMDF_QUEUE_STATE_DEVICE_LOST,
                Some(_) => AMDF_QUEUE_STATE_FAILED,
                None => AMDF_QUEUE_STATE_ACTIVE,
            },
            terminal_status,
            ..Default::default()
        }
    }
}

pub(crate) unsafe extern "C" fn create(
    pointer: *mut amdf_device_t,
    create_info: *const amdf_gpu_kernel_queue_create_info_t,
    out: *mut *mut amdf_kernel_queue_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        output_pointer(out)?;
        let create_info = input(
            create_info,
            AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
        )?;
        let device = object(pointer.cast::<Device>())?;
        let family = instance::family(device.native.endpoint(), create_info.queue_family_ordinal)?;
        if family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL == 0 {
            return Err(UNSUPPORTED);
        }
        let format = match family.command_type {
            AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 => KernelQueueFormat::Pm4,
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA => KernelQueueFormat::Sdma,
            _ => return Err(UNSUPPORTED),
        };
        let instance = instance::device_instance(device);
        let slot =
            Owned::<KernelQueue>::try_new_uninit(instance.allocator).map_err(|_| EXHAUSTED)?;
        register(&device.queues)?;
        let gpu = match device.native.gpu() {
            Ok(gpu) => gpu,
            Err(error) => {
                unregister(&device.queues);
                return Err(device.native_error_status(&error));
            }
        };
        let native_queue = match gpu.create_kernel_queue(format) {
            Ok(queue) => queue,
            Err(error) => {
                unregister(&device.queues);
                return Err(device.native_error_status(&error));
            }
        };
        let info = amdf_kernel_queue_info_t {
            device_id: device.id,
            reset_epoch: device.current_reset_epoch(),
            queue_family_ordinal: create_info.queue_family_ordinal,
            command_type: family.command_type,
            maximum_pending_submission_count: if create_info.maximum_pending_submission_count == 0 {
                AMDF_GPU_KERNEL_QUEUE_DEFAULT_PENDING_SUBMISSION_COUNT
            } else {
                create_info.maximum_pending_submission_count
            },
            maximum_command_count: 1,
            notification_types: 0,
            ..Default::default()
        };
        let owner = slot.write(KernelQueue {
            native: native_queue,
            device: pointer.cast(),
            info,
            accepted: AtomicU64::new(0),
            destroying: AtomicBool::new(false),
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn info(
    pointer: *mut amdf_kernel_queue_t,
    out: *mut amdf_kernel_queue_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let out = output(out, AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO)?;
        let queue = object(pointer.cast::<KernelQueue>())?;
        queue.require_usable()?;
        out.publish(queue.info);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn status(
    pointer: *mut amdf_kernel_queue_t,
    out: *mut amdf_kernel_queue_status_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let out = output(out, AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS)?;
        let queue = object(pointer.cast::<KernelQueue>())?;
        queue.require_usable()?;
        out.publish(queue.public_status(queue.native.status()));
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn refresh_status(
    pointer: *mut amdf_kernel_queue_t,
    out: *mut amdf_kernel_queue_status_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let out = output(out, AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS)?;
        let queue = object(pointer.cast::<KernelQueue>())?;
        queue.require_usable()?;
        let sample = queue
            .native
            .refresh_status()
            .map_err(|error| queue.observe_error(&error))?;
        out.publish(queue.public_status(sample));
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn request_notification(
    pointer: *mut amdf_kernel_queue_t,
    submission: u64,
    event: *const amdf_native_event_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        if submission == 0 {
            return Err(INVALID);
        }
        let event = object(event)?;
        if event.reserved != 0 {
            return Err(INVALID);
        }
        match event.r#type {
            AMDF_NATIVE_EVENT_TYPE_EVENTFD => {
                if event.payload.file_descriptor < 0
                    || event.payload.file_descriptor > i64::from(i32::MAX)
                {
                    return Err(INVALID);
                }
            }
            AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT => {
                let handle = event.payload.native_handle as usize;
                if handle == 0 || handle == usize::MAX {
                    return Err(INVALID);
                }
            }
            _ => return Err(INVALID),
        }
        let queue = object(pointer.cast::<KernelQueue>())?;
        queue.require_usable()?;
        if submission > queue.accepted.load(Ordering::Acquire) {
            return Err(RANGE);
        }
        Err(UNSUPPORTED)
    })
}

pub(crate) unsafe extern "C" fn wait(
    pointer: *mut amdf_kernel_queue_t,
    submission: u64,
    timeout_nanoseconds: u64,
    poll_duration_nanoseconds: u64,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let start = Instant::now();
        let queue = object(pointer.cast::<KernelQueue>())?;
        queue.require_usable()?;
        let remaining = if timeout_nanoseconds == AMDF_TIMEOUT_INFINITE {
            AMDF_TIMEOUT_INFINITE
        } else {
            timeout_nanoseconds
                .saturating_sub(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX))
        };
        match queue.native.wait(
            submission,
            remaining,
            poll_duration_nanoseconds.min(remaining),
        ) {
            Ok(KernelQueueWait::Retired) => Ok(()),
            Ok(KernelQueueWait::TimedOut) => Err(DEADLINE),
            Err(error) => Err(queue.observe_error(&error)),
        }
    })
}

pub(crate) unsafe extern "C" fn submit(
    pointer: *mut amdf_kernel_queue_t,
    submit_info: *const amdf_gpu_kernel_queue_submission_info_t,
    out: *mut u64,
) -> u64 {
    crate::support::boundary(|| unsafe {
        output_pointer(out)?;
        let submit_info = input(
            submit_info,
            AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
        )?;
        if submit_info.reserved != 0 || submit_info.command_count != 1 {
            return Err(INVALID);
        }
        let descriptors = array(submit_info.commands, submit_info.command_count)?;
        let queue = object(pointer.cast::<KernelQueue>())?;
        queue.require_usable()?;
        if (*queue.device).current_reset_epoch() != queue.info.reset_epoch {
            return Err(LOST);
        }
        let descriptor = &descriptors[0];
        if descriptor.reserved != 0 {
            return Err(INVALID);
        }
        let command = memory::kernel_command(
            descriptor.memory,
            descriptor.access_ordinal,
            queue.device,
            queue.info.reset_epoch,
            descriptor.byte_offset,
            descriptor.byte_length,
        )?;
        // SAFETY: kernel_command validates current attachment and execute
        // permission. The AMDF submit contract requires the caller to keep
        // command storage and indirect dependencies live and unchanged until
        // the accepted submission retires, including an uncertain outcome.
        let submission = queue
            .native
            .submit(command)
            .map_err(|error| queue.observe_error(&error))?;
        queue.accepted.store(submission, Ordering::Release);
        out.write(submission);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn destroy(pointer: *mut amdf_kernel_queue_t) -> u64 {
    crate::support::boundary(|| unsafe {
        let queue = exclusive(pointer.cast::<KernelQueue>())?;
        if !queue.destroying.load(Ordering::Acquire) {
            let accepted = queue.accepted.load(Ordering::Acquire);
            if accepted > queue.native.status().retired_submission {
                let _ = queue.native.wait(accepted, 0, 0);
                if accepted > queue.native.status().retired_submission {
                    return Err(BUSY);
                }
            }
            queue.destroying.store(true, Ordering::Release);
        }
        let result = queue
            .native
            .destroy()
            .map_err(|error| queue.observe_error(&error));
        unregister(&(*queue.device).queues);
        let owner = Owned::from_raw(pointer.cast::<KernelQueue>());
        if result.is_err() {
            // A failed native cleanup consumes the public handle. Retain the
            // native owner and its dependencies without a second release.
            std::mem::forget(owner);
        } else {
            drop(owner);
        }
        result
    })
}
