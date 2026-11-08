// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Public GPU native events over the shared KFD signal-page service.

use crate::generated::amdf::*;
use crate::instance::{self, Device};
use crate::support::*;
use rocddi::device::gpu::event::linux::{SignalEvent, SignalEventAddressing};
use rocddi::memory::storage::Owned;

struct Event {
    native: SignalEvent,
    device: *mut Device,
    info: amdf_gpu_event_info_t,
}

pub(crate) unsafe extern "C" fn create(
    pointer: *mut amdf_device_t,
    info: *const amdf_gpu_event_create_info_t,
    out: *mut *mut amdf_gpu_event_t,
) -> u64 {
    boundary(|| unsafe {
        output_pointer(out)?;
        let create_info = input(info, AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO)?;
        if create_info.reserved != 0 {
            return Err(INVALID);
        }
        let device = object(pointer.cast::<Device>())?;
        let instance = instance::device_instance(device);
        if !instance.process_lifetime || instance.process_id != std::process::id() {
            return Err(UNSUPPORTED);
        }
        let reset_epoch = device.current_reset_epoch();
        let slot = instance.owner_slot::<Event>()?;
        register(&device.events)?;
        let result = (|| {
            instance
                .event_service
                .lock()
                .map_err(|_| INTERNAL)?
                .create(&device.native, SignalEventAddressing::Mapped)
                .map_err(|error| device.native_error_status(&error))
        })();
        let (mut event, mailbox) = match result {
            Ok(value) => value,
            Err(status) => {
                unregister(&device.events);
                return Err(status);
            }
        };
        if device.current_reset_epoch() != reset_epoch {
            let _ = event.destroy();
            unregister(&device.events);
            return Err(LOST);
        }
        let owner = slot.write(Event {
            info: amdf_gpu_event_info_t {
                device_id: device.id,
                reset_epoch,
                format: AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL,
                format_version: 1,
                publication_modes: AMDF_QUEUE_PUBLICATION_MODE_USER,
                mailbox_address: mailbox.device,
                trigger_data: event.info().kfd_event_id,
                interrupt_context: event.info().kfd_event_id,
                ..Default::default()
            },
            native: event,
            device: pointer.cast(),
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn info(
    pointer: *mut amdf_gpu_event_t,
    out: *mut amdf_gpu_event_info_t,
) -> u64 {
    boundary(|| unsafe {
        let event = object(pointer.cast::<Event>())?;
        output(out, AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO)?.publish_from(&raw const event.info);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn signal(pointer: *mut amdf_gpu_event_t) -> u64 {
    boundary(|| unsafe {
        let event = object(pointer.cast::<Event>())?;
        if (*event.device).current_reset_epoch() != event.info.reset_epoch {
            return Err(LOST);
        }
        event
            .native
            .signal()
            .map_err(|error| (*event.device).native_error_status(&error))
    })
}

pub(crate) unsafe extern "C" fn wait(pointer: *mut amdf_gpu_event_t, timeout: u64) -> u64 {
    boundary(|| unsafe {
        let event = object(pointer.cast::<Event>())?;
        let device = &*event.device;
        if device.current_reset_epoch() != event.info.reset_epoch {
            return Err(LOST);
        }
        if device.event_observation_failed() {
            return Err(INTERNAL);
        }
        let result = event.native.wait(timeout);
        if device.current_reset_epoch() != event.info.reset_epoch {
            return Err(LOST);
        }
        if let Err(error) = &result {
            if error.kind() == rocddi::ErrorKind::DeviceLost {
                device.observe_loss(event.info.reset_epoch);
                return Err(LOST);
            }
        }
        match result {
            Ok(true) => Ok(()),
            Ok(false) | Err(_) if device.event_observation_failed() => Err(INTERNAL),
            Ok(false) => Err(DEADLINE),
            Err(error) => Err(device.native_error_status(&error)),
        }
    })
}

pub(crate) unsafe extern "C" fn destroy(pointer: *mut amdf_gpu_event_t) -> u64 {
    boundary(|| unsafe {
        let event = exclusive(pointer.cast::<Event>())?;
        event
            .native
            .destroy()
            .map_err(|error| (*event.device).native_error_status(&error))?;
        unregister(&(*event.device).events);
        drop(Owned::from_raw(pointer.cast::<Event>()));
        Ok(())
    })
}
