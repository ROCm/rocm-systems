// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Owned GPU native events and KFD signal-page preparation.

use crate::generated::amdf::*;
use crate::instance::{self, Device, EventPageState};
use crate::support::*;
use rocddi::device::gpu::event::linux::{
    self, SignalEvent, SignalEventPage, lock_process_signal_event_page,
};
use rocddi::memory::device::{DeviceAccess, MemoryKind};
use rocddi::memory::storage::Owned;

struct Event {
    native: SignalEvent,
    device: *mut Device,
    info: amdf_gpu_event_info_t,
}

unsafe fn prepare_page(device: &Device) -> Result<EventPageState, u64> {
    // SAFETY: A live device borrows its endpoint and instance.
    let instance = unsafe { instance::device_instance(device) };
    let page_size = instance.host_page_size;
    let length = 32_768_u64.div_ceil(page_size) * page_size;
    let allocation = device
        .native
        .allocate(
            MemoryKind::System,
            length,
            page_size,
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .map_err(|error| device.native_error_status(&error))?;
    let address = allocation.info().device_address;
    let host_address = allocation.info().host_address.ok_or(UNSUPPORTED)?;
    Ok(EventPageState {
        page: SignalEventPage::new(allocation),
        endpoint_id: device.native.endpoint().id,
        address,
        host_address,
    })
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
        let gpu = device
            .native
            .gpu()
            .map_err(|error| device.native_error_status(&error))?;
        let reset_epoch = device.current_reset_epoch();
        let slot = Owned::<Event>::try_new_uninit().map_err(|_| EXHAUSTED)?;
        register(&device.events)?;
        let result = (|| {
            let mut process = lock_process_signal_event_page().map_err(|_| INTERNAL)?;
            let mut guard = instance.event_page.lock().map_err(|_| INTERNAL)?;
            let endpoint_id = device.native.endpoint().id;
            let (event, address) = if let Some(address) = process.device_address(endpoint_id) {
                (linux::create_signal_event(gpu, None), address)
            } else if process.host_address().is_some() {
                return Err(UNSUPPORTED);
            } else {
                if guard.as_ref().is_some_and(|state| {
                    state.endpoint_id != endpoint_id && !state.page.was_offered()
                }) {
                    // No native call could have installed the old candidate.
                    *guard = None;
                }
                if guard.is_none() {
                    *guard = Some(prepare_page(device)?);
                }
                let state = guard.as_mut().ok_or(INTERNAL)?;
                if state.endpoint_id != endpoint_id {
                    return Err(UNSUPPORTED);
                }
                let mut mappings = Vec::new();
                mappings.try_reserve_exact(1).map_err(|_| EXHAUSTED)?;
                mappings.push((endpoint_id, state.address));
                let first = linux::create_signal_event(gpu, Some(&mut state.page));
                let offered = state.page.was_offered();
                if offered {
                    process
                        .record_offer(state.host_address, mappings)
                        .map_err(|error| device.native_error_status(&error))?;
                }
                let event = match first {
                    Err(_) if offered && !process.is_confirmed() => {
                        linux::create_signal_event(gpu, None)
                    }
                    result => result,
                };
                (event, state.address)
            };
            let event = event.map_err(|error| device.native_error_status(&error))?;
            if process.host_address().is_none() {
                return Err(INTERNAL);
            }
            process.confirm();
            let mailbox = address
                .checked_add(u64::from(event.info().event_page_slot_index) * 8)
                .ok_or(RANGE)?;
            Ok((event, mailbox))
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
                mailbox_address: mailbox,
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
        let result = event.native.wait(timeout);
        if device.current_reset_epoch() != event.info.reset_epoch {
            return Err(LOST);
        }
        match result {
            Ok(true) => Ok(()),
            Ok(false) => Err(DEADLINE),
            Err(error) if error.kind() == rocddi::ErrorKind::DeviceLost => {
                device.observe_loss(event.info.reset_epoch);
                Err(LOST)
            }
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
