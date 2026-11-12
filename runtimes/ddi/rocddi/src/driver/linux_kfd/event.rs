// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Owned KFD signal events used by interrupt-capable HSA signals.
//!
//! Creation validates every identifier and mailbox field returned by KFD before
//! publication. Destruction is explicitly retryable for ordinary failures. An
//! interrupted destroy has an unknowable native outcome, so the owner records
//! that uncertainty and refuses to replay the event ID.

use super::{errno, error, native_error, sys, uapi};
use crate::device::gpu::event::linux::SignalEventInfo;
use crate::memory::storage::{Owned, Shared};
use crate::{Error, ErrorKind};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// One KFD event and the shared native endpoint required to destroy it.
pub(crate) struct KfdSignalEvent {
    /// Native connection that created this event.
    kfd: Shared<sys::Kfd>,
    /// Event ID until native destruction succeeds.
    event_id: Option<u32>,
    /// Cached mailbox and trigger data for HSA signal construction.
    info: SignalEventInfo,
    /// An interrupted destroy may already have released the event ID.
    uncertain: bool,
    /// Waits are caller-serialized; fault events exist before publication.
    wait_state: Mutex<SignalWaitState>,
}

struct SignalWaitState {
    last_age: u64,
    memory_id: Option<u32>,
    hardware_id: Option<u32>,
    memory_uncertain: bool,
    hardware_uncertain: bool,
}

impl SignalWaitState {
    fn prepare_event(
        kfd: &sys::Kfd,
        event_type: u32,
        id: &mut Option<u32>,
        uncertain: &mut bool,
    ) -> Result<(), Error> {
        if *uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD fault event state is uncertain",
            ));
        }
        if id.is_some() {
            return Ok(());
        }
        let mut args = uapi::CreateEvent::default();
        if let Err(source) = kfd.create_exception_event(event_type, &mut args) {
            if args.event_id != 0 && kfd.destroy_event(args.event_id).is_err() {
                *id = Some(args.event_id);
                *uncertain = true;
            }
            return Err(signal_event_error("KFD fault event creation", source));
        }
        if args.event_id == 0 {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD returned an invalid fault event",
            ));
        }
        *id = Some(args.event_id);
        Ok(())
    }

    fn prepare(&mut self, kfd: &sys::Kfd) -> Result<(u32, u32), Error> {
        Self::prepare_event(
            kfd,
            uapi::MEMORY_EXCEPTION,
            &mut self.memory_id,
            &mut self.memory_uncertain,
        )?;
        Self::prepare_event(
            kfd,
            uapi::HW_EXCEPTION,
            &mut self.hardware_id,
            &mut self.hardware_uncertain,
        )?;
        Ok((
            self.memory_id
                .ok_or_else(|| error(ErrorKind::Internal, "KFD memory fault event is missing"))?,
            self.hardware_id
                .ok_or_else(|| error(ErrorKind::Internal, "KFD hardware fault event is missing"))?,
        ))
    }
}

impl KfdSignalEvent {
    pub(super) fn create(
        kfd: Shared<sys::Kfd>,
        event_page_handle: Option<u64>,

        page_offered: &mut bool,
    ) -> Result<Owned<Self>, Error> {
        *page_offered = false;
        let storage = Owned::try_new_uninit()?;
        let mut args = uapi::CreateEvent::default();
        if let Err(source) = kfd.create_signal_event(event_page_handle, &mut args, page_offered) {
            if args.event_id != 0 && kfd.destroy_event(args.event_id).is_err() {
                std::mem::forget(kfd);
            }
            return Err(signal_event_error("KFD signal event creation", source));
        }
        if args.event_id == 0
            || args.trigger_data != args.event_id
            || args.page_offset == 0
            || args.slot_index >= uapi::SIGNAL_EVENT_LIMIT
        {
            if args.event_id != 0 && kfd.destroy_event(args.event_id).is_err() {
                std::mem::forget(kfd);
            }
            return Err(error(
                ErrorKind::DriverContract,
                "KFD returned an invalid signal event",
            ));
        }
        let event = storage.write(Self {
            kfd,
            event_id: Some(args.event_id),
            info: SignalEventInfo {
                kfd_event_id: args.event_id,
                event_page_slot_index: args.slot_index,
            },
            uncertain: false,
            wait_state: Mutex::new(SignalWaitState {
                last_age: 1,
                memory_id: None,
                hardware_id: None,
                memory_uncertain: false,
                hardware_uncertain: false,
            }),
        });
        // Register fault events before publishing the signal owner. A fault
        // between publication and the first wait must wake that wait too.
        // Dropping a partially prepared owner releases every known event ID.
        event
            .wait_state
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "KFD signal wait state lock poisoned"))?
            .prepare(&event.kfd)?;
        Ok(event)
    }

    pub(crate) fn info(&self) -> SignalEventInfo {
        self.info
    }

    pub(crate) fn signal(&self) -> Result<(), Error> {
        let event_id = self.event_id.ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "KFD signal event has been destroyed",
            )
        })?;
        self.kfd
            .set_event(event_id)
            .map_err(|source| signal_event_error("KFD signal event set", source))
    }

    pub(crate) fn wait(&self, timeout_ns: u64) -> Result<bool, Error> {
        let event_id = self.event_id.ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "KFD signal event has been destroyed",
            )
        })?;
        let mut state = self
            .wait_state
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "KFD signal wait state lock poisoned"))?;
        let memory_id = state
            .memory_id
            .ok_or_else(|| error(ErrorKind::Internal, "KFD memory fault event is missing"))?;
        let hardware_id = state
            .hardware_id
            .ok_or_else(|| error(ErrorKind::Internal, "KFD hardware fault event is missing"))?;
        let infinite = timeout_ns == u64::MAX;
        let duration = Duration::from_nanos(timeout_ns);
        let start = Instant::now();
        loop {
            let remaining = if infinite {
                duration
            } else {
                duration.saturating_sub(start.elapsed())
            };
            let milliseconds = if infinite {
                u32::MAX
            } else {
                let millis =
                    remaining.as_millis() + u128::from(remaining.subsec_nanos() % 1_000_000 != 0);
                u32::try_from(millis.min(i32::MAX as u128)).unwrap_or(i32::MAX as u32)
            };
            match self.kfd.wait_signal_event(
                event_id,
                state.last_age,
                memory_id,
                hardware_id,
                milliseconds,
            ) {
                Ok(Some((_, true))) => {
                    return Err(error(
                        ErrorKind::DeviceLost,
                        "KFD reported a device fault while waiting",
                    ));
                }
                Ok(Some((new_age, false))) => {
                    state.last_age = new_age;
                    return Ok(true);
                }
                Ok(None) if !infinite && start.elapsed() >= duration => return Ok(false),
                Ok(None) => {}
                Err(source) if source.raw_os_error() == Some(errno::EINTR) => return Ok(true),
                Err(source)
                    if matches!(source.raw_os_error(), Some(errno::EIO | errno::ENODEV)) =>
                {
                    return Err(error(
                        ErrorKind::DeviceLost,
                        "KFD context failed while waiting for a signal",
                    ));
                }
                Err(source) => return Err(signal_event_error("KFD signal event wait", source)),
            }
        }
    }

    pub(crate) fn destroy(&mut self) -> Result<(), Error> {
        let state = self
            .wait_state
            .get_mut()
            .map_err(|_| error(ErrorKind::Internal, "KFD signal wait state lock poisoned"))?;
        Self::destroy_id(&self.kfd, &mut state.memory_id, &mut state.memory_uncertain)?;
        Self::destroy_id(
            &self.kfd,
            &mut state.hardware_id,
            &mut state.hardware_uncertain,
        )?;
        Self::destroy_id(&self.kfd, &mut self.event_id, &mut self.uncertain)
    }

    fn destroy_id(kfd: &sys::Kfd, id: &mut Option<u32>, uncertain: &mut bool) -> Result<(), Error> {
        let Some(event_id) = *id else {
            return Ok(());
        };
        if *uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD signal event destruction outcome is uncertain",
            ));
        }
        match kfd.destroy_event(event_id) {
            Ok(()) => {
                *id = None;
                Ok(())
            }
            Err(source) => {
                *uncertain = source.raw_os_error() == Some(errno::EINTR);
                Err(signal_event_error("KFD signal event destruction", source))
            }
        }
    }
}

impl Drop for KfdSignalEvent {
    fn drop(&mut self) {
        if self.destroy().is_err() {
            std::mem::forget(self.kfd.clone());
        }
    }
}

fn signal_event_error(operation: &'static str, source: std::io::Error) -> Error {
    match source.raw_os_error() {
        Some(errno::EINVAL) => Error::NativeOperation {
            kind: ErrorKind::InvalidArgument,
            operation,
            source,
        },
        Some(errno::EAGAIN | errno::EBUSY) => Error::NativeOperation {
            kind: ErrorKind::Busy,
            operation,
            source,
        },
        Some(errno::ENOMEM | errno::ENOSPC) => Error::NativeOperation {
            kind: ErrorKind::ResourceExhausted,
            operation,
            source,
        },
        _ => native_error(operation, source),
    }
}

#[cfg(test)]
#[allow(unsafe_code, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::driver::linux_kfd::sys::{Call, IoctlHook, Kfd};
    use std::ffi::c_void;
    use std::fs::File;
    use std::sync::{Arc, Mutex};

    fn endpoint(hook: IoctlHook) -> Shared<Kfd> {
        Shared::new(Kfd::with_hook(File::open("/dev/null").unwrap(), hook)).unwrap()
    }

    #[test]
    fn signal_event_propagates_id_and_slot_and_destroys_once() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = calls.clone();
        let kfd = endpoint(Arc::new(move |call| {
            match call {
                Call::CreateEvent(args) => {
                    observed
                        .lock()
                        .unwrap()
                        .push((0, u64::from(args.event_type)));
                    match args.event_type {
                        uapi::SIGNAL_EVENT => {
                            assert_eq!(args.page_offset, 0x1234);
                            assert_eq!(args.auto_reset, 1);
                            args.page_offset = 0x8000_0000_0000_0000;
                            args.trigger_data = 19;
                            args.event_id = 19;
                            args.slot_index = 7;
                        }
                        uapi::MEMORY_EXCEPTION => args.event_id = 20,
                        uapi::HW_EXCEPTION => args.event_id = 21,
                        _ => panic!("unexpected event type"),
                    }
                }
                Call::DestroyEvent(args) => {
                    observed.lock().unwrap().push((args.event_id, 0));
                }
                Call::WaitSignal(args, events) => {
                    assert_eq!(
                        (events[0].event_id, events[1].event_id, events[2].event_id),
                        (19, 20, 21)
                    );
                    assert_eq!(args.timeout, u32::MAX);
                    observed.lock().unwrap().push((99, 0));
                    args.result = uapi::WAIT_COMPLETE;
                    events[1].payload[0] = 0;
                }
                _ => panic!("unexpected event call"),
            }
            Ok(())
        }));
        let mut page_offered = false;
        let mut event = KfdSignalEvent::create(kfd, Some(0x1234), &mut page_offered).unwrap();
        assert!(page_offered);
        assert_eq!(
            event.info(),
            SignalEventInfo {
                kfd_event_id: 19,
                event_page_slot_index: 7,
            }
        );
        assert_eq!(
            event.wait(u64::MAX).unwrap_err().kind(),
            ErrorKind::DeviceLost
        );
        event.destroy().unwrap();
        drop(event);
        assert_eq!(
            *calls.lock().unwrap(),
            [
                (0, u64::from(uapi::SIGNAL_EVENT)),
                (0, u64::from(uapi::MEMORY_EXCEPTION)),
                (0, u64::from(uapi::HW_EXCEPTION)),
                (99, 0),
                (20, 0),
                (21, 0),
                (19, 0),
            ]
        );
    }

    #[test]
    fn first_event_failure_can_return_without_an_event_id() {
        let calls = Arc::new(Mutex::new(0));
        let observed = calls.clone();
        let kfd = endpoint(Arc::new(move |call| match call {
            Call::CreateEvent(args) => {
                assert_eq!(args.page_offset, 0x1234);
                *observed.lock().unwrap() += 1;
                Err(std::io::Error::from_raw_os_error(errno::ENOMEM))
            }
            _ => panic!("unexpected event call"),
        }));
        let mut page_offered = false;
        let error = KfdSignalEvent::create(kfd, Some(0x1234), &mut page_offered)
            .err()
            .unwrap();
        assert!(page_offered);
        assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn fault_registration_failure_releases_prepared_and_signal_events() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = calls.clone();
        let kfd = endpoint(Arc::new(move |call| {
            match call {
                Call::CreateEvent(args) => match args.event_type {
                    uapi::SIGNAL_EVENT => {
                        observed.lock().unwrap().push("signal");
                        args.page_offset = 1;
                        args.trigger_data = 31;
                        args.event_id = 31;
                        args.slot_index = 2;
                    }
                    uapi::MEMORY_EXCEPTION => {
                        observed.lock().unwrap().push("memory");
                        args.event_id = 32;
                    }
                    uapi::HW_EXCEPTION => {
                        observed.lock().unwrap().push("hardware");
                        return Err(std::io::Error::from_raw_os_error(errno::ENOMEM));
                    }
                    _ => panic!("unexpected event type"),
                },
                Call::DestroyEvent(args) => match args.event_id {
                    32 => observed.lock().unwrap().push("destroy memory"),
                    31 => observed.lock().unwrap().push("destroy signal"),
                    _ => panic!("unexpected event ID"),
                },
                _ => panic!("unexpected event call"),
            }
            Ok(())
        }));
        let mut page_offered = false;
        let error = KfdSignalEvent::create(kfd, Some(0x1234), &mut page_offered)
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
        assert!(page_offered);
        assert_eq!(
            *calls.lock().unwrap(),
            [
                "signal",
                "memory",
                "hardware",
                "destroy memory",
                "destroy signal"
            ]
        );
    }

    #[test]
    fn invalid_page_handle_is_rejected_before_offer() {
        let kfd = endpoint(Arc::new(|_| panic!("invalid page reached KFD")));
        let mut page_offered = true;
        assert!(KfdSignalEvent::create(kfd, Some(0), &mut page_offered).is_err());
        assert!(!page_offered);
    }

    #[test]
    fn closed_kfd_rejects_event_before_page_offer() {
        let mut kfd = endpoint(Arc::new(|_| panic!("closed KFD reached ioctl")));
        Shared::get_mut(&mut kfd).unwrap().close().unwrap();
        let mut page_offered = true;
        assert!(KfdSignalEvent::create(kfd, Some(0x1234), &mut page_offered).is_err());
        assert!(!page_offered);
    }

    #[test]
    fn metadata_failure_does_not_offer_the_page() {
        unsafe extern "C" fn reject(_: *mut c_void, _: u64, _: u64) -> *mut c_void {
            std::ptr::null_mut()
        }
        unsafe extern "C" fn unused_free(_: *mut c_void, _: *mut c_void) {}

        let kfd = endpoint(Arc::new(|_| panic!("allocation failure reached KFD")));
        // SAFETY: The rejection callback is valid for this call and never
        // publishes storage; free cannot be invoked.
        let allocator = unsafe {
            crate::memory::storage::Allocator::from_callbacks(
                std::ptr::null_mut(),
                Some(reject),
                Some(unused_free),
            )
            .unwrap()
        };
        let mut page_offered = true;
        allocator.with_scope(|| {
            let error = KfdSignalEvent::create(kfd, Some(0x1234), &mut page_offered)
                .err()
                .unwrap();
            assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
            assert!(!page_offered);
        });
    }

    #[test]
    fn interrupted_destroy_is_not_replayed() {
        let destroys = Arc::new(Mutex::new(0));
        let observed = destroys.clone();
        let kfd = endpoint(Arc::new(move |call| match call {
            Call::CreateEvent(args) => {
                match args.event_type {
                    uapi::SIGNAL_EVENT => {
                        assert_eq!(args.page_offset, 0);
                        args.page_offset = 1;
                        args.trigger_data = 23;
                        args.event_id = 23;
                        args.slot_index = 9;
                    }
                    uapi::MEMORY_EXCEPTION => args.event_id = 24,
                    uapi::HW_EXCEPTION => args.event_id = 25,
                    _ => panic!("unexpected event type"),
                }
                Ok(())
            }
            Call::DestroyEvent(_) => {
                *observed.lock().unwrap() += 1;
                Err(std::io::Error::from_raw_os_error(errno::EINTR))
            }
            _ => panic!("unexpected event call"),
        }));
        let mut page_offered = false;
        let mut event = KfdSignalEvent::create(kfd, None, &mut page_offered).unwrap();
        assert!(!page_offered);
        assert_eq!(event.destroy().unwrap_err().kind(), ErrorKind::Driver);
        assert_eq!(
            event.destroy().unwrap_err().kind(),
            ErrorKind::DriverContract
        );
        drop(event);
        assert_eq!(*destroys.lock().unwrap(), 1);
    }
}
