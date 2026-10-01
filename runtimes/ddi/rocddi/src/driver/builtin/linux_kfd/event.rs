//! Owned KFD signal events used by interrupt-capable HSA signals.
//!
//! Creation validates every identifier and mailbox field returned by KFD before
//! publication. Destruction is explicitly retryable for ordinary failures. An
//! interrupted destroy has an unknowable native outcome, so the owner records
//! that uncertainty and refuses to replay the event ID.

use super::memory::{error, native_error};
use super::{sys, uapi};
use crate::event::SignalEventInfo;
use crate::host_storage::{Allocator, Owned, Shared};
use crate::{Error, ErrorKind};

/// One KFD event and the shared native endpoint required to destroy it.
pub(crate) struct KfdSignalEvent {
    kfd: Shared<sys::Kfd>,
    event_id: Option<u32>,
    info: SignalEventInfo,
    uncertain: bool,
}

impl KfdSignalEvent {
    pub(super) fn create(
        kfd: Shared<sys::Kfd>,
        event_page_handle: Option<u64>,
        allocator: Allocator,
    ) -> Result<Owned<Self>, Error> {
        let storage = Owned::try_new_uninit(allocator)?;
        let mut args = uapi::CreateEvent::default();
        if let Err(source) = kfd.create_signal_event(event_page_handle, &mut args) {
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
        Ok(storage.write(Self {
            kfd,
            event_id: Some(args.event_id),
            info: SignalEventInfo {
                kfd_event_id: args.event_id,
                event_page_slot_index: args.slot_index,
            },
            uncertain: false,
        }))
    }

    pub(crate) fn info(&self) -> SignalEventInfo {
        self.info
    }

    pub(crate) fn destroy(&mut self) -> Result<(), Error> {
        let Some(event_id) = self.event_id else {
            return Ok(());
        };
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD signal event destruction outcome is uncertain",
            ));
        }
        match self.kfd.destroy_event(event_id) {
            Ok(()) => {
                self.event_id = None;
                Ok(())
            }
            Err(source) => {
                self.uncertain = source.raw_os_error() == Some(4);
                Err(signal_event_error("KFD signal event destruction", source))
            }
        }
    }
}

impl Drop for KfdSignalEvent {
    fn drop(&mut self) {
        if self.destroy().is_err() && self.event_id.is_some() {
            std::mem::forget(self.kfd.clone());
        }
    }
}

fn signal_event_error(operation: &'static str, source: std::io::Error) -> Error {
    match source.raw_os_error() {
        Some(22) => Error::NativeOperation {
            kind: ErrorKind::InvalidArgument,
            operation,
            source,
        },
        Some(11 | 16) => Error::NativeOperation {
            kind: ErrorKind::Busy,
            operation,
            source,
        },
        Some(12 | 28) => Error::NativeOperation {
            kind: ErrorKind::ResourceExhausted,
            operation,
            source,
        },
        _ => native_error(operation, source),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::driver::builtin::linux_kfd::sys::{Call, IoctlHook, Kfd};
    use std::fs::File;
    use std::sync::{Arc, Mutex};

    fn endpoint(hook: IoctlHook) -> Shared<Kfd> {
        Shared::new(
            Kfd::with_hook(File::open("/dev/null").unwrap(), hook),
            Allocator::default(),
        )
        .unwrap()
    }

    #[test]
    fn signal_event_propagates_id_and_slot_and_destroys_once() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = calls.clone();
        let kfd = endpoint(Arc::new(move |call| {
            match call {
                Call::CreateEvent(args) => {
                    observed.lock().unwrap().push((0, args.page_offset));
                    assert_eq!(args.event_type, uapi::SIGNAL_EVENT);
                    assert_eq!(args.auto_reset, 1);
                    args.page_offset = 0x8000_0000_0000_0000;
                    args.trigger_data = 19;
                    args.event_id = 19;
                    args.slot_index = 7;
                }
                Call::DestroyEvent(args) => {
                    observed.lock().unwrap().push((args.event_id, 0));
                }
                _ => panic!("unexpected event call"),
            }
            Ok(())
        }));
        let mut event = KfdSignalEvent::create(kfd, Some(0x1234), Allocator::default()).unwrap();
        assert_eq!(
            event.info(),
            SignalEventInfo {
                kfd_event_id: 19,
                event_page_slot_index: 7,
            }
        );
        event.destroy().unwrap();
        drop(event);
        assert_eq!(*calls.lock().unwrap(), [(0, 0x1234), (19, 0)]);
    }

    #[test]
    fn interrupted_destroy_is_not_replayed() {
        let destroys = Arc::new(Mutex::new(0));
        let observed = destroys.clone();
        let kfd = endpoint(Arc::new(move |call| match call {
            Call::CreateEvent(args) => {
                assert_eq!(args.page_offset, 0);
                args.page_offset = 1;
                args.trigger_data = 23;
                args.event_id = 23;
                args.slot_index = 9;
                Ok(())
            }
            Call::DestroyEvent(_) => {
                *observed.lock().unwrap() += 1;
                Err(std::io::Error::from_raw_os_error(4))
            }
            _ => panic!("unexpected event call"),
        }));
        let mut event = KfdSignalEvent::create(kfd, None, Allocator::default()).unwrap();
        assert_eq!(event.destroy().unwrap_err().kind(), ErrorKind::Driver);
        assert_eq!(
            event.destroy().unwrap_err().kind(),
            ErrorKind::DriverContract
        );
        drop(event);
        assert_eq!(*destroys.lock().unwrap(), 1);
    }
}
