// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! SDMA callbacks for native failures affecting the queue's device VM.
//!
//! KFD reports process memory faults and device exceptions without an SDMA
//! queue identifier. Each queue observes its own endpoint's stream; it never
//! interprets these events as an AQL inactive-signal or scratch request.

use super::*;
use rocddi::device::event::{DeviceEvent, DeviceEventSubscription};

/// Observes independent native events until queue teardown stops and joins us.
pub(super) fn worker(
    mut subscription: DeviceEventSubscription,
    stop: &AtomicBool,
    alive: &AtomicBool,
    endpoint: [u8; 16],
    key: usize,
) {
    worker_with(
        |timeout| subscription.wait(timeout),
        thread::sleep,
        |status| notify(key, status),
        stop,
        alive,
        endpoint,
    );
}

/// Invokes a callback outside the registry lock while teardown retains it.
fn notify(key: usize, status: Status) {
    let pending = {
        let Ok(guard) = lock() else { return };
        let Some(runtime) = guard.as_ref() else {
            return;
        };
        let Some(queue) = runtime.queue_state.sdma_queues.get(&key) else {
            return;
        };
        if !queue.event_alive.load(Ordering::Acquire) {
            return;
        }
        queue
            .callback
            .map(|callback| (callback, queue.callback_data))
    };
    if let Some((callback, data)) = pending {
        let _scope = CallbackScope::enter();
        // SAFETY: Queue destruction removes the registry entry, then joins
        // this worker before the descriptor or caller callback data may expire.
        unsafe { callback(status, key as *mut HsaQueue, data.as_ptr()) };
    }
}

/// Filters process/device failures to the endpoint owning this SDMA queue.
fn event_status(event: DeviceEvent, endpoint: [u8; 16]) -> Option<Status> {
    match event {
        DeviceEvent::GpuMemoryFault(fault) if fault.endpoint_id == Some(endpoint) => {
            Some(MEMORY_FAULT)
        }
        DeviceEvent::GpuHardwareException(exception) if exception.endpoint_id == Some(endpoint) => {
            Some(EXCEPTION)
        }
        _ => None,
    }
}

/// Runs observation and retry policy with injectable native operations.
fn worker_with(
    mut wait: impl FnMut(Duration) -> Result<Option<DeviceEvent>, rocddi::Error>,
    mut pause: impl FnMut(Duration),
    mut notify: impl FnMut(Status),
    stop: &AtomicBool,
    alive: &AtomicBool,
    endpoint: [u8; 16],
) {
    let mut retry_delay = Duration::from_millis(1);
    let mut driver_failures = 0u8;
    while !stop.load(Ordering::Acquire) && alive.load(Ordering::Acquire) {
        let Some(_lease) = crate::runtime::enter_runtime_service() else {
            pause(Duration::from_millis(1));
            continue;
        };
        let status = match wait(Duration::from_millis(1)) {
            Ok(event) => {
                retry_delay = Duration::from_millis(1);
                driver_failures = 0;
                event.and_then(|event| event_status(event, endpoint))
            }
            Err(error) if error.kind() == rocddi::ErrorKind::Busy => {
                driver_failures = 0;
                pause(retry_delay);
                retry_delay = (retry_delay * 2).min(Duration::from_millis(20));
                None
            }
            Err(error) if error.kind() == rocddi::ErrorKind::Driver && driver_failures < 16 => {
                driver_failures += 1;
                pause(retry_delay);
                retry_delay = (retry_delay * 2).min(Duration::from_millis(20));
                None
            }
            Err(error) => {
                if !stop.load(Ordering::Acquire) && alive.load(Ordering::Acquire) {
                    notify(map_error(error));
                }
                return;
            }
        };
        if let Some(status) = status {
            if stop.load(Ordering::Acquire) || !alive.load(Ordering::Acquire) {
                return;
            }
            // A device failure terminates this queue's event stream. Deliver
            // it once even when multiple native fault records are retained.
            notify(status);
            return;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::unreachable)]
mod tests {
    use super::*;
    use rocddi::device::event::{
        GpuHardwareException, GpuMemoryFault, GpuMemoryFaultCause, GpuResetCause, GpuResetScope,
    };
    use std::sync::mpsc;

    fn fault(endpoint_id: Option<[u8; 16]>) -> DeviceEvent {
        DeviceEvent::GpuMemoryFault(GpuMemoryFault {
            endpoint_id,
            virtual_address: 0x1000,
            page_not_present: true,
            read_only: false,
            no_execute: false,
            imprecise: false,
            cause: GpuMemoryFaultCause::None,
        })
    }

    #[test]
    fn endpoint_events_are_filtered_and_terminal_fault_is_delivered_once() {
        let stop = AtomicBool::new(false);
        let alive = AtomicBool::new(true);
        let mut events = [
            fault(Some([2; 16])),
            fault(None),
            fault(Some([1; 16])),
            fault(Some([1; 16])),
        ]
        .into_iter();
        let mut statuses = Vec::new();
        worker_with(
            |_| Ok(events.next()),
            |_| unreachable!(),
            |status| {
                statuses.push(status);
            },
            &stop,
            &alive,
            [1; 16],
        );
        assert_eq!(statuses, [MEMORY_FAULT]);
        assert!(events.next().is_some());
        assert_eq!(
            event_status(
                DeviceEvent::GpuHardwareException(GpuHardwareException {
                    endpoint_id: Some([1; 16]),
                    origin_id: None,
                    scope: GpuResetScope::WholeGpu,
                    memory_lost: true,
                    cause: GpuResetCause::Hang,
                }),
                [1; 16]
            ),
            Some(EXCEPTION)
        );
    }

    #[test]
    fn stopped_worker_does_not_invoke_a_callback_from_a_completed_wait() {
        let stop = AtomicBool::new(false);
        let alive = AtomicBool::new(true);
        worker_with(
            |_| {
                alive.store(false, Ordering::Release);
                Ok(Some(fault(Some([1; 16]))))
            },
            |_| unreachable!(),
            |_| panic!("stopped worker must not invoke callback"),
            &stop,
            &alive,
            [1; 16],
        );
    }

    #[test]
    fn native_poll_failure_retries_then_reports_once() {
        let stop = AtomicBool::new(false);
        let alive = AtomicBool::new(true);
        let mut polls = 0;
        let mut pauses = Vec::new();
        let mut statuses = Vec::new();
        worker_with(
            |_| {
                polls += 1;
                Err(rocddi::Error::Operation {
                    kind: rocddi::ErrorKind::Driver,
                    detail: "injected poll failure",
                })
            },
            |delay| pauses.push(delay),
            |status| {
                statuses.push(status);
            },
            &stop,
            &alive,
            [1; 16],
        );
        assert_eq!(polls, 17);
        assert_eq!(pauses.len(), 16);
        assert_eq!(pauses[0], Duration::from_millis(1));
        assert_eq!(pauses[15], Duration::from_millis(20));
        assert_eq!(statuses, [ERROR]);
    }

    #[test]
    fn queue_teardown_waits_for_an_inflight_sdma_callback() {
        let alive = Arc::new(AtomicBool::new(true));
        let worker_alive = alive.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            worker_with(
                |_| Ok(Some(fault(Some([1; 16])))),
                |_| unreachable!(),
                |_| {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                },
                &AtomicBool::new(false),
                &worker_alive,
                [1; 16],
            );
        });
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let (joined_tx, joined_rx) = mpsc::channel();
        let teardown = thread::spawn(move || {
            stop_queue_event_worker(&alive, &mut Some(worker)).unwrap();
            joined_tx.send(()).unwrap();
        });
        assert!(joined_rx.recv_timeout(Duration::from_millis(10)).is_err());
        release_tx.send(()).unwrap();
        joined_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        teardown.join().unwrap();
    }
}
