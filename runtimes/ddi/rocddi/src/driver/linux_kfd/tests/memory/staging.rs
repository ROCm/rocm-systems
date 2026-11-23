// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host-only native-boundary failures for a four-MiB owned staging allocation.

use super::*;

#[test]
fn owned_host_staging_busy_errors_preserve_native_cause_and_rollback() {
    for (replies, operation) in [
        (
            vec![Reply::Allocate(0, Some(errno::EAGAIN))],
            "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU",
        ),
        (
            vec![Reply::Allocate(17, Some(errno::EAGAIN)), Reply::Free(None)],
            "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU",
        ),
        (
            vec![
                Reply::Allocate(17, None),
                Reply::Map(0, 0, Some(errno::EAGAIN)),
                Reply::Map(0, 1, None),
                Reply::Unmap(0, 1, None),
                Reply::Free(None),
            ],
            "AMDKFD_IOC_MAP_MEMORY_TO_GPU",
        ),
        (
            vec![
                Reply::Allocate(17, None),
                Reply::Map(0, 1, Some(errno::EAGAIN)),
                Reply::Map(1, 1, None),
                Reply::Unmap(0, 1, None),
                Reply::Free(None),
            ],
            "AMDKFD_IOC_MAP_MEMORY_TO_GPU",
        ),
    ] {
        let pending = Arc::new(Mutex::new(VecDeque::from(replies)));
        let kfd = staging_connection(pending.clone());
        let vm = queue_fixture(kfd, File::open("/dev/null").unwrap(), node());
        let device = super::super::super::DeviceState {
            vm: vm.clone(),
            native: node(),
            asic_family_id: 0,
            gpu_counter_frequency_hz: 0,
        };
        let result = super::super::super::LinuxKfdDriver::with_context_lifetime(
            crate::session::DriverContextLifetime::Process,
            crate::os::ProcessIdentity::capture().unwrap(),
        )
        .allocate_owned(
            &device,
            &[],
            MemoryKind::OwnedHost {
                policy: HostMappingPolicy::Coarse,
            },
            4 << 20,
            os_memory::page_size().unwrap() as u64,
            DeviceAccess::READ | DeviceAccess::WRITE,
        );
        let error = result.err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Busy);
        assert_eq!(
            error
                .native_status()
                .and_then(crate::NativeStatus::errno_code),
            Some(errno::EAGAIN)
        );
        let Error::NativeOperation {
            operation: actual, ..
        } = error
        else {
            panic!("staging allocation lost its native operation");
        };
        assert_eq!(actual, operation);
        assert!(
            pending.lock().unwrap().is_empty(),
            "incomplete native rollback"
        );
        drop(device);
        assert_eq!(Shared::strong_count(&vm), 1, "rollback retained a VM owner");
    }
}

/// Scripted KFD calls retain the production four-MiB USERPTR arguments.
fn staging_connection(script: Arc<Mutex<VecDeque<Reply>>>) -> Shared<connection::Kfd> {
    shared(connection::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| {
            if let connection::Call::Wait(args, _) = call {
                args.result = uapi::WAIT_TIMEOUT;
                return Ok(());
            }
            if let connection::Call::DestroyEvent(_) = call {
                return Ok(());
            }
            let reply = script.lock().unwrap().pop_front().unwrap();
            let failure = match (call, reply) {
                (connection::Call::Allocate(args), Reply::Allocate(handle, failure)) => {
                    assert_eq!(args.size, 4 << 20);
                    assert_eq!(args.va % os_memory::page_size().unwrap() as u64, 0);
                    assert_eq!(args.mmap_offset, args.va);
                    assert_eq!(args.gpu_id, 42);
                    assert_eq!(
                        args.flags,
                        uapi::USERPTR | uapi::NO_SUBSTITUTE | uapi::WRITABLE
                    );
                    args.handle = handle;
                    failure
                }
                (connection::Call::Map(args, devices), Reply::Map(before, after, failure))
                | (connection::Call::Unmap(args, devices), Reply::Unmap(before, after, failure)) => {
                    assert_eq!(args.handle, 17);
                    assert_eq!(*devices, [42]);
                    assert_eq!(args.success, before);
                    args.success = after;
                    failure
                }
                (connection::Call::Free(args), Reply::Free(failure)) => {
                    assert_eq!(args.handle, 17);
                    failure
                }
                _ => panic!("unexpected staging allocation operation"),
            };
            failure.map_or(Ok(()), |failure| Err(io::Error::from_raw_os_error(failure)))
        }),
    ))
}
