//! Exercise kernel output and retry contracts through the typed call boundary.

#![allow(clippy::unwrap_used, clippy::panic)]

use super::*;
use std::os::fd::{AsRawFd, IntoRawFd};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn endpoint(hook: IoctlHook) -> Kfd {
    Kfd::with_hook(File::open("/dev/null").unwrap(), hook)
}

#[test]
fn clock_query_preserves_the_correlated_kernel_sample() {
    let kfd = endpoint(Arc::new(|call| {
        let Call::ClockCounters(counters) = call else {
            panic!("unexpected ioctl")
        };
        assert_eq!(counters.gpu_id, 42);
        counters.gpu_clock_counter = 11;
        counters.cpu_clock_counter = 22;
        counters.system_clock_counter = 33;
        counters.system_clock_frequency = 44;
        Ok(())
    }));
    let counters = kfd.clock_counters(42).unwrap();
    assert_eq!(counters.gpu_clock_counter, 11);
    assert_eq!(counters.cpu_clock_counter, 22);
    assert_eq!(counters.system_clock_counter, 33);
    assert_eq!(counters.system_clock_frequency, 44);
}

#[test]
fn available_memory_query_preserves_the_kernel_result() {
    let kfd = endpoint(Arc::new(|call| {
        let Call::AvailableMemory(memory) = call else {
            panic!("unexpected ioctl")
        };
        assert_eq!(memory.gpu_id, 42);
        assert_eq!(memory.pad, 0);
        memory.available = 0x3f5c_00000;
        Ok(())
    }));
    assert_eq!(kfd.available_memory(42).unwrap(), 0x3f5c_00000);
}

#[test]
fn mmio_allocation_requires_the_exact_native_contract() {
    let kfd = endpoint(Arc::new(|call| {
        let Call::Allocate(args) = call else {
            panic!("unexpected ioctl")
        };
        assert_ne!(args.va, 0);
        assert_eq!(args.va % 4096, 0);
        assert_eq!(args.size, 4096);
        assert_eq!(args.gpu_id, 42);
        assert_eq!(args.mmap_offset, 0);
        assert_eq!(
            args.flags,
            uapi::MMIO_REMAP | uapi::WRITABLE | uapi::COHERENT
        );
        args.handle = 17;
        args.mmap_offset = 3_u64 << 62;
        Ok(())
    }));
    let mut args = uapi::AllocMemory {
        va: 0x1_0000,
        size: 4096,
        gpu_id: 42,
        flags: uapi::MMIO_REMAP | uapi::WRITABLE | uapi::COHERENT,
        ..uapi::AllocMemory::default()
    };
    kfd.allocate_mmio(&mut args).unwrap();
    assert_eq!(args.handle, 17);
    assert_eq!(args.mmap_offset, 3_u64 << 62);

    args.flags |= uapi::NO_SUBSTITUTE;
    assert_eq!(
        kfd.allocate_mmio(&mut args).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn trap_handler_preserves_device_and_gpu_addresses() {
    let kfd = endpoint(Arc::new(|call| {
        let Call::SetTrapHandler(args) = call else {
            panic!("unexpected ioctl")
        };
        assert_eq!(args.tba_address, 0x1_0000);
        assert_eq!(args.tma_address, 0x2_0000);
        assert_eq!(args.gpu_id, 42);
        assert_eq!(args.pad, 0);
        Ok(())
    }));
    kfd.set_trap_handler(42, 0x1_0000, 0x2_0000).unwrap();
}

#[test]
fn trap_handler_rejects_missing_device_identity() {
    let kfd = endpoint(Arc::new(|_| {
        panic!("invalid trap handler call reached KFD")
    }));
    assert_eq!(
        kfd.set_trap_handler(0, 1, 2).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn allocation_accepts_exactly_one_supported_backing_kind() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let kfd = endpoint(Arc::new(move |call| {
        let Call::Allocate(_) = call else {
            panic!("unexpected ioctl")
        };
        observed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }));
    for kind in [uapi::VRAM, uapi::GTT, uapi::USERPTR] {
        let mut args = uapi::AllocMemory {
            flags: kind,
            ..uapi::AllocMemory::default()
        };
        kfd.allocate(&mut args).unwrap();
    }
    for flags in [
        0,
        uapi::VRAM | uapi::GTT,
        uapi::GTT | uapi::USERPTR,
        uapi::VRAM | uapi::USERPTR,
        uapi::VRAM | uapi::GTT | uapi::USERPTR,
        uapi::USERPTR | uapi::DOORBELL,
    ] {
        let mut args = uapi::AllocMemory {
            flags,
            ..uapi::AllocMemory::default()
        };
        assert_eq!(
            kfd.allocate(&mut args).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}

#[test]
fn doorbell_allocation_requires_the_exact_gpuvm_contract() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let kfd = endpoint(Arc::new(move |call| {
        let Call::Allocate(args) = call else {
            panic!("unexpected ioctl")
        };
        assert_eq!(args.va, 0x10000);
        assert_eq!(args.size, 8192);
        assert_eq!(args.gpu_id, 42);
        observed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }));
    let required = uapi::DOORBELL | uapi::WRITABLE | uapi::COHERENT | uapi::NO_SUBSTITUTE;
    let mut args = uapi::AllocMemory {
        va: 0x10000,
        size: 8192,
        gpu_id: 42,
        flags: required,
        ..uapi::AllocMemory::default()
    };
    kfd.allocate_doorbells(&mut args).unwrap();
    for mut invalid in [
        uapi::AllocMemory { va: 0, ..args },
        uapi::AllocMemory { size: 0, ..args },
        uapi::AllocMemory {
            flags: required & !uapi::COHERENT,
            ..args
        },
    ] {
        assert_eq!(
            kfd.allocate_doorbells(&mut invalid).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn dma_buf_metadata_size_is_retried_with_owned_storage() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let kfd = endpoint(Arc::new(move |call| {
        let Call::DmaBufInfo(args, metadata) = call else {
            panic!("unexpected DMA-BUF query call")
        };
        assert_eq!(args.descriptor, 7);
        if observed.fetch_add(1, Ordering::Relaxed) == 0 {
            assert!(metadata.is_empty());
            args.metadata_size = 4;
            Err(io::Error::from_raw_os_error(22))
        } else {
            assert_eq!(metadata.len(), 4);
            metadata.copy_from_slice(&[1, 2, 3, 4]);
            args.size = 8192;
            args.gpu_id = 42;
            args.flags = uapi::GTT;
            Ok(())
        }
    }));
    let details = kfd.dma_buf_info(7).unwrap();
    assert_eq!(details.info.size, 8192);
    assert_eq!(details.info.gpu_id, 42);
    assert_eq!(details.info.flags, uapi::GTT);
    assert_eq!(details.metadata.as_slice(), &[1, 2, 3, 4]);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn export_owns_the_returned_descriptor_on_success_and_failure() {
    use std::io::Read;
    use std::os::unix::net::UnixStream;

    for errno in [None, Some(5)] {
        let (mut observer, returned_stream) = UnixStream::pair().unwrap();
        observer.set_nonblocking(true).unwrap();
        let returned = returned_stream.into_raw_fd();
        let kfd = endpoint(Arc::new(move |call| {
            let Call::ExportDmaBuf(args) = call else {
                panic!("unexpected DMA-BUF export call")
            };
            assert_eq!(args.handle, 17);
            assert_eq!(args.flags, O_CLOEXEC);
            args.descriptor = u32::try_from(returned).unwrap();
            errno.map_or(Ok(()), |code| Err(io::Error::from_raw_os_error(code)))
        }));
        let result = kfd.export_dma_buf(17);
        if errno.is_none() {
            let file = result.unwrap();
            assert_eq!(file.as_raw_fd(), returned);
            drop(file);
        } else {
            assert_eq!(result.unwrap_err().raw_os_error(), errno);
        }
        // Observe closure of the underlying socket. Another test may reopen
        // the same integer fd immediately after it is released.
        let mut byte = [0];
        assert_eq!(observer.read(&mut byte).unwrap(), 0);
    }
}

#[test]
fn ipc_export_preserves_the_allocation_identity_and_flags() {
    let kfd = endpoint(Arc::new(|call| {
        let Call::IpcExportHandle(args) = call else {
            panic!("unexpected IPC export call")
        };
        assert_eq!(args.handle, 17);
        assert_eq!(args.gpu_id, 42);
        assert_eq!(args.flags, uapi::GTT | uapi::WRITABLE);
        args.share_handle = [1, 2, 3, 4];
        Ok(())
    }));
    assert_eq!(
        kfd.export_ipc_handle(17, 42, uapi::GTT | uapi::WRITABLE)
            .unwrap(),
        [1, 2, 3, 4]
    );
}

#[test]
fn ipc_handles_reject_invalid_inputs_and_empty_kernel_outputs() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let kfd = endpoint(Arc::new(move |call| {
        let Call::IpcExportHandle(_) = call else {
            panic!("unexpected IPC call")
        };
        observed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }));
    assert_eq!(
        kfd.export_ipc_handle(0, 42, uapi::GTT).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        kfd.export_ipc_handle(17, 0, uapi::GTT).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        kfd.export_ipc_handle(17, 42, uapi::GTT).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let kfd = endpoint(Arc::new(|_| panic!("invalid IPC import reached KFD")));
    for mut args in [
        uapi::IpcImportHandle {
            share_handle: [1, 2, 3, 4],
            gpu_id: 42,
            ..uapi::IpcImportHandle::default()
        },
        uapi::IpcImportHandle {
            va_addr: 0x10000,
            gpu_id: 42,
            ..uapi::IpcImportHandle::default()
        },
        uapi::IpcImportHandle {
            va_addr: 0x10000,
            share_handle: [1, 2, 3, 4],
            ..uapi::IpcImportHandle::default()
        },
    ] {
        assert_eq!(
            kfd.import_ipc_handle(&mut args).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}

#[test]
fn ipc_import_preserves_kernel_outputs_on_failure() {
    let kfd = endpoint(Arc::new(|call| {
        let Call::IpcImportHandle(args) = call else {
            panic!("unexpected IPC import call")
        };
        assert_eq!(args.va_addr, 0x10000);
        assert_eq!(args.share_handle, [1, 2, 3, 4]);
        assert_eq!(args.gpu_id, 42);
        args.handle = 17;
        args.mmap_offset = 0x20000;
        args.flags = uapi::GTT | uapi::WRITABLE;
        Err(io::Error::from_raw_os_error(12))
    }));
    let mut args = uapi::IpcImportHandle {
        va_addr: 0x10000,
        share_handle: [1, 2, 3, 4],
        gpu_id: 42,
        ..uapi::IpcImportHandle::default()
    };
    assert_eq!(
        kfd.import_ipc_handle(&mut args).unwrap_err().raw_os_error(),
        Some(12)
    );
    assert_eq!(args.handle, 17);
    assert_eq!(args.mmap_offset, 0x20000);
    assert_eq!(args.flags, uapi::GTT | uapi::WRITABLE);
}

#[test]
fn svm_attributes_preserve_the_flexible_array_contract() {
    assert_eq!(uapi::svm_request(0), Some(uapi::SVM));
    assert_eq!(uapi::svm_request(2), Some(uapi::SVM + (16_u64 << 16)));
    assert!(uapi::svm_request(2044).is_some());
    assert!(uapi::svm_request(2045).is_none());

    let kfd = endpoint(Arc::new(|call| {
        let Call::Svm(args, attributes) = call else {
            panic!("unexpected SVM call")
        };
        assert_eq!(args.start_address, 0x10000);
        assert_eq!(args.size, 8192);
        assert_eq!(args.operation, uapi::SVM_OP_GET_ATTR);
        assert_eq!(args.attribute_count, 2);
        assert_eq!(attributes[0].attribute_type, uapi::SVM_ATTR_SET_FLAGS);
        assert_eq!(attributes[1].attribute_type, uapi::SVM_ATTR_ACCESS);
        assert_eq!(attributes[1].value, 42);
        attributes[0].value = uapi::SVM_FLAG_COHERENT;
        attributes[1].attribute_type = uapi::SVM_ATTR_ACCESS_IN_PLACE;
        Ok(())
    }));
    let mut attributes = [
        uapi::SvmAttribute {
            attribute_type: uapi::SVM_ATTR_SET_FLAGS,
            value: 0,
        },
        uapi::SvmAttribute {
            attribute_type: uapi::SVM_ATTR_ACCESS,
            value: 42,
        },
    ];
    kfd.svm_attributes(0x10000, 8192, uapi::SVM_OP_GET_ATTR, &mut attributes)
        .unwrap();
    assert_eq!(attributes[0].value, uapi::SVM_FLAG_COHERENT);
    assert_eq!(attributes[1].attribute_type, uapi::SVM_ATTR_ACCESS_IN_PLACE);
}

#[test]
fn svm_attributes_reject_invalid_ranges_before_ioctl() {
    let kfd = endpoint(Arc::new(|_| panic!("invalid SVM request reached KFD")));
    let mut attribute = [uapi::SvmAttribute {
        attribute_type: uapi::SVM_ATTR_SET_FLAGS,
        value: uapi::SVM_FLAG_COHERENT,
    }];
    for (address, size, operation) in [
        (0, 4096, uapi::SVM_OP_SET_ATTR),
        (0x10001, 4096, uapi::SVM_OP_SET_ATTR),
        (0x10000, 0, uapi::SVM_OP_SET_ATTR),
        (0x10000, 4097, uapi::SVM_OP_SET_ATTR),
        (0x10000, 4096, 2),
    ] {
        assert_eq!(
            kfd.svm_attributes(address, size, operation, &mut attribute)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
}

#[test]
fn spm_preserves_the_complete_kernel_record() {
    let kfd = endpoint(Arc::new(|call| {
        let Call::Spm(args) = call else {
            panic!("unexpected SPM call")
        };
        assert_eq!(args.destination, 0x10_000);
        assert_eq!(args.size, 8192);
        assert_eq!(args.operation, uapi::SPM_OP_SET_DESTINATION);
        assert_eq!(args.timeout, 50);
        assert_eq!(args.gpu_id, 42);
        assert_eq!(args.bytes_copied, 0);
        assert_eq!(args.has_data_loss, 0);
        args.timeout = 7;
        args.bytes_copied = 4096;
        args.has_data_loss = 2;
        Err(io::Error::from_raw_os_error(5))
    }));
    let mut args = uapi::Spm {
        destination: 0x10_000,
        size: 8192,
        operation: uapi::SPM_OP_SET_DESTINATION,
        timeout: 50,
        gpu_id: 42,
        bytes_copied: 0,
        has_data_loss: 0,
    };
    assert_eq!(kfd.spm(&mut args).unwrap_err().raw_os_error(), Some(5));
    assert_eq!(args.timeout, 7);
    assert_eq!(args.bytes_copied, 4096);
    assert_eq!(args.has_data_loss, 2);
}

#[test]
fn spm_rejects_invalid_operations_before_ioctl() {
    let kfd = endpoint(Arc::new(|_| panic!("invalid SPM request reached KFD")));
    for (gpu_id, operation) in [(0, uapi::SPM_OP_ACQUIRE), (42, 3)] {
        let mut args = uapi::Spm {
            operation,
            gpu_id,
            ..uapi::Spm::default()
        };
        assert_eq!(
            kfd.spm(&mut args).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}

#[test]
fn pc_sampling_capabilities_use_owned_two_pass_storage() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let kfd = endpoint(Arc::new(move |call| {
        let Call::PcSampling(args, configurations) = call else {
            panic!("unexpected PC sampling call")
        };
        assert_eq!(args.operation, uapi::PC_SAMPLE_OP_QUERY_CAPABILITIES);
        assert_eq!(args.gpu_id, 42);
        assert_eq!(args.trace_id, 0);
        assert_eq!(args.flags, 0);
        assert_eq!(args.version, 0);
        if observed.fetch_add(1, Ordering::Relaxed) == 0 {
            assert!(configurations.is_empty());
            assert_eq!(args.sample_info_count, 0);
            args.sample_info_count = 1;
        } else {
            assert_eq!(configurations.len(), 1);
            assert_eq!(args.sample_info_count, 1);
            configurations[0] = uapi::PcSampleInfo {
                interval: 0,
                interval_min: 512,
                interval_max: u64::MAX,
                flags: 0,
                method: uapi::PC_SAMPLE_METHOD_HOSTTRAP,
                sample_type: uapi::PC_SAMPLE_TYPE_TIME_US,
            };
        }
        Ok(())
    }));
    let configurations = kfd.pc_sampling_capabilities(42).unwrap();
    assert_eq!(configurations.len(), 1);
    assert_eq!(
        configurations[0],
        uapi::PcSampleInfo {
            interval: 0,
            interval_min: 512,
            interval_max: u64::MAX,
            flags: 0,
            method: uapi::PC_SAMPLE_METHOD_HOSTTRAP,
            sample_type: uapi::PC_SAMPLE_TYPE_TIME_US,
        }
    );
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn pc_sampling_rejects_a_growing_capability_result() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let kfd = endpoint(Arc::new(move |call| {
        let Call::PcSampling(args, configurations) = call else {
            panic!("unexpected PC sampling call")
        };
        if observed.fetch_add(1, Ordering::Relaxed) == 0 {
            assert!(configurations.is_empty());
            args.sample_info_count = 1;
        } else {
            assert_eq!(configurations.len(), 1);
            args.sample_info_count = 2;
        }
        Ok(())
    }));
    assert_eq!(
        kfd.pc_sampling_capabilities(42).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn pc_sampling_create_preserves_trace_id_on_failure() {
    let kfd = endpoint(Arc::new(|call| {
        let Call::PcSampling(args, configurations) = call else {
            panic!("unexpected PC sampling call")
        };
        assert_eq!(args.operation, uapi::PC_SAMPLE_OP_CREATE);
        assert_eq!(args.gpu_id, 42);
        assert_eq!(args.sample_info_count, 1);
        assert_eq!(args.trace_id, 0);
        assert_eq!(configurations.len(), 1);
        assert_eq!(configurations[0].interval, 512);
        assert_eq!(configurations[0].method, uapi::PC_SAMPLE_METHOD_HOSTTRAP);
        assert_eq!(configurations[0].sample_type, uapi::PC_SAMPLE_TYPE_TIME_US);
        args.trace_id = 17;
        Err(io::Error::from_raw_os_error(4))
    }));
    let mut configuration = uapi::PcSampleInfo {
        interval: 512,
        method: uapi::PC_SAMPLE_METHOD_HOSTTRAP,
        sample_type: uapi::PC_SAMPLE_TYPE_TIME_US,
        ..uapi::PcSampleInfo::default()
    };
    let mut trace_id = 0;
    assert_eq!(
        kfd.pc_sampling_create(42, &mut configuration, &mut trace_id)
            .unwrap_err()
            .raw_os_error(),
        Some(4)
    );
    assert_eq!(trace_id, 17);
}

#[test]
fn pc_sampling_controls_preserve_device_and_trace_identity() {
    let operations = Arc::new(Mutex::new(Vec::new()));
    let observed = operations.clone();
    let kfd = endpoint(Arc::new(move |call| {
        let Call::PcSampling(args, configurations) = call else {
            panic!("unexpected PC sampling call")
        };
        assert!(configurations.is_empty());
        assert_eq!(args.sample_info_count, 0);
        assert_eq!(args.gpu_id, 42);
        assert_eq!(args.trace_id, 17);
        assert_eq!(args.flags, 0);
        assert_eq!(args.version, 0);
        observed.lock().unwrap().push(args.operation);
        Ok(())
    }));
    kfd.pc_sampling_start(42, 17).unwrap();
    kfd.pc_sampling_stop(42, 17).unwrap();
    kfd.pc_sampling_destroy(42, 17).unwrap();
    assert_eq!(
        *operations.lock().unwrap(),
        [
            uapi::PC_SAMPLE_OP_START,
            uapi::PC_SAMPLE_OP_STOP,
            uapi::PC_SAMPLE_OP_DESTROY
        ]
    );
}

#[test]
fn pc_sampling_rejects_invalid_requests_before_ioctl() {
    let kfd = endpoint(Arc::new(|_| panic!("invalid PC sampling call reached KFD")));
    assert_eq!(
        kfd.pc_sampling_capabilities(0).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    let valid = uapi::PcSampleInfo {
        interval: 512,
        method: uapi::PC_SAMPLE_METHOD_HOSTTRAP,
        sample_type: uapi::PC_SAMPLE_TYPE_TIME_US,
        ..uapi::PcSampleInfo::default()
    };
    for (gpu_id, mut configuration, mut trace_id) in [
        (0, valid, 0),
        (
            42,
            uapi::PcSampleInfo {
                interval: 0,
                ..valid
            },
            0,
        ),
        (42, uapi::PcSampleInfo { method: 0, ..valid }, 0),
        (
            42,
            uapi::PcSampleInfo {
                sample_type: 3,
                ..valid
            },
            0,
        ),
        (42, valid, 1),
    ] {
        assert_eq!(
            kfd.pc_sampling_create(gpu_id, &mut configuration, &mut trace_id)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
    for (gpu_id, trace_id) in [(0, 17), (42, 0)] {
        assert_eq!(
            kfd.pc_sampling_start(gpu_id, trace_id).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            kfd.pc_sampling_stop(gpu_id, trace_id).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            kfd.pc_sampling_destroy(gpu_id, trace_id)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
}

#[test]
fn mapping_retries_preserve_zero_partial_and_full_prefixes() {
    for map in [false, true] {
        for progress in 0..=3 {
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = calls.clone();
            let kfd = endpoint(Arc::new(move |call| {
                let (args, devices) = match call {
                    Call::Map(args, devices) if map => (args, devices),
                    Call::Unmap(args, devices) if !map => (args, devices),
                    _ => panic!("wrong ioctl during mapping retry"),
                };
                assert_eq!(*devices, &[42, 53, 64]);
                assert_eq!(args.handle, 17);
                assert_eq!(args.count, 3);
                if observed.fetch_add(1, Ordering::Relaxed) == 0 {
                    assert_eq!(args.success, 0);
                    args.success = progress;
                    Err(io::Error::from_raw_os_error(4))
                } else {
                    assert_eq!(args.success, progress);
                    args.success = 3;
                    Ok(())
                }
            }));
            let mut completed = 0;
            assert_eq!(
                kfd.transfer(17, &[42, 53, 64], &mut completed, map)
                    .unwrap_err()
                    .raw_os_error(),
                Some(4)
            );
            assert_eq!(completed, progress);
            assert_eq!(
                calls.load(Ordering::Relaxed),
                1,
                "EINTR must reach the owner"
            );
            kfd.transfer(17, &[42, 53, 64], &mut completed, map)
                .unwrap();
            assert_eq!(completed, 3);
        }
    }
}

#[test]
fn malformed_prefixes_and_aperture_counts_are_rejected() {
    for (initial, returned, succeeds) in [(1, 0, false), (0, 4, false), (0, 1, true)] {
        let kfd = endpoint(Arc::new(move |call| {
            let Call::Map(args, _) = call else {
                panic!("expected MAP")
            };
            args.success = returned;
            if succeeds {
                Ok(())
            } else {
                Err(io::Error::from_raw_os_error(5))
            }
        }));
        let mut completed = initial;
        assert_eq!(
            kfd.transfer(17, &[42, 53, 64], &mut completed, true)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(completed, returned);
    }
    let kfd = endpoint(Arc::new(|call| {
        let Call::Apertures(args, entries) = call else {
            panic!("expected apertures")
        };
        assert_eq!(args.pad, 0);
        args.count = if entries.is_empty() { 1 } else { 2 };
        Ok(())
    }));
    assert_eq!(
        kfd.apertures().unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn hardware_loss_poll_uses_persistent_events_and_decodes_memory_loss() {
    for (wait_result, memory_lost, expected) in [
        (uapi::WAIT_TIMEOUT, 0_u32, false),
        (uapi::WAIT_COMPLETE, 0, false),
        (uapi::WAIT_COMPLETE, 1, true),
    ] {
        let kfd = endpoint(Arc::new(move |call| {
            match call {
                Call::CreateEvent(args) => {
                    assert_eq!(args.event_type, uapi::HW_EXCEPTION);
                    assert_eq!(args.auto_reset, 0);
                    assert_eq!(args.page_offset, 0);
                    args.event_id = 19;
                }
                Call::Wait(args, event) => {
                    assert_eq!(args.count, 1);
                    assert_eq!(args.timeout, 0);
                    assert_eq!(event.event_id, 19);
                    let mut bytes = [0; 8];
                    bytes[..4].copy_from_slice(&memory_lost.to_ne_bytes());
                    event.payload[1] = u64::from_ne_bytes(bytes);
                    args.result = wait_result;
                }
                _ => panic!("unexpected event call"),
            }
            Ok(())
        }));
        kfd.create_exception_event(uapi::HW_EXCEPTION, &mut uapi::CreateEvent::default())
            .unwrap();
        assert_eq!(kfd.hardware_memory_lost(19).unwrap(), expected);
    }
}

#[test]
fn memory_exception_poll_decodes_the_complete_uapi_payload() {
    let kfd = endpoint(Arc::new(|call| {
        match call {
            Call::CreateEvent(args) => {
                assert_eq!(args.event_type, uapi::MEMORY_EXCEPTION);
                assert_eq!(args.auto_reset, 0);
                assert_eq!(args.page_offset, 0);
                args.event_id = 20;
            }
            Call::Wait(args, event) => {
                assert_eq!(args.count, 1);
                assert_eq!(args.timeout, 0);
                assert_eq!(event.event_id, 20);
                event.payload[0] = u64::from_ne_bytes([1, 0, 0, 0, 1, 0, 0, 0]);
                event.payload[1] = u64::from_ne_bytes([0, 0, 0, 0, 1, 0, 0, 0]);
                event.payload[2] = 0x1234_5678_9abc_def0;
                event.payload[3] = u64::from_ne_bytes([42, 0, 0, 0, 3, 0, 0, 0]);
                args.result = uapi::WAIT_COMPLETE;
            }
            _ => panic!("unexpected event call"),
        }
        Ok(())
    }));
    kfd.create_exception_event(uapi::MEMORY_EXCEPTION, &mut uapi::CreateEvent::default())
        .unwrap();
    assert_eq!(
        kfd.memory_exception(20).unwrap(),
        Some(MemoryException {
            not_present: 1,
            read_only: 1,
            no_execute: 0,
            imprecise: 1,
            address: 0x1234_5678_9abc_def0,
            gpu_id: 42,
            error_type: 3,
        })
    );
}

#[test]
fn unsupported_exception_type_is_rejected_before_the_ioctl() {
    let kfd = endpoint(Arc::new(|_| panic!("invalid event reached ioctl")));
    assert_eq!(
        kfd.create_exception_event(7, &mut uapi::CreateEvent::default())
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn forked_endpoints_and_reservations_reject_native_work() {
    let mut kfd = endpoint(Arc::new(|_| panic!("child reached ioctl")));
    kfd.process = std::process::id().wrapping_add(1);
    assert_eq!(
        kfd.version().unwrap_err().kind(),
        io::ErrorKind::Unsupported
    );
    assert_eq!(
        kfd.create_queue(&mut uapi::CreateQueue::default())
            .unwrap_err()
            .kind(),
        io::ErrorKind::Unsupported
    );
    assert_eq!(
        kfd.destroy_queue(0).unwrap_err().kind(),
        io::ErrorKind::Unsupported
    );
    assert_eq!(
        kfd.map_doorbells(0, 8192, (0, u64::MAX))
            .err()
            .unwrap()
            .kind(),
        io::ErrorKind::Unsupported
    );
    let page = page_size().unwrap();
    let mut reservation = Reservation::new(page, page * 4, (0, u64::MAX), false).unwrap();
    assert_eq!(reservation.address() % (page * 4), 0);
    let length = reservation.length;
    reservation.process = kfd.process;
    assert_eq!(
        reservation.release().unwrap_err().kind(),
        io::ErrorKind::Unsupported
    );
    assert_eq!(reservation.length, length);
    reservation.process = std::process::id();
    reservation.fail_release_once(12);
    assert!(reservation.release().is_err());
    assert_eq!(reservation.length, length);
    reservation.release().unwrap();
    assert_eq!(reservation.length, 0);
    reservation.release().unwrap();
}

#[test]
fn doorbell_mmap_preserves_kfds_high_type_bits() {
    let kfd = endpoint(Arc::new(|_| panic!("mapping does not issue an ioctl")));
    let error = kfd
        .map_doorbells(3_u64 << 62, 8192, (0, u64::MAX))
        .err()
        .unwrap();
    // /dev/null cannot map a doorbell, but the tagged token must reach mmap.
    // Rejecting it during signed integer conversion loses valid KFD offsets.
    assert!(error.raw_os_error().is_some());
    assert_ne!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn render_mapping_replaces_only_the_owned_usable_extent() {
    use std::os::unix::fs::FileExt;
    let page = page_size().unwrap();
    let path = std::env::temp_dir().join(format!("rocddi-render-map-{}", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    file.set_len(page as u64).unwrap();
    file.write_all_at(&[0x5a], 0).unwrap();
    let mut reservation = Reservation::new(page, page * 4, (0, u64::MAX), false).unwrap();
    let base = reservation.base;
    let length = reservation.length;
    reservation.map_render(&file, 0).unwrap();
    assert_eq!(reservation.base, base);
    assert_eq!(reservation.length, length);
    assert!(reservation.address >= base && reservation.address + page <= base + length);
    // SAFETY: The live writable file mapping covers this byte and no other
    // thread uses the private file. No Rust reference aliases the mapped bytes.
    unsafe {
        let byte = reservation.address as *mut u8;
        assert_eq!(byte.read_volatile(), 0x5a);
        byte.write_volatile(0xa5);
    }
    reservation.release().unwrap();
    let mut bytes = [0];
    file.read_exact_at(&mut bytes, 0).unwrap();
    assert_eq!(bytes, [0xa5]);
}

#[test]
fn inaccessible_view_restores_its_parent_reservation() {
    let page = page_size().unwrap();
    let path = std::env::temp_dir().join(format!(
        "rocddi-inaccessible-render-map-{}",
        std::process::id()
    ));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    file.set_len(page as u64).unwrap();
    let parent = Reservation::new(page * 2, page, (0, u64::MAX), false).unwrap();
    let mut view = Reservation::view(parent.address(), page);

    view.map_render_inaccessible(&file, 0).unwrap();
    assert_eq!(view.length, page);
    assert!(!view.writable);
    view.release().unwrap();
    assert_eq!(view.length, 0);

    drop(view);
    drop(parent);
}

#[test]
fn requested_virtual_address_falls_back_without_replacing_live_memory() {
    let page = page_size().unwrap();
    let occupied = Reservation::new(page, page, (0, u64::MAX), false).unwrap();
    let requested = occupied.address();
    let fallback = Reservation::new_at(page, page, (0, u64::MAX), requested).unwrap();

    assert_ne!(fallback.address(), requested);
    assert_eq!(occupied.address(), requested);
}

#[test]
fn virtual_host_mapping_rejects_unknown_permission_bits() {
    let page = page_size().unwrap();
    let parent = Reservation::new(page, page, (0, u64::MAX), false).unwrap();
    let file = File::open("/dev/zero").unwrap();
    let mut view = Reservation::view(parent.address(), page);

    assert_eq!(
        view.map_dma_buf_with_permissions(&file, 0, 4)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn explicit_endpoint_close_consumes_descriptor_once() {
    let mut kfd = endpoint(Arc::new(|_| panic!("closed endpoint reached ioctl")));
    kfd.close().unwrap();
    assert!(kfd.file.is_none());
    let replacement = File::open("/dev/null").unwrap();
    kfd.close().unwrap();
    assert!(replacement.metadata().is_ok());
    assert_eq!(
        kfd.version().unwrap_err().kind(),
        io::ErrorKind::NotConnected
    );
}

#[test]
fn process_lifetime_never_selects_a_secondary_context() {
    let mut kfd = endpoint(Arc::new(|_| panic!("process policy issued a KFD ioctl")));
    kfd.prepare_context(SessionLifetime::Process).unwrap();
    kfd.prepare_context(SessionLifetime::Process).unwrap();
    kfd.close().unwrap();
}

#[test]
fn concurrent_instance_activation_selects_one_secondary_context() {
    let selections = Arc::new(AtomicUsize::new(0));
    let observed = selections.clone();
    let mut kfd = endpoint(Arc::new(move |call| {
        match call {
            Call::Version(version) => {
                version.major = 1;
                version.minor = 19;
            }
            Call::CreateProcess(args) => {
                assert_eq!((args.flags, args.pad), (0, 0));
                observed.fetch_add(1, Ordering::Relaxed);
            }
            Call::RuntimeEnable(args) => assert!(matches!(args.mode_mask, 0 | 1)),
            _ => panic!("unexpected context lifecycle ioctl"),
        }
        Ok(())
    }));
    std::thread::scope(|scope| {
        let threads = (0..8)
            .map(|_| scope.spawn(|| kfd.prepare_context(SessionLifetime::Session).unwrap()))
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
    });
    assert_eq!(selections.load(Ordering::Relaxed), 1);
    kfd.enable_runtime().unwrap();
    kfd.close().unwrap();
    assert_eq!(selections.load(Ordering::Relaxed), 1);
}

#[test]
fn unsupported_version_leaves_context_selection_retryable() {
    let versions = Arc::new(AtomicUsize::new(0));
    let observed = versions.clone();
    let selections = Arc::new(AtomicUsize::new(0));
    let selected = selections.clone();
    let mut kfd = endpoint(Arc::new(move |call| {
        match call {
            Call::Version(version) => {
                version.major = 1;
                version.minor = if observed.fetch_add(1, Ordering::Relaxed) == 0 {
                    18
                } else {
                    19
                };
            }
            Call::CreateProcess(_) => {
                selected.fetch_add(1, Ordering::Relaxed);
            }
            _ => panic!("unsupported UAPI reached a native mutation"),
        }
        Ok(())
    }));
    assert_eq!(
        kfd.prepare_context(SessionLifetime::Session)
            .unwrap_err()
            .kind(),
        io::ErrorKind::Unsupported
    );
    assert_eq!(selections.load(Ordering::Relaxed), 0);
    kfd.prepare_context(SessionLifetime::Session).unwrap();
    assert_eq!(selections.load(Ordering::Relaxed), 1);
    kfd.close().unwrap();
}

#[test]
fn ambiguous_secondary_selection_requires_descriptor_teardown() {
    let selections = Arc::new(AtomicUsize::new(0));
    let observed = selections.clone();
    let mut kfd = endpoint(Arc::new(move |call| match call {
        Call::Version(version) => {
            version.major = 1;
            version.minor = 19;
            Ok(())
        }
        Call::CreateProcess(_) => {
            observed.fetch_add(1, Ordering::Relaxed);
            Err(io::Error::from_raw_os_error(5))
        }
        _ => panic!("ambiguous selection replayed or touched runtime"),
    }));
    assert_eq!(
        kfd.prepare_context(SessionLifetime::Session)
            .unwrap_err()
            .raw_os_error(),
        Some(5)
    );
    assert_eq!(
        kfd.prepare_context(SessionLifetime::Session)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(
        kfd.enable_runtime().unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    kfd.close().unwrap();
    assert_eq!(selections.load(Ordering::Relaxed), 1);
}

#[test]
fn runtime_enable_is_serialized_once_and_close_disables_once() {
    let modes = Arc::new(Mutex::new(Vec::new()));
    let observed = modes.clone();
    let mut kfd = endpoint(Arc::new(move |call| {
        let Call::RuntimeEnable(args) = call else {
            panic!("unexpected runtime lifecycle call")
        };
        assert_eq!(args.r_debug, 0);
        assert_eq!(args.capabilities_mask, 0);
        observed.lock().unwrap().push(args.mode_mask);
        Ok(())
    }));
    std::thread::scope(|scope| {
        let threads = (0..8)
            .map(|_| scope.spawn(|| kfd.enable_runtime().unwrap()))
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
    });
    assert_eq!(*modes.lock().unwrap(), [1]);
    kfd.close().unwrap();
    kfd.close().unwrap();
    assert_eq!(*modes.lock().unwrap(), [1, 0]);
}

#[test]
fn interrupted_runtime_enable_and_failed_disable_preserve_retry_state() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let mut kfd = endpoint(Arc::new(move |call| {
        let Call::RuntimeEnable(args) = call else {
            panic!("unexpected runtime lifecycle call")
        };
        let call = observed.fetch_add(1, Ordering::Relaxed);
        match (call, args.mode_mask) {
            (0, 1) => Err(io::Error::from_raw_os_error(4)),
            (1, 1) | (3, 0) => Ok(()),
            (2, 0) => Err(io::Error::from_raw_os_error(5)),
            _ => panic!("unexpected runtime retry"),
        }
    }));
    assert_eq!(kfd.enable_runtime().unwrap_err().raw_os_error(), Some(4));
    kfd.enable_runtime().unwrap();
    assert_eq!(kfd.close().unwrap_err().raw_os_error(), Some(5));
    assert!(kfd.file.is_some());
    kfd.close().unwrap();
    assert!(kfd.file.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 4);
}

#[test]
fn foreign_runtime_rejection_never_disables_unowned_state() {
    for errno in [16, 17] {
        let modes = Arc::new(Mutex::new(Vec::new()));
        let observed = modes.clone();
        let mut kfd = endpoint(Arc::new(move |call| {
            let Call::RuntimeEnable(args) = call else {
                panic!("unexpected runtime lifecycle call")
            };
            observed.lock().unwrap().push(args.mode_mask);
            assert_eq!(args.mode_mask, 1, "core disabled a foreign runtime");
            Err(io::Error::from_raw_os_error(errno))
        }));
        assert_eq!(
            kfd.enable_runtime().unwrap_err().raw_os_error(),
            Some(errno)
        );
        kfd.close().unwrap();
        assert!(kfd.file.is_none());
        assert_eq!(*modes.lock().unwrap(), [1]);
    }
}

#[test]
fn ambiguous_runtime_enable_requires_cleanup_without_replaying_enable() {
    let modes = Arc::new(Mutex::new(Vec::new()));
    let observed = modes.clone();
    let mut kfd = endpoint(Arc::new(move |call| {
        let Call::RuntimeEnable(args) = call else {
            panic!("unexpected runtime lifecycle call")
        };
        observed.lock().unwrap().push(args.mode_mask);
        match args.mode_mask {
            1 => Err(io::Error::from_raw_os_error(5)),
            0 => Ok(()),
            _ => panic!("invalid runtime mode"),
        }
    }));
    assert_eq!(kfd.enable_runtime().unwrap_err().raw_os_error(), Some(5));
    assert_eq!(
        kfd.enable_runtime().unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    kfd.close().unwrap();
    assert!(kfd.file.is_none());
    assert_eq!(*modes.lock().unwrap(), [1, 0]);
}
