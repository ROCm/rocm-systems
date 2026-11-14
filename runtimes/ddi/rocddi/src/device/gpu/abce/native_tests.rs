// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU-free owner, publication and retention tests. No packet execution is
//! emulated; tests explicitly advance fences and native read frontiers.

#![allow(clippy::unwrap_used, clippy::panic)]

use super::*;
use crate::device::gpu::abce::{abce_copy_op_initialize, abi::ABCE_ENDPOINT_KIND_DEVICE};
use crate::driver::{DriverInstance, test_driver::TestDriver};
use crate::memory::storage::Shared;
use crate::session::{DriverContextLifetime, Session};
use crate::topology::{
    EndpointKind, GpuCopyTransport, GpuInfo, GpuQueueCapabilities, GpuTargetProfile,
    GpuUserQueueCapabilities,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture(ids: &[u8], multi: bool) -> (Session, Vec<Arc<AtomicUsize>>) {
    let mut destroys = Vec::new();
    let drivers = ids
        .iter()
        .map(|&id| {
            let driver = TestDriver::new(
                [id; 16],
                EndpointKind::Gpu {
                    info: GpuInfo {
                        gfx_major: 9,
                        target_profile: GpuTargetProfile::Gfx9,
                        queues: GpuQueueCapabilities {
                            copy: Some(GpuCopyTransport::User),
                            sdma: Some(GpuUserQueueCapabilities::default()),
                            sdma_engine_count: 2,
                            sdma_multi_engine_copy: multi,
                            ..GpuQueueCapabilities::default()
                        },
                        ..GpuInfo::default()
                    },
                },
                DriverContextLifetime::Session,
            )
            .with_sdma_memory();
            destroys.push(driver.queue_destroys.clone());
            DriverInstance::Test(Shared::new(driver).unwrap())
        })
        .collect();
    (
        Session::from_drivers(drivers, DriverContextLifetime::Session).unwrap(),
        destroys,
    )
}

fn operations() -> [abce_copy_op_t; 2] {
    std::array::from_fn(|i| {
        let mut op = unsafe { std::mem::zeroed::<abce_copy_op_t>() };
        unsafe {
            abce_copy_op_initialize(&raw mut op);
        }
        op.src = (0x1000 + i * 0x100) as *const _;
        op.dst = (0x2000 + i * 0x100) as *mut _;
        op.size = 64;
        op.src_end.kind = ABCE_ENDPOINT_KIND_DEVICE;
        op.dst_end.kind = ABCE_ENDPOINT_KIND_DEVICE;
        op
    })
}

#[test]
fn engines_remain_inside_the_activated_gpu_and_session() {
    let (session, destroys) = fixture(&[1, 2], true);
    let a = session
        .activate(&session.open_endpoint([1; 16]).unwrap())
        .unwrap();
    let b = session
        .activate(&session.open_endpoint([2; 16]).unwrap())
        .unwrap();
    let a_service = a.gpu().unwrap().copy_engine().unwrap();
    let b_service = b.gpu().unwrap().copy_engine().unwrap();
    let a_engines = a_service.engines().unwrap();
    let b_engines = b_service.engines().unwrap();
    assert_eq!(a_engines[0].selector(), b_engines[0].selector());
    let rejected =
        unsafe { a_service.prepare_raw(&b_engines[..1], &operations(), CopyOptions::default()) };
    assert_eq!(rejected.err().unwrap().kind(), ErrorKind::InvalidArgument);
    assert!(destroys.iter().all(|n| n.load(Ordering::Acquire) == 0));
    let (other, _) = fixture(&[1], true);
    let foreign = other
        .activate(&other.open_endpoint([1; 16]).unwrap())
        .unwrap();
    let foreign_engines = foreign
        .gpu()
        .unwrap()
        .copy_engine()
        .unwrap()
        .engines()
        .unwrap();
    assert!(
        unsafe {
            a_service.prepare_raw(&foreign_engines[..1], &operations(), CopyOptions::default())
        }
        .is_err()
    );
    assert!(std::ptr::eq(
        a.gpu().unwrap().resources,
        a.clone().gpu().unwrap().resources
    ));
}

#[test]
fn direct_fanout_publishes_all_rings_before_any_retirement_wait() {
    let (session, _) = fixture(&[1], true);
    let device = session
        .activate(&session.open_endpoint([1; 16]).unwrap())
        .unwrap();
    let service = device.gpu().unwrap().copy_engine().unwrap();
    let engines = service.engines().unwrap();
    let prepared = unsafe {
        service.prepare_raw(
            &engines,
            &operations(),
            CopyOptions {
                placement: Placement::FanOut,
                ..CopyOptions::default()
            },
        )
    }
    .unwrap();
    assert_eq!(prepared.work.plan.num_frames, 2);
    assert!(
        prepared
            .work
            .contexts
            .iter()
            .all(|c| c.ring.control.max_write_index == 0)
    );
    let mut submission = prepared.submit().unwrap();
    assert_eq!(submission.poll().unwrap(), CopyProgress::Pending);
    for context in &submission.work.contexts {
        let NativeQueue::Direct(queue) = &*context.queue else {
            panic!("expected direct queue")
        };
        let info = queue.info();
        assert!(
            unsafe {
                AtomicU64::from_ptr(info.doorbell_host_address as *mut u64).load(Ordering::Acquire)
            } > 0
        );
    }
    // One retired frame is insufficient to release shared coordination.
    retire_context(&mut submission.work.contexts[0]);
    assert_eq!(submission.poll().unwrap(), CopyProgress::Pending);
    retire_context(&mut submission.work.contexts[1]);
    assert_eq!(submission.poll().unwrap(), CopyProgress::Retired);
    drop(submission);
    assert_eq!(
        service
            .gpu
            .resources
            .abce_pool
            .idle
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.is_some())
            .count(),
        2
    );
}

fn retire_context(context: &mut Context) {
    let NativeQueue::Direct(queue) = &*context.queue else {
        panic!("expected direct queue")
    };
    let info = queue.info();
    unsafe {
        AtomicU32::from_ptr(context.command_host as *mut u32).store(1, Ordering::Release);
        AtomicU64::from_ptr(info.read_index_host_address as *mut u64)
            .store(context.pending.unwrap(), Ordering::Release);
    }
}

#[test]
fn timeout_and_drop_do_not_return_accepted_queues_to_the_pool() {
    let (session, destroys) = fixture(&[1], true);
    let device = session
        .activate(&session.open_endpoint([1; 16]).unwrap())
        .unwrap();
    let service = device.gpu().unwrap().copy_engine().unwrap();
    let engines = service.engines().unwrap();
    let prepared =
        unsafe { service.prepare_raw(&engines[..1], &operations(), CopyOptions::default()) }
            .unwrap();
    let mut submission = prepared.submit().unwrap();
    assert_eq!(
        submission.wait(Duration::ZERO).unwrap(),
        CopyProgress::Pending
    );
    drop(submission);
    assert!(
        service
            .gpu
            .resources
            .abce_pool
            .idle
            .lock()
            .unwrap()
            .iter()
            .all(Option::is_none)
    );
    assert_eq!(destroys[0].load(Ordering::Acquire), 0);
}

#[test]
fn unqualified_multi_engine_rejection_precedes_resource_creation() {
    let (session, destroys) = fixture(&[1], false);
    let device = session
        .activate(&session.open_endpoint([1; 16]).unwrap())
        .unwrap();
    let service = device.gpu().unwrap().copy_engine().unwrap();
    let engines = service.engines().unwrap();
    let rejected = unsafe { service.prepare_raw(&engines, &operations(), CopyOptions::default()) };
    assert_eq!(rejected.err().unwrap().kind(), ErrorKind::Unsupported);
    assert_eq!(destroys[0].load(Ordering::Acquire), 0);
}

fn kernel_fixture() -> (
    Session,
    Arc<crate::driver::test_driver::kernel::KernelControls>,
) {
    let driver = TestDriver::new(
        [3; 16],
        EndpointKind::Gpu {
            info: GpuInfo {
                gfx_major: 12,
                gfx_stepping: 1,
                target_profile: GpuTargetProfile::Gfx1201,
                queues: GpuQueueCapabilities {
                    copy: Some(GpuCopyTransport::Kernel),
                    kernel_sdma: true,
                    sdma_system_cache_control: true,
                    ..GpuQueueCapabilities::default()
                },
                ..GpuInfo::default()
            },
        },
        DriverContextLifetime::Session,
    )
    .with_kernel_sdma();
    let controls = driver.kernel_controls.clone().unwrap();
    let session = Session::from_drivers(
        vec![DriverInstance::Test(Shared::new(driver).unwrap())],
        DriverContextLifetime::Session,
    )
    .unwrap();
    (session, controls)
}

#[test]
fn kernel_submission_is_executable_and_failure_stays_visible_after_retirement() {
    let (session, controls) = kernel_fixture();
    let device = session
        .activate(&session.open_endpoint([3; 16]).unwrap())
        .unwrap();
    let service = device.gpu().unwrap().copy_engine().unwrap();
    let engines = service.engines().unwrap();
    let prepared =
        unsafe { service.prepare_raw(&engines, &operations(), CopyOptions::default()) }.unwrap();
    // The test driver rejects a command unless its allocation has EXECUTE.
    let mut submission = prepared.submit().unwrap();
    assert_eq!(submission.poll().unwrap(), CopyProgress::Pending);
    controls.status.lock().unwrap().terminal = Some(ErrorKind::Driver);
    assert_eq!(submission.poll().unwrap_err().kind(), ErrorKind::Driver);
    assert!(!submission.is_retired());
    controls.status.lock().unwrap().retired_submission = controls.submitted.load(Ordering::Acquire);
    assert_eq!(submission.poll().unwrap_err().kind(), ErrorKind::Driver);
    assert!(submission.is_retired());
    assert_eq!(
        submission.wait(Duration::ZERO).unwrap_err().kind(),
        ErrorKind::Driver
    );
}

#[test]
fn rejected_kernel_submission_has_no_accepted_storage_obligation() {
    let (session, controls) = kernel_fixture();
    let device = session
        .activate(&session.open_endpoint([3; 16]).unwrap())
        .unwrap();
    let service = device.gpu().unwrap().copy_engine().unwrap();
    let engines = service.engines().unwrap();
    let prepared =
        unsafe { service.prepare_raw(&engines, &operations(), CopyOptions::default()) }.unwrap();
    controls.reject.store(true, Ordering::Release);
    assert_eq!(prepared.submit().err().unwrap().kind(), ErrorKind::Driver);
    assert_eq!(controls.submitted.load(Ordering::Acquire), 0);
    assert_eq!(
        service
            .gpu
            .resources
            .abce_pool
            .idle
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.is_some())
            .count(),
        1
    );
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn fork_child_rejects_before_mapped_writes_or_inherited_pool_lock() {
    // Forking concurrently with another test can inherit its descriptors and
    // delay observable EOF. Run the actual fork in an isolated test process.
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "device::gpu::abce::native::tests::fork_child_pool_probe",
            "--ignored",
            "--test-threads=1",
        ])
        .status()
        .unwrap();
    assert!(status.success());
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
#[ignore = "invoked in an isolated process by the fork regression"]
fn fork_child_pool_probe() {
    let (session, _) = fixture(&[1], false);
    let device = session
        .activate(&session.open_endpoint([1; 16]).unwrap())
        .unwrap();
    let service = device.gpu().unwrap().copy_engine().unwrap();
    let engines = service.engines().unwrap();
    let mut prepared =
        unsafe { service.prepare_raw(&engines[..1], &operations(), CopyOptions::default()) }
            .unwrap();
    let dropped =
        unsafe { service.prepare_raw(&engines[..1], &operations(), CopyOptions::default()) }
            .unwrap();
    // A shared sentinel makes a forbidden child store visible to the parent.
    let page = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            4096,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(page, libc::MAP_FAILED);
    let sentinel = unsafe { AtomicU64::from_ptr(page.cast()) };
    sentinel.store(0x1234, Ordering::Release);
    prepared.work.sync_host = page as usize;
    let held = service.gpu.resources.abce_pool.idle.lock().unwrap();
    // The child only checks/drops inherited ABCE objects, then _exit avoids
    // unrelated test-harness destructors. Alarm bounds an inherited-lock bug.
    let child = unsafe { libc::syscall(libc::SYS_fork) };
    assert!(child >= 0);
    if child == 0 {
        unsafe {
            libc::alarm(3);
        }
        let rejects = service.engines().is_err() && prepared.submit().is_err();
        drop(dropped);
        unsafe {
            libc::_exit(i32::from(!rejects));
        }
    }
    let pid = i32::try_from(child).unwrap();
    let mut status = 0;
    assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
    assert_eq!(status, 0);
    assert_eq!(sentinel.load(Ordering::Acquire), 0x1234);
    drop(held);
    drop(prepared);
    drop(dropped);
    assert_eq!(unsafe { libc::munmap(page, 4096) }, 0);
}
