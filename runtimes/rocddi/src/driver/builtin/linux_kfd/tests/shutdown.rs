//! Teardown must retain the session while abandoned native owners still depend
//! on its caller allocator. The raw resource owners below model native-failure
//! retention, with test-only recovery so every callback allocation is reclaimed.
#![allow(unsafe_code, clippy::unwrap_used, clippy::panic)]

use super::super::LinuxKfdDriver;
use super::*;
use crate::driver::ProviderDriver;
use crate::test_support::allocator::State;
use std::sync::Arc;

fn controller(allocator: Allocator) -> LinuxKfdDriver {
    let mut controller = LinuxKfdDriver::new(allocator);
    let mut endpoint = sys::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(|call| match call {
            sys::Call::RuntimeEnable(args) if args.mode_mask == 0 => Ok(()),
            sys::Call::DestroyEvent(_) => Ok(()),
            _ => panic!("unexpected native call during shutdown"),
        }),
    );
    endpoint.mark_runtime_enabled_for_test();
    let kfd = Shared::new(endpoint, allocator).unwrap();
    let loss = Shared::new(
        LossEvent {
            kfd: kfd.clone(),
            hardware_event_id: AtomicU32::new(19),
            memory_event_id: AtomicU32::new(20),
            hardware_destroy_uncertain: AtomicBool::new(false),
            memory_destroy_uncertain: AtomicBool::new(false),
            memory_event_claimed: Mutex::new(false),
            lost: AtomicBool::new(false),
        },
        allocator,
    )
    .unwrap();
    let vm = Shared::new(
        DeviceVm {
            loss: loss.clone(),
            render: Some(File::open("/dev/null").unwrap()),
            system_dma_buf_import: false,
            gpu_id: 42,
            render_minor: 128,
            identity: [0; 16],
            unique_id: Some(123),
            base: 0x10000,
            limit: isize::MAX as u64,
            lds_base: 0x1000_0000_0000,
            scratch_base: 0x2000_0000_0000,
            scratch: Mutex::new(ScratchPool::new(
                sysfs::NativeQueueProperties {
                    gfx_target: 120_001,
                    xcc_count: 1,
                    ..sysfs::NativeQueueProperties::default()
                },
                allocator,
            )),
            vmem: Mutex::new(super::super::vmem::VmState::new(allocator)),
            version: uapi::Version {
                major: 1,
                minor: 23,
            },
            doorbells: super::super::queue::Doorbells::default(),
        },
        allocator,
    )
    .unwrap();
    assert!(controller.kfd.set(kfd).is_ok());
    let bindings = controller.bindings.bindings.get_mut().unwrap();
    bindings.loss = Some(loss);
    bindings.devices.try_push(vm).unwrap();
    controller
}

#[test]
fn forgotten_callback_resource_keeps_its_vm_and_instance_alive() {
    let callbacks = State::default();
    // SAFETY: Callback state remains stationary until all owners are reclaimed.
    let allocator = unsafe { callbacks.allocator() };
    let mut driver = controller(allocator);
    let vm = driver.bindings.bindings.get_mut().unwrap().devices[0].clone();
    // This callback-owned resource has the same retained VM dependency as a
    // forgotten allocation or queue backing. No normal destructor owns it now.
    let abandoned = Owned::new(vm, allocator).unwrap().into_raw();
    let allocations = callbacks.allocations.load(Ordering::Relaxed);
    let frees = callbacks.frees.load(Ordering::Relaxed);
    for _ in 0..3 {
        assert_eq!(
            driver.shutdown().unwrap_err().kind(),
            ErrorKind::DriverContract
        );
        assert!(driver.closing);
        let bindings = driver.bindings.bindings.get_mut().unwrap();
        assert_eq!(bindings.devices.len(), 1);
        assert_eq!(Shared::strong_count(&bindings.devices[0]), 2);
        assert!(bindings.devices[0].render.is_some());
        assert_eq!(
            bindings
                .loss
                .as_ref()
                .unwrap()
                .hardware_event_id
                .load(Ordering::Relaxed),
            19
        );
        assert_eq!(
            bindings
                .loss
                .as_ref()
                .unwrap()
                .memory_event_id
                .load(Ordering::Relaxed),
            20
        );
        assert!(driver.kfd.get().is_some());
        assert_eq!(callbacks.allocations.load(Ordering::Relaxed), allocations);
        assert_eq!(callbacks.frees.load(Ordering::Relaxed), frees);
    }
    // SAFETY: Recover the sole raw owner created above, only after proving that
    // repeated session destruction cannot abandon or free its dependencies.
    unsafe {
        drop(Owned::<Shared<DeviceVm>>::from_raw(abandoned));
    }
    driver.shutdown().unwrap();
    assert!(driver.kfd.get().is_none());
    assert!(
        driver
            .bindings
            .bindings
            .get_mut()
            .unwrap()
            .devices
            .is_empty()
    );
    drop(driver);
    assert_eq!(
        callbacks.allocations.load(Ordering::Relaxed),
        callbacks.frees.load(Ordering::Relaxed)
    );
}

#[test]
fn retained_loss_owner_preserves_the_remaining_shutdown_records() {
    let callbacks = State::default();
    // SAFETY: Callback state outlives the controller and recovered resource.
    let allocator = unsafe { callbacks.allocator() };
    let mut driver = controller(allocator);
    let loss = driver
        .bindings
        .bindings
        .get_mut()
        .unwrap()
        .loss
        .as_ref()
        .unwrap()
        .clone();
    let abandoned = Owned::new(loss, allocator).unwrap().into_raw();
    assert_eq!(
        driver.shutdown().unwrap_err().kind(),
        ErrorKind::DriverContract
    );
    let bindings = driver.bindings.bindings.get_mut().unwrap();
    assert!(bindings.devices.is_empty());
    assert_eq!(Shared::strong_count(bindings.loss.as_ref().unwrap()), 2);
    assert_eq!(
        bindings
            .loss
            .as_ref()
            .unwrap()
            .hardware_event_id
            .load(Ordering::Relaxed),
        19
    );
    assert_eq!(
        bindings
            .loss
            .as_ref()
            .unwrap()
            .memory_event_id
            .load(Ordering::Relaxed),
        20
    );
    assert!(driver.kfd.get().is_some());
    let frees = callbacks.frees.load(Ordering::Relaxed);
    assert_eq!(
        driver.shutdown().unwrap_err().kind(),
        ErrorKind::DriverContract
    );
    assert_eq!(callbacks.frees.load(Ordering::Relaxed), frees);
    // SAFETY: This pointer still denotes the sole unconsumed raw resource owner.
    unsafe {
        drop(Owned::<Shared<LossEvent>>::from_raw(abandoned));
    }
    driver.shutdown().unwrap();
    drop(driver);
    assert_eq!(
        callbacks.allocations.load(Ordering::Relaxed),
        callbacks.frees.load(Ordering::Relaxed)
    );
}

#[test]
fn retained_kfd_owner_prevents_success_after_other_cleanup_finishes() {
    let callbacks = State::default();
    // SAFETY: Callback state outlives the controller and recovered resource.
    let allocator = unsafe { callbacks.allocator() };
    let mut driver = controller(allocator);
    let kfd = driver.kfd.get().unwrap().clone();
    let abandoned = Owned::new(kfd, allocator).unwrap().into_raw();
    assert_eq!(
        driver.shutdown().unwrap_err().kind(),
        ErrorKind::DriverContract
    );
    assert!(driver.bindings.bindings.get_mut().unwrap().loss.is_none());
    assert_eq!(Shared::strong_count(driver.kfd.get().unwrap()), 2);
    let frees = callbacks.frees.load(Ordering::Relaxed);
    assert_eq!(
        driver.shutdown().unwrap_err().kind(),
        ErrorKind::DriverContract
    );
    assert_eq!(callbacks.frees.load(Ordering::Relaxed), frees);
    assert!(driver.kfd.get().is_some());
    // SAFETY: This pointer still denotes the sole unconsumed raw resource owner.
    unsafe {
        drop(Owned::<Shared<sys::Kfd>>::from_raw(abandoned));
    }
    driver.shutdown().unwrap();
    assert!(driver.kfd.get().is_none());
    drop(driver);
    assert_eq!(
        callbacks.allocations.load(Ordering::Relaxed),
        callbacks.frees.load(Ordering::Relaxed)
    );
}
