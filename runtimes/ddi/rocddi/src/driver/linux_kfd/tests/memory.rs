// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native ownership tests use real anonymous reservations and scripted KFD
//! replies. Scripts model errno and output progress independently, including
//! failures after the kernel has finished every per-device mapping operation.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::driver::AllocationOperations;
use crate::driver::linux_kfd::target::KfdTargetProfile;
use crate::memory::host::HostAllocation;
use crate::memory::{HostMappingPolicy, MemoryKind};
use std::os::fd::{AsRawFd, IntoRawFd};
use std::os::unix::fs::FileExt;
use std::sync::Arc;
fn shared<T>(value: T) -> Shared<T> {
    Shared::new(value).unwrap()
}
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize};

enum Reply {
    Xnack(i32, Option<i32>),
    SetScratch(Option<i32>),
    Allocate(u64, Option<i32>),
    DmaBufInfo(u64, u32, u32, &'static [u8], Option<i32>),
    ImportDmaBuf(u64, u32, Option<i32>),
    ExportDmaBuf(i32, Option<i32>),
    IpcImport(u64, u64, u32, Option<i32>),
    IpcExport([u32; 4], Option<i32>),
    Ais(uapi::AisInput, uapi::AisOutput, Option<i32>),
    Map(u32, u32, Option<i32>),
    Unmap(u32, u32, Option<i32>),
    MapDevices(&'static [u32], u32, u32, Option<i32>),
    UnmapDevices(&'static [u32], u32, u32, Option<i32>),
    Free(Option<i32>),
    Svm(u64, u64, &'static [(u32, u32)], Option<i32>),
    Userptr(u64, u64, u64, Option<i32>),
}

enum FixtureMemoryKind {
    Owned(MemoryKind),
    RegisteredHost {
        address: usize,
        policy: HostMappingPolicy,
    },
}

impl From<MemoryKind> for FixtureMemoryKind {
    fn from(kind: MemoryKind) -> Self {
        Self::Owned(kind)
    }
}

struct Fixture {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    lost: Arc<AtomicBool>,
    hardware: Arc<AtomicU32>,
    hardware_reset_errno: Arc<AtomicU32>,
    fault_gpu_id: Arc<AtomicU32>,
    memory_wait_errno: Arc<AtomicU32>,
    native_waits: Arc<AtomicUsize>,
    scratch_base: Arc<AtomicU64>,
    vm: Shared<DeviceVm>,
}

fn node() -> sysfs::KfdNode {
    sysfs::KfdNode {
        debug: sysfs::KfdDebugProperties::default(),
        host_atomics: false,
        node: 1,
        gpu_id: 42,
        render_minor: Some(128),
        unique_id: Some(123),
        identity: [0; 16],
        queues: sysfs::KfdQueueProperties::default(),
        local_memory_bytes: 1 << 30,
        public_memory_bytes: 0,
    }
}

fn desc() -> AllocationDesc {
    AllocationDesc {
        size: 65536,
        alignment: 65536,
    }
}

fn scratch_pool() -> Mutex<ScratchPool> {
    Mutex::new(ScratchPool::new(sysfs::KfdQueueProperties {
        gfx_target: 120_001,
        target_profile: crate::topology::GpuTargetProfile::Gfx1201,
        xcc_count: 1,
        ..sysfs::KfdQueueProperties::default()
    }))
}

impl Fixture {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self::with_flags(replies, uapi::VRAM | uapi::WRITABLE | uapi::NO_SUBSTITUTE)
    }

    fn with_flags(replies: impl IntoIterator<Item = Reply>, expected_flags: u32) -> Self {
        Self::with_devices(
            replies,
            expected_flags,
            &[42],
            File::open("/dev/null").unwrap(),
        )
    }

    fn with_owned_userptr(replies: impl IntoIterator<Item = Reply>, expected_flags: u32) -> Self {
        Self::with_devices_and_userptr_source(
            replies,
            expected_flags,
            &[42],
            File::open("/dev/null").unwrap(),
            true,
            0x12345,
        )
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the scripted native-call matcher keeps each ownership test explicit"
    )]
    fn with_devices(
        replies: impl IntoIterator<Item = Reply>,
        expected_flags: u32,
        expected_devices: &[u32],
        render: File,
    ) -> Self {
        Self::with_devices_and_userptr_source(
            replies,
            expected_flags,
            expected_devices,
            render,
            false,
            0x12345,
        )
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the scripted native-call matcher keeps each ownership test explicit"
    )]
    fn with_devices_and_userptr_source(
        replies: impl IntoIterator<Item = Reply>,
        expected_flags: u32,
        expected_devices: &[u32],
        render: File,
        owned_userptr: bool,
        expected_userptr_address: usize,
    ) -> Self {
        let replies = Arc::new(Mutex::new(replies.into_iter().collect::<VecDeque<_>>()));
        let pending = replies.clone();
        let expected_devices = expected_devices.to_vec();
        let lost = Arc::new(AtomicBool::new(false));
        let loss_signal = lost.clone();
        let hardware = Arc::new(AtomicU32::new(0));
        let hardware_signal = hardware.clone();
        let hardware_reset_errno = Arc::new(AtomicU32::new(0));
        let injected_reset_error = hardware_reset_errno.clone();
        let fault_gpu_id = Arc::new(AtomicU32::new(42));
        let reported_gpu_id = fault_gpu_id.clone();
        let memory_wait_errno = Arc::new(AtomicU32::new(0));
        let injected_memory_error = memory_wait_errno.clone();
        let native_waits = Arc::new(AtomicUsize::new(0));
        let observed_native_waits = native_waits.clone();
        let scratch_base = Arc::new(AtomicU64::new(0));
        let observed_scratch_base = scratch_base.clone();
        let kfd = shared(connection::Kfd::with_hook(
            File::open("/dev/null").unwrap(),
            Arc::new(move |call| {
                match call {
                    connection::Call::Wait(args, event) if event.event_id == 19 => {
                        let state = hardware_signal.load(Ordering::Acquire);
                        if state == 0 {
                            args.result = uapi::WAIT_TIMEOUT;
                        } else {
                            event.payload[0] =
                                u64::from(reported_gpu_id.load(Ordering::Acquire)) | (1_u64 << 32);
                            event.payload[1] = u64::from(state == 2) | (1_u64 << 32);
                            args.result = uapi::WAIT_COMPLETE;
                        }
                        return Ok(());
                    }
                    connection::Call::Wait(args, event) if event.event_id == 20 => {
                        let errno = injected_memory_error.load(Ordering::Acquire);
                        if errno != 0 {
                            return Err(io::Error::from_raw_os_error(
                                i32::try_from(errno).unwrap(),
                            ));
                        }
                        if loss_signal.load(Ordering::Relaxed) {
                            event.payload[0] = u64::from_ne_bytes([0, 0, 0, 0, 1, 0, 0, 0]);
                            event.payload[2] = 0x1234_5000;
                            event.payload[3] = u64::from(reported_gpu_id.load(Ordering::Acquire));
                            args.result = uapi::WAIT_COMPLETE;
                        } else {
                            args.result = uapi::WAIT_TIMEOUT;
                        }
                        return Ok(());
                    }
                    connection::Call::WaitExceptions(args, events) => {
                        assert_eq!(args.count, 2);
                        assert_eq!([events[0].event_id, events[1].event_id], [19, 20]);
                        observed_native_waits.fetch_add(1, Ordering::Relaxed);
                        args.result = uapi::WAIT_TIMEOUT;
                        return Ok(());
                    }
                    connection::Call::ResetEvent(args) if args.event_id == 19 => {
                        let errno = injected_reset_error.swap(0, Ordering::AcqRel);
                        if errno != 0 {
                            return Err(io::Error::from_raw_os_error(
                                i32::try_from(errno).unwrap(),
                            ));
                        }
                        hardware_signal.store(0, Ordering::Release);
                        return Ok(());
                    }
                    connection::Call::ResetEvent(args) if args.event_id == 20 => {
                        loss_signal.store(false, Ordering::Release);
                        return Ok(());
                    }
                    connection::Call::DestroyEvent(_) => return Ok(()),
                    _ => {}
                }
                let reply = pending
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("unexpected native call");
                let errno = match (call, reply) {
                    (
                        connection::Call::Allocate(args),
                        Reply::Userptr(address, size, handle, errno),
                    ) => {
                        assert_eq!(args.mmap_offset, address);
                        assert_eq!(args.size, size);
                        assert_eq!(
                            args.flags,
                            uapi::USERPTR | uapi::COHERENT | uapi::NO_SUBSTITUTE | uapi::WRITABLE
                        );
                        assert_eq!(args.gpu_id, 42);
                        args.handle = handle;
                        errno
                    }
                    (
                        connection::Call::Svm(args, attributes),
                        Reply::Svm(start, size, expected, errno),
                    ) => {
                        assert_eq!(args.start_address, start);
                        assert_eq!(args.size, size);
                        assert_eq!(args.operation, uapi::SVM_OP_SET_ATTR);
                        let actual: Vec<_> = attributes
                            .iter()
                            .map(|attribute| (attribute.attribute_type, attribute.value))
                            .collect();
                        assert_eq!(actual, expected);
                        errno
                    }
                    (connection::Call::SetScratchBackingVa(args), Reply::SetScratch(errno)) => {
                        assert_eq!(args.gpu_id, 42);
                        assert_eq!(args.pad, 0);
                        assert_ne!(args.va_address, 0);
                        observed_scratch_base.store(args.va_address << 16, Ordering::Relaxed);
                        errno
                    }
                    (connection::Call::Allocate(args), Reply::Allocate(handle, errno)) => {
                        assert_eq!(args.va % 65536, 0);
                        let page = os_memory::page_size().unwrap() as u64;
                        let expected_size = if expected_flags & uapi::USERPTR != 0 && !owned_userptr
                        {
                            (65536 + (expected_userptr_address as u64 & (page - 1))).div_ceil(page)
                                * page
                        } else {
                            65536
                        };
                        assert_eq!(args.size, expected_size);
                        assert_eq!(args.gpu_id, 42);
                        assert_eq!(args.flags, expected_flags);
                        let expected_mmap_offset = if expected_flags & uapi::USERPTR == 0 {
                            0
                        } else if owned_userptr {
                            args.va
                        } else {
                            expected_userptr_address as u64 & !(page - 1)
                        };
                        assert_eq!(args.mmap_offset, expected_mmap_offset);
                        args.handle = handle;
                        errno
                    }
                    (
                        connection::Call::DmaBufInfo(args, metadata),
                        Reply::DmaBufInfo(size, gpu_id, flags, expected_metadata, errno),
                    ) => {
                        assert!(i32::try_from(args.descriptor).is_ok());
                        if metadata.is_empty() && !expected_metadata.is_empty() {
                            args.metadata_size = u32::try_from(expected_metadata.len()).unwrap();
                            pending.lock().unwrap().push_front(Reply::DmaBufInfo(
                                size,
                                gpu_id,
                                flags,
                                expected_metadata,
                                errno,
                            ));
                            return Err(io::Error::from_raw_os_error(errno::EINVAL));
                        }
                        assert_eq!(metadata.len(), expected_metadata.len());
                        metadata.copy_from_slice(expected_metadata);
                        args.size = size;
                        args.gpu_id = gpu_id;
                        args.flags = flags;
                        args.metadata_size = u32::try_from(expected_metadata.len()).unwrap();
                        errno
                    }
                    (
                        connection::Call::ImportDmaBuf(args),
                        Reply::ImportDmaBuf(handle, gpu_id, errno),
                    ) => {
                        assert_eq!(args.va % 65536, 0);
                        assert_eq!(args.gpu_id, gpu_id);
                        assert!(i32::try_from(args.descriptor).is_ok());
                        args.handle = handle;
                        errno
                    }
                    (connection::Call::ExportDmaBuf(args), Reply::ExportDmaBuf(fd, errno)) => {
                        assert_eq!(args.handle, 17);
                        args.descriptor = u32::try_from(fd).unwrap();
                        errno
                    }
                    (
                        connection::Call::IpcImportHandle(args),
                        Reply::IpcImport(handle, offset, flags, errno),
                    ) => {
                        assert_eq!(args.va_addr % 4096, 0);
                        assert_eq!(args.share_handle, [1, 2, 3, 4]);
                        assert_eq!(args.gpu_id, 42);
                        args.handle = handle;
                        args.mmap_offset = offset;
                        args.flags = flags;
                        errno
                    }
                    (connection::Call::IpcExportHandle(args), Reply::IpcExport(handle, errno)) => {
                        assert_eq!(args.handle, 17);
                        assert_eq!(args.gpu_id, 42);
                        assert_eq!(args.flags, expected_flags);
                        args.share_handle = handle;
                        errno
                    }
                    (connection::Call::Ais(args), Reply::Ais(expected, output, errno)) => {
                        assert_eq!(args.requested_input(), expected);
                        args.set_completed_output(output);
                        errno
                    }
                    (connection::Call::Map(args, devices), Reply::Map(before, after, errno))
                    | (
                        connection::Call::Unmap(args, devices),
                        Reply::Unmap(before, after, errno),
                    ) => {
                        assert_eq!(*devices, expected_devices.as_slice());
                        assert_eq!(args.handle, 17);
                        assert_eq!(args.success, before);
                        args.success = after;
                        errno
                    }
                    (
                        connection::Call::Map(args, devices),
                        Reply::MapDevices(expected, before, after, errno),
                    )
                    | (
                        connection::Call::Unmap(args, devices),
                        Reply::UnmapDevices(expected, before, after, errno),
                    ) => {
                        assert_eq!(devices, &expected);
                        assert_eq!(args.handle, 17);
                        assert_eq!(args.success, before);
                        args.success = after;
                        errno
                    }
                    (connection::Call::XnackMode(args), Reply::Xnack(enabled, errno)) => {
                        assert_eq!(args.xnack_enabled, -1);
                        args.xnack_enabled = enabled;
                        errno
                    }
                    (connection::Call::Free(args), Reply::Free(errno)) => {
                        assert_eq!(args.handle, 17);
                        errno
                    }
                    _ => panic!("cleanup issued the wrong native operation"),
                };
                errno.map_or(Ok(()), |errno| Err(io::Error::from_raw_os_error(errno)))
            }),
        ));
        let vm = Shared::new(DeviceVm {
            loss: Shared::new(LossEvent {
                kfd,
                hardware_event_id: AtomicU32::new(19),
                memory_event_id: AtomicU32::new(20),
                hardware_destroy_uncertain: AtomicBool::new(false),
                memory_destroy_uncertain: AtomicBool::new(false),
                observed: Mutex::new(ObservedEvents::new().unwrap()),
                host_aliases: Mutex::new(Vec::new()),
                svm_registrations: Mutex::new(registered_svm::Registry::default()),
                lost: AtomicBool::new(false),
            })
            .unwrap(),
            render: Some(render),
            system_dma_buf_import: false,
            gpu_id: 42,
            render_minor: 128,
            identity: [0; 16],
            unique_id: Some(123),
            base: 0x10000,
            limit: isize::MAX as u64,
            lds_base: 0x1000_0000_0000,
            scratch_base: 0x2000_0000_0000,
            sdma_next_engine: AtomicU32::new(0),
            scratch: scratch_pool(),
            vmem: Mutex::new(super::super::vmem::VmState::new()),
            version: uapi::Version {
                major: 1,
                minor: 23,
            },
            doorbells: super::super::queue::Doorbells::default(),
        })
        .unwrap();
        Self {
            replies,
            lost,
            hardware,
            hardware_reset_errno,
            fault_gpu_id,
            memory_wait_errno,
            native_waits,
            scratch_base,
            vm,
        }
    }

    fn peer(&self, gpu_id: u32, base: u64, limit: u64) -> Shared<DeviceVm> {
        shared(DeviceVm {
            loss: self.vm.loss.clone(),
            render: Some(File::open("/dev/null").unwrap()),
            system_dma_buf_import: false,
            gpu_id,
            render_minor: 129,
            identity: [u8::try_from(gpu_id).unwrap(); 16],
            unique_id: Some(u64::from(gpu_id)),
            base,
            limit,
            lds_base: 0x3000_0000_0000,
            scratch_base: 0x4000_0000_0000,
            sdma_next_engine: AtomicU32::new(0),
            scratch: scratch_pool(),
            vmem: Mutex::new(super::super::vmem::VmState::new()),
            version: self.vm.version,
            doorbells: super::super::queue::Doorbells::default(),
        })
    }

    fn create(&self) -> Result<Owned<KfdAllocation>, Error> {
        KfdAllocation::create(
            self.vm.clone(),
            desc(),
            BufferKind::Vram {
                public: false,
                coherent: false,
                extended_coherent: false,
                uncached: false,
                contiguous: false,
            },
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
    }

    #[allow(unsafe_code)]
    fn allocate(
        &self,
        kind: impl Into<FixtureMemoryKind>,
        permissions: DeviceAccess,
    ) -> Result<Owned<KfdAllocation>, Error> {
        let kind = match kind.into() {
            FixtureMemoryKind::Owned(MemoryKind::System) => BufferKind::Gtt,
            FixtureMemoryKind::Owned(MemoryKind::OwnedHost { policy }) => {
                BufferKind::OwnedUserptr { cache: policy }
            }
            FixtureMemoryKind::RegisteredHost { address, policy } => {
                // SAFETY: Scripted KFD replies cannot access this synthetic
                // address; the fixture tests metadata and rollback only.
                BufferKind::Userptr(unsafe { BorrowedHostPages::new(address, policy) })
            }
            FixtureMemoryKind::Owned(MemoryKind::DeviceLocal {
                host_visible,
                coherent,
                extended_coherent,
                uncached,
                contiguous,
            }) => BufferKind::Vram {
                public: host_visible,
                coherent,
                extended_coherent,
                uncached,
                contiguous,
            },
        };
        KfdAllocation::create(self.vm.clone(), desc(), kind, permissions)
    }

    fn allocate_with_lifetime(
        &self,
        lifetime: crate::session::DriverContextLifetime,
        kind: MemoryKind,
        permissions: DeviceAccess,
    ) -> Result<Owned<super::super::LinuxAllocation>, Error> {
        let device = super::super::DeviceState {
            vm: self.vm.clone(),
            native: sysfs::KfdNode {
                public_memory_bytes: 1 << 28,
                ..node()
            },
            asic_family_id: 0,
            gpu_counter_frequency_hz: 0,
        };
        super::super::LinuxKfdDriver::with_context_lifetime(
            lifetime,
            crate::os::ProcessIdentity::capture().expect("test process identity"),
        )
        .allocate_owned(
            &device,
            &[],
            kind,
            desc().size,
            desc().alignment,
            permissions,
        )
    }

    fn import(
        &self,
        descriptor: i32,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<KfdAllocation>, Error> {
        KfdAllocation::import_dma_buf(
            self.vm.clone(),
            descriptor,
            source_offset,
            byte_length,
            alignment,
            permissions,
        )
    }

    fn import_graphics(
        &self,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        descriptor: i32,
        size_hint: u64,
    ) -> Result<Owned<KfdAllocation>, Error> {
        KfdAllocation::import_graphics_dma_buf(self.vm.clone(), peers, descriptor, size_hint)
    }

    fn import_ipc(
        &self,
        mappings: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        words: [u32; 8],
        size: u64,
    ) -> Result<Owned<KfdAllocation>, Error> {
        KfdAllocation::import_ipc(self.vm.clone(), mappings, words, size)
    }

    fn exhausted(&self) {
        assert!(
            self.replies.lock().unwrap().is_empty(),
            "native cleanup stopped early"
        );
    }
}

#[test]
fn xnack_query_reads_live_mode_and_rejects_lost_devices() {
    use crate::device::{Device, DeviceResources};
    use crate::driver::test_driver::TestDriver;
    use crate::driver::{DeviceDriverState, Driver, EndpointSelector};
    use crate::session::DriverContextLifetime;
    use crate::topology::{EndpointKind, GpuInfo};

    let fixture = Fixture::new([
        Reply::Xnack(1, None),
        Reply::Xnack(0, None),
        Reply::Xnack(0, Some(errno::EIO)),
    ]);
    let driver = shared(super::super::LinuxKfdDriver::new());
    let mut endpoint = TestDriver::new(
        node().identity,
        EndpointKind::Gpu {
            info: GpuInfo::default(),
        },
        DriverContextLifetime::Session,
    )
    .open_endpoint(node().identity)
    .unwrap();
    endpoint.driver_instance = driver.driver_instance();
    endpoint.selector = EndpointSelector::LinuxKfd {
        native: node(),
        access_targets: None,
    };
    let driver_state = DeviceDriverState::LinuxKfd {
        driver,
        state: super::super::DeviceState {
            vm: fixture.vm.clone(),
            native: node(),
            asic_family_id: 0,
            gpu_counter_frequency_hz: 0,
        },
    };
    let device = Device {
        resources: DeviceResources::for_activated(&endpoint, &driver_state).unwrap(),
        driver_state,
        endpoint,
    };
    let gpu = device.gpu().unwrap();
    assert!(gpu.xnack_enabled().unwrap());
    assert!(!gpu.xnack_enabled().unwrap());
    assert_eq!(
        gpu.xnack_enabled()
            .unwrap_err()
            .native_status()
            .and_then(crate::NativeStatus::errno_code),
        Some(errno::EIO)
    );
    fixture.hardware.store(2, Ordering::Release);
    assert_eq!(
        gpu.xnack_enabled().unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
    fixture.exhausted();
}

#[test]
fn ais_uses_the_mapped_vram_handle_and_preserves_native_failure() {
    let fixture = Fixture::new([
        Reply::Allocate(17, None),
        Reply::Map(0, 1, None),
        Reply::Ais(
            uapi::AisInput {
                handle: 17,
                handle_offset: 4096 + 32,
                file_offset: 8192,
                size: 128,
                operation: uapi::AIS_READ,
                descriptor: 3,
            },
            uapi::AisOutput {
                size_copied: 96,
                status: 0,
                pad: 0,
            },
            None,
        ),
        Reply::Ais(
            uapi::AisInput {
                handle: 17,
                handle_offset: 4096 + 256,
                file_offset: 16384,
                size: 512,
                operation: uapi::AIS_WRITE,
                descriptor: 4,
            },
            uapi::AisOutput {
                size_copied: 64,
                status: -errno::EIO,
                pad: 0,
            },
            Some(errno::EIO),
        ),
        Reply::Unmap(0, 1, None),
        Reply::Free(None),
    ]);
    let mut allocation = fixture.create().unwrap();
    // Model a logical slice of a larger KFD handle, as with imported VRAM.
    allocation.device_byte_offset = 4096;
    allocation.logical_size = 8192;
    let result = allocation
        .ais_transfer(3, 32, 128, 8192, uapi::AIS_READ)
        .unwrap();
    assert_eq!(result.size_copied, 96);
    assert_eq!(result.status, 0);

    assert_eq!(
        allocation
            .ais_transfer(3, 8191, 2, 0, uapi::AIS_READ)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        allocation
            .ais_transfer(-1, 32, 128, 0, uapi::AIS_READ)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidArgument
    );
    let failure = allocation
        .ais_transfer(4, 256, 512, 16384, uapi::AIS_WRITE)
        .unwrap_err();
    assert_eq!(
        failure
            .native_status()
            .and_then(crate::NativeStatus::errno_code),
        Some(errno::EIO)
    );
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn ais_rejects_non_vram_backing_without_a_native_transfer() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        backing_file(desc().size),
    );
    let mut allocation = fixture
        .allocate(MemoryKind::System, DeviceAccess::READ | DeviceAccess::WRITE)
        .unwrap();
    assert_eq!(
        allocation
            .ais_transfer(3, 0, 4096, 0, uapi::AIS_READ)
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn bo_backed_host_alias_reuses_mapping_and_blocks_source_release() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        backing_file(desc().size),
    );
    let mut source = fixture
        .allocate(MemoryKind::System, DeviceAccess::READ | DeviceAccess::WRITE)
        .unwrap();
    source.set_host_alias_identity([7, 11]).unwrap();
    let host = source.host_address().unwrap();
    let mut backing = None;
    let mut alias = KfdHostAlias::try_create(
        &fixture.vm,
        &[],
        host + 17,
        31,
        HostMappingPolicy::Fine,
        DeviceAccess::READ | DeviceAccess::WRITE,
        &mut backing,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        alias.cached_info().device_address,
        source.device_address(&fixture.vm).unwrap() + 17
    );
    assert_eq!(alias.cached_info().physical_backing_id, [7, 11]);
    assert_eq!(source.free().unwrap_err().kind(), ErrorKind::Busy);
    assert_eq!(
        source.set_access(&[&fixture.vm]).unwrap_err().kind(),
        ErrorKind::Busy
    );
    alias.free();
    source.set_access(&[&fixture.vm]).unwrap();
    let mut backing = None;
    let mut next_alias = KfdHostAlias::try_create(
        &fixture.vm,
        &[],
        host + 17,
        31,
        HostMappingPolicy::Fine,
        DeviceAccess::READ | DeviceAccess::WRITE,
        &mut backing,
    )
    .unwrap()
    .unwrap();
    assert_eq!(next_alias.cached_info().physical_backing_id, [7, 11]);
    next_alias.free();
    source.free().unwrap();
    drop(alias);
    drop(source);
    fixture.exhausted();
}

#[test]
fn ais_file_read_rejects_read_only_vram_before_native_transfer() {
    let fixture = Fixture::with_flags(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::VRAM | uapi::NO_SUBSTITUTE,
    );
    let mut allocation = fixture
        .allocate(
            MemoryKind::DeviceLocal {
                host_visible: false,
                coherent: false,
                extended_coherent: false,
                uncached: false,
                contiguous: false,
            },
            DeviceAccess::READ,
        )
        .unwrap();
    assert_eq!(
        allocation
            .ais_transfer(3, 0, 4096, 0, uapi::AIS_READ)
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn secondary_context_rejects_owned_userptr_before_native_allocation() {
    let fixture = Fixture::new([]);
    assert_eq!(
        fixture
            .allocate_with_lifetime(
                crate::session::DriverContextLifetime::Session,
                MemoryKind::OwnedHost {
                    policy: HostMappingPolicy::Fine,
                },
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Unsupported
    );
    fixture.exhausted();
}

#[test]
#[allow(unsafe_code)]
fn secondary_extended_registration_reaches_drm_only_for_gfx1201() {
    let fixture = Fixture::new([]);
    let make_device = |gfx_target| super::super::DeviceState {
        vm: fixture.vm.clone(),
        native: sysfs::KfdNode {
            debug: sysfs::KfdDebugProperties::default(),
            host_atomics: false,
            queues: sysfs::KfdQueueProperties {
                gfx_target,
                target_profile: crate::topology::GpuTargetProfile::from_kfd_target(gfx_target),
                ..sysfs::KfdQueueProperties::default()
            },
            ..node()
        },
        asic_family_id: 0,
        gpu_counter_frequency_hz: 0,
    };
    let driver = super::super::LinuxKfdDriver::new();
    let gfx1201 = make_device(120_001);
    let other = make_device(110_000);
    let request = || crate::memory::HostRegistration {
        address: 0x10000,
        policy: HostMappingPolicy::Extended,
        size: desc().size,
        alignment: desc().alignment,
        permissions: DeviceAccess::READ | DeviceAccess::WRITE,
        backing: None,
    };

    // SAFETY: /dev/null rejects GEM USERPTR before it can retain the
    // synthetic page cover.
    let error = unsafe { driver.register_host(&gfx1201, &[], request()) }
        .err()
        .unwrap();
    assert!(matches!(
        error,
        Error::NativeOperation {
            operation: "DRM GEM USERPTR registration",
            ..
        }
    ));
    // SAFETY: The unsupported targets are rejected before native access.
    let error = unsafe { driver.register_host(&other, &[], request()) }
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    // SAFETY: An unsupported peer is rejected before native access.
    let error = unsafe { driver.register_host(&gfx1201, &[&other], request()) }
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Unsupported);

    drop(gfx1201);
    drop(other);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
}

#[test]
#[allow(unsafe_code)]
fn failed_secondary_drm_registration_releases_its_vm_dependency() {
    let fixture = Fixture::new([]);
    // SAFETY: The scripted render endpoint rejects registration before it
    // can retain or access the synthetic host address.
    let result = unsafe {
        super::super::registered_host::DrmRegisteredHost::create(
            fixture.vm.clone(),
            std::iter::empty(),
            desc(),
            0x10000,
            DeviceAccess::READ | DeviceAccess::WRITE,
            false,
            None,
        )
    };
    // The scripted VM uses /dev/null as its render endpoint. GEM USERPTR must
    // fail without retaining that VM or its speculative GPU VA.
    assert_eq!(result.err().unwrap().kind(), ErrorKind::Unsupported);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
}

#[test]
fn ipc_export_encodes_the_native_share_handle_and_full_backing_extent() {
    let flags = uapi::VRAM | uapi::WRITABLE | uapi::NO_SUBSTITUTE;
    let fixture = Fixture::with_flags(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::IpcExport([1, 2, 3, 4], None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        flags,
    );
    let mut allocation = fixture.create().unwrap();
    assert_eq!(
        allocation.export_ipc_memory().unwrap().words,
        [1, 2, 3, 4, IPC_APERTURE_DGPU, 0, 16, 42]
    );
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn ipc_import_maps_requested_devices_and_owns_cpu_visible_backing() {
    let flags = uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::WRITABLE;
    let fixture = Fixture::with_devices(
        [
            Reply::IpcImport(17, 65536, flags, None),
            Reply::Map(0, 2, None),
            Reply::Unmap(0, 2, None),
            Reply::Free(None),
        ],
        0,
        &[42, 53],
        backing_file(2 * desc().size),
    );
    let peer = fixture.peer(53, 0x20000, isize::MAX as u64);
    let mut allocation = fixture
        .import_ipc(
            [fixture.vm.clone(), peer.clone()].into_iter(),
            [1, 2, 3, 4, IPC_APERTURE_DGPU, 0, 16, 42],
            4097,
        )
        .unwrap();
    let info = allocation.cached_info();
    assert_eq!(info.size, 4097);
    assert_eq!(info.native_size, 65536);
    assert_eq!(
        info.host_address,
        Some(usize::try_from(info.device_address).unwrap())
    );
    assert_eq!(
        allocation.device_address(&fixture.vm).unwrap(),
        info.device_address
    );
    assert_eq!(
        allocation.device_address(&peer).unwrap(),
        info.device_address
    );
    assert_eq!(allocation.remaining_backing_byte_length(), 0);
    allocation
        .reservation
        .as_mut()
        .unwrap()
        .fail_release_once(errno::ENOMEM);
    assert_eq!(
        allocation.free().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert!(allocation.handle.is_none());
    assert_eq!(allocation.remaining_backing_byte_length(), 0);
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&peer), 1);
}

#[test]
fn ipc_import_rejects_fragments_and_inconsistent_extents_before_kfd() {
    let fixture = Fixture::new([]);
    for (words, size) in [
        ([0; 8], 4096),
        (
            [1, 2, 3, 4, IPC_APERTURE_DGPU, 0, IPC_FRAGMENT | 16, 42],
            4096,
        ),
        ([1, 2, 3, 4, IPC_APERTURE_DGPU, 1, 16, 42], 4096),
        ([1, 2, 3, 4, IPC_APERTURE_DGPU, 0, 16, 53], 4096),
        ([1, 2, 3, 4, IPC_APERTURE_DGPU, 0, 16, 42], 0),
        ([1, 2, 3, 4, IPC_APERTURE_DGPU, 0, 1, 42], 4097),
    ] {
        assert_eq!(
            fixture
                .import_ipc(std::iter::empty(), words, size)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidArgument
        );
    }
    fixture.exhausted();
}

#[test]
#[allow(unsafe_code)]
fn scratch_owner_metadata_failure_does_not_acquire_a_native_range() {
    let allocator_state = crate::test_support::allocator::State::default();
    // SAFETY: The allocator state remains live until after the fixture drops.
    let fixture = Fixture::with_devices_and_userptr_source(
        [],
        uapi::VRAM | uapi::WRITABLE | uapi::NO_SUBSTITUTE,
        &[42],
        File::open("/dev/null").unwrap(),
        false,
        0x12345,
    );
    unsafe { allocator_state.allocator() }.with_scope(|| {
        let before = allocator_state.allocations.load(Ordering::Relaxed);
        allocator_state.fail_at.store(before + 2, Ordering::Relaxed);
        let result = KfdAllocation::create_scratch(
            &fixture.vm,
            AllocationDesc {
                size: desc().size,
                alignment: 4096,
            },
        );
        assert_eq!(result.err().unwrap().kind(), ErrorKind::ResourceExhausted);
        assert_eq!(
            allocator_state.allocations.load(Ordering::Relaxed),
            before + 2
        );
        assert_eq!(allocator_state.frees.load(Ordering::Relaxed), 1);
        assert!(fixture.vm.scratch.lock().unwrap().reservation.is_none());
        fixture.exhausted();
    });
}

#[test]
fn scratch_backing_programs_one_process_base_and_reuses_released_ranges() {
    let fixture = Fixture::with_devices(
        [
            Reply::SetScratch(None),
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::VRAM | uapi::WRITABLE | uapi::NO_SUBSTITUTE,
        &[42],
        backing_file(desc().size),
    );
    let scratch_desc = AllocationDesc {
        size: desc().size,
        alignment: 4096,
    };
    let mut first = KfdAllocation::create_scratch(&fixture.vm, scratch_desc).unwrap();
    let first_address = first.cached_info().device_address;
    assert_eq!(first_address, fixture.scratch_base.load(Ordering::Relaxed));
    assert_eq!(first.cached_info().native_size, desc().size);
    assert!(matches!(
        first.export_dma_buf(),
        Err(error) if error.kind() == ErrorKind::Unsupported
    ));
    first.free().unwrap();
    drop(first);

    let mut second = KfdAllocation::create_scratch(&fixture.vm, scratch_desc).unwrap();
    assert_eq!(second.cached_info().device_address, first_address);
    second.free().unwrap();
    drop(second);

    let scratch = fixture.vm.scratch.lock().unwrap();
    assert_eq!(scratch.ranges.len(), 1);
    assert!(!scratch.ranges[0].allocated);
    assert_eq!(scratch.ranges[0].size, 8_u64 << 30);
    drop(scratch);
    fixture.exhausted();
}

fn backing_file(byte_length: u64) -> File {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "rocddi-dma-buf-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(path).unwrap();
    file.set_len(byte_length).unwrap();
    file
}

#[test]
fn system_allocation_maps_every_vm_at_one_exact_address() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 2, None),
            Reply::Unmap(0, 2, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42, 53],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x20000, isize::MAX as u64);
    let unrelated = fixture.peer(64, 0x20000, isize::MAX as u64);
    let mut allocation = KfdAllocation::create_with_peers(
        fixture.vm.clone(),
        std::iter::once(peer.clone()),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
        None,
    )
    .unwrap();
    let primary_address = allocation.device_address(&fixture.vm).unwrap();
    assert_eq!(allocation.device_address(&peer).unwrap(), primary_address);
    assert_eq!(allocation.cached_info().device_address, primary_address);
    assert!(primary_address >= 0x20000);
    assert_eq!(
        allocation.device_address(&unrelated).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
    assert_eq!(Shared::strong_count(&peer), 1);
}

#[test]
fn installed_signal_event_page_transfers_to_process_lifetime() {
    let fixture = Fixture::with_devices(
        [Reply::Allocate(17, None), Reply::Map(0, 1, None)],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        backing_file(desc().size),
    );
    let mut allocation = KfdAllocation::create(
        fixture.vm.clone(),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
    )
    .unwrap();
    assert_eq!(
        allocation.signal_event_page_handle(&fixture.vm).unwrap(),
        17
    );
    fixture.hardware.store(2, Ordering::Relaxed);
    allocation.retain_signal_event_page().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
}

#[test]
fn repeated_logical_devices_reuse_each_native_vm_mapping() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 2, None),
            Reply::Unmap(0, 2, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42, 53],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x20000, isize::MAX as u64);
    let mut allocation = KfdAllocation::create_with_peers(
        fixture.vm.clone(),
        [fixture.vm.clone(), peer.clone(), peer.clone()].into_iter(),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
        None,
    )
    .unwrap();
    let address = allocation.device_address(&fixture.vm).unwrap();
    assert_eq!(allocation.device_address(&peer).unwrap(), address);
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
    assert_eq!(Shared::strong_count(&peer), 1);
}

#[test]
fn local_allocation_maps_every_qualified_peer_vm() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 2, None),
            Reply::Unmap(0, 2, None),
            Reply::Free(None),
        ],
        uapi::VRAM | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42, 53],
        File::open("/dev/null").unwrap(),
    );
    let peer = fixture.peer(53, 0x20000, isize::MAX as u64);
    let mut allocation = KfdAllocation::create_with_peers(
        fixture.vm.clone(),
        std::iter::once(peer.clone()),
        desc(),
        BufferKind::Vram {
            public: false,
            coherent: false,
            extended_coherent: false,
            uncached: false,
            contiguous: false,
        },
        DeviceAccess::READ | DeviceAccess::WRITE,
        None,
    )
    .unwrap();
    let address = allocation.device_address(&fixture.vm).unwrap();
    assert_eq!(allocation.device_address(&peer).unwrap(), address);
    assert_eq!(
        allocation.host_address().unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
    assert_eq!(Shared::strong_count(&peer), 1);
}

#[test]
fn access_replacement_maps_new_peer_before_unmapping_the_old_device() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::MapDevices(&[53], 0, 1, None),
            Reply::UnmapDevices(&[42], 0, 1, None),
            Reply::MapDevices(&[42], 0, 1, None),
            Reply::UnmapDevices(&[53], 0, 1, None),
            Reply::UnmapDevices(&[42], 0, 1, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x20000, isize::MAX as u64);
    let mut allocation = KfdAllocation::create(
        fixture.vm.clone(),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
    )
    .unwrap();
    allocation.set_access(&[&peer]).unwrap();
    assert!(allocation.device_address(&peer).is_ok());
    assert_eq!(
        allocation.device_address(&fixture.vm).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    allocation.set_access(&[&fixture.vm]).unwrap();
    assert!(allocation.device_address(&fixture.vm).is_ok());
    assert_eq!(
        allocation.device_address(&peer).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&peer), 1);
}

#[test]
fn failed_peer_map_keeps_old_access_and_retries_its_completed_prefix() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::MapDevices(&[53], 0, 1, Some(errno::EINTR)),
            Reply::MapDevices(&[53], 1, 1, None),
            Reply::UnmapDevices(&[42], 0, 1, None),
            Reply::UnmapDevices(&[53], 0, 1, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x20000, isize::MAX as u64);
    let mut allocation = KfdAllocation::create(
        fixture.vm.clone(),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
    )
    .unwrap();
    assert!(allocation.set_access(&[&peer]).is_err());
    allocation.set_host_alias_identity([19, 31]).unwrap();
    assert!(allocation.set_host_alias_identity([19, 32]).is_err());
    assert!(allocation.device_address(&fixture.vm).is_ok());
    assert_eq!(
        allocation.device_address(&peer).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    allocation.set_access(&[&peer]).unwrap();
    assert!(allocation.device_address(&peer).is_ok());
    let mut backing = None;
    let mut alias = KfdHostAlias::try_create(
        &peer,
        &[],
        allocation.host_address().unwrap(),
        desc().size,
        HostMappingPolicy::Fine,
        DeviceAccess::READ | DeviceAccess::WRITE,
        &mut backing,
    )
    .unwrap()
    .unwrap();
    assert_eq!(alias.cached_info().physical_backing_id, [19, 31]);
    alias.free();
    drop(alias);
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&peer), 1);
}

#[test]
fn failed_peer_unmap_is_not_reported_as_available_until_retry() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::MapDevices(&[53], 0, 1, None),
            Reply::UnmapDevices(&[42], 0, 1, Some(errno::EINTR)),
            Reply::UnmapDevices(&[42], 1, 1, None),
            Reply::UnmapDevices(&[53], 0, 1, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x20000, isize::MAX as u64);
    let mut allocation = KfdAllocation::create(
        fixture.vm.clone(),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
    )
    .unwrap();
    assert!(allocation.set_access(&[&peer]).is_err());
    assert!(allocation.device_address(&peer).is_ok());
    assert_eq!(
        allocation.device_address(&fixture.vm).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    allocation.set_access(&[&peer]).unwrap();
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn access_replacement_rejects_an_incompatible_peer_before_mapping() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        backing_file(desc().size),
    );
    let mut allocation = KfdAllocation::create(
        fixture.vm.clone(),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
    )
    .unwrap();
    let address = allocation.cached_info().device_address;
    let peer = fixture.peer(53, address + desc().size, address + desc().size * 2 - 1);
    assert_eq!(
        allocation.set_access(&[&peer]).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    assert!(allocation.device_address(&fixture.vm).is_ok());
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn free_completes_a_pending_peer_map_before_releasing_backing() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::MapDevices(&[53], 0, 1, Some(errno::EINTR)),
            Reply::MapDevices(&[53], 1, 1, None),
            Reply::UnmapDevices(&[42, 53], 0, 2, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x20000, isize::MAX as u64);
    let mut allocation = KfdAllocation::create(
        fixture.vm.clone(),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
    )
    .unwrap();
    assert!(allocation.set_access(&[&peer]).is_err());
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&peer), 1);
}

#[test]
fn incompatible_vm_apertures_fail_before_native_allocation() {
    let fixture = Fixture::with_devices(
        [],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42, 53],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x20000, 0x2ffff);
    let isolated = shared(DeviceVm {
        base: 0x40000,
        limit: 0x4ffff,
        loss: fixture.vm.loss.clone(),
        render: Some(File::open("/dev/null").unwrap()),
        system_dma_buf_import: false,
        gpu_id: 64,
        render_minor: 130,
        identity: [64; 16],
        unique_id: Some(64),
        lds_base: 0x5000_0000_0000,
        scratch_base: 0x6000_0000_0000,
        sdma_next_engine: AtomicU32::new(0),
        scratch: scratch_pool(),
        vmem: Mutex::new(super::super::vmem::VmState::new()),
        version: fixture.vm.version,
        doorbells: super::super::queue::Doorbells::default(),
    });
    assert_eq!(
        KfdAllocation::create_with_peers(
            peer,
            std::iter::once(isolated),
            desc(),
            BufferKind::Gtt,
            DeviceAccess::READ | DeviceAccess::WRITE,
            None,
        )
        .err()
        .unwrap()
        .kind(),
        ErrorKind::Unsupported
    );
    fixture.exhausted();
}

#[test]
fn multi_vm_mapping_and_unmapping_resume_from_the_reported_prefix() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, Some(errno::EINTR)),
            Reply::Map(1, 2, None),
            Reply::Unmap(0, 2, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42, 53],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x10000, isize::MAX as u64);
    assert_eq!(
        KfdAllocation::create_with_peers(
            fixture.vm.clone(),
            std::iter::once(peer.clone()),
            desc(),
            BufferKind::Gtt,
            DeviceAccess::READ | DeviceAccess::WRITE,
            None,
        )
        .err()
        .unwrap()
        .kind(),
        ErrorKind::Driver
    );
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
    assert_eq!(Shared::strong_count(&peer), 1);

    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 2, None),
            Reply::Unmap(0, 1, Some(errno::EINTR)),
            Reply::Unmap(1, 2, None),
            Reply::Free(None),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42, 53],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x10000, isize::MAX as u64);
    let mut allocation = KfdAllocation::create_with_peers(
        fixture.vm.clone(),
        std::iter::once(peer.clone()),
        desc(),
        BufferKind::Gtt,
        DeviceAccess::READ | DeviceAccess::WRITE,
        None,
    )
    .unwrap();
    assert_eq!(allocation.free().unwrap_err().kind(), ErrorKind::Driver);
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
    assert_eq!(Shared::strong_count(&peer), 1);
}

#[test]
fn ambiguous_multi_vm_mapping_retains_every_vm_dependency() {
    let fixture = Fixture::with_devices(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, Some(errno::EFAULT)),
        ],
        uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42, 53],
        backing_file(desc().size),
    );
    let peer = fixture.peer(53, 0x10000, isize::MAX as u64);
    assert!(
        KfdAllocation::create_with_peers(
            fixture.vm.clone(),
            std::iter::once(peer.clone()),
            desc(),
            BufferKind::Gtt,
            DeviceAccess::READ | DeviceAccess::WRITE,
            None,
        )
        .is_err()
    );
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 2);
    assert_eq!(Shared::strong_count(&peer), 2);
}

#[test]
fn allocation_permissions_reach_kfd_without_widening_access() {
    for (permissions, flags) in [
        (DeviceAccess::READ, 0),
        (DeviceAccess::READ | DeviceAccess::WRITE, uapi::WRITABLE),
        (DeviceAccess::READ | DeviceAccess::EXECUTE, uapi::EXECUTABLE),
        (
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            uapi::WRITABLE | uapi::EXECUTABLE,
        ),
    ] {
        let fixture = Fixture::with_flags(
            [
                Reply::Allocate(17, None),
                Reply::Map(0, 1, None),
                Reply::Unmap(0, 1, None),
                Reply::Free(None),
            ],
            uapi::VRAM | uapi::NO_SUBSTITUTE | flags,
        );
        let mut allocation = fixture
            .allocate(
                MemoryKind::DeviceLocal {
                    host_visible: false,
                    coherent: false,
                    extended_coherent: false,
                    uncached: false,
                    contiguous: false,
                },
                permissions,
            )
            .unwrap();
        allocation.free().unwrap();
        drop(allocation);
        fixture.exhausted();
        assert_eq!(Shared::strong_count(&fixture.vm), 1);

        for (kind, placement) in [
            (
                MemoryKind::System,
                uapi::GTT | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE,
            ),
            (
                MemoryKind::OwnedHost {
                    policy: HostMappingPolicy::Coarse,
                },
                uapi::USERPTR | uapi::NO_SUBSTITUTE,
            ),
            (
                MemoryKind::OwnedHost {
                    policy: HostMappingPolicy::Fine,
                },
                uapi::USERPTR | uapi::COHERENT | uapi::NO_SUBSTITUTE,
            ),
            (
                MemoryKind::OwnedHost {
                    policy: HostMappingPolicy::Extended,
                },
                uapi::USERPTR | uapi::COHERENT | uapi::EXT_COHERENT | uapi::NO_SUBSTITUTE,
            ),
            (
                MemoryKind::OwnedHost {
                    policy: HostMappingPolicy::Uncached,
                },
                uapi::USERPTR | uapi::COHERENT | uapi::UNCACHED | uapi::NO_SUBSTITUTE,
            ),
            (
                MemoryKind::DeviceLocal {
                    host_visible: true,
                    coherent: false,
                    extended_coherent: false,
                    uncached: false,
                    contiguous: false,
                },
                uapi::VRAM | uapi::NO_SUBSTITUTE | uapi::PUBLIC,
            ),
        ] {
            // Verify exact flags before an injected allocation failure, without
            // needing a real render device for these CPU-visible placements.
            let fixture = if matches!(kind, MemoryKind::OwnedHost { .. }) {
                Fixture::with_owned_userptr(
                    [Reply::Allocate(0, Some(errno::ENOMEM))],
                    placement | flags,
                )
            } else {
                Fixture::with_flags([Reply::Allocate(0, Some(errno::ENOMEM))], placement | flags)
            };
            assert_eq!(
                fixture.allocate(kind, permissions).err().unwrap().kind(),
                ErrorKind::ResourceExhausted
            );
            fixture.exhausted();
            assert_eq!(Shared::strong_count(&fixture.vm), 1);
        }
    }
}

#[test]
fn dma_buf_import_maps_the_full_backing_and_exposes_the_logical_subrange() {
    let file = backing_file(2 * 65536);
    let identity = os_file::dma_buf_file_info(&file).unwrap().physical_id;
    let exported = os_file::duplicate_file(file.as_raw_fd())
        .unwrap()
        .into_raw_fd();
    let fixture = Fixture::new([
        Reply::DmaBufInfo(2 * 65536, 42, uapi::GTT, &[], None),
        Reply::ImportDmaBuf(17, 42, None),
        Reply::Map(0, 1, None),
        Reply::ExportDmaBuf(exported, None),
        Reply::Unmap(0, 1, None),
        Reply::Free(None),
    ]);
    let original_descriptor = file.as_raw_fd();
    let mut allocation = fixture
        .import(
            original_descriptor,
            65536,
            4097,
            65536,
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
        )
        .unwrap();
    assert_eq!(file.as_raw_fd(), original_descriptor);
    let info = allocation.cached_info();
    assert_eq!(info.size, 4097);
    assert_eq!(info.native_size, 2 * 65536);
    assert_eq!(info.physical_backing_id, identity);
    assert_eq!(info.host_address, Some(allocation.host_address().unwrap()));
    assert_eq!(
        info.device_address,
        allocation.device_address(&fixture.vm).unwrap()
    );
    assert_eq!(info.device_address % 65536, 0);
    allocation
        .reservation
        .as_mut()
        .unwrap()
        .write_bytes(65536, &[0x5a])
        .unwrap();
    let mut byte = [0];
    file.read_exact_at(&mut byte, 65536).unwrap();
    assert_eq!(byte, [0x5a]);
    let dma_buf = allocation.export_dma_buf().unwrap();
    assert_eq!(dma_buf.info().byte_length, 2 * 65536);
    assert_eq!(dma_buf.info().source_offset, 65536);
    assert_eq!(dma_buf.info().physical_backing_id, identity);
    drop(dma_buf);
    let mut backing = None;
    let mut alias = KfdHostAlias::try_create(
        &fixture.vm,
        &[],
        info.host_address.unwrap() + 17,
        31,
        HostMappingPolicy::Coarse,
        DeviceAccess::READ | DeviceAccess::WRITE,
        &mut backing,
    )
    .unwrap()
    .unwrap();
    assert_eq!(alias.source_offset(), 65536 + 17);
    assert_eq!(alias.cached_info().native_size, 2 * 65536);
    assert_eq!(alias.cached_info().physical_backing_id, identity);
    assert_eq!(allocation.free().unwrap_err().kind(), ErrorKind::Busy);
    alias.free();
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn dma_buf_import_validates_permissions_origin_placement_and_range() {
    let page = os_memory::page_size().unwrap() as u64;
    let file = backing_file(2 * 65536);
    let descriptor = file.as_raw_fd();
    let fixture = Fixture::new([]);
    assert_eq!(
        fixture
            .import(
                descriptor,
                0,
                page,
                page,
                DeviceAccess::READ | DeviceAccess::WRITE
            )
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Unsupported
    );
    fixture.exhausted();

    let fixture = Fixture::new([]);
    assert_eq!(
        fixture
            .import(
                descriptor,
                1,
                page,
                page,
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidArgument
    );
    fixture.exhausted();

    for (gpu_id, flags, offset, length, expected) in [
        (53, uapi::GTT, 0, page, ErrorKind::Unsupported),
        (42, uapi::VRAM, 0, page, ErrorKind::Unsupported),
        (42, uapi::GTT | (1 << 12), 0, page, ErrorKind::Unsupported),
        (42, uapi::GTT, 2 * 65536, 1, ErrorKind::InvalidArgument),
    ] {
        let fixture = Fixture::new([Reply::DmaBufInfo(2 * 65536, gpu_id, flags, &[], None)]);
        assert_eq!(
            fixture
                .import(
                    descriptor,
                    offset,
                    length,
                    page,
                    DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
                )
                .err()
                .unwrap()
                .kind(),
            expected
        );
        fixture.exhausted();
    }
}

#[test]
fn graphics_dma_buf_import_retains_metadata_and_maps_every_requested_gpu() {
    let file = backing_file(2 * 65536);
    let peer = 53;
    let fixture = Fixture::with_devices(
        [
            Reply::DmaBufInfo(2 * 65536, 64, uapi::GTT, &[1, 2, 3, 4], None),
            Reply::ImportDmaBuf(17, 64, None),
            Reply::Map(0, 2, None),
            Reply::Unmap(0, 2, None),
            Reply::Free(None),
        ],
        0,
        &[42, peer],
        File::open("/dev/null").unwrap(),
    );
    let peer_vm = fixture.peer(peer, 0x20_000, isize::MAX as u64);
    let mut allocation = fixture
        .import_graphics(std::iter::once(peer_vm.clone()), file.as_raw_fd(), 1234)
        .unwrap();
    let info = allocation.cached_info();
    assert_eq!(info.size, 2 * 65536);
    assert_eq!(info.native_size, 2 * 65536);
    assert_eq!(
        info.host_address,
        Some(usize::try_from(info.device_address).unwrap())
    );
    assert_eq!(allocation.metadata(), &[1, 2, 3, 4]);
    assert!(!allocation.is_owned_by(&fixture.vm));
    assert!(!allocation.is_owned_by(&peer_vm));
    assert_eq!(
        allocation.device_address(&fixture.vm).unwrap(),
        info.device_address
    );
    assert_eq!(
        allocation.device_address(&peer_vm).unwrap(),
        info.device_address
    );
    allocation
        .reservation
        .as_mut()
        .unwrap()
        .write_bytes(65536, &[0x5a])
        .unwrap();
    let mut byte = [0];
    file.read_exact_at(&mut byte, 65536).unwrap();
    assert_eq!(byte, [0x5a]);
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&peer_vm), 1);
}

#[test]
fn private_vram_graphics_import_has_no_cpu_mapping() {
    let file = backing_file(65536);
    let fixture = Fixture::new([
        Reply::DmaBufInfo(65536, 42, uapi::VRAM, &[], None),
        Reply::ImportDmaBuf(17, 42, None),
        Reply::Map(0, 1, None),
        Reply::Unmap(0, 1, None),
        Reply::Free(None),
    ]);
    let mut allocation = fixture
        .import_graphics(std::iter::empty(), file.as_raw_fd(), 0)
        .unwrap();
    assert_eq!(allocation.cached_info().host_address, None);
    assert!(allocation.is_owned_by(&fixture.vm));
    assert!(allocation.metadata().is_empty());
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn failed_dma_buf_import_owns_returned_handles_without_consuming_the_input() {
    let file = backing_file(65536);
    let descriptor = file.as_raw_fd();
    let fixture = Fixture::new([
        Reply::DmaBufInfo(65536, 42, uapi::GTT, &[], None),
        Reply::ImportDmaBuf(17, 42, Some(errno::ENOMEM)),
        Reply::Free(None),
    ]);
    assert_eq!(
        fixture
            .import(
                descriptor,
                0,
                65536,
                65536,
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )
            .err()
            .unwrap()
            .kind(),
        ErrorKind::ResourceExhausted
    );
    assert!(file.metadata().is_ok());
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);

    let fixture = Fixture::new([
        Reply::DmaBufInfo(65536, 42, uapi::GTT, &[], None),
        Reply::ImportDmaBuf(17, 42, Some(errno::EFAULT)),
    ]);
    assert!(
        fixture
            .import(
                descriptor,
                0,
                65536,
                65536,
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )
            .is_err()
    );
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 2);
}

#[test]
fn registered_host_pages_keep_the_caller_address_and_an_independent_gpu_va() {
    let fixture = Fixture::with_flags(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::USERPTR | uapi::COHERENT | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
    );
    let mut allocation = fixture
        .allocate(
            FixtureMemoryKind::RegisteredHost {
                address: 0x12345,
                policy: HostMappingPolicy::Fine,
            },
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .unwrap();
    let reservation = allocation.reservation.as_ref().unwrap().address();
    let info = allocation.cached_info();
    assert_eq!(info.host_address, Some(0x12345));
    let page = os_memory::page_size().unwrap() as u64;
    assert_eq!(
        info.device_address,
        reservation as u64 + (0x12345 & (page - 1))
    );
    assert_eq!(info.size, 65536);
    assert_eq!(
        info.native_size,
        (65536 + (0x12345 & (page - 1))).div_ceil(page) * page
    );
    assert_ne!(info.device_address, 0x12345);
    assert_eq!(
        allocation.device_address(&fixture.vm).unwrap(),
        info.device_address
    );
    assert_eq!(allocation.host_address().unwrap(), 0x12345);
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
}

#[test]
#[allow(unsafe_code)]
fn ambiguous_userptr_creation_retains_owned_host_pages() {
    let mut host = HostAllocation::allocate(desc().size, desc().alignment).unwrap();
    let fixture = Fixture::with_devices_and_userptr_source(
        [Reply::Allocate(17, Some(errno::EFAULT))],
        uapi::USERPTR | uapi::COHERENT | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        &[42],
        File::open("/dev/null").unwrap(),
        false,
        host.info().host_address,
    );
    // SAFETY: The registration lease moves into the native owner below.
    let pages =
        unsafe { BorrowedHostPages::new(host.info().host_address, HostMappingPolicy::Fine) };
    assert!(
        KfdAllocation::create_with_peers(
            fixture.vm.clone(),
            std::iter::empty(),
            desc(),
            BufferKind::Userptr(pages),
            DeviceAccess::READ | DeviceAccess::WRITE,
            Some(host.registration_lease().unwrap()),
        )
        .is_err()
    );
    assert_eq!(host.free().unwrap_err().kind(), ErrorKind::Busy);
    fixture.exhausted();
}

#[test]
fn registered_host_cache_policies_reach_kfd() {
    for (cache, flags) in [
        (HostMappingPolicy::Coarse, 0),
        (HostMappingPolicy::Fine, uapi::COHERENT),
        (
            HostMappingPolicy::Extended,
            uapi::COHERENT | uapi::EXT_COHERENT,
        ),
        (HostMappingPolicy::Uncached, uapi::COHERENT | uapi::UNCACHED),
    ] {
        let fixture = Fixture::with_flags(
            [
                Reply::Allocate(17, None),
                Reply::Map(0, 1, None),
                Reply::Unmap(0, 1, None),
                Reply::Free(None),
            ],
            uapi::USERPTR | flags | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
        );
        let mut allocation = fixture
            .allocate(
                FixtureMemoryKind::RegisteredHost {
                    address: 0x12345,
                    policy: cache,
                },
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .unwrap();
        assert_eq!(allocation.remaining_backing_byte_length(), 0);
        allocation.free().unwrap();
        drop(allocation);
        fixture.exhausted();
    }
}

#[test]
fn owned_system_pages_use_one_cpu_and_gpu_address() {
    let fixture = Fixture::with_owned_userptr(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::USERPTR | uapi::COHERENT | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
    );
    let mut allocation = fixture
        .allocate(
            MemoryKind::OwnedHost {
                policy: HostMappingPolicy::Fine,
            },
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .unwrap();
    let info = allocation.cached_info();
    assert_eq!(
        info.host_address,
        Some(usize::try_from(info.device_address).unwrap())
    );
    assert_eq!(info.size, 65536);
    assert_eq!(info.native_size, 65536);
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
}

#[test]
fn owned_host_pages_remain_accounted_after_kfd_free_when_cpu_unmap_fails() {
    let fixture = Fixture::with_owned_userptr(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::USERPTR | uapi::COHERENT | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
    );
    let mut allocation = fixture
        .allocate(
            MemoryKind::OwnedHost {
                policy: HostMappingPolicy::Fine,
            },
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .unwrap();
    assert_eq!(allocation.remaining_backing_byte_length(), desc().size);
    allocation
        .reservation
        .as_mut()
        .unwrap()
        .fail_release_once(errno::ENOMEM);
    assert_eq!(
        allocation.free().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert!(allocation.handle.is_none());
    assert!(
        allocation
            .reservation
            .as_ref()
            .unwrap()
            .has_writable_mapping()
    );
    assert_eq!(allocation.remaining_backing_byte_length(), desc().size);
    allocation.free().unwrap();
    assert_eq!(allocation.remaining_backing_byte_length(), 0);
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn invalid_registered_host_address_fails_before_native_observation() {
    let fixture = Fixture::new([]);
    fixture.hardware.store(2, Ordering::Relaxed);
    assert_eq!(
        fixture
            .allocate(
                FixtureMemoryKind::RegisteredHost {
                    address: 0,
                    policy: HostMappingPolicy::Fine,
                },
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert!(!fixture.vm.loss.lost.load(Ordering::Relaxed));
    fixture.exhausted();
}

#[test]
fn permissions_without_read_fail_before_loss_polling_or_native_acquisition() {
    for permissions in [
        DeviceAccess::NONE,
        DeviceAccess::WRITE,
        DeviceAccess::EXECUTE,
        DeviceAccess::WRITE | DeviceAccess::EXECUTE,
    ] {
        let fixture = Fixture::new([]);
        fixture.hardware.store(2, Ordering::Relaxed);
        assert_eq!(
            fixture
                .allocate(MemoryKind::System, permissions)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
        assert!(!fixture.vm.loss.lost.load(Ordering::Relaxed));
        fixture.exhausted();
        assert_eq!(Shared::strong_count(&fixture.vm), 1);
    }
}

#[test]
fn failed_creation_finishes_mapping_before_unmapping_even_with_a_full_prefix() {
    for prefix in [0, 1] {
        let fixture = Fixture::new([
            Reply::Allocate(17, None),
            Reply::Map(0, prefix, Some(errno::EINTR)),
            Reply::Map(prefix, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ]);
        let result = fixture.create();
        assert_eq!(result.err().unwrap().kind(), ErrorKind::Driver);
        fixture.exhausted();
        assert_eq!(
            Shared::strong_count(&fixture.vm),
            1,
            "successful cleanup retained a VM reference"
        );
    }
}

#[test]
fn failed_free_retries_completion_then_only_the_remaining_native_work() {
    for prefix in [0, 1] {
        let fixture = Fixture::new([
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, prefix, Some(errno::EINTR)),
            Reply::Unmap(prefix, 1, None),
            Reply::Free(Some(errno::EBUSY)),
            Reply::Free(None),
        ]);
        let mut allocation = fixture.create().unwrap();
        assert_eq!(allocation.device_address(&fixture.vm).unwrap() % 65536, 0);
        assert_eq!(
            allocation.host_address().unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(allocation.free().unwrap_err().kind(), ErrorKind::Driver);
        assert!(allocation.device_address(&fixture.vm).is_err());
        assert_eq!(allocation.free().unwrap_err().kind(), ErrorKind::Busy);
        assert!(allocation.reservation.is_some());
        assert_eq!(allocation.handle, Some(17));
        assert_eq!(allocation.remaining_backing_byte_length(), desc().size);
        allocation
            .reservation
            .as_mut()
            .unwrap()
            .fail_release_once(errno::ENOMEM);
        assert_eq!(
            allocation.free().unwrap_err().kind(),
            ErrorKind::ResourceExhausted
        );
        assert!(
            allocation.handle.is_none(),
            "successful FREE must not be replayed"
        );
        assert_eq!(allocation.remaining_backing_byte_length(), 0);
        allocation.free().unwrap();
        allocation.free().unwrap();
        drop(allocation);
        fixture.exhausted();
    }
}

#[test]
fn failed_creation_owns_returned_handles_and_uncertain_native_results() {
    let fixture = Fixture::new([Reply::Allocate(17, Some(errno::ENOMEM)), Reply::Free(None)]);
    assert_eq!(
        fixture.create().err().unwrap().kind(),
        ErrorKind::ResourceExhausted
    );
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);

    let fixture = Fixture::new([Reply::Allocate(0, Some(errno::ENOMEM))]);
    assert!(fixture.create().is_err());
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);

    // No FREE is safe when copyout concealed the handle, or when a successful
    // allocation returned an invalid handle. Retain dependencies until exit.
    for errno in [Some(errno::EFAULT), None] {
        let fixture = Fixture::new([Reply::Allocate(0, errno)]);
        assert!(fixture.create().is_err());
        fixture.exhausted();
        assert_eq!(Shared::strong_count(&fixture.vm), 2);
    }
    let fixture = Fixture::new([
        Reply::Allocate(17, None),
        Reply::Map(0, 1, Some(errno::EIO)),
        Reply::Map(1, 1, Some(errno::EIO)),
    ]);
    assert!(fixture.create().is_err());
    fixture.exhausted();
    assert_eq!(
        Shared::strong_count(&fixture.vm),
        2,
        "failed Drop released native dependencies"
    );
}

#[test]
fn malformed_or_copy_fault_mapping_outputs_are_not_replayed() {
    for (prefix, errno) in [(2, Some(errno::EIO)), (0, None), (1, Some(errno::EFAULT))] {
        let fixture = Fixture::new([Reply::Allocate(17, None), Reply::Map(0, prefix, errno)]);
        assert!(fixture.create().is_err());
        fixture.exhausted();
        assert_eq!(Shared::strong_count(&fixture.vm), 2);
    }
}

#[test]
fn local_allocation_properties_reach_kfd_exactly() {
    for (coherent, extended_coherent, uncached, contiguous, expected) in [
        (false, false, false, false, 0),
        (true, false, false, false, uapi::COHERENT),
        (
            true,
            true,
            false,
            false,
            uapi::COHERENT | uapi::EXT_COHERENT,
        ),
        (false, false, true, false, uapi::UNCACHED),
        (false, false, false, true, uapi::CONTIGUOUS),
        (
            true,
            false,
            true,
            true,
            uapi::COHERENT | uapi::UNCACHED | uapi::CONTIGUOUS,
        ),
    ] {
        let fixture = Fixture::with_flags(
            [Reply::Allocate(0, Some(errno::ENOMEM))],
            uapi::VRAM | uapi::NO_SUBSTITUTE | uapi::PUBLIC | uapi::WRITABLE | expected,
        );
        let result = fixture.allocate(
            MemoryKind::DeviceLocal {
                host_visible: true,
                coherent,
                extended_coherent,
                uncached,
                contiguous,
            },
            DeviceAccess::READ | DeviceAccess::WRITE,
        );
        assert_eq!(result.err().unwrap().kind(), ErrorKind::ResourceExhausted);
        fixture.exhausted();
        assert_eq!(Shared::strong_count(&fixture.vm), 1);
    }
}

#[test]
fn failed_public_vram_cpu_mapping_releases_the_buffer() {
    let fixture = Fixture::with_flags(
        [Reply::Allocate(17, None), Reply::Free(None)],
        uapi::VRAM | uapi::WRITABLE | uapi::NO_SUBSTITUTE | uapi::PUBLIC,
    );
    // /dev/null cannot provide a render mapping. The native buffer is freed
    // before any GPU MAP, and the failed file mmap leaves its VA reservation.
    let result = KfdAllocation::create(
        fixture.vm.clone(),
        desc(),
        BufferKind::Vram {
            public: true,
            coherent: false,
            extended_coherent: false,
            uncached: false,
            contiguous: false,
        },
        DeviceAccess::READ | DeviceAccess::WRITE,
    );
    let Error::NativeOperation { operation, .. } = result.err().unwrap() else {
        panic!("CPU mapping failure lost its native operation");
    };
    assert_eq!(operation, "KFD buffer CPU mmap");
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);
}

#[test]
fn native_loss_latches_without_preventing_explicit_cleanup() {
    let fixture = Fixture::new([
        Reply::Allocate(17, None),
        Reply::Map(0, 1, None),
        Reply::Unmap(0, 1, None),
        Reply::Free(None),
    ]);
    let mut allocation = fixture.create().unwrap();
    assert!(!fixture.vm.has_latched_loss());
    fixture.hardware.store(2, Ordering::Relaxed);
    assert!(!fixture.vm.has_latched_loss());
    assert_eq!(
        allocation.device_address(&fixture.vm).unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
    assert!(fixture.vm.has_latched_loss());
    fixture.hardware.store(0, Ordering::Relaxed);
    assert_eq!(
        allocation.device_address(&fixture.vm).unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
    assert_eq!(
        fixture.create().err().unwrap().kind(),
        ErrorKind::DeviceLost
    );
    allocation.free().unwrap();
    fixture.exhausted();
}

#[test]
fn concurrent_hardware_loss_observers_all_receive_the_sticky_loss() {
    let fixture = Fixture::new([]);
    fixture.hardware.store(2, Ordering::Release);
    std::thread::scope(|scope| {
        let observers = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    assert_eq!(
                        fixture.vm.check().unwrap_err().kind(),
                        ErrorKind::DeviceLost
                    );
                })
            })
            .collect::<Vec<_>>();
        for observer in observers {
            observer.join().unwrap();
        }
    });
    fixture.hardware.store(0, Ordering::Release);
    assert_eq!(
        fixture.vm.check().unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
}

#[test]
fn memory_fault_reaches_each_subscriber_without_inventing_device_loss() {
    let fixture = Fixture::new([]);
    fixture.vm.loss.register_identity(42, [0; 16]).unwrap();
    let mut first = fixture.vm.subscribe_events().unwrap();
    let mut second = fixture.vm.subscribe_events().unwrap();
    fixture.lost.store(true, Ordering::Release);
    let expected = DeviceEvent::GpuMemoryFault(GpuMemoryFault {
        endpoint_id: Some([0; 16]),
        virtual_address: 0x1234_5000,
        page_not_present: false,
        read_only: true,
        no_execute: false,
        imprecise: false,
        cause: GpuMemoryFaultCause::None,
    });
    assert_eq!(first.poll().unwrap(), Some(expected));
    assert_eq!(second.poll().unwrap(), Some(expected));
    assert_eq!(first.poll().unwrap(), None);
    assert_eq!(second.poll().unwrap(), None);
    assert!(fixture.vm.check().is_ok());
    assert!(!fixture.vm.has_latched_loss());
    fixture.lost.store(false, Ordering::Release);
}

#[test]
fn event_subscription_wait_uses_native_exception_wait() {
    let fixture = Fixture::new([]);
    let mut subscription = fixture.vm.subscribe_events().unwrap();
    assert_eq!(subscription.poll().unwrap(), None);
    subscription
        .wait_native(std::time::Duration::from_micros(100))
        .unwrap();
    assert_eq!(fixture.native_waits.load(Ordering::Relaxed), 1);
    fixture.exhausted();
}

#[test]
fn sequential_memory_faults_reach_each_subscriber_in_order() {
    let fixture = Fixture::new([]);
    fixture.vm.loss.register_identity(42, [42; 16]).unwrap();
    fixture.vm.loss.register_identity(77, [77; 16]).unwrap();
    let mut first = fixture.vm.subscribe_events().unwrap();
    let mut second = fixture.vm.subscribe_events().unwrap();

    fixture.lost.store(true, Ordering::Release);
    let Some(DeviceEvent::GpuMemoryFault(first_fault)) = first.poll().unwrap() else {
        panic!("first memory fault was not delivered");
    };
    assert_eq!(first_fault.endpoint_id, Some([42; 16]));

    fixture.fault_gpu_id.store(77, Ordering::Release);
    fixture.lost.store(true, Ordering::Release);
    let Some(DeviceEvent::GpuMemoryFault(second_fault)) = first.poll().unwrap() else {
        panic!("second memory fault was not delivered");
    };
    assert_eq!(second_fault.endpoint_id, Some([77; 16]));
    assert_eq!(
        second.poll().unwrap(),
        Some(DeviceEvent::GpuMemoryFault(first_fault))
    );
    assert_eq!(
        second.poll().unwrap(),
        Some(DeviceEvent::GpuMemoryFault(second_fault))
    );
    assert_eq!(first.poll().unwrap(), None);
    assert_eq!(second.poll().unwrap(), None);
}

#[test]
fn slow_subscriber_backs_up_native_polling_without_dropping_exceptions() {
    let fixture = Fixture::new([]);
    let mut slow = fixture.vm.subscribe_events().unwrap();
    let mut fast = fixture.vm.subscribe_events().unwrap();
    for _ in 0..MAX_RETAINED_EXCEPTIONS {
        fixture.hardware.store(1, Ordering::Release);
        assert!(matches!(
            fast.poll().unwrap(),
            Some(DeviceEvent::GpuHardwareException(_))
        ));
    }
    fixture.hardware.store(1, Ordering::Release);
    assert_eq!(fast.poll().unwrap_err().kind(), ErrorKind::Busy);
    assert!(matches!(
        slow.poll().unwrap(),
        Some(DeviceEvent::GpuHardwareException(_))
    ));
    assert!(matches!(
        fast.poll().unwrap(),
        Some(DeviceEvent::GpuHardwareException(_))
    ));
}

#[test]
fn operation_check_keeps_fault_details_for_late_subscribers() {
    let fixture = Fixture::new([]);
    fixture.vm.loss.register_identity(42, [42; 16]).unwrap();
    fixture.lost.store(true, Ordering::Release);
    assert!(fixture.vm.check().is_ok());
    fixture.lost.store(false, Ordering::Release);
    let mut late = fixture.vm.subscribe_events().unwrap();
    let Some(DeviceEvent::GpuMemoryFault(fault)) = late.poll().unwrap() else {
        panic!("late subscriber missed the memory fault");
    };
    assert_eq!(fault.endpoint_id, Some([42; 16]));
    assert_eq!(fault.virtual_address, 0x1234_5000);
    assert_eq!(late.poll().unwrap(), None);
}

#[test]
fn late_subscriber_receives_the_latest_fault_without_unbounded_history() {
    let fixture = Fixture::new([]);
    fixture.vm.loss.register_identity(42, [42; 16]).unwrap();
    fixture.vm.loss.register_identity(77, [77; 16]).unwrap();
    fixture.lost.store(true, Ordering::Release);
    assert!(fixture.vm.check().is_ok());
    fixture.fault_gpu_id.store(77, Ordering::Release);
    fixture.lost.store(true, Ordering::Release);
    assert!(fixture.vm.check().is_ok());
    assert_eq!(fixture.vm.loss.observed.lock().unwrap().events.len(), 1);

    let mut late = fixture.vm.subscribe_events().unwrap();
    let Some(DeviceEvent::GpuMemoryFault(fault)) = late.poll().unwrap() else {
        panic!("late subscriber missed the latest fault");
    };
    assert_eq!(fault.endpoint_id, Some([77; 16]));
    assert_eq!(late.poll().unwrap(), None);
}

#[test]
fn hardware_event_keeps_native_origin_before_endpoint_activation() {
    let fixture = Fixture::new([]);
    let origin = crate::device::event::DeviceEventOriginId::from_driver_key(42);
    assert_eq!(fixture.vm.event_origin_id(), origin);
    let mut early = fixture.vm.subscribe_events().unwrap();
    fixture.hardware.store(1, Ordering::Release);
    let Some(DeviceEvent::GpuHardwareException(unattributed)) = early.poll().unwrap() else {
        panic!("hardware exception was not delivered");
    };
    assert_eq!(unattributed.endpoint_id, None);
    assert_eq!(unattributed.origin_id, Some(origin));

    fixture.vm.loss.register_identity(42, [42; 16]).unwrap();
    let mut late = fixture.vm.subscribe_events().unwrap();
    let Some(DeviceEvent::GpuHardwareException(attributed)) = late.poll().unwrap() else {
        panic!("retained hardware exception was not replayed");
    };
    assert_eq!(attributed.endpoint_id, Some([42; 16]));
    assert_eq!(attributed.origin_id, Some(origin));
}

#[test]
fn later_hardware_loss_is_reported_after_a_nonterminal_exception() {
    let fixture = Fixture::new([]);
    fixture.vm.loss.register_identity(42, [42; 16]).unwrap();
    fixture.hardware.store(1, Ordering::Release);
    assert!(fixture.vm.check().is_ok());
    let mut subscriber = fixture.vm.subscribe_events().unwrap();
    assert_eq!(
        subscriber.poll().unwrap(),
        Some(DeviceEvent::GpuHardwareException(GpuHardwareException {
            endpoint_id: Some([42; 16]),
            origin_id: Some(crate::device::event::DeviceEventOriginId::from_driver_key(
                42
            )),
            scope: GpuResetScope::Other,
            memory_lost: false,
            cause: GpuResetCause::Ecc,
        }))
    );
    assert_eq!(subscriber.poll().unwrap(), None);
    assert!(!fixture.vm.has_latched_loss());
    fixture.hardware.store(2, Ordering::Release);
    assert_eq!(
        fixture.vm.check().unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
    assert_eq!(
        subscriber.poll().unwrap(),
        Some(DeviceEvent::GpuHardwareException(GpuHardwareException {
            endpoint_id: Some([42; 16]),
            origin_id: Some(crate::device::event::DeviceEventOriginId::from_driver_key(
                42
            )),
            scope: GpuResetScope::Other,
            memory_lost: true,
            cause: GpuResetCause::Ecc,
        }))
    );
    assert_eq!(subscriber.poll().unwrap(), None);
}

#[test]
fn failed_native_rearm_retries_without_replaying_the_record() {
    let fixture = Fixture::new([]);
    let mut subscriber = fixture.vm.subscribe_events().unwrap();
    fixture
        .hardware_reset_errno
        .store(errno::EIO as u32, Ordering::Release);
    fixture.hardware.store(1, Ordering::Release);
    assert!(matches!(
        subscriber.poll().unwrap(),
        Some(DeviceEvent::GpuHardwareException(_))
    ));
    assert_eq!(subscriber.poll().unwrap(), None);
    fixture.hardware.store(1, Ordering::Release);
    assert!(matches!(
        subscriber.poll().unwrap(),
        Some(DeviceEvent::GpuHardwareException(_))
    ));
}

#[test]
fn memory_fault_identity_uses_the_reported_gpu_across_a_shared_connection() {
    let fixture = Fixture::new([]);
    fixture.vm.loss.register_identity(42, [42; 16]).unwrap();
    fixture.vm.loss.register_identity(77, [77; 16]).unwrap();
    fixture.fault_gpu_id.store(77, Ordering::Release);
    fixture.lost.store(true, Ordering::Release);
    let mut subscriber = fixture.vm.subscribe_events().unwrap();
    let Some(DeviceEvent::GpuMemoryFault(fault)) = subscriber.poll().unwrap() else {
        panic!("subscriber missed the second GPU's memory fault");
    };
    assert_eq!(fault.endpoint_id, Some([77; 16]));
    assert!(fixture.vm.check().is_ok());
}

#[test]
fn concurrent_subscribers_receive_the_same_manual_reset_event_once() {
    let fixture = Fixture::new([]);
    fixture.vm.loss.register_identity(42, [42; 16]).unwrap();
    fixture.hardware.store(2, Ordering::Release);
    std::thread::scope(|scope| {
        let observers = (0..8)
            .map(|_| {
                let mut subscription = fixture.vm.subscribe_events().unwrap();
                scope.spawn(move || {
                    let Some(DeviceEvent::GpuHardwareException(exception)) =
                        subscription.poll().unwrap()
                    else {
                        panic!("concurrent subscriber missed the hardware exception");
                    };
                    assert_eq!(exception.endpoint_id, Some([42; 16]));
                    assert!(exception.memory_lost);
                    assert_eq!(subscription.poll().unwrap(), None);
                })
            })
            .collect::<Vec<_>>();
        for observer in observers {
            observer.join().unwrap();
        }
    });
    assert_eq!(
        fixture.vm.check().unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
}

#[test]
fn late_identity_registration_resolves_an_already_observed_source() {
    let fixture = Fixture::new([]);
    fixture.lost.store(true, Ordering::Release);
    assert!(fixture.vm.check().is_ok());
    fixture.vm.loss.register_identity(42, [42; 16]).unwrap();
    let mut subscription = fixture.vm.subscribe_events().unwrap();
    let Some(DeviceEvent::GpuMemoryFault(fault)) = subscription.poll().unwrap() else {
        panic!("late subscriber missed the memory fault");
    };
    assert_eq!(fault.endpoint_id, Some([42; 16]));
}

#[test]
fn terminal_hardware_loss_survives_a_second_event_poll_error() {
    let fixture = Fixture::new([]);
    fixture.hardware.store(2, Ordering::Release);
    fixture
        .memory_wait_errno
        .store(errno::EIO as u32, Ordering::Release);
    assert_eq!(
        fixture.vm.check().unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
    let mut subscription = fixture.vm.subscribe_events().unwrap();
    let Some(DeviceEvent::GpuHardwareException(exception)) = subscription.poll().unwrap() else {
        panic!("hardware exception was lost after the memory poll failed");
    };
    assert!(exception.memory_lost);
    assert_eq!(
        subscription
            .poll()
            .unwrap_err()
            .native_status()
            .and_then(crate::NativeStatus::errno_code),
        Some(errno::EIO)
    );
    fixture.memory_wait_errno.store(0, Ordering::Release);
    assert_eq!(subscription.poll().unwrap(), None);
}

#[test]
fn failed_memory_event_creation_reclaims_every_returned_event_id() {
    let destroyed = Arc::new(Mutex::new(Vec::new()));
    let observed = destroyed.clone();
    let kfd = shared(connection::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| match call {
            connection::Call::CreateEvent(args) if args.event_type == uapi::HW_EXCEPTION => {
                args.event_id = 19;
                Ok(())
            }
            connection::Call::CreateEvent(args) if args.event_type == uapi::MEMORY_EXCEPTION => {
                args.event_id = 20;
                Err(io::Error::from_raw_os_error(errno::EIO))
            }
            connection::Call::DestroyEvent(args) => {
                observed.lock().unwrap().push(args.event_id);
                Ok(())
            }
            _ => panic!("unexpected exception lifecycle operation"),
        }),
    ));
    let error = LossEvent::create(kfd).err().unwrap();
    assert_eq!(
        error
            .native_status()
            .and_then(crate::NativeStatus::errno_code),
        Some(errno::EIO)
    );
    assert_eq!(*destroyed.lock().unwrap(), [20, 19]);
}

#[test]
fn native_errors_keep_operation_and_errno_without_inventing_device_loss() {
    for code in [errno::EIO, errno::ENODEV, errno::EINVAL] {
        let error = native_error("native test", io::Error::from_raw_os_error(code));
        assert_eq!(error.kind(), ErrorKind::Driver);
        let Error::NativeOperation {
            operation, source, ..
        } = error
        else {
            panic!("lost native cause")
        };
        assert_eq!(operation, "native test");
        assert_eq!(source.raw_os_error(), Some(code));
    }
}

#[test]
fn warmed_native_address_and_allocation_limits_allocate_no_heap_storage() {
    let fixture = Fixture::new([
        Reply::Allocate(17, None),
        Reply::Map(0, 1, None),
        Reply::Unmap(0, 1, None),
        Reply::Free(None),
    ]);
    let mut allocation = fixture.create().unwrap();
    allocation.device_address(&fixture.vm).unwrap();
    let count = crate::test_support::allocation_counter::allocations(|| {
        for _ in 0..100 {
            assert!(allocation.device_address(&fixture.vm).is_ok());
            assert_eq!(allocation.cached_info().size, 65536);
        }
    });
    assert_eq!(count, 0);
    allocation.free().unwrap();
    fixture.exhausted();
}

fn initialization_endpoint(acquisitions: Arc<AtomicUsize>, busy: bool) -> Shared<connection::Kfd> {
    shared(connection::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| {
            match call {
                connection::Call::Version(args) => {
                    args.major = 1;
                    args.minor = 23;
                }
                connection::Call::Apertures(args, entries) => {
                    args.count = 2;
                    for (entry, gpu_id) in entries.iter_mut().zip([42, 43]) {
                        entry.gpu_id = gpu_id;
                        entry.gpuvm_base = 0x10000;
                        entry.gpuvm_limit = isize::MAX as u64;
                    }
                }
                connection::Call::CreateEvent(args) => {
                    args.event_id = match args.event_type {
                        uapi::HW_EXCEPTION => 19,
                        uapi::MEMORY_EXCEPTION => 20,
                        _ => panic!("unexpected exception event type"),
                    };
                }
                connection::Call::Wait(args, _) => args.result = uapi::WAIT_TIMEOUT,
                connection::Call::AcquireVm(args) => {
                    assert!(matches!(args.gpu_id, 42 | 43));
                    acquisitions.fetch_add(1, Ordering::Relaxed);
                    if busy {
                        return Err(io::Error::from_raw_os_error(errno::EBUSY));
                    }
                }
                connection::Call::RuntimeEnable(args) => {
                    assert!(matches!(args.mode_mask, 0 | 1));
                    assert_eq!(args.r_debug, 0);
                    assert_eq!(args.capabilities_mask, 0);
                }
                connection::Call::DestroyEvent(_) => {}
                _ => panic!("unexpected initialization ioctl"),
            }
            Ok(())
        }),
    ))
}

#[test]
fn concurrent_initialization_keeps_the_same_render_file_after_public_owners_drop() {
    let bindings = VmBindings::new();
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let kfd = initialization_endpoint(acquisitions.clone(), false);
    let native = node();
    let owners = std::thread::scope(|scope| {
        (0..8)
            .map(|_| {
                scope.spawn(|| {
                    bindings
                        .device_with_render(&kfd, &native, None, None, |minor| {
                            assert_eq!(minor, 128);
                            File::open("/dev/null")
                        })
                        .unwrap()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    for owner in &owners {
        assert!(Shared::ptr_eq(owner, &owners[0]));
    }
    let retained_address = std::ptr::from_ref(&*owners[0]);
    drop(owners);
    drop(kfd);
    assert_eq!(bindings.bindings.lock().unwrap().devices.len(), 1);
    let fresh = initialization_endpoint(acquisitions.clone(), false);
    let vm = bindings
        .device_with_render(&fresh, &native, None, None, |_| {
            panic!("reopened a bound render file")
        })
        .unwrap();
    assert_eq!(acquisitions.load(Ordering::Relaxed), 1);
    assert_eq!(std::ptr::from_ref(&*vm), retained_address);
    let mut replacement = native;
    replacement.unique_id = Some(999);
    assert_eq!(
        bindings
            .device_with_render(&fresh, &replacement, None, None, |_| panic!(
                "opened replaced device"
            ))
            .err()
            .unwrap()
            .kind(),
        ErrorKind::DeviceLost
    );
}

#[test]
fn existing_foreign_vm_is_busy_and_fork_rejection_precedes_the_inherited_lock() {
    let bindings = VmBindings::new();
    let kfd = initialization_endpoint(Arc::new(AtomicUsize::new(0)), true);
    assert_eq!(
        bindings
            .device_with_render(&kfd, &node(), None, None, |_| File::open("/dev/null"))
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Busy
    );
    assert!(bindings.bindings.lock().unwrap().devices.is_empty());
    bindings
        .process
        .store(std::process::id().wrapping_add(1), Ordering::Relaxed);
    let held = bindings.bindings.lock().unwrap();
    assert_eq!(
        bindings
            .device_with_render(&kfd, &node(), None, None, |_| panic!(
                "fork child opened render node"
            ))
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Unsupported
    );
    drop(held);
}

#[test]
fn retained_primary_vm_cannot_reactivate_after_sticky_device_loss() {
    let fixture = Fixture::new([]);
    let bindings = VmBindings::new();
    bindings
        .bindings
        .lock()
        .unwrap()
        .devices
        .push(fixture.vm.clone());
    fixture.hardware.store(2, Ordering::Release);
    for _ in 0..2 {
        assert_eq!(
            bindings
                .device_with_render(&fixture.vm.kfd_owner(), &node(), None, None, |_| {
                    panic!("device loss reopened an existing render endpoint")
                })
                .err()
                .unwrap()
                .kind(),
            ErrorKind::DeviceLost
        );
        // A later native wait reporting no new event must not clear the latch.
        fixture.hardware.store(0, Ordering::Release);
    }
    fixture.exhausted();
}

#[test]
fn runtime_reenable_reuses_exact_vm_and_allows_late_distinct_device_activation() {
    let mut bindings = VmBindings::new();
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let mut kfd = initialization_endpoint(acquisitions.clone(), false);
    let native = node();
    let first = bindings
        .device_with_render(&kfd, &native, None, None, |_| File::open("/dev/null"))
        .unwrap();
    kfd.disable_runtime().unwrap();
    let recreated = bindings
        .device_with_render(&kfd, &native, None, None, |_| {
            panic!("reopened the bound device")
        })
        .unwrap();
    let second = sysfs::KfdNode {
        debug: sysfs::KfdDebugProperties::default(),
        host_atomics: false,
        node: 2,
        gpu_id: 43,
        render_minor: Some(129),
        unique_id: Some(456),
        identity: [1; 16],
        ..native
    };
    let second = bindings
        .device_with_render(&kfd, &second, None, None, |minor| {
            assert_eq!(minor, 129);
            File::open("/dev/null")
        })
        .unwrap();
    assert!(Shared::ptr_eq(&first, &recreated));
    assert!(!Shared::ptr_eq(&first, &second));
    assert_eq!(first.gpu_id, 42);
    assert_eq!(second.gpu_id, 43);
    assert_eq!(acquisitions.load(Ordering::Relaxed), 2);
    drop(first);
    drop(recreated);
    drop(second);
    assert_eq!(bindings.bindings.lock().unwrap().devices.len(), 2);
    bindings.shutdown().unwrap();
    Shared::get_mut(&mut kfd).unwrap().close().unwrap();
}

#[test]
fn instance_shutdown_quarantines_ambiguous_event_destroy_without_replaying_id() {
    let destroys = Arc::new(AtomicUsize::new(0));
    let observed = destroys.clone();
    let kfd = shared(connection::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| {
            if let connection::Call::DestroyEvent(_) = call {
                let attempt = observed.fetch_add(1, Ordering::Relaxed);
                if attempt == 0 {
                    Err(io::Error::from_raw_os_error(errno::EIO))
                } else {
                    Ok(())
                }
            } else {
                panic!("unexpected shutdown ioctl")
            }
        }),
    ));
    let mut bindings = VmBindings::new();
    bindings.bindings.get_mut().unwrap().loss = Some(shared(LossEvent {
        kfd,
        hardware_event_id: AtomicU32::new(19),
        memory_event_id: AtomicU32::new(20),
        hardware_destroy_uncertain: AtomicBool::new(false),
        memory_destroy_uncertain: AtomicBool::new(false),
        observed: Mutex::new(ObservedEvents::new().unwrap()),
        host_aliases: Mutex::new(Vec::new()),
        svm_registrations: Mutex::new(registered_svm::Registry::default()),
        lost: AtomicBool::new(false),
    }));
    assert_eq!(
        bindings
            .shutdown()
            .unwrap_err()
            .native_status()
            .and_then(crate::NativeStatus::errno_code),
        Some(errno::EIO)
    );
    assert!(bindings.bindings.get_mut().unwrap().loss.is_some());
    let loss = bindings.bindings.get_mut().unwrap().loss.as_ref().unwrap();
    assert_eq!(loss.memory_event_id.load(Ordering::Relaxed), 20);
    assert!(loss.memory_destroy_uncertain.load(Ordering::Relaxed));
    assert_eq!(loss.hardware_event_id.load(Ordering::Relaxed), 0);
    assert_eq!(
        bindings.shutdown().unwrap_err().kind(),
        ErrorKind::DriverContract
    );
    assert!(bindings.bindings.get_mut().unwrap().loss.is_some());
    assert_eq!(destroys.load(Ordering::Relaxed), 2);
}

const SVM_FINE_GRANT: &[(u32, u32)] = &[
    (
        uapi::SVM_ATTR_SET_FLAGS,
        uapi::SVM_FLAG_HOST_ACCESS | uapi::SVM_FLAG_COHERENT,
    ),
    (
        uapi::SVM_ATTR_CLEAR_FLAGS,
        uapi::SVM_FLAG_EXT_COHERENT | uapi::SVM_FLAG_GPU_READ_ONLY | uapi::SVM_FLAG_GPU_EXECUTE,
    ),
    (uapi::SVM_ATTR_ACCESS_IN_PLACE, 42),
];
const SVM_REVOKE: &[(u32, u32)] = &[(uapi::SVM_ATTR_NO_ACCESS, 42)];

fn svm_registration(
    fixture: &Fixture,
    address: usize,
    size: u64,
    backing: Option<crate::memory::host::HostBackingLease>,
) -> Result<Owned<KfdRegisteredHost>, Error> {
    KfdRegisteredHost::create(
        fixture.vm.clone(),
        std::iter::empty(),
        AllocationDesc {
            size,
            alignment: 4096,
        },
        address,
        HostMappingPolicy::Fine,
        DeviceAccess::READ | DeviceAccess::WRITE,
        backing,
    )
}

#[test]
fn svm_host_registration_reuses_advised_address_without_userptr_bo() {
    let fixture = Fixture::new([
        Reply::Svm(0x12000, 0x2000, SVM_FINE_GRANT, None),
        Reply::Svm(
            0x12000,
            0x2000,
            &[(uapi::SVM_ATTR_SET_FLAGS, uapi::SVM_FLAG_GPU_READ_MOSTLY)],
            None,
        ),
        Reply::Svm(0x12000, 0x2000, SVM_REVOKE, None),
        Reply::Svm(0x12000, 0x2000, SVM_FINE_GRANT, None),
        Reply::Svm(0x12000, 0x2000, SVM_REVOKE, None),
    ]);
    let mut first = svm_registration(&fixture, 0x12345, 4096, None).unwrap();
    let info = first.cached_info();
    assert_eq!(info.device_address, 0x12345);
    assert_eq!(info.host_address, Some(0x12345));
    assert_eq!(info.size, 4096);
    assert_eq!(info.native_size, 8192);
    fixture
        .vm
        .kfd()
        .svm_attributes(
            0x12000,
            0x2000,
            uapi::SVM_OP_SET_ATTR,
            &mut [uapi::SvmAttribute {
                attribute_type: uapi::SVM_ATTR_SET_FLAGS,
                value: uapi::SVM_FLAG_GPU_READ_MOSTLY,
            }],
        )
        .unwrap();
    first.free().unwrap();
    first.free().unwrap();
    let mut second = svm_registration(&fixture, 0x12345, 4096, None).unwrap();
    second.free().unwrap();
    fixture.exhausted();
}

#[test]
fn svm_host_unregister_preserves_shared_page_cover_and_gpu_grants() {
    let fixture = Fixture::new([
        Reply::Svm(0x10000, 0x4000, SVM_FINE_GRANT, None),
        Reply::Svm(0x11000, 0x2000, SVM_FINE_GRANT, None),
        Reply::Svm(0x10000, 0x1000, SVM_REVOKE, None),
        Reply::Svm(0x13000, 0x1000, SVM_REVOKE, None),
        Reply::Svm(0x11000, 0x2000, SVM_REVOKE, None),
    ]);
    let mut outer = svm_registration(&fixture, 0x10000, 0x4000, None).unwrap();
    let mut inner = svm_registration(&fixture, 0x11010, 0x1000, None).unwrap();
    outer.free().unwrap();
    assert_eq!(inner.device_address(&fixture.vm).unwrap(), 0x11010);
    inner.free().unwrap();
    fixture.exhausted();
}

#[test]
fn svm_host_unregister_retry_accounts_for_new_registration() {
    let fixture = Fixture::new([
        Reply::Svm(0x10000, 0x3000, SVM_FINE_GRANT, None),
        Reply::Svm(0x10000, 0x3000, SVM_REVOKE, Some(errno::EIO)),
        Reply::Svm(0x11000, 0x1000, SVM_FINE_GRANT, None),
        Reply::Svm(0x10000, 0x1000, SVM_REVOKE, None),
        Reply::Svm(0x12000, 0x1000, SVM_REVOKE, None),
        Reply::Svm(0x11000, 0x1000, SVM_REVOKE, None),
    ]);
    let mut first = svm_registration(&fixture, 0x10000, 0x3000, None).unwrap();
    assert!(first.free().is_err());
    assert!(first.device_address(&fixture.vm).is_err());
    let mut newer = svm_registration(&fixture, 0x11000, 0x1000, None).unwrap();
    first.free().unwrap();
    assert_eq!(newer.device_address(&fixture.vm).unwrap(), 0x11000);
    newer.free().unwrap();
    fixture.exhausted();
}

#[test]
fn svm_host_registration_rejects_conflicting_page_policy_before_native_changes() {
    let fixture = Fixture::new([
        Reply::Svm(0x10000, 0x1000, SVM_FINE_GRANT, None),
        Reply::Svm(0x10000, 0x1000, SVM_REVOKE, None),
    ]);
    let mut first = svm_registration(&fixture, 0x10000, 16, None).unwrap();
    let result = KfdRegisteredHost::create(
        fixture.vm.clone(),
        std::iter::empty(),
        AllocationDesc {
            size: 16,
            alignment: 4096,
        },
        0x10100,
        HostMappingPolicy::Coarse,
        DeviceAccess::READ | DeviceAccess::WRITE,
        None,
    );
    assert_eq!(result.err().unwrap().kind(), ErrorKind::Unsupported);
    first.free().unwrap();
    fixture.exhausted();
}

#[test]
fn svm_host_access_replacement_retains_both_device_sets_until_retry() {
    let fixture = Fixture::new([
        Reply::Svm(0x10000, 0x1000, SVM_FINE_GRANT, None),
        Reply::Svm(
            0x10000,
            0x1000,
            &[(uapi::SVM_ATTR_ACCESS_IN_PLACE, 43)],
            None,
        ),
        Reply::Svm(0x10000, 0x1000, SVM_REVOKE, Some(errno::EIO)),
        Reply::Svm(0x10000, 0x1000, SVM_REVOKE, None),
        Reply::Svm(0x10000, 0x1000, &[(uapi::SVM_ATTR_NO_ACCESS, 43)], None),
    ]);
    let peer = fixture.peer(43, 0x10000, isize::MAX as u64);
    let mut allocation = svm_registration(&fixture, 0x10000, 0x1000, None).unwrap();
    assert!(allocation.set_access(&[&peer]).is_err());
    assert!(allocation.device_address(&fixture.vm).is_err());
    assert!(allocation.device_address(&peer).is_err());
    assert_eq!(
        allocation.set_access(&[&fixture.vm]).unwrap_err().kind(),
        ErrorKind::Busy
    );
    allocation.set_access(&[&peer]).unwrap();
    assert!(allocation.device_address(&fixture.vm).is_err());
    assert_eq!(allocation.device_address(&peer).unwrap(), 0x10000);
    allocation.free().unwrap();
    fixture.exhausted();
}

#[test]
fn svm_host_failed_grant_runs_revoke_before_returning_backing() {
    let fixture = Fixture::new([
        Reply::Svm(0x10000, 0x1000, SVM_FINE_GRANT, Some(errno::EIO)),
        Reply::Svm(0x10000, 0x1000, SVM_REVOKE, None),
    ]);
    assert!(svm_registration(&fixture, 0x10000, 0x1000, None).is_err());
    fixture.exhausted();
}

#[test]
fn svm_host_release_failure_retains_owned_page_lease_until_retry() {
    let mut backing = HostAllocation::allocate(4096, 4096).unwrap();
    let address = backing.info().host_address;
    let fixture = Fixture::new([
        Reply::Svm(address as u64, 4096, SVM_FINE_GRANT, None),
        Reply::Svm(address as u64, 4096, SVM_REVOKE, Some(errno::EIO)),
        Reply::Svm(address as u64, 4096, SVM_REVOKE, None),
    ]);
    let mut registration = svm_registration(
        &fixture,
        address,
        4096,
        Some(backing.registration_lease().unwrap()),
    )
    .unwrap();
    assert!(registration.free().is_err());
    assert_eq!(backing.free().unwrap_err().kind(), ErrorKind::Busy);
    registration.free().unwrap();
    backing.free().unwrap();
    fixture.exhausted();
}

#[test]
#[allow(unsafe_code)]
fn svm_host_process_registration_routes_to_svm_and_does_not_retry_userptr() {
    let fixture = Fixture::new([
        Reply::Userptr(0x12000, 0x2000, 0, Some(libc::EADDRINUSE)),
        Reply::Svm(0x12000, 0x2000, SVM_FINE_GRANT, None),
        Reply::Svm(0x12000, 0x2000, SVM_REVOKE, None),
        Reply::Userptr(0x12000, 0x2000, 0, Some(libc::EADDRINUSE)),
        Reply::Svm(0x12000, 0x2000, SVM_FINE_GRANT, Some(errno::EIO)),
        Reply::Svm(0x12000, 0x2000, SVM_REVOKE, None),
    ]);
    let driver = super::super::LinuxKfdDriver::with_context_lifetime(
        crate::session::DriverContextLifetime::Process,
        crate::os::ProcessIdentity::capture().unwrap(),
    );
    let device = super::super::DeviceState {
        vm: fixture.vm.clone(),
        native: node(),
        asic_family_id: 0,
        gpu_counter_frequency_hz: 0,
    };
    let request = || crate::memory::HostRegistration {
        address: 0x12345,
        policy: HostMappingPolicy::Fine,
        size: 4096,
        alignment: 4096,
        permissions: DeviceAccess::READ | DeviceAccess::WRITE,
        backing: None,
    };
    // SAFETY: The scripted native backend never dereferences the synthetic pages.
    let mut registration = unsafe { driver.register_host(&device, &[], request()) }.unwrap();
    assert_eq!(registration.cached_info().device_address, 0x12345);
    registration.free().unwrap();
    // SAFETY: The script models a partial native grant followed by revocation.
    assert!(unsafe { driver.register_host(&device, &[], request()) }.is_err());
    fixture.exhausted();
}

#[test]
fn svm_host_unregister_preserves_overlap_only_for_the_registered_gpu() {
    let fixture = Fixture::new([
        Reply::Svm(
            0x10000,
            0x4000,
            &[
                (
                    uapi::SVM_ATTR_SET_FLAGS,
                    uapi::SVM_FLAG_HOST_ACCESS | uapi::SVM_FLAG_COHERENT,
                ),
                (
                    uapi::SVM_ATTR_CLEAR_FLAGS,
                    uapi::SVM_FLAG_EXT_COHERENT
                        | uapi::SVM_FLAG_GPU_READ_ONLY
                        | uapi::SVM_FLAG_GPU_EXECUTE,
                ),
                (uapi::SVM_ATTR_ACCESS_IN_PLACE, 42),
                (uapi::SVM_ATTR_ACCESS_IN_PLACE, 43),
            ],
            None,
        ),
        Reply::Svm(0x11000, 0x2000, SVM_FINE_GRANT, None),
        Reply::Svm(0x10000, 0x1000, SVM_REVOKE, None),
        Reply::Svm(0x13000, 0x1000, SVM_REVOKE, None),
        Reply::Svm(0x10000, 0x4000, &[(uapi::SVM_ATTR_NO_ACCESS, 43)], None),
        Reply::Svm(0x11000, 0x2000, SVM_REVOKE, None),
    ]);
    let peer = fixture.peer(43, 0x10000, isize::MAX as u64);
    let mut outer = KfdRegisteredHost::create(
        fixture.vm.clone(),
        std::iter::once(peer),
        AllocationDesc {
            size: 0x4000,
            alignment: 4096,
        },
        0x10000,
        HostMappingPolicy::Fine,
        DeviceAccess::READ | DeviceAccess::WRITE,
        None,
    )
    .unwrap();
    let mut inner = svm_registration(&fixture, 0x11000, 0x2000, None).unwrap();
    outer.free().unwrap();
    assert_eq!(inner.device_address(&fixture.vm).unwrap(), 0x11000);
    inner.free().unwrap();
    fixture.exhausted();
}

#[test]
#[allow(unsafe_code)]
fn svm_host_fallback_keeps_the_original_owned_backing_lease() {
    let mut backing = HostAllocation::allocate(4096, 4096).unwrap();
    let address = backing.info().host_address;
    let fixture = Fixture::new([
        Reply::Userptr(address as u64, 4096, 0, Some(libc::EADDRINUSE)),
        Reply::Svm(address as u64, 4096, SVM_FINE_GRANT, None),
        Reply::Svm(address as u64, 4096, SVM_REVOKE, None),
    ]);
    let driver = super::super::LinuxKfdDriver::with_context_lifetime(
        crate::session::DriverContextLifetime::Process,
        crate::os::ProcessIdentity::capture().unwrap(),
    );
    let device = super::super::DeviceState {
        vm: fixture.vm.clone(),
        native: node(),
        asic_family_id: 0,
        gpu_counter_frequency_hz: 0,
    };
    // SAFETY: The owned lease retains the live pages throughout both mechanisms.
    let mut registration = unsafe {
        driver.register_host(
            &device,
            &[],
            crate::memory::HostRegistration {
                address,
                policy: HostMappingPolicy::Fine,
                size: 4096,
                alignment: 4096,
                permissions: DeviceAccess::READ | DeviceAccess::WRITE,
                backing: Some(backing.registration_lease().unwrap()),
            },
        )
    }
    .unwrap();
    assert_eq!(backing.free().unwrap_err().kind(), ErrorKind::Busy);
    registration.free().unwrap();
    backing.free().unwrap();
    fixture.exhausted();
}

#[test]
#[allow(unsafe_code)]
fn svm_host_fallback_rejects_errors_with_uncertain_or_returned_native_objects() {
    for (handle, errno) in [(0, errno::EFAULT), (17, libc::EADDRINUSE)] {
        let mut replies = vec![Reply::Userptr(0x12000, 0x2000, handle, Some(errno))];
        if handle != 0 {
            replies.push(Reply::Free(None));
        }
        let fixture = Fixture::new(replies);
        let driver = super::super::LinuxKfdDriver::with_context_lifetime(
            crate::session::DriverContextLifetime::Process,
            crate::os::ProcessIdentity::capture().unwrap(),
        );
        let device = super::super::DeviceState {
            vm: fixture.vm.clone(),
            native: node(),
            asic_family_id: 0,
            gpu_counter_frequency_hz: 0,
        };
        // SAFETY: The script models output ambiguity without native page access.
        let result = unsafe {
            driver.register_host(
                &device,
                &[],
                crate::memory::HostRegistration {
                    address: 0x12345,
                    policy: HostMappingPolicy::Fine,
                    size: 4096,
                    alignment: 4096,
                    permissions: DeviceAccess::READ | DeviceAccess::WRITE,
                    backing: None,
                },
            )
        };
        assert!(result.is_err());
        fixture.exhausted();
    }
}
