//! Native ownership tests use real anonymous reservations and scripted KFD
//! replies. Scripts model errno and output progress independently, including
//! failures after the kernel has finished every per-device mapping operation.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::driver::AllocationDriver;
use crate::host_storage::Allocator;
use crate::memory::MemoryKind;
use std::os::fd::{AsRawFd, IntoRawFd};
use std::os::unix::fs::FileExt;
use std::sync::Arc;
fn shared<T>(value: T) -> Shared<T> {
    Shared::new(value, Allocator::default()).unwrap()
}
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize};

enum Reply {
    SetScratch(Option<i32>),
    Allocate(u64, Option<i32>),
    DmaBufInfo(u64, u32, u32, &'static [u8], Option<i32>),
    ImportDmaBuf(u64, u32, Option<i32>),
    ExportDmaBuf(i32, Option<i32>),
    IpcImport(u64, u64, u32, Option<i32>),
    IpcExport([u32; 4], Option<i32>),
    Map(u32, u32, Option<i32>),
    Unmap(u32, u32, Option<i32>),
    Free(Option<i32>),
}

struct Fixture {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    lost: Arc<AtomicBool>,
    scratch_base: Arc<AtomicU64>,
    vm: Shared<DeviceVm>,
}

fn node() -> sysfs::NativeNode {
    sysfs::NativeNode {
        node: 1,
        gpu_id: 42,
        render_minor: Some(128),
        unique_id: Some(123),
        identity: [0; 16],
        queues: sysfs::NativeQueueProperties::default(),
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
    Mutex::new(ScratchPool::new(
        sysfs::NativeQueueProperties {
            gfx_target: 120_001,
            xcc_count: 1,
            ..sysfs::NativeQueueProperties::default()
        },
        Allocator::default(),
    ))
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
    ) -> Self {
        let replies = Arc::new(Mutex::new(replies.into_iter().collect::<VecDeque<_>>()));
        let pending = replies.clone();
        let expected_devices = expected_devices.to_vec();
        let lost = Arc::new(AtomicBool::new(false));
        let loss_signal = lost.clone();
        let scratch_base = Arc::new(AtomicU64::new(0));
        let observed_scratch_base = scratch_base.clone();
        let kfd = shared(sys::Kfd::with_hook(
            File::open("/dev/null").unwrap(),
            Arc::new(move |call| {
                match call {
                    sys::Call::Wait(args, event) if event.event_id == 19 => {
                        args.result = uapi::WAIT_TIMEOUT;
                        return Ok(());
                    }
                    sys::Call::Wait(args, event) if event.event_id == 20 => {
                        if loss_signal.load(Ordering::Relaxed) {
                            event.payload[0] = u64::from_ne_bytes([0, 0, 0, 0, 1, 0, 0, 0]);
                            event.payload[2] = 0x1234_5000;
                            event.payload[3] = u64::from_ne_bytes([42, 0, 0, 0, 0, 0, 0, 0]);
                            args.result = uapi::WAIT_COMPLETE;
                        } else {
                            args.result = uapi::WAIT_TIMEOUT;
                        }
                        return Ok(());
                    }
                    sys::Call::DestroyEvent(_) => return Ok(()),
                    _ => {}
                }
                let reply = pending
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("unexpected native call");
                let errno = match (call, reply) {
                    (sys::Call::SetScratchBackingVa(args), Reply::SetScratch(errno)) => {
                        assert_eq!(args.gpu_id, 42);
                        assert_eq!(args.pad, 0);
                        assert_ne!(args.va_address, 0);
                        observed_scratch_base.store(args.va_address << 16, Ordering::Relaxed);
                        errno
                    }
                    (sys::Call::Allocate(args), Reply::Allocate(handle, errno)) => {
                        assert_eq!(args.va % 65536, 0);
                        assert_eq!(args.size, 65536);
                        assert_eq!(args.gpu_id, 42);
                        assert_eq!(args.flags, expected_flags);
                        let expected_mmap_offset = if expected_flags & uapi::USERPTR == 0 {
                            0
                        } else if owned_userptr {
                            args.va
                        } else {
                            0x12000
                        };
                        assert_eq!(args.mmap_offset, expected_mmap_offset);
                        args.handle = handle;
                        errno
                    }
                    (
                        sys::Call::DmaBufInfo(args, metadata),
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
                            return Err(io::Error::from_raw_os_error(22));
                        }
                        assert_eq!(metadata.len(), expected_metadata.len());
                        metadata.copy_from_slice(expected_metadata);
                        args.size = size;
                        args.gpu_id = gpu_id;
                        args.flags = flags;
                        args.metadata_size = u32::try_from(expected_metadata.len()).unwrap();
                        errno
                    }
                    (sys::Call::ImportDmaBuf(args), Reply::ImportDmaBuf(handle, gpu_id, errno)) => {
                        assert_eq!(args.va % 65536, 0);
                        assert_eq!(args.gpu_id, gpu_id);
                        assert!(i32::try_from(args.descriptor).is_ok());
                        args.handle = handle;
                        errno
                    }
                    (sys::Call::ExportDmaBuf(args), Reply::ExportDmaBuf(fd, errno)) => {
                        assert_eq!(args.handle, 17);
                        args.descriptor = u32::try_from(fd).unwrap();
                        errno
                    }
                    (
                        sys::Call::IpcImportHandle(args),
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
                    (sys::Call::IpcExportHandle(args), Reply::IpcExport(handle, errno)) => {
                        assert_eq!(args.handle, 17);
                        assert_eq!(args.gpu_id, 42);
                        assert_eq!(args.flags, expected_flags);
                        args.share_handle = handle;
                        errno
                    }
                    (sys::Call::Map(args, devices), Reply::Map(before, after, errno))
                    | (sys::Call::Unmap(args, devices), Reply::Unmap(before, after, errno)) => {
                        assert_eq!(*devices, expected_devices.as_slice());
                        assert_eq!(args.handle, 17);
                        assert_eq!(args.success, before);
                        args.success = after;
                        errno
                    }
                    (sys::Call::Free(args), Reply::Free(errno)) => {
                        assert_eq!(args.handle, 17);
                        errno
                    }
                    _ => panic!("cleanup issued the wrong native operation"),
                };
                errno.map_or(Ok(()), |errno| Err(io::Error::from_raw_os_error(errno)))
            }),
        ));
        let vm = shared(DeviceVm {
            loss: shared(LossEvent {
                kfd,
                hardware_event_id: AtomicU32::new(19),
                memory_event_id: AtomicU32::new(20),
                hardware_destroy_uncertain: AtomicBool::new(false),
                memory_destroy_uncertain: AtomicBool::new(false),
                memory_event_claimed: Mutex::new(false),
                lost: AtomicBool::new(false),
            }),
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
            scratch: scratch_pool(),
            vmem: Mutex::new(super::super::vmem::VmState::new(Allocator::default())),
            version: uapi::Version {
                major: 1,
                minor: 23,
            },
            doorbells: super::super::queue::Doorbells::default(),
        });
        Self {
            replies,
            lost,
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
            scratch: scratch_pool(),
            vmem: Mutex::new(super::super::vmem::VmState::new(Allocator::default())),
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
                uncached: false,
                contiguous: false,
            },
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
    }

    fn allocate(
        &self,
        kind: MemoryKind,
        permissions: DeviceAccess,
    ) -> Result<Owned<KfdAllocation>, Error> {
        let kind = match kind {
            MemoryKind::System => BufferKind::Gtt,
            MemoryKind::OwnedHost => BufferKind::OwnedUserptr { uncached: false },
            MemoryKind::RegisteredHost { address, uncached } => {
                BufferKind::Userptr { address, uncached }
            }
            MemoryKind::DeviceLocal {
                host_visible,
                coherent,
                uncached,
                contiguous,
            } => BufferKind::Vram {
                public: host_visible,
                coherent,
                uncached,
                contiguous,
            },
        };
        KfdAllocation::create(self.vm.clone(), desc(), kind, permissions)
    }

    fn allocate_with_lifetime(
        &self,
        lifetime: crate::session::SessionLifetime,
        kind: MemoryKind,
        permissions: DeviceAccess,
    ) -> Result<Owned<super::super::NativeAllocation>, Error> {
        let device = super::super::DeviceState {
            vm: self.vm.clone(),
            native: sysfs::NativeNode {
                public_memory_bytes: 1 << 28,
                ..node()
            },
            lifetime,
        };
        super::super::LinuxKfdDriver::new(Allocator::default()).allocate(
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
fn secondary_context_rejects_owned_userptr_before_native_allocation() {
    let fixture = Fixture::new([]);
    assert_eq!(
        fixture
            .allocate_with_lifetime(
                crate::session::SessionLifetime::Session,
                MemoryKind::OwnedHost,
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
fn failed_secondary_drm_registration_releases_its_vm_dependency() {
    let fixture = Fixture::new([]);
    let result = super::super::registered_host::DrmRegisteredHost::create(
        fixture.vm.clone(),
        std::iter::empty(),
        desc(),
        0x10000,
        DeviceAccess::READ | DeviceAccess::WRITE,
        false,
    );
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
        allocation.export_ipc_memory().unwrap().words(),
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
    assert_eq!(scratch.ranges[0].size, GFX12_SCRATCH_BYTES_PER_XCC);
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
    fixture.lost.store(true, Ordering::Relaxed);
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
            uncached: false,
            contiguous: false,
        },
        DeviceAccess::READ | DeviceAccess::WRITE,
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
        scratch: scratch_pool(),
        vmem: Mutex::new(super::super::vmem::VmState::new(Allocator::default())),
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
            Reply::Map(0, 1, Some(4)),
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
            Reply::Unmap(0, 1, Some(4)),
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
        [Reply::Allocate(17, None), Reply::Map(0, 1, Some(14))],
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
                MemoryKind::OwnedHost,
                uapi::USERPTR | uapi::COHERENT | uapi::NO_SUBSTITUTE,
            ),
            (
                MemoryKind::DeviceLocal {
                    host_visible: true,
                    coherent: false,
                    uncached: false,
                    contiguous: false,
                },
                uapi::VRAM | uapi::NO_SUBSTITUTE | uapi::PUBLIC,
            ),
        ] {
            // Verify exact flags before an injected allocation failure, without
            // needing a real render device for these CPU-visible placements.
            let fixture = if kind == MemoryKind::OwnedHost {
                Fixture::with_owned_userptr([Reply::Allocate(0, Some(12))], placement | flags)
            } else {
                Fixture::with_flags([Reply::Allocate(0, Some(12))], placement | flags)
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
    let identity = util::dma_buf_file_info(&file).unwrap().physical_id;
    let exported = util::duplicate_file(file.as_raw_fd())
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
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
}

#[test]
fn dma_buf_import_validates_permissions_origin_placement_and_range() {
    let file = backing_file(2 * 65536);
    let descriptor = file.as_raw_fd();
    let fixture = Fixture::new([]);
    assert_eq!(
        fixture
            .import(
                descriptor,
                0,
                4096,
                4096,
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
                4096,
                4096,
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidArgument
    );
    fixture.exhausted();

    for (gpu_id, flags, offset, length, expected) in [
        (53, uapi::GTT, 0, 4096, ErrorKind::Unsupported),
        (42, uapi::VRAM, 0, 4096, ErrorKind::Unsupported),
        (42, uapi::GTT | (1 << 12), 0, 4096, ErrorKind::Unsupported),
        (42, uapi::GTT, 2 * 65536, 1, ErrorKind::InvalidArgument),
    ] {
        let fixture = Fixture::new([Reply::DmaBufInfo(2 * 65536, gpu_id, flags, &[], None)]);
        assert_eq!(
            fixture
                .import(
                    descriptor,
                    offset,
                    length,
                    4096,
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
        Reply::ImportDmaBuf(17, 42, Some(12)),
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
        Reply::ImportDmaBuf(17, 42, Some(14)),
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
            MemoryKind::RegisteredHost {
                address: 0x12345,
                uncached: false,
            },
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .unwrap();
    let reservation = allocation.reservation.as_ref().unwrap().address();
    let info = allocation.cached_info();
    assert_eq!(info.host_address, Some(0x12345));
    assert_eq!(info.device_address, reservation as u64 + 0x345);
    assert_eq!(info.size, 65536);
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
fn registered_uncached_host_pages_reach_kfd_without_coherent_caching() {
    let fixture = Fixture::with_flags(
        [
            Reply::Allocate(17, None),
            Reply::Map(0, 1, None),
            Reply::Unmap(0, 1, None),
            Reply::Free(None),
        ],
        uapi::USERPTR | uapi::UNCACHED | uapi::NO_SUBSTITUTE | uapi::WRITABLE,
    );
    let mut allocation = fixture
        .allocate(
            MemoryKind::RegisteredHost {
                address: 0x12345,
                uncached: true,
            },
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .unwrap();
    allocation.free().unwrap();
    drop(allocation);
    fixture.exhausted();
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
            MemoryKind::OwnedHost,
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
fn invalid_registered_host_address_fails_before_native_observation() {
    let fixture = Fixture::new([]);
    fixture.lost.store(true, Ordering::Relaxed);
    assert_eq!(
        fixture
            .allocate(
                MemoryKind::RegisteredHost {
                    address: 0,
                    uncached: false,
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
        fixture.lost.store(true, Ordering::Relaxed);
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
            Reply::Map(0, prefix, Some(4)),
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
            Reply::Unmap(0, prefix, Some(4)),
            Reply::Unmap(prefix, 1, None),
            Reply::Free(Some(16)),
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
        allocation
            .reservation
            .as_mut()
            .unwrap()
            .fail_release_once(12);
        assert_eq!(
            allocation.free().unwrap_err().kind(),
            ErrorKind::ResourceExhausted
        );
        assert!(
            allocation.handle.is_none(),
            "successful FREE must not be replayed"
        );
        allocation.free().unwrap();
        allocation.free().unwrap();
        drop(allocation);
        fixture.exhausted();
    }
}

#[test]
fn failed_creation_owns_returned_handles_and_uncertain_native_results() {
    let fixture = Fixture::new([Reply::Allocate(17, Some(12)), Reply::Free(None)]);
    assert_eq!(
        fixture.create().err().unwrap().kind(),
        ErrorKind::ResourceExhausted
    );
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);

    let fixture = Fixture::new([Reply::Allocate(0, Some(12))]);
    assert!(fixture.create().is_err());
    fixture.exhausted();
    assert_eq!(Shared::strong_count(&fixture.vm), 1);

    // No FREE is safe when copyout concealed the handle, or when a successful
    // allocation returned an invalid handle. Retain dependencies until exit.
    for errno in [Some(14), None] {
        let fixture = Fixture::new([Reply::Allocate(0, errno)]);
        assert!(fixture.create().is_err());
        fixture.exhausted();
        assert_eq!(Shared::strong_count(&fixture.vm), 2);
    }
    let fixture = Fixture::new([
        Reply::Allocate(17, None),
        Reply::Map(0, 1, Some(5)),
        Reply::Map(1, 1, Some(5)),
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
    for (prefix, errno) in [(2, Some(5)), (0, None), (1, Some(14))] {
        let fixture = Fixture::new([Reply::Allocate(17, None), Reply::Map(0, prefix, errno)]);
        assert!(fixture.create().is_err());
        fixture.exhausted();
        assert_eq!(Shared::strong_count(&fixture.vm), 2);
    }
}

#[test]
fn local_allocation_properties_reach_kfd_exactly() {
    for (coherent, uncached, contiguous, expected) in [
        (false, false, false, 0),
        (true, false, false, uapi::COHERENT),
        (false, true, false, uapi::UNCACHED),
        (false, false, true, uapi::CONTIGUOUS),
        (
            true,
            true,
            true,
            uapi::COHERENT | uapi::UNCACHED | uapi::CONTIGUOUS,
        ),
    ] {
        let fixture = Fixture::with_flags(
            [Reply::Allocate(0, Some(12))],
            uapi::VRAM | uapi::NO_SUBSTITUTE | uapi::PUBLIC | uapi::WRITABLE | expected,
        );
        let result = fixture.allocate(
            MemoryKind::DeviceLocal {
                host_visible: true,
                coherent,
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
fn failed_public_vram_cpu_mapping_releases_the_buffer_before_its_reservation() {
    let fixture = Fixture::with_flags(
        [Reply::Allocate(17, None), Reply::Free(None)],
        uapi::VRAM | uapi::WRITABLE | uapi::NO_SUBSTITUTE | uapi::PUBLIC,
    );
    // /dev/null cannot provide a render mapping. The real mmap error must
    // leave cleanup with the native owner before any GPU MAP is attempted.
    let result = KfdAllocation::create(
        fixture.vm.clone(),
        desc(),
        BufferKind::Vram {
            public: true,
            coherent: false,
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
    assert!(!fixture.vm.has_observed_loss());
    fixture.lost.store(true, Ordering::Relaxed);
    assert!(!fixture.vm.has_observed_loss());
    assert_eq!(
        allocation.device_address(&fixture.vm).unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
    assert!(fixture.vm.has_observed_loss());
    fixture.lost.store(false, Ordering::Relaxed);
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
fn concurrent_memory_fault_observers_all_receive_the_sticky_loss() {
    let fixture = Fixture::new([]);
    fixture.lost.store(true, Ordering::Release);
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
    fixture.lost.store(false, Ordering::Release);
    assert_eq!(
        fixture.vm.check().unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
}

#[test]
fn claimed_memory_fault_is_reported_without_latching_device_loss() {
    let fixture = Fixture::new([]);
    fixture.lost.store(true, Ordering::Release);
    assert_eq!(
        fixture.vm.poll_memory_fault().unwrap(),
        Some(GpuMemoryFault {
            kfd_gpu_id: 42,
            virtual_address: 0x1234_5000,
            page_not_present: false,
            read_only: true,
            no_execute: false,
            imprecise: false,
            error_type: 0,
        })
    );
    assert!(fixture.vm.check().is_ok());
    assert!(!fixture.vm.has_observed_loss());
    fixture.lost.store(false, Ordering::Release);
}

#[test]
fn failed_memory_event_creation_reclaims_every_returned_event_id() {
    let destroyed = Arc::new(Mutex::new(Vec::new()));
    let observed = destroyed.clone();
    let kfd = shared(sys::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| match call {
            sys::Call::CreateEvent(args) if args.event_type == uapi::HW_EXCEPTION => {
                args.event_id = 19;
                Ok(())
            }
            sys::Call::CreateEvent(args) if args.event_type == uapi::MEMORY_EXCEPTION => {
                args.event_id = 20;
                Err(io::Error::from_raw_os_error(5))
            }
            sys::Call::DestroyEvent(args) => {
                observed.lock().unwrap().push(args.event_id);
                Ok(())
            }
            _ => panic!("unexpected exception lifecycle operation"),
        }),
    ));
    let error = LossEvent::create(kfd).err().unwrap();
    assert_eq!(error.native_error_code(), Some(5));
    assert_eq!(*destroyed.lock().unwrap(), [20, 19]);
}

#[test]
fn native_errors_keep_operation_and_errno_without_inventing_device_loss() {
    for errno in [5, 19, 22] {
        let error = native_error("native test", io::Error::from_raw_os_error(errno));
        assert_eq!(error.kind(), ErrorKind::Driver);
        let Error::NativeOperation {
            operation, source, ..
        } = error
        else {
            panic!("lost native cause")
        };
        assert_eq!(operation, "native test");
        assert_eq!(source.raw_os_error(), Some(errno));
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

fn initialization_endpoint(acquisitions: Arc<AtomicUsize>, busy: bool) -> Shared<sys::Kfd> {
    shared(sys::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| {
            match call {
                sys::Call::Version(args) => {
                    args.major = 1;
                    args.minor = 23;
                }
                sys::Call::Apertures(args, entries) => {
                    args.count = 2;
                    for (entry, gpu_id) in entries.iter_mut().zip([42, 43]) {
                        entry.gpu_id = gpu_id;
                        entry.gpuvm_base = 0x10000;
                        entry.gpuvm_limit = isize::MAX as u64;
                    }
                }
                sys::Call::CreateEvent(args) => {
                    args.event_id = match args.event_type {
                        uapi::HW_EXCEPTION => 19,
                        uapi::MEMORY_EXCEPTION => 20,
                        _ => panic!("unexpected exception event type"),
                    };
                }
                sys::Call::Wait(args, _) => args.result = uapi::WAIT_TIMEOUT,
                sys::Call::AcquireVm(args) => {
                    assert!(matches!(args.gpu_id, 42 | 43));
                    acquisitions.fetch_add(1, Ordering::Relaxed);
                    if busy {
                        return Err(io::Error::from_raw_os_error(16));
                    }
                }
                sys::Call::RuntimeEnable(args) => {
                    assert!(matches!(args.mode_mask, 0 | 1));
                    assert_eq!(args.r_debug, 0);
                    assert_eq!(args.capabilities_mask, 0);
                }
                sys::Call::DestroyEvent(_) => {}
                _ => panic!("unexpected initialization ioctl"),
            }
            Ok(())
        }),
    ))
}

#[test]
fn concurrent_initialization_keeps_the_same_render_file_after_public_owners_drop() {
    let bindings = VmBindings::new(Allocator::default());
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let kfd = initialization_endpoint(acquisitions.clone(), false);
    let native = node();
    let owners = std::thread::scope(|scope| {
        (0..8)
            .map(|_| {
                scope.spawn(|| {
                    bindings
                        .device_with_render(&kfd, &native, |minor| {
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
        .device_with_render(&fresh, &native, |_| panic!("reopened a bound render file"))
        .unwrap();
    assert_eq!(acquisitions.load(Ordering::Relaxed), 1);
    assert_eq!(std::ptr::from_ref(&*vm), retained_address);
    let mut replacement = native;
    replacement.unique_id = Some(999);
    assert_eq!(
        bindings
            .device_with_render(&fresh, &replacement, |_| panic!("opened replaced device"))
            .err()
            .unwrap()
            .kind(),
        ErrorKind::DeviceLost
    );
}

#[test]
fn existing_foreign_vm_is_busy_and_fork_rejection_precedes_the_inherited_lock() {
    let bindings = VmBindings::new(Allocator::default());
    let kfd = initialization_endpoint(Arc::new(AtomicUsize::new(0)), true);
    assert_eq!(
        bindings
            .device_with_render(&kfd, &node(), |_| File::open("/dev/null"))
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
            .device_with_render(&kfd, &node(), |_| panic!("fork child opened render node"))
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Unsupported
    );
    drop(held);
}

#[test]
fn runtime_enable_allows_recreation_and_late_distinct_device_activation() {
    let mut bindings = VmBindings::new(Allocator::default());
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let mut kfd = initialization_endpoint(acquisitions.clone(), false);
    let native = node();
    let first = bindings
        .device_with_render(&kfd, &native, |_| File::open("/dev/null"))
        .unwrap();
    kfd.enable_runtime().unwrap();
    let recreated = bindings
        .device_with_render(&kfd, &native, |_| panic!("reopened the bound device"))
        .unwrap();
    let second = sysfs::NativeNode {
        node: 2,
        gpu_id: 43,
        render_minor: Some(129),
        unique_id: Some(456),
        identity: [1; 16],
        ..native
    };
    let second = bindings
        .device_with_render(&kfd, &second, |minor| {
            assert_eq!(minor, 129);
            File::open("/dev/null")
        })
        .unwrap();
    assert!(Shared::ptr_eq(&first, &recreated));
    assert!(!Shared::ptr_eq(&first, &second));
    assert_eq!(first.gpu_id(), 42);
    assert_eq!(second.gpu_id(), 43);
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
    let kfd = shared(sys::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| {
            if let sys::Call::DestroyEvent(_) = call {
                let attempt = observed.fetch_add(1, Ordering::Relaxed);
                if attempt == 0 {
                    Err(io::Error::from_raw_os_error(5))
                } else {
                    Ok(())
                }
            } else {
                panic!("unexpected shutdown ioctl")
            }
        }),
    ));
    let mut bindings = VmBindings::new(Allocator::default());
    bindings.bindings.get_mut().unwrap().loss = Some(shared(LossEvent {
        kfd,
        hardware_event_id: AtomicU32::new(19),
        memory_event_id: AtomicU32::new(20),
        hardware_destroy_uncertain: AtomicBool::new(false),
        memory_destroy_uncertain: AtomicBool::new(false),
        memory_event_claimed: Mutex::new(false),
        lost: AtomicBool::new(false),
    }));
    assert_eq!(
        bindings.shutdown().unwrap_err().native_error_code(),
        Some(5)
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
