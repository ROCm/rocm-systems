// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Immutable normal traps retained by the shared primary KFD context.

use super::{DeviceState, KfdContextResources, LinuxKfdDriver, memory};
use crate::device::gpu::CopyFailure;
use crate::error::error;
use crate::memory::storage::{Owned, Shared};
use crate::memory::{AllocationDesc, DeviceAccess};
use crate::os::native_error;
use crate::session::DriverContextLifetime;
use crate::{Error, ErrorKind};
use std::sync::atomic::{Ordering, fence};

/// Native backing without a session driver reference or a VM ownership cycle.
pub(crate) struct KfdProcessTrap {
    /// Exact GPU VM, also retained by the allocation.
    vm: Shared<memory::DeviceVm>,
    /// Taken only to retain backing if its process owner is dropped early.
    allocation: Option<Owned<memory::KfdAllocation>>,
    /// Host copy used to reject changes to an already occupied trap slot.
    code: Vec<u8>,
}

impl KfdProcessTrap {
    fn create(vm: &Shared<memory::DeviceVm>, code: &[u8]) -> Result<Shared<Self>, Error> {
        if code.is_empty() {
            return Err(error(ErrorKind::InvalidArgument, "trap code is empty"));
        }
        // Reserve fallible owner metadata before acquiring native backing;
        // no failed publication may leave a candidate outside the cache.
        let owner = Shared::try_new_uninit()?;
        let mut saved_code = Vec::new();
        saved_code.try_reserve_exact(code.len())?;
        saved_code.extend_from_slice(code);
        let page = crate::memory::host_page_size()?;
        let allocation = memory::KfdAllocation::create(
            vm.clone(),
            AllocationDesc {
                size: code.len() as u64,
                alignment: page.max(256),
            },
            memory::BufferKind::Gtt,
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
        )?;
        let host = allocation.cached_info().host_address.ok_or_else(|| {
            error(
                ErrorKind::DriverContract,
                "trap backing is not host visible",
            )
        })?;
        // SAFETY: The unpublished native allocation owns this writable extent.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), host as *mut u8, code.len());
        }
        fence(Ordering::SeqCst);
        Ok(owner.write(Self {
            vm: vm.clone(),
            allocation: Some(allocation),
            code: saved_code,
        }))
    }

    pub(crate) fn address(&self, device: &DeviceState) -> Result<u64, Error> {
        self.address_in(&device.vm)
    }

    fn address_in(&self, vm: &Shared<memory::DeviceVm>) -> Result<u64, Error> {
        if !Shared::ptr_eq(&self.vm, vm) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "trap belongs to another GPU VM",
            ));
        }
        self.allocation
            .as_ref()
            .ok_or_else(|| error(ErrorKind::Internal, "process trap lost backing"))?
            .device_address(vm)
    }
}

impl Drop for KfdProcessTrap {
    fn drop(&mut self) {
        // Neither runtime disable nor trap unbind retires already-entered waves.
        // The primary context normally lasts until process exit. Even an early
        // drop must not recycle executable backing without that retirement.
        if let Some(allocation) = self.allocation.take() {
            std::mem::forget(allocation);
        }
    }
}

#[derive(Default)]
enum Publication {
    #[default]
    Pending,
    Ready,
    Uncertain,
}

/// The context keeps a candidate even if publication or binding fails.
pub(super) struct ProcessTrapEntry {
    trap: Shared<KfdProcessTrap>,
    publication: Publication,
}

impl KfdContextResources {
    fn install_trap(
        &self,
        vm: &Shared<memory::DeviceVm>,
        code: &[u8],
        publish: impl FnOnce(u64, u64) -> Result<(), CopyFailure>,
    ) -> Result<Shared<KfdProcessTrap>, Error> {
        vm.check()?;
        let mut traps = self
            .traps
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "process trap lock poisoned"))?;
        let index = if let Some(index) = traps
            .iter()
            .position(|entry| Shared::ptr_eq(&entry.trap.vm, vm))
        {
            index
        } else {
            traps.try_reserve(1)?;
            let trap = KfdProcessTrap::create(vm, code)?;
            traps.push(ProcessTrapEntry {
                trap,
                publication: Publication::Pending,
            });
            traps.len() - 1
        };
        let entry = &mut traps[index];
        if entry.trap.code != code {
            return Err(error(
                ErrorKind::Busy,
                "process trap code cannot be replaced",
            ));
        }
        let address = entry.trap.address_in(vm)?;
        match entry.publication {
            Publication::Uncertain => {
                return Err(error(
                    ErrorKind::DriverContract,
                    "trap publication is uncertain",
                ));
            }
            Publication::Pending => {
                // A panic or ambiguous queue retirement must retain the candidate
                // and must never let a retry skip an unfinished publication.
                entry.publication = Publication::Uncertain;
                if let Err(failure) = publish(address, code.len() as u64) {
                    if !failure.operands_may_be_live {
                        entry.publication = Publication::Pending;
                    }
                    return Err(failure.error);
                }
                entry.publication = Publication::Ready;
            }
            Publication::Ready => {}
        }
        // Rebind on every generation: runtime disable or another trap user may
        // have changed the native binding since this code was first published.
        vm.kfd()
            .set_trap_handler(vm.gpu_id, address, 0)
            .map_err(|source| native_error("AMDKFD_IOC_SET_TRAP_HANDLER", source))?;
        Ok(entry.trap.clone())
    }
}

impl LinuxKfdDriver {
    pub(crate) fn install_process_trap(
        &self,
        device: &DeviceState,
        registration_address: usize,
        code: &[u8],
        publish: impl FnOnce(u64, u64) -> Result<(), CopyFailure>,
    ) -> Result<Shared<KfdProcessTrap>, Error> {
        self.ensure_open()?;
        if self.lifetime != DriverContextLifetime::Process {
            return Err(error(
                ErrorKind::Unsupported,
                "resident traps require a process context",
            ));
        }
        if !device.debugger_registration_matches(registration_address, true)? {
            return Err(error(
                ErrorKind::Busy,
                "process debugger registration differs",
            ));
        }
        self.with_context_resources(|resources| resources.install_trap(&device.vm, code, publish))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::super::{ProcessSlot, ProcessState, sys, sysfs, uapi};
    use super::*;
    use crate::os::ProcessIdentity;
    use std::fs::{File, OpenOptions};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Condvar, Mutex};

    static FILE_ID: AtomicUsize = AtomicUsize::new(0);
    const CODE: &[u8] = &[0xaa, 0xbb, 0xcc, 0xdd];

    struct Fixture {
        slot: ProcessSlot,
        resources: Arc<KfdContextResources>,
        vm: Shared<memory::DeviceVm>,
        allocations: Arc<AtomicUsize>,
        frees: Arc<AtomicUsize>,
        bindings: Arc<Mutex<Vec<u64>>>,
        fail_bind: Arc<AtomicBool>,
        disables: Arc<AtomicUsize>,
    }

    impl Fixture {
        fn new() -> Self {
            Self::with_retired_cleanup(false)
        }

        #[allow(
            clippy::too_many_lines,
            reason = "native failure accounting stays beside the scripted calls"
        )]
        fn with_retired_cleanup(allow_cleanup: bool) -> Self {
            let path = std::env::temp_dir().join(format!(
                "rocddi-process-trap-{}-{}",
                std::process::id(),
                FILE_ID.fetch_add(1, Ordering::Relaxed)
            ));
            let render = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            std::fs::remove_file(path).unwrap();
            render
                .set_len(crate::memory::host_page_size().unwrap())
                .unwrap();
            let allocations = Arc::new(AtomicUsize::new(0));
            let observed_allocations = allocations.clone();
            let frees = Arc::new(AtomicUsize::new(0));
            let observed_frees = frees.clone();
            let bindings = Arc::new(Mutex::new(Vec::new()));
            let observed_bindings = bindings.clone();
            let fail_bind = Arc::new(AtomicBool::new(false));
            let observed_failure = fail_bind.clone();
            let disables = Arc::new(AtomicUsize::new(0));
            let observed_disables = disables.clone();
            let mut kfd = sys::Kfd::with_hook(
                File::open("/dev/null").unwrap(),
                Arc::new(move |call| {
                    match call {
                        sys::Call::Allocate(args) => {
                            assert_eq!(
                                args.flags,
                                uapi::GTT
                                    | uapi::COHERENT
                                    | uapi::UNCACHED
                                    | uapi::NO_SUBSTITUTE
                                    | uapi::WRITABLE
                                    | uapi::EXECUTABLE
                            );
                            args.handle = 1;
                            args.mmap_offset = 0;
                            observed_allocations.fetch_add(1, Ordering::Relaxed);
                        }
                        sys::Call::Map(args, ids) => {
                            assert_eq!(ids, &[42]);
                            args.success = args.count;
                        }
                        sys::Call::Unmap(args, _) if allow_cleanup => args.success = args.count,
                        sys::Call::Free(_) if allow_cleanup => {
                            observed_frees.fetch_add(1, Ordering::Relaxed);
                        }
                        sys::Call::Wait(args, _) => args.result = uapi::WAIT_TIMEOUT,
                        sys::Call::DestroyEvent(_) => {}
                        sys::Call::SetTrapHandler(args) => {
                            assert_eq!(args.gpu_id, 42);
                            assert_ne!(args.tba_address, 0);
                            assert_eq!(args.tma_address, 0);
                            observed_bindings.lock().unwrap().push(args.tba_address);
                            if observed_failure.swap(false, Ordering::Relaxed) {
                                return Err(std::io::Error::from_raw_os_error(
                                    super::super::errno::EFAULT,
                                ));
                            }
                        }
                        sys::Call::RuntimeEnable(args) => {
                            if args.mode_mask == 0 {
                                observed_disables.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        // Any trap unbind, UNMAP, or FREE violates the lifetime test.
                        _ => panic!("unexpected native trap operation"),
                    }
                    Ok(())
                }),
            );
            kfd.mark_runtime_enabled_for_test();
            let kfd = Shared::new(kfd).unwrap();
            let vm = memory::queue_fixture(kfd.clone(), render, node());
            let process = ProcessIdentity::capture().unwrap();
            let resources = Arc::new(KfdContextResources::new(process.clone()));
            assert!(resources.kfd.set(kfd).is_ok());
            let slot = ProcessSlot {
                process,
                state: Mutex::new(ProcessState {
                    resources: Some(resources.clone()),
                    joined_sessions: 0,
                    pending_disable: false,
                    disabling: false,
                }),
                changed: Condvar::new(),
            };
            Self {
                slot,
                resources,
                vm,
                allocations,
                frees,
                bindings,
                fail_bind,
                disables,
            }
        }

        fn install(
            &self,
            publish: impl FnOnce(u64, u64) -> Result<(), CopyFailure>,
        ) -> Result<Shared<KfdProcessTrap>, Error> {
            self.resources.install_trap(&self.vm, CODE, publish)
        }
    }

    fn node() -> sysfs::KfdNode {
        sysfs::KfdNode {
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

    fn publication_failure(uncertain: bool) -> CopyFailure {
        CopyFailure {
            error: error(ErrorKind::Busy, "injected publication failure"),
            operands_may_be_live: uncertain,
        }
    }

    #[test]
    fn process_trap_metadata_failure_releases_every_unpublished_candidate() {
        // Sweep every callback allocation until the first uninjected success.
        // Later host-alias metadata failures may require native rollback.
        let mut reached_success = false;
        for failure in 1..=32 {
            let fixture = Fixture::with_retired_cleanup(true);
            // Keep callbacks valid even if a regression retains an orphan owner.
            let state = Box::leak(Box::new(crate::test_support::allocator::State::default()));
            state.fail_at.store(failure, Ordering::Relaxed);
            // SAFETY: The callback state lives until process exit.
            #[allow(unsafe_code)]
            let allocator = unsafe { state.allocator() };
            let result = allocator.with_scope(|| fixture.install(|_, _| Ok(())));
            if result.is_ok() {
                assert_eq!(state.allocations.load(Ordering::Relaxed), failure - 1);
                assert_eq!(fixture.allocations.load(Ordering::Relaxed), 1);
                assert_eq!(fixture.bindings.lock().unwrap().len(), 1);
                reached_success = true;
                break;
            }
            assert_eq!(result.err().unwrap().kind(), ErrorKind::ResourceExhausted);
            assert_eq!(state.allocations.load(Ordering::Relaxed), failure);
            assert_eq!(state.frees.load(Ordering::Relaxed), failure - 1);
            let acquired = fixture.allocations.load(Ordering::Relaxed);
            assert_eq!(fixture.frees.load(Ordering::Relaxed), acquired);
            assert!(fixture.bindings.lock().unwrap().is_empty());
            assert!(fixture.resources.traps.lock().unwrap().is_empty());
            // A retry outside the failing allocator acquires exactly one page.
            assert!(fixture.install(|_, _| Ok(())).is_ok());
            assert_eq!(fixture.allocations.load(Ordering::Relaxed), acquired + 1);
        }
        assert!(reached_success);
    }

    #[test]
    fn process_trap_survives_foreign_session_and_reuses_code_after_rejoin() {
        let fixture = Fixture::new();
        let hsa = AtomicBool::new(false);
        let native = AtomicBool::new(false);
        fixture.slot.join(&hsa, &fixture.slot.process).unwrap();
        let trap = fixture
            .install(|address, size| {
                assert_eq!(size, CODE.len() as u64);
                // SAFETY: The production allocator owns this host-visible range.
                #[allow(unsafe_code)]
                let contents =
                    unsafe { std::slice::from_raw_parts(address as *const u8, CODE.len()) };
                assert_eq!(contents, CODE);
                Ok(())
            })
            .unwrap();
        let address = trap.address_in(&fixture.vm).unwrap();
        fixture.slot.join(&native, &fixture.slot.process).unwrap();
        // The sampling restore token shares only the native allocation/VM.
        let restore = trap.clone();
        drop(trap);
        fixture.slot.leave(&hsa).unwrap();
        assert_eq!(fixture.disables.load(Ordering::Relaxed), 0);
        assert_eq!(restore.address_in(&fixture.vm).unwrap(), address);
        drop(restore);
        for _ in 0..2 {
            let rejoined = fixture.slot.join(&hsa, &fixture.slot.process).unwrap();
            assert!(Arc::ptr_eq(&rejoined, &fixture.resources));
            rejoined.kfd().unwrap().enable_runtime().unwrap();
            let next = fixture
                .install(|_, _| panic!("reused code was republished"))
                .unwrap();
            assert_eq!(next.address_in(&fixture.vm).unwrap(), address);
            drop(next);
            fixture.slot.leave(&hsa).unwrap();
            // End the foreign session before the second rejoin so both shared
            // and completely disabled generations exercise the same cache.
            fixture.slot.leave(&native).unwrap();
        }
        assert_eq!(fixture.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(*fixture.bindings.lock().unwrap(), vec![address; 3]);
        assert_eq!(fixture.disables.load(Ordering::Relaxed), 2);
        assert_eq!(fixture.slot.state.lock().unwrap().joined_sessions, 0);
        assert_eq!(fixture.resources.traps.lock().unwrap().len(), 1);
    }

    #[test]
    fn process_trap_retries_retired_publication_and_failed_bind_at_same_address() {
        let fixture = Fixture::new();
        assert!(
            fixture
                .install(|_, _| Err(publication_failure(false)))
                .is_err()
        );
        assert!(fixture.bindings.lock().unwrap().is_empty());
        fixture.fail_bind.store(true, Ordering::Relaxed);
        assert!(fixture.install(|_, _| Ok(())).is_err());
        let attempted = fixture.bindings.lock().unwrap()[0];
        let trap = fixture
            .install(|_, _| panic!("completed publication repeated"))
            .unwrap();
        assert_eq!(trap.address_in(&fixture.vm).unwrap(), attempted);
        assert_eq!(fixture.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(*fixture.bindings.lock().unwrap(), vec![attempted; 2]);
        assert!(
            fixture
                .resources
                .install_trap(&fixture.vm, &[1, 2, 3, 4], |_, _| panic!(
                    "changed code published"
                ))
                .is_err()
        );
        let other = memory::queue_fixture(
            fixture.resources.kfd().unwrap(),
            File::open("/dev/null").unwrap(),
            node(),
        );
        assert_eq!(
            trap.address_in(&other).unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn process_trap_retains_uncertain_or_unwound_publication_without_binding() {
        for unwind in [false, true] {
            let fixture = Fixture::new();
            if unwind {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = fixture.install(|_, _| panic!("publication unwind"));
                    }))
                    .is_err()
                );
            } else {
                assert!(
                    fixture
                        .install(|_, _| Err(publication_failure(true)))
                        .is_err()
                );
            }
            assert!(
                fixture
                    .install(|_, _| panic!("uncertain publication retried"))
                    .is_err()
            );
            assert!(fixture.bindings.lock().unwrap().is_empty());
            assert_eq!(fixture.allocations.load(Ordering::Relaxed), 1);
            // Even dropping all context/test handles cannot release backing
            // whose native publication may still be live.
        }
    }
}
