// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Queue requests, transport descriptions, and owned native queues.
//!
//! Queue objects retain their backing and platform state until explicit
//! destruction succeeds. Packet-format policy and public ABI behavior remain
//! responsibilities of the frontend using rocddi.

mod types;
pub use types::*;

use crate::driver::{self, CachedInfo, UserQueueDriver};
use crate::gpu::GpuDevice;
use crate::host_storage::{Owned, Shared};
use crate::{Error, ErrorKind};

/// Drains host packet stores before notifying a GPU queue through its doorbell.
///
/// CPU mappings of local rings can be write-combined. An ordinary release
/// fence does not drain those writes on x86, so the engine could observe the
/// doorbell before the packet. On the x86-64 host path, this
/// operation orders earlier stores before the 64-bit notification, including
/// stores to write-combined ring memory. Other hosts use a sequentially
/// consistent fence and require native queue-publication qualification.
///
/// # Safety
/// `address` must be a live, aligned, writable 64-bit queue doorbell mapping.
/// The caller must own the queue's packet-publication protocol and keep its
/// ring, index storage, and doorbell mapping live through this call.
#[allow(unsafe_code)]
pub unsafe fn ring_doorbell(address: usize, value: u64) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: x86-64 has SSE2, and SFENCE drains earlier write-combined stores.
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
    #[cfg(not(target_arch = "x86_64"))]
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    // SAFETY: The caller retains the live, aligned MMIO mapping.
    unsafe { (address as *mut u64).write_volatile(value) };
}

/// Driver-owned queue transport and cleanup state, independent of the
/// selected GPU driver's queue representation.
pub(crate) struct DriverQueue<D: UserQueueDriver> {
    // Drop the queue resource before its shared driver instance.
    inner: Owned<D::Queue>,
    driver: Shared<D>,
    info: QueueTransport,
}

impl<D: UserQueueDriver> DriverQueue<D> {
    pub(crate) fn new(driver: Shared<D>, inner: Owned<D::Queue>) -> Self {
        let info = inner.cached_info();
        Self {
            inner,
            driver,
            info,
        }
    }

    pub(crate) fn info(&self) -> QueueTransport {
        self.info
    }

    pub(crate) fn map_device(
        &self,
        device_driver: &Shared<D>,
        state: &D::DeviceState,
    ) -> Result<QueueTransport, Error> {
        if !Shared::ptr_eq(&self.driver, device_driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "queue producer must belong to one session",
            });
        }
        D::map_queue(&self.inner, state)
    }

    pub(crate) fn progress(&self) -> Result<(u64, u64), Error> {
        D::queue_progress(&self.inner)
    }

    pub(crate) fn inactivate(&mut self) -> Result<(), Error> {
        D::inactivate_queue(&mut self.inner)
    }

    pub(crate) fn set_priority(&mut self, priority: QueuePriority) -> Result<(), Error> {
        D::set_queue_priority(&mut self.inner, priority)
    }

    pub(crate) fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error> {
        D::set_queue_cu_mask(&mut self.inner, mask)
    }

    /// # Safety
    /// Firmware must have stopped the queue. The caller retains the new
    /// scratch backing and synchronizes its inactive signal until replacement
    /// or conclusive queue teardown.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error> {
        // SAFETY: The caller keeps the stopped-queue and backing obligations.
        unsafe { D::set_queue_scratch(&mut self.inner, scratch) }
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        D::check_queue(&self.inner)
    }

    /// # Safety
    /// Producers and public transport mappings must be retired before native
    /// destruction can release queue backing.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn destroy(&mut self) -> Result<(), Error> {
        // SAFETY: The caller retired every producer and transport mapping.
        unsafe { D::destroy_queue(&mut self.inner) }
    }
}

/// Owns a native queue, its independently allocated backing, and teardown state.
/// Explicit destruction reports failures. Final Drop retains any backing that
/// the kernel may still reach and never retries an ambiguously released ID.
/// Dropping a live queue without explicit destruction retains its native
/// backing through process teardown because producers may still publish.
pub struct Queue {
    inner: DriverQueue<driver::KfdDriver>,
}

/// Keep externally supplied backing alive if native queue creation may have
/// succeeded without returning a trustworthy queue ID.
pub(crate) fn retain_create_dependencies<T, D: 'static>(
    create: impl FnOnce() -> Result<T, Error>,
    dependencies: D,
) -> Result<(T, D), Error> {
    // Native creation may unwind after firmware acquires these addresses.
    let dependencies = std::mem::ManuallyDrop::new(dependencies);
    match create() {
        Ok(queue) => Ok((queue, std::mem::ManuallyDrop::into_inner(dependencies))),
        Err(error @ Error::QueueBackingMayBeLive { .. }) => Err(error),
        Err(error) => {
            drop(std::mem::ManuallyDrop::into_inner(dependencies));
            Err(error)
        }
    }
}

/// Discards an unpublished queue only after native destruction succeeds.
/// A failed or unwinding destructor may leave firmware references live.
#[allow(unsafe_code)]
pub(crate) fn abandon_unpublished<T, D: 'static>(
    queue: T,
    dependencies: D,
    destroy: impl FnOnce(&mut T) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut queue = std::mem::ManuallyDrop::new(queue);
    let mut dependencies = std::mem::ManuallyDrop::new(dependencies);
    let result = destroy(&mut queue);
    if result.is_ok() {
        // SAFETY: Native destruction proved the queue no longer reaches either
        // owner. Drop the native queue before its external dependencies.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut queue);
            std::mem::ManuallyDrop::drop(&mut dependencies);
        }
    }
    result
}

impl Queue {
    /// Copies the ring, index, and doorbell facts captured at creation.
    /// It performs no health poll or progress read. The adapter permits address
    /// use only while a live public queue mapping borrows this native owner.
    #[must_use]
    pub fn info(&self) -> QueueTransport {
        self.inner.info()
    }
    /// Establishes producer-local queue mappings in one activated device VM.
    /// The returned addresses remain owned by this queue. Queue-backing peer
    /// mappings are retained until queue destruction, while the shared process
    /// doorbell retains its peer attachment until the owning VM is closed.
    ///
    /// # Errors
    /// Rejects a device from another core session or one whose address
    /// aperture cannot contain the queue. Native peer attachment can report an
    /// unsupported route, allocation failure, device loss, or ambiguous state.
    pub fn map_device(&self, device: GpuDevice<'_>) -> Result<QueueTransport, Error> {
        self.inner.map_device(device.driver, device.state)
    }
    /// Observes native loss, then acquire-loads the consumed and producer indices
    /// from the queue's control mapping. The pair uses PM4 dword counts, AQL
    /// packet counts, or SDMA byte counts and is a sample taken during possible
    /// concurrent publication. PM4's ring-relative native read pointer is
    /// expanded into the stable producer window. For AQL multiple-producer
    /// queues the producer frontier can include reserved slots; it is not proof
    /// that every packet is published. This call allocates nothing and does not
    /// wait for device progress.
    ///
    /// # Errors
    /// Returns `DeviceLost` for observed loss, `Unsupported` when the queue transport
    /// is no longer live, or the native observation error. No progress pair is
    /// returned on failure.
    pub fn progress(&self) -> Result<(u64, u64), Error> {
        self.inner.progress()
    }
    /// Stops native processing while retaining the queue and its backing for
    /// later destruction.
    ///
    /// # Errors
    /// Returns a native failure when the active backend cannot deactivate the
    /// queue.
    pub fn inactivate(&mut self) -> Result<(), Error> {
        self.inner.inactivate()
    }
    /// Changes the native scheduling priority of an active queue.
    ///
    /// # Errors
    /// Returns a native failure or rejects an unavailable queue.
    pub fn set_priority(&mut self, priority: QueuePriority) -> Result<(), Error> {
        self.inner.set_priority(priority)
    }
    /// Applies a non-empty native CU mask expressed as whole 32-bit words.
    ///
    /// # Errors
    /// Returns a native failure or rejects a queue that cannot accept masking.
    pub fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error> {
        self.inner.set_cu_mask(mask)
    }
    /// Replaces the fixed scratch description while firmware has the AQL queue
    /// stopped for an insufficient-scratch event. The caller must retain the
    /// described allocation until this queue is destroyed or scratch is
    /// replaced again, and must release the queue's inactive signal only after
    /// this update succeeds.
    ///
    /// # Errors
    /// Rejects non-AQL queues, invalid target geometry, unavailable queues, or
    /// scratch ranges that cannot be represented by the native control block.
    ///
    /// # Safety
    /// Firmware must have stopped this queue for a scratch fault. The GPU
    /// address must remain backed and writable until the queue is destroyed or
    /// a later stopped-queue update replaces it. The caller must synchronize
    /// release of the inactive signal with this update.
    #[allow(unsafe_code)]
    pub unsafe fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error> {
        // SAFETY: The public caller supplies the stopped-queue and backing contract.
        unsafe { self.inner.set_scratch(scratch) }
    }
    /// Checks loss and transport availability without reading queue progress.
    /// Success establishes no completion frontier and does not refresh the
    /// cached mapping information.
    ///
    /// # Errors
    /// Returns `DeviceLost` for observed loss, `Unsupported` during teardown or an
    /// uncertain native outcome, or the native error from the loss source.
    pub fn check(&self) -> Result<(), Error> {
        self.inner.check()
    }
    /// Destroys a queue after producers stop, then releases its ring and control
    /// backing. An active queue must have no unconsumed packets; an inactivated
    /// queue may discard pending work. The adapter's public mappings must be
    /// destroyed before this call. It samples progress once and performs no
    /// wait. Native loss permits cleanup, but is not itself proof that commands
    /// retired.
    ///
    /// # Errors
    /// Returns `Busy` before destruction while producer and consumer frontiers differ.
    /// Native cleanup failures preserve unfinished state. Retry resumes only
    /// operations whose ownership is known; an ambiguous DESTROY result retains
    /// backing and never resubmits a possibly recycled queue ID.
    ///
    /// # Safety
    /// All producers must have stopped publishing to this queue, and the
    /// adapter must have retired every public transport mapping. A single
    /// progress sample cannot rule out a producer publishing after the sample.
    #[allow(unsafe_code)]
    pub unsafe fn destroy(&mut self) -> Result<(), Error> {
        // SAFETY: The public caller has retired producers and mappings.
        unsafe { self.inner.destroy() }
    }

    /// Destroys a queue that was never published to producers. The native queue
    /// and externally supplied signal or scratch backing are retained together
    /// if cleanup fails or unwinds; successful cleanup drops both owners.
    ///
    /// # Errors
    /// Returns the native destruction error after retaining both owners.
    ///
    /// # Safety
    /// `dependencies` must own every external GPU address supplied at queue
    /// creation. The queue must never have been published to a producer, so no
    /// packet can still be in flight when native destruction succeeds.
    #[allow(unsafe_code)]
    pub unsafe fn abandon_unpublished_with_dependencies<D: 'static>(
        self,
        dependencies: D,
    ) -> Result<(), Error> {
        // SAFETY: The caller guarantees that no producer has ever observed this
        // queue, and its transport was never published.
        abandon_unpublished(self, dependencies, |queue| unsafe { queue.destroy() })
    }
}

impl GpuDevice<'_> {
    /// Creates a new native queue and initializes its ring and control storage
    /// before exposing addresses. Each queue owns its backing independently;
    /// only the device's native doorbell mapping is shared. Producers remain
    /// responsible for the selected format's publication protocol.
    ///
    /// # Errors
    /// Rejects unsupported formats, priorities, placement, or queue context
    /// sizes before acquisition. Allocation and native queue setup can fail;
    /// cleanup preserves the exact acquired state, including backing that a
    /// CREATE copy fault may have left reachable without a trustworthy ID.
    ///
    /// # Safety
    /// Any raw GPU address in an AQL request must refer to live, suitably
    /// aligned backing with the required firmware access. The caller must keep
    /// the inactive signal and scratch backing alive until queue destruction,
    /// and obey the selected producer publication protocol. If the queue is
    /// dropped without successful destruction, retain that external backing
    /// through process teardown.
    #[allow(unsafe_code)]
    pub unsafe fn create_queue(&self, desc: QueueRequest) -> Result<Queue, Error> {
        // SAFETY: The caller retains every raw address in the request, even
        // when native creation has an ambiguous result.
        let inner = unsafe { self.driver.create_queue(self.state, desc) }?;
        Ok(Queue {
            inner: DriverQueue::new(self.driver.clone(), inner),
        })
    }

    /// Creates a queue while carrying every frontend-owned signal and scratch
    /// dependency through native acquisition. A creation error with uncertain
    /// native ownership or an unwind retains `dependencies` until process teardown.
    /// A successful result returns the dependencies with the queue so the
    /// frontend can keep them live through queue destruction.
    ///
    /// # Errors
    /// Returns the same validation and native errors as `create_queue`.
    ///
    /// # Safety
    /// `dependencies` must own all externally supplied GPU addresses in
    /// `desc`. The caller must keep the returned owner with the queue until
    /// native destruction proves those addresses are no longer in use, and
    /// must obey the selected producer publication protocol. Dropping an
    /// undestroyed queue requires retaining those dependencies through process
    /// teardown.
    #[allow(unsafe_code)]
    pub unsafe fn create_queue_with_dependencies<D: 'static>(
        &self,
        desc: QueueRequest,
        dependencies: D,
    ) -> Result<(Queue, D), Error> {
        // SAFETY: The caller supplies and retains the described backing.
        retain_create_dependencies(|| unsafe { self.create_queue(desc) }, dependencies)
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod driver_contract_tests {
    use super::*;
    use crate::driver::{CachedInfo, DeviceStateType};
    use crate::host_storage::Allocator;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Clone)]
    struct FakeDevice;

    struct FakeGpu {
        dropped: Arc<AtomicBool>,
        queue_drops: Arc<AtomicUsize>,
    }

    impl Drop for FakeGpu {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Relaxed);
        }
    }

    struct FakeQueue {
        info: QueueTransport,
        driver_dropped: Arc<AtomicBool>,
        queue_drops: Arc<AtomicUsize>,
        fail_destroy: bool,
        released: bool,
    }

    impl Drop for FakeQueue {
        fn drop(&mut self) {
            assert!(!self.driver_dropped.load(Ordering::Relaxed));
            self.queue_drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl CachedInfo for FakeQueue {
        type Info = QueueTransport;
        fn cached_info(&self) -> QueueTransport {
            self.info
        }
    }

    impl DeviceStateType for FakeGpu {
        type DeviceState = FakeDevice;
    }

    fn transport() -> QueueTransport {
        QueueTransport {
            ring_host_address: 0x1000,
            ring_device_address: 0x2000,
            ring_size_bytes: 4096,
            sdma_engine_id: None,
            read_index_host_address: 0x3000,
            read_index_device_address: 0x4000,
            write_index_host_address: 0x5000,
            write_index_device_address: 0x6000,
            read_index_width: QueueAccessWidth::Bits64,
            write_index_width: QueueAccessWidth::Bits64,
            index_unit_bytes: 4,
            read_index_wraps: false,
            doorbell_host_address: 0x7000,
            doorbell_device_address: None,
            doorbell_width: QueueAccessWidth::Bits32,
        }
    }

    fn failure(detail: &'static str) -> Error {
        Error::Operation {
            kind: ErrorKind::Driver,
            detail,
        }
    }

    impl UserQueueDriver for FakeGpu {
        type Queue = FakeQueue;

        fn supports_expert_scheduling(&self, _: &FakeDevice) -> Result<bool, Error> {
            Ok(false)
        }

        fn check_queue(queue: &FakeQueue) -> Result<(), Error> {
            if queue.released {
                Err(failure("fake queue released"))
            } else {
                Ok(())
            }
        }

        fn queue_progress(queue: &FakeQueue) -> Result<(u64, u64), Error> {
            Self::check_queue(queue)?;
            Ok((0, 0))
        }

        fn inactivate_queue(queue: &mut FakeQueue) -> Result<(), Error> {
            Self::check_queue(queue)
        }

        fn set_queue_priority(queue: &mut FakeQueue, _: QueuePriority) -> Result<(), Error> {
            Self::check_queue(queue)
        }

        fn set_queue_cu_mask(queue: &mut FakeQueue, _: &[u32]) -> Result<(), Error> {
            Self::check_queue(queue)
        }

        unsafe fn set_queue_scratch(_: &mut FakeQueue, _: QueueScratch) -> Result<(), Error> {
            Err(failure("fake queue has no scratch"))
        }

        unsafe fn destroy_queue(queue: &mut FakeQueue) -> Result<(), Error> {
            if queue.fail_destroy {
                queue.fail_destroy = false;
                return Err(failure("injected queue destruction failure"));
            }
            queue.released = true;
            Ok(())
        }

        unsafe fn create_queue(
            &self,
            _: &FakeDevice,
            request: QueueRequest,
        ) -> Result<Owned<FakeQueue>, Error> {
            if request.parameters != QueueParameters::Pm4 {
                return Err(failure("unsupported fake format"));
            }
            Owned::new(
                FakeQueue {
                    info: transport(),
                    driver_dropped: self.dropped.clone(),
                    queue_drops: self.queue_drops.clone(),
                    fail_destroy: true,
                    released: false,
                },
                Allocator::system(),
            )
            .map_err(Into::into)
        }

        fn map_queue(queue: &FakeQueue, _: &FakeDevice) -> Result<QueueTransport, Error> {
            Self::check_queue(queue)?;
            Ok(queue.info)
        }
    }

    #[test]
    fn driver_queue_retains_native_owner_and_driver_through_failed_cleanup() -> Result<(), Error> {
        let driver_dropped = Arc::new(AtomicBool::new(false));
        let queue_drops = Arc::new(AtomicUsize::new(0));
        let driver = Shared::new(
            FakeGpu {
                dropped: driver_dropped.clone(),
                queue_drops: queue_drops.clone(),
            },
            Allocator::system(),
        )?;
        let state = FakeDevice;
        let request = QueueRequest {
            ring_size_bytes: 4096,
            parameters: QueueParameters::Pm4,
            priority: QueuePriority::Normal,
            device_producer: false,
        };
        // SAFETY: This PM4 request contains no raw external GPU addresses.
        let driver_queue = unsafe { driver.create_queue(&state, request) }?;
        let mut queue = DriverQueue::new(driver.clone(), driver_queue);
        assert_eq!(queue.info(), transport());
        let foreign = Shared::new(
            FakeGpu {
                dropped: Arc::new(AtomicBool::new(false)),
                queue_drops: Arc::new(AtomicUsize::new(0)),
            },
            Allocator::system(),
        )?;
        assert!(matches!(
            queue.map_device(&foreign, &state),
            Err(error) if error.kind() == ErrorKind::InvalidArgument
        ));
        assert_eq!(queue.map_device(&driver, &state)?, transport());
        assert_eq!(queue.progress()?, (0, 0));
        queue.inactivate()?;
        drop(driver);

        // SAFETY: No producer or public mapping was created by this fake.
        assert!(unsafe { queue.destroy() }.is_err());
        assert_eq!(queue_drops.load(Ordering::Relaxed), 0);
        assert!(!driver_dropped.load(Ordering::Relaxed));
        // SAFETY: No producer or public mapping was created by this fake.
        unsafe { queue.destroy() }?;
        drop(queue);
        assert_eq!(queue_drops.load(Ordering::Relaxed), 1);
        assert!(driver_dropped.load(Ordering::Relaxed));
        Ok(())
    }
}
