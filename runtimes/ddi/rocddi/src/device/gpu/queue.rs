// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Queue requests, transport descriptions, and owned native queues.
//!
//! Queue objects retain their backing and platform state until explicit
//! destruction succeeds. Packet-format policy and public ABI behavior remain
//! responsibilities of the frontend using rocddi.

mod exception;
mod layout;
mod scratch;
mod types;
pub use super::kernel_queue::*;
pub use exception::{
    AqlQueueException, aql_scratch_lanes, decode_aql_queue_exception, replace_scratch_owner,
};
pub(crate) use layout::{AqlControlLayout, AqlQueueHeader};
pub use scratch::{MAX_PRIVATE_SEGMENT_BYTES, ScratchPlan, plan_scratch};
pub use types::*;

use crate::device::gpu::GpuDevice;
use crate::driver::{self, Driver, UserQueueResource};
use crate::memory::storage::{Owned, Shared};
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
pub(crate) struct DriverQueue<D: Driver, Q: UserQueueResource<DeviceState = D::DeviceState>> {
    /// Native queue dropped before its shared driver instance.
    inner: Owned<Q>,
    /// Keeps the exact creating driver alive through queue cleanup.
    driver: Shared<D>,
    /// Transport addresses cached at queue creation.
    info: QueueTransport,
    /// Native teardown completed and cannot still depend on this driver.
    released: bool,
}

impl<D: Driver, Q: UserQueueResource<DeviceState = D::DeviceState>> DriverQueue<D, Q> {
    pub(crate) fn new(driver: Shared<D>, inner: Owned<Q>) -> Self {
        let info = inner.cached_info();
        Self {
            inner,
            driver,
            info,
            released: false,
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
        self.inner.map_device(state)
    }

    pub(crate) fn progress(&self) -> Result<(u64, u64), Error> {
        self.inner.progress()
    }

    pub(crate) fn inactivate(&mut self) -> Result<(), Error> {
        self.inner.inactivate()
    }

    pub(crate) fn set_priority(&mut self, priority: QueuePriority) -> Result<(), Error> {
        self.inner.set_priority(priority)
    }

    pub(crate) fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error> {
        self.inner.set_cu_mask(mask)
    }

    /// # Safety
    /// The command processor must have stopped the queue. The caller retains
    /// the new scratch backing and synchronizes its inactive signal until
    /// replacement or conclusive queue teardown.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error> {
        // SAFETY: The caller keeps the stopped-queue and backing obligations.
        unsafe { self.inner.set_scratch(scratch) }
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        self.inner.check()
    }

    /// # Safety
    /// Producers and public transport mappings must be retired before native
    /// destruction can release queue backing.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn destroy(&mut self) -> Result<(), Error> {
        // SAFETY: The caller retired every producer and transport mapping.
        unsafe { self.inner.destroy()? };
        self.released = true;
        Ok(())
    }

    /// # Safety
    /// Producers and public transport mappings must be retired. The callback
    /// may release external queue dependencies after native retirement.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn destroy_with_retained_transport(
        &mut self,
        callback: &mut dyn FnMut(),
    ) -> Result<(), Error> {
        // SAFETY: The caller retired producers and retained dependencies until
        // the native queue is conclusively removed.
        unsafe { self.inner.destroy_with_retained_transport(callback)? };
        self.released = true;
        Ok(())
    }
}

impl<D: Driver, Q: UserQueueResource<DeviceState = D::DeviceState>> Drop for DriverQueue<D, Q> {
    fn drop(&mut self) {
        if !self.released {
            // The native queue may still hold this session after its own Drop
            // quarantines backing. Keep the creating driver alive as well.
            std::mem::forget(self.driver.clone());
        }
    }
}

/// Owns a native queue, its independently allocated backing, and teardown state.
/// Explicit destruction reports failures. Final Drop retains any backing that
/// the kernel may still reach and never retries an ambiguously released ID.
/// Dropping a live queue without explicit destruction retains its native
/// backing through process teardown because producers may still publish.
pub struct Queue {
    pub(crate) inner: driver::QueueState,
}

/// Native queue paired with every frontend-owned address it can still read.
/// The dependency bundle is released only after explicit native destruction
/// succeeds. Dropping a live or uncertain queue retains the whole bundle
/// through process teardown along with its native transport.
pub struct RetainedQueue<D, Q = Queue> {
    queue: std::mem::ManuallyDrop<Q>,
    dependencies: std::mem::ManuallyDrop<D>,
    retired: bool,
}

impl<D, Q> RetainedQueue<D, Q> {
    fn new(queue: Q, dependencies: D) -> Self {
        Self {
            queue: std::mem::ManuallyDrop::new(queue),
            dependencies: std::mem::ManuallyDrop::new(dependencies),
            retired: false,
        }
    }

    fn retire_with_dependencies(
        &mut self,
        retire: impl FnOnce(&mut Q, &mut D) -> Result<(), Error>,
    ) -> Result<(), Error> {
        retire(&mut self.queue, &mut self.dependencies)?;
        self.retired = true;
        Ok(())
    }
}

impl<D> RetainedQueue<D> {
    /// Copies the native transport facts while this owner is live.
    #[must_use]
    pub fn info(&self) -> QueueTransport {
        self.queue.info()
    }

    /// Samples native read and write positions without retiring work.
    ///
    /// # Errors
    /// Propagates native queue or device errors.
    pub fn progress(&self) -> Result<(u64, u64), Error> {
        self.queue.progress()
    }

    /// Maps the retained queue transport into another compatible device view.
    ///
    /// # Errors
    /// Propagates ownership, routing, and native mapping failures.
    pub fn map_device(&self, device: GpuDevice<'_>) -> Result<QueueTransport, Error> {
        self.queue.map_device(device)
    }

    /// Stops native execution while keeping its dependency bundle live.
    ///
    /// # Errors
    /// Propagates native queue or device errors.
    pub fn inactivate(&mut self) -> Result<(), Error> {
        self.queue.inactivate()
    }

    /// Changes the native queue's scheduling priority.
    ///
    /// # Errors
    /// Propagates native queue or device errors.
    pub fn set_priority(&mut self, priority: QueuePriority) -> Result<(), Error> {
        self.queue.set_priority(priority)
    }

    /// Changes the native queue's compute-unit mask.
    ///
    /// # Errors
    /// Propagates native queue or device errors.
    pub fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error> {
        self.queue.set_cu_mask(mask)
    }

    /// Provides separate mutable borrows for a staged scratch update.
    ///
    /// # Safety
    /// The caller must not separate this queue from its active dependencies,
    /// and must retain each candidate backing before publishing its address.
    #[allow(unsafe_code)]
    pub unsafe fn parts_mut(&mut self) -> (&mut Queue, &mut D) {
        (&mut self.queue, &mut self.dependencies)
    }

    /// Retires native state after the frontend has stopped all producers and
    /// public transport mappings. The bundle stays owned until this succeeds.
    ///
    /// # Errors
    /// Leaves the bundle retained if native retirement or cleanup fails.
    ///
    /// # Safety
    /// Producers and public transport mappings must be retired.
    #[allow(unsafe_code)]
    pub unsafe fn destroy(&mut self) -> Result<(), Error> {
        // SAFETY: The caller upholds the public producer contract.
        self.retire_with_dependencies(|queue, _| unsafe { queue.destroy() })
    }

    /// Invokes `callback` once at native retirement while the public control
    /// mapping and dependency bundle remain live. The callback may release
    /// backing no longer reachable by the stopped queue.
    ///
    /// # Errors
    /// Leaves remaining dependencies retained on failure.
    ///
    /// # Safety
    /// Producers and public transport mappings must be retired.
    #[allow(unsafe_code)]
    pub unsafe fn destroy_with_retained_transport(
        &mut self,
        callback: &mut dyn FnMut(&mut D),
    ) -> Result<(), Error> {
        // SAFETY: The caller upholds the public producer contract. The
        // dependency bundle remains live throughout this native call.
        self.retire_with_dependencies(|queue, dependencies| unsafe {
            queue.destroy_with_retained_transport(&mut || callback(dependencies))
        })
    }

    /// Destroys a queue never published to a producer. A failed cleanup or
    /// unwind retains its native and external backing through process exit.
    ///
    /// # Errors
    /// Returns the native destruction failure.
    ///
    /// # Safety
    /// No producer has observed the queue or any public transport mapping.
    #[allow(unsafe_code)]
    pub unsafe fn abandon_unpublished(mut self) -> Result<(), Error> {
        // SAFETY: The caller guarantees that the queue was not published.
        unsafe { self.destroy() }
    }
}

impl<D, Q> Drop for RetainedQueue<D, Q> {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: Native teardown has either succeeded or Queue's destructor
        // conservatively retains native backing. External dependencies are
        // released only in the former case and only after the native owner.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.queue) };
        if self.retired {
            unsafe { std::mem::ManuallyDrop::drop(&mut self.dependencies) };
        }
    }
}

/// Keep externally supplied backing alive if native queue creation may have
/// succeeded without returning a trustworthy queue ID.
pub(crate) fn retain_create_dependencies<T, D: 'static>(
    create: impl FnOnce() -> Result<T, Error>,
    dependencies: D,
) -> Result<(T, D), Error> {
    // Native creation may unwind after a GPU engine acquires these addresses.
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
/// A failed or unwinding destructor may leave GPU references live.
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
        self.inner.map_device(device.device)
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
    /// Replaces the fixed scratch description while the command processor has
    /// stopped the AQL queue for an insufficient-scratch event. The caller
    /// must retain the described allocation until this queue is destroyed or
    /// scratch is replaced again. It must release the queue's inactive signal
    /// only after this update succeeds.
    ///
    /// # Errors
    /// Rejects non-AQL queues, invalid target geometry, unavailable queues, or
    /// scratch ranges that cannot be represented by the native control block.
    ///
    /// # Safety
    /// The command processor must have stopped this queue for a scratch fault.
    /// The GPU address must remain backed and writable until the queue is
    /// destroyed or a later stopped-queue update replaces it. The caller must
    /// synchronize release of the inactive signal with this update.
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

    /// Retires the native queue and runs `callback` once before its public
    /// control mapping is released. The callback can release external backing
    /// and report that release while the original queue header is still valid.
    /// Native retirement failure leaves the callback pending for a retry.
    ///
    /// # Errors
    /// Returns a native retirement or backing cleanup failure. The callback
    /// is not repeated if backing cleanup fails after native retirement.
    ///
    /// # Safety
    /// Producers and public transport mappings must be retired. The callback
    /// must only release dependencies no longer reachable by the native queue.
    #[allow(unsafe_code)]
    pub unsafe fn destroy_with_retained_transport(
        &mut self,
        callback: &mut dyn FnMut(),
    ) -> Result<(), Error> {
        // SAFETY: The caller supplies the producer and dependency contract.
        unsafe { self.inner.destroy_with_retained_transport(callback) }
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
    /// aligned backing with the required GPU access. The caller must keep
    /// the inactive signal and scratch backing alive until queue destruction,
    /// and obey the selected producer publication protocol. If the queue is
    /// dropped without successful destruction, retain that external backing
    /// through process teardown.
    #[allow(unsafe_code)]
    pub unsafe fn create_queue(&self, desc: QueueRequest) -> Result<Queue, Error> {
        // SAFETY: The caller retains every raw address in the request, even
        // when native creation has an ambiguous result.
        Ok(Queue {
            inner: unsafe { self.driver.create_queue(desc)? },
        })
    }

    /// Creates a queue while carrying every frontend-owned signal and scratch
    /// dependency through native acquisition. A creation error with uncertain
    /// native ownership or an unwind retains `dependencies` until process teardown.
    /// A successful result retains the dependencies inside the native queue
    /// owner through explicit destruction.
    ///
    /// # Errors
    /// Returns the same validation and native errors as `create_queue`.
    ///
    /// # Safety
    /// `dependencies` must own all externally supplied GPU addresses in
    /// `desc`. The caller must obey the selected producer publication protocol.
    #[allow(unsafe_code)]
    pub unsafe fn create_queue_with_dependencies<D: 'static>(
        &self,
        desc: QueueRequest,
        dependencies: D,
    ) -> Result<RetainedQueue<D>, Error> {
        // SAFETY: The caller supplies backing for every external GPU address.
        let (queue, dependencies) =
            retain_create_dependencies(|| unsafe { self.create_queue(desc) }, dependencies)?;
        Ok(RetainedQueue::new(queue, dependencies))
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod queue_resource_tests {
    use super::*;
    use crate::driver::{CachedInfo, Driver};
    use crate::session::DriverContextLifetime;
    use crate::topology::Endpoint;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Clone)]
    struct FakeDevice;

    struct FakeDriver {
        dropped: Arc<AtomicBool>,
        queue_drops: Arc<AtomicUsize>,
    }

    impl Drop for FakeDriver {
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

    impl Driver for FakeDriver {
        type DeviceState = FakeDevice;

        fn driver_instance(&self) -> crate::driver::DriverInstanceId {
            crate::driver::DriverInstanceId::TEST
        }

        fn context_lifetime(&self) -> DriverContextLifetime {
            DriverContextLifetime::Session
        }

        fn shutdown(&mut self) -> Result<(), Error> {
            Ok(())
        }

        fn enumerate(
            &self,
            _visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn open_endpoint(&self, _id: [u8; 16]) -> Result<Endpoint, Error> {
            Err(failure("queue test has no endpoints"))
        }

        fn activate(&self, _endpoint: &Endpoint) -> Result<FakeDevice, Error> {
            Err(failure("queue test has no activated endpoints"))
        }
    }

    fn transport() -> QueueTransport {
        QueueTransport {
            ring_host_address: 0x1000,
            ring_device_address: 0x2000,
            ring_size_bytes: 4096,
            aql_control: None,
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

    #[test]
    fn retained_queue_releases_dependencies_only_after_proven_retirement() {
        struct NativeProbe(Arc<AtomicBool>);
        impl Drop for NativeProbe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        struct BackingProbe {
            native_dropped: Arc<AtomicBool>,
            drops: Arc<AtomicUsize>,
        }
        impl Drop for BackingProbe {
            fn drop(&mut self) {
                assert!(self.native_dropped.load(Ordering::Acquire));
                self.drops.fetch_add(1, Ordering::Release);
            }
        }

        let native_dropped = Arc::new(AtomicBool::new(false));
        let backing_drops = Arc::new(AtomicUsize::new(0));
        let mut uncertain = RetainedQueue::new(
            NativeProbe(native_dropped.clone()),
            BackingProbe {
                native_dropped: native_dropped.clone(),
                drops: backing_drops.clone(),
            },
        );
        assert!(
            uncertain
                .retire_with_dependencies(|_, _| Err(failure("uncertain native destroy")))
                .is_err()
        );
        drop(uncertain);
        assert!(native_dropped.load(Ordering::Acquire));
        assert_eq!(backing_drops.load(Ordering::Acquire), 0);

        let native_dropped = Arc::new(AtomicBool::new(false));
        let mut retired = RetainedQueue::new(
            NativeProbe(native_dropped.clone()),
            BackingProbe {
                native_dropped: native_dropped.clone(),
                drops: backing_drops.clone(),
            },
        );
        assert!(retired.retire_with_dependencies(|_, _| Ok(())).is_ok());
        drop(retired);
        assert_eq!(backing_drops.load(Ordering::Acquire), 1);
    }

    impl UserQueueResource for FakeQueue {
        type DeviceState = FakeDevice;

        fn check(&self) -> Result<(), Error> {
            if self.released {
                Err(failure("fake queue released"))
            } else {
                Ok(())
            }
        }

        fn progress(&self) -> Result<(u64, u64), Error> {
            self.check()?;
            Ok((0, 0))
        }

        fn inactivate(&mut self) -> Result<(), Error> {
            self.check()
        }

        fn set_priority(&mut self, _: QueuePriority) -> Result<(), Error> {
            self.check()
        }

        fn set_cu_mask(&mut self, _: &[u32]) -> Result<(), Error> {
            self.check()
        }

        unsafe fn set_scratch(&mut self, _: QueueScratch) -> Result<(), Error> {
            Err(failure("fake queue has no scratch"))
        }

        unsafe fn destroy(&mut self) -> Result<(), Error> {
            if self.fail_destroy {
                self.fail_destroy = false;
                return Err(failure("injected queue destruction failure"));
            }
            self.released = true;
            Ok(())
        }

        unsafe fn destroy_with_retained_transport(
            &mut self,
            callback: &mut dyn FnMut(),
        ) -> Result<(), Error> {
            // SAFETY: The fake driver has no external transport mappings.
            unsafe { self.destroy()? };
            callback();
            Ok(())
        }

        fn map_device(&self, _: &FakeDevice) -> Result<QueueTransport, Error> {
            self.check()?;
            Ok(self.info)
        }
    }

    #[test]
    fn driver_queue_retains_native_owner_and_driver_through_failed_cleanup() -> Result<(), Error> {
        let driver_dropped = Arc::new(AtomicBool::new(false));
        let queue_drops = Arc::new(AtomicUsize::new(0));
        let driver = Shared::new(FakeDriver {
            dropped: driver_dropped.clone(),
            queue_drops: queue_drops.clone(),
        })?;
        let state = FakeDevice;
        let driver_queue = Owned::new(FakeQueue {
            info: transport(),
            driver_dropped: driver.dropped.clone(),
            queue_drops: driver.queue_drops.clone(),
            fail_destroy: true,
            released: false,
        })?;
        let mut queue = DriverQueue::new(driver.clone(), driver_queue);
        assert_eq!(queue.info(), transport());
        let foreign = Shared::new(FakeDriver {
            dropped: Arc::new(AtomicBool::new(false)),
            queue_drops: Arc::new(AtomicUsize::new(0)),
        })?;
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

    #[test]
    fn failed_queue_drop_keeps_its_creating_driver_live() -> Result<(), Error> {
        let driver_dropped = Arc::new(AtomicBool::new(false));
        let queue_drops = Arc::new(AtomicUsize::new(0));
        let driver = Shared::new(FakeDriver {
            dropped: driver_dropped.clone(),
            queue_drops: queue_drops.clone(),
        })?;
        let inner = Owned::new(FakeQueue {
            info: transport(),
            driver_dropped: driver_dropped.clone(),
            queue_drops: queue_drops.clone(),
            fail_destroy: true,
            released: false,
        })?;
        let mut queue = DriverQueue::new(driver.clone(), inner);
        drop(driver);
        // SAFETY: This fake never published transport to a producer.
        assert!(unsafe { queue.destroy() }.is_err());
        drop(queue);
        assert_eq!(queue_drops.load(Ordering::Relaxed), 1);
        assert!(!driver_dropped.load(Ordering::Relaxed));
        Ok(())
    }
}
