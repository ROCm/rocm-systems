//! Queue requests, transport descriptions, and owned native queues.
//!
//! Queue objects retain their backing and platform state until explicit
//! destruction succeeds. Packet-format policy and public ABI behavior remain
//! responsibilities of the frontend using rocddi.

mod types;
pub use types::*;

use crate::driver::{self, QueueDriver};
use crate::gpu::GpuDevice;
use crate::host_storage::{Owned, Shared};
use crate::{Error, ErrorKind};

/// Owns a native queue, its independently allocated backing, and teardown state.
/// Explicit destruction reports failures. Final Drop retains any backing that
/// the kernel may still reach and never retries an ambiguously released ID.
pub struct Queue {
    pub(crate) driver: Shared<driver::PlatformDriver>,
    pub(crate) inner: Owned<driver::NativeQueue>,
    pub(crate) info: QueueTransport,
}

impl Queue {
    /// Copies the ring, index, and doorbell facts captured at creation.
    /// It performs no health poll or progress read. The adapter permits address
    /// use only while a live public queue mapping borrows this native owner.
    #[must_use]
    pub fn info(&self) -> QueueTransport {
        self.info
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
        if !Shared::ptr_eq(&self.driver, &device.device.driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "queue producer must belong to one session",
            });
        }
        driver::PlatformDriver::map_queue(&self.inner, &device.device.state)
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
        driver::PlatformDriver::queue_progress(&self.inner)
    }
    /// Stops native processing while retaining the queue and its backing for
    /// later destruction.
    ///
    /// # Errors
    /// Returns a native failure when the active backend cannot deactivate the
    /// queue.
    pub fn inactivate(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::inactivate_queue(&mut self.inner)
    }
    /// Changes the native scheduling priority of an active queue.
    ///
    /// # Errors
    /// Returns a native failure or rejects an unavailable queue.
    pub fn set_priority(&mut self, priority: QueuePriority) -> Result<(), Error> {
        driver::PlatformDriver::set_queue_priority(&mut self.inner, priority)
    }
    /// Applies a non-empty native CU mask expressed as whole 32-bit words.
    ///
    /// # Errors
    /// Returns a native failure or rejects a queue that cannot accept masking.
    pub fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error> {
        driver::PlatformDriver::set_queue_cu_mask(&mut self.inner, mask)
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
    pub fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error> {
        driver::PlatformDriver::set_queue_scratch(&mut self.inner, scratch)
    }
    /// Checks loss and transport availability without reading queue progress.
    /// Success establishes no completion frontier and does not refresh the
    /// cached mapping information.
    ///
    /// # Errors
    /// Returns `DeviceLost` for observed loss, `Unsupported` during teardown or an
    /// uncertain native outcome, or the native error from the loss source.
    pub fn check(&self) -> Result<(), Error> {
        driver::PlatformDriver::check_queue(&self.inner)
    }
    /// Destroys an idle queue, then releases its ring and control backing.
    /// Producers must be stopped and the adapter's public mappings destroyed
    /// before this call. It samples progress once and performs no wait. Native
    /// loss permits cleanup, but is not itself proof that commands retired.
    ///
    /// # Errors
    /// Returns `Busy` before destruction while producer and consumer frontiers differ.
    /// Native cleanup failures preserve unfinished state. Retry resumes only
    /// operations whose ownership is known; an ambiguous DESTROY result retains
    /// backing and never resubmits a possibly recycled queue ID.
    pub fn destroy(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::destroy_queue(&mut self.inner)
    }
}

impl GpuDevice<'_> {
    /// Creates a new native queue and initializes its ring and control storage
    /// before exposing addresses. Each queue owns its backing independently;
    /// only the device's native doorbell mapping is shared. Producers remain
    /// responsible for the selected format's publication protocol.
    ///
    /// # Errors
    /// Rejects unsupported formats, priorities, placement, or native context
    /// sizes before acquisition. Allocation and native queue setup can fail;
    /// cleanup preserves the exact acquired state, including backing that a
    /// CREATE copy fault may have left reachable without a trustworthy ID.
    pub fn create_queue(&self, desc: QueueRequest) -> Result<Queue, Error> {
        let inner = self.device.driver.create_queue(&self.device.state, desc)?;
        let info = inner.cached_info();
        Ok(Queue {
            driver: self.device.driver.clone(),
            inner,
            info,
        })
    }
}
