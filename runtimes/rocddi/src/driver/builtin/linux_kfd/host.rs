//! CPU-only backing implemented as an owned anonymous Linux mapping.
//!
//! Host allocation is independent of GPU activation and therefore works for
//! both session and process native-lifetime policies. The mapping remains owned
//! after a failed `munmap`, allowing explicit destruction to retry without
//! publishing a dangling address or silently leaking the ownership record.

use super::{
    memory::{error, native_error},
    sys, util,
};
use crate::host_storage::{Allocator, Owned};
use crate::memory::HostAllocationInfo;
use crate::{Error, ErrorKind};

/// Page-aligned CPU storage with retryable mapping destruction.
pub(crate) struct HostAllocation {
    mapping: Option<sys::Reservation>,
}

impl HostAllocation {
    pub(super) fn create(
        size: u64,
        alignment: u64,
        allocator: Allocator,
        process: u32,
    ) -> Result<Owned<Self>, Error> {
        let size = usize::try_from(size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "host allocation size exceeds address width",
            )
        })?;
        let alignment = usize::try_from(alignment).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "host allocation alignment exceeds address width",
            )
        })?;
        let page = util::page_size().map_err(|e| native_error("host page size", e))?;
        if size == 0
            || size % page != 0
            || size > isize::MAX as usize
            || !alignment.is_power_of_two()
            || alignment < page
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid host allocation extent or alignment",
            ));
        }
        let owner = Owned::<Self>::try_new_uninit(allocator)?;
        let mapping =
            sys::Reservation::new_host_in_process(size, alignment, (0, isize::MAX as u64), process)
                .map_err(|e| native_error("host allocation mmap", e))?;
        Ok(owner.write(Self {
            mapping: Some(mapping),
        }))
    }
    pub(crate) fn cached_info(&self) -> HostAllocationInfo {
        HostAllocationInfo {
            host_address: self.mapping.as_ref().map_or(0, sys::Reservation::address),
            size: self.mapping.as_ref().map_or(0, |m| m.usable_size() as u64),
        }
    }
    pub(super) fn free(&mut self) -> Result<(), Error> {
        if let Some(mapping) = self.mapping.as_mut() {
            mapping
                .release()
                .map_err(|e| native_error("host allocation munmap", e))?;
            self.mapping = None;
        }
        Ok(())
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(mapping) = self.mapping.take() {
                std::mem::forget(mapping);
            }
        }
    }
}

#[cfg(test)]
#[allow(unsafe_code, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::memory::{host_cache_control, host_cache_line_size, host_page_size};
    use crate::session::{Session, SessionLifetime};
    use std::sync::atomic::Ordering;
    #[test]
    fn cpu_storage_supports_both_lifetimes_and_retains_custom_allocator() {
        for policy in [SessionLifetime::Session, SessionLifetime::Process] {
            let callbacks = crate::test_support::allocator::State::default();
            // SAFETY: State remains stationary until all native owners drop.
            let allocator = unsafe { callbacks.allocator() };
            let count = crate::test_support::allocation_counter::allocations(|| {
                let mut session = Session::with_allocator(policy, allocator).unwrap();
                let page = host_page_size().unwrap();
                let size = page * 2;
                let alignment = page * 16;
                let mut allocation = session.allocate_host(size, alignment).unwrap();
                assert_eq!(allocation.info().host_address as u64 % alignment, 0);
                assert_eq!(allocation.info().size, size);
                if let Ok(line) = host_cache_line_size() {
                    // SAFETY: The newly allocated extent remains mapped.
                    unsafe {
                        host_cache_control(allocation.info().host_address, size, line).unwrap();
                    }
                }
                allocation.free().unwrap();
                allocation.free().unwrap();
                drop(allocation);
                session.destroy().unwrap();
                assert!(session.allocate_host(page, page).is_err());
            });
            assert_eq!(count, 0);
            assert_eq!(callbacks.allocations.load(Ordering::Relaxed), 2);
            assert_eq!(callbacks.frees.load(Ordering::Relaxed), 2);
        }
    }
    #[test]
    fn failed_host_free_preserves_the_mapping_and_retries_only_remaining_work() {
        let page = host_page_size().unwrap();
        let mut allocation =
            HostAllocation::create(page, page, Allocator::default(), std::process::id()).unwrap();
        let info = allocation.cached_info();
        allocation.mapping.as_mut().unwrap().fail_release_once(5);
        assert_eq!(allocation.free().unwrap_err().native_error_code(), Some(5));
        assert_eq!(allocation.cached_info(), info);
        allocation.free().unwrap();
        assert!(allocation.mapping.is_none());
    }
}
