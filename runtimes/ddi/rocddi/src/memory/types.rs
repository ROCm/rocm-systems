//! Native allocation requests and limits, independent of kernel handle formats.

/// One native memory allocation request with explicit size and alignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationDesc {
    /// Requested bytes, satisfying the native allocator's size granularity.
    pub size: u64,
    /// Required base alignment in bytes, expressed as a nonzero power of two.
    pub alignment: u64,
}

/// Native allocation limits for one backing kind. A successful query describes support,
/// not a promise that sufficient memory will still be available when allocating.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationLimits {
    /// Minimum base alignment in bytes, expressed as a power of two.
    pub alignment: u64,
    /// Required size multiple in bytes. Native requests are not silently rounded.
    pub granularity: u64,
    /// Largest request supported by this allocator, independent of free capacity.
    pub maximum_size: u64,
}

impl AllocationLimits {
    /// Checks a request against these facts without querying or allocating on a
    /// device. The allocator still validates the request against current native
    /// state; retaining this value does not keep a removed device operational.
    #[must_use]
    pub fn supports(self, desc: AllocationDesc) -> bool {
        self.is_valid()
            && desc.size != 0
            && desc.size <= self.maximum_size
            && desc.size % self.granularity == 0
            && desc.alignment.is_power_of_two()
            && desc.alignment >= self.alignment
    }

    pub(crate) fn is_valid(self) -> bool {
        self.alignment.is_power_of_two()
            && self.granularity != 0
            && self.maximum_size >= self.granularity
    }
}
