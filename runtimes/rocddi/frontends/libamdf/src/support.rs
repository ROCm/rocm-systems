//! Shared admission and output-publication rules for the AMDF entry points.
//!
//! The imported C contract requires each handle to come from this provider and
//! remain alive, with its borrowed parents, throughout the operation. It also
//! requires destruction to be serialized against other uses. Pointer alignment
//! checks cannot prove those properties; adding a handle registry here would
//! change both ownership and the cost of ordinary metadata queries.
//!
//! A caller supplies the declared readable input extent and writable output
//! extent. Output payload fields may be uninitialized: only their header is read
//! before success publishes a complete record. Larger caller extents and their
//! trailing bytes survive unchanged. Count/prefix enumeration is the explicit
//! exception to the usual rule that failure leaves every output untouched.
//! Entry-point unsafe blocks rely on these common obligations; comments at
//! acquisition, publication, and destruction explain the additional ownership
//! transitions that require care.

use crate::generated::amdf::*;
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const INVALID: u64 = AMDF_STATUS_CODE_INVALID_ARGUMENT as u64;
pub(crate) const RANGE: u64 = AMDF_STATUS_CODE_OUT_OF_RANGE as u64;
pub(crate) const UNSUPPORTED: u64 = AMDF_STATUS_CODE_UNSUPPORTED as u64;
pub(crate) const NOT_FOUND: u64 = AMDF_STATUS_CODE_NOT_FOUND as u64;
pub(crate) const EXHAUSTED: u64 = AMDF_STATUS_CODE_RESOURCE_EXHAUSTED as u64;
pub(crate) const BUSY: u64 = AMDF_STATUS_CODE_BUSY as u64;
pub(crate) const DEADLINE: u64 = AMDF_STATUS_CODE_DEADLINE_EXCEEDED as u64;
pub(crate) const LOST: u64 = AMDF_STATUS_CODE_DEVICE_LOST as u64;
pub(crate) const VERSION: u64 = AMDF_STATUS_CODE_VERSION_MISMATCH as u64;
pub(crate) const SMALL: u64 = AMDF_STATUS_CODE_BUFFER_TOO_SMALL as u64;
pub(crate) const PRECONDITION: u64 = AMDF_STATUS_CODE_FAILED_PRECONDITION as u64;
pub(crate) const INTERNAL: u64 = AMDF_STATUS_CODE_INTERNAL as u64;

pub(crate) fn boundary(body: impl FnOnce() -> Result<(), u64>) -> u64 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => 0,
        Ok(Err(status)) => status,
        Err(_) => INTERNAL,
    }
}

pub(crate) fn output_pointer<T>(p: *mut T) -> Result<(), u64> {
    if p.is_null() || (p as usize) % align_of::<T>() != 0 {
        Err(INVALID)
    } else {
        Ok(())
    }
}

pub(crate) unsafe fn object<'a, T>(p: *const T) -> Result<&'a T, u64> {
    output_pointer(p.cast_mut())?;
    // SAFETY: The caller supplies a live object of this exact type.
    Ok(unsafe { &*p })
}

pub(crate) unsafe fn exclusive<'a, T>(p: *mut T) -> Result<&'a mut T, u64> {
    output_pointer(p)?;
    // SAFETY: Mutating operations require exclusive access in the C contract.
    Ok(unsafe { &mut *p })
}

pub(crate) unsafe fn array<'a, T>(p: *const T, count: u32) -> Result<&'a [T], u64> {
    if count == 0 {
        return Ok(&[]);
    }
    output_array(p.cast_mut(), count)?;
    // SAFETY: The caller provides the complete contiguous array extent.
    Ok(unsafe { std::slice::from_raw_parts(p, count as usize) })
}

/// Output arrays may contain uninitialized bytes. Check only their raw extent;
/// constructing a Rust slice would incorrectly assert initialized elements.
pub(crate) fn output_array<T>(p: *mut T, count: u32) -> Result<(), u64> {
    if count == 0 {
        return Ok(());
    }
    output_pointer(p)?;
    if (count as usize)
        .checked_mul(size_of::<T>())
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(INVALID);
    }
    Ok(())
}

#[repr(C)]
/// Universal prefix read before a caller-provided extensible record is trusted.
#[derive(Clone, Copy)]
struct Prefix {
    kind: u32,
    size: u32,
}

unsafe fn validate<T>(p: *const T, kind: u32) -> Result<amdf_input_structure_t, u64> {
    output_pointer(p.cast_mut())?;
    // Only the universal eight-byte prefix is read until its extent is valid.
    // In particular, a short record can safely end at a guard page.
    let prefix = unsafe { p.cast::<Prefix>().read() };
    if prefix.kind != kind || (prefix.size as usize) < size_of::<T>() {
        return Err(INVALID);
    }
    // SAFETY: Every typed record has this prefix, now covered by its size.
    let header = unsafe { p.cast::<amdf_input_structure_t>().read() };
    if !header.next.is_null() {
        return Err(UNSUPPORTED);
    }
    Ok(header)
}

pub(crate) unsafe fn input<T: Copy>(p: *const T, kind: u32) -> Result<T, u64> {
    unsafe { validate(p, kind)? };
    // SAFETY: Header admission precedes reading the complete typed record.
    Ok(unsafe { p.read() })
}

/// Validated output slot retaining the caller's original extensible header.
///
/// Publication preserves the extension pointer and any trailing bytes beyond
/// this ABI's structure size. A retained record can publish its payload without
/// copying the header or materializing a full temporary.
pub(crate) struct Output<T> {
    pointer: *mut T,
    header: amdf_input_structure_t,
}

pub(crate) unsafe fn output<T>(p: *mut T, kind: u32) -> Result<Output<T>, u64> {
    Ok(Output {
        pointer: p,
        header: unsafe { validate(p, kind)? },
    })
}

impl<T> Output<T> {
    pub(crate) unsafe fn publish(self, mut value: T) {
        // Preserve the caller's larger extent and extension pointer. Unknown
        // trailing bytes are never overwritten by this provider version.
        unsafe {
            (&raw mut value)
                .cast::<amdf_input_structure_t>()
                .write(self.header);
            self.pointer.write(value);
        }
    }

    /// Publish a retained ABI record while keeping the caller's validated
    /// extensible header and any bytes beyond the known record untouched.
    pub(crate) unsafe fn publish_from(self, source: *const T)
    where
        T: Copy,
    {
        let header_size = size_of::<amdf_input_structure_t>();
        // SAFETY: The caller supplied a complete writable T slot, source is
        // initialized, and every ABI record begins with the common header.
        // copy permits overlapping source and destination ranges.
        unsafe {
            std::ptr::copy(
                source.cast::<u8>().add(header_size),
                self.pointer.cast::<u8>().add(header_size),
                size_of::<T>() - header_size,
            );
        }
    }
}

pub(crate) fn range(offset: u64, length: u64, extent: u64) -> Result<(), u64> {
    if offset > extent || length > extent - offset {
        Err(RANGE)
    } else {
        Ok(())
    }
}

pub(crate) fn next_id(counter: &AtomicU64) -> Result<u64, u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| EXHAUSTED)
}

pub(crate) fn register(counter: &AtomicU64) -> Result<(), u64> {
    next_id(counter).map(|_| ())
}
pub(crate) fn unregister(counter: &AtomicU64) {
    counter.fetch_sub(1, Ordering::Release);
}

pub(crate) fn native(error: &rocddi::Error) -> u64 {
    if let Some(code) = error.native_error_code() {
        return (u64::from(AMDF_STATUS_DOMAIN_ERRNO) << 32) | u64::from(code.unsigned_abs());
    }
    match error.kind() {
        rocddi::ErrorKind::InvalidArgument => INVALID,
        rocddi::ErrorKind::Unsupported => UNSUPPORTED,
        rocddi::ErrorKind::PermissionDenied => u64::from(AMDF_STATUS_CODE_PERMISSION_DENIED),
        rocddi::ErrorKind::ResourceExhausted => EXHAUSTED,
        rocddi::ErrorKind::Busy => BUSY,
        rocddi::ErrorKind::DeviceLost => LOST,
        _ => INTERNAL,
    }
}
