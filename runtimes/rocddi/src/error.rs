//! Error taxonomy shared by discovery and native resource operations.
//!
//! [`Error`] preserves detailed context for diagnostics while [`ErrorKind`] is
//! the stable, coarse classification intended for programmatic decisions. A
//! caller should display the full error chain for humans and branch only on the
//! kind values for which it has an explicit recovery strategy.

use std::error::Error as StdError;
use std::fmt;
use std::io;

/// Stable, coarse error classification for programmatic recovery decisions.
///
/// This enum deliberately omits paths and driver text. It is suitable for
/// branching, telemetry aggregation, and translation to the C status ABI. Match
/// expressions must include a wildcard because new categories may be added.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A caller-provided argument is invalid.
    InvalidArgument,
    /// The requested operation is not supported on this platform.
    Unsupported,
    /// The process lacks permission for a required operation.
    PermissionDenied,
    /// A kernel driver or operating-system operation failed.
    Driver,
    /// A driver violated its advertised interface contract.
    DriverContract,
    /// Topology or other implementation data is malformed or inconsistent.
    InvalidData,
    /// A concurrent state change prevented a consistent result.
    ConcurrentModification,
    /// A finite identity, object, or native resource space was exhausted.
    ResourceExhausted,
    /// The native core detected a violation of one of its own internal invariants.
    Internal,
    /// Reset or device removal invalidated a resource.
    DeviceLost,
    /// Native state still has a dependency that prevents cleanup.
    Busy,
}

/// Failure from a rocddi mechanism. Static context preserves an allocator
/// failure without allocating a second diagnostic object.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Metadata allocation or another bounded resource could not be obtained.
    Capacity {
        /// Resource whose capacity was exhausted.
        resource: &'static str,
    },
    /// A portable core validation or state failure.
    Operation {
        /// Portable recovery category.
        kind: ErrorKind,
        /// Static description of the failed operation.
        detail: &'static str,
    },
    /// A native call failed, preserving the original operating-system error.
    NativeOperation {
        /// Portable recovery category.
        kind: ErrorKind,
        /// Native operation that failed.
        operation: &'static str,
        /// Original native error including a platform error code when available.
        source: io::Error,
    },
}

impl Error {
    /// Original platform error code when supplied by the operating system.
    ///
    /// The numeric domain is platform-defined: Linux callers receive an errno
    /// value, while a Windows backend may return a Win32 or NT-derived code.
    /// Portable recovery decisions must use [`Self::kind`] instead.
    #[must_use]
    pub fn native_error_code(&self) -> Option<i32> {
        match self {
            Self::NativeOperation { source, .. } => source.raw_os_error(),
            _ => None,
        }
    }
    /// Portable recovery category without diagnostic allocation.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Capacity { .. } => ErrorKind::ResourceExhausted,
            Self::Operation { kind, .. } | Self::NativeOperation { kind, .. } => *kind,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity { resource } => write!(f, "rocddi exhausted {resource}"),
            Self::Operation { detail, .. } => f.write_str(detail),
            Self::NativeOperation {
                operation, source, ..
            } => write!(f, "{operation} failed: {source}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::NativeOperation { source, .. } => Some(source),
            _ => None,
        }
    }
}
