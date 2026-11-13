// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Error taxonomy shared by discovery and native resource operations.
//!
//! [`Error`] preserves detailed context for diagnostics while [`ErrorKind`] is
//! the stable, coarse classification intended for programmatic decisions. A
//! caller should display the full error chain for humans and branch only on the
//! kind values for which it has an explicit recovery strategy.

use std::error::Error as StdError;
use std::fmt;
use std::io;

/// Native status namespace. Equal numeric codes in different domains have
/// unrelated meanings; frontend adapters must preserve both fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeStatusDomain {
    /// POSIX error number.
    Errno,
    /// Windows system error returned by `GetLastError`.
    Win32,
    /// Windows NT status, preserving all 32 bits.
    NtStatus,
    /// COM or graphics API HRESULT, preserving all 32 bits.
    Hresult,
    /// Provider-defined device firmware status.
    Firmware,
}

/// An uninterpreted native status and the namespace that defines its bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeStatus {
    /// Namespace of `code`.
    pub domain: NativeStatusDomain,
    /// Original native bits, without signed magnitude conversion.
    pub code: u32,
}

impl NativeStatus {
    /// Extracts a POSIX error number only from the errno domain.
    #[must_use]
    pub fn errno_code(self) -> Option<i32> {
        (self.domain == NativeStatusDomain::Errno)
            .then(|| i32::try_from(self.code).ok())
            .flatten()
    }

    pub(crate) fn from_io(source: &io::Error) -> Option<Self> {
        if let Some(nested) = source
            .get_ref()
            .and_then(|source| source.downcast_ref::<Error>())
        {
            return nested.native_status();
        }
        let code = source.raw_os_error()?;
        Some(Self {
            domain: if cfg!(windows) {
                NativeStatusDomain::Win32
            } else {
                NativeStatusDomain::Errno
            },
            code: u32::from_ne_bytes(code.to_ne_bytes()),
        })
    }
}

impl StdError for NativeStatus {}

impl fmt::Display for NativeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} status {:#010x}", self.domain, self.code)
    }
}

/// Constructs a DDI validation or state error without a native I/O source.
pub(crate) fn error(kind: ErrorKind, detail: &'static str) -> Error {
    Error::Operation { kind, detail }
}

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
    /// The requested endpoint or native object no longer exists.
    NotFound,
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
    /// A provider call returned a native status outside the OS I/O namespace.
    DriverOperation {
        /// Portable recovery category determined by the provider.
        kind: ErrorKind,
        /// Operation whose native protocol returned this status.
        operation: &'static str,
        /// Original status namespace and bits.
        status: NativeStatus,
    },
    /// Queue creation failed while native state may still reach caller-owned
    /// inactive-signal or scratch backing. Frontends must retain that backing.
    QueueBackingMayBeLive {
        /// Queue acquisition or rollback step that left ownership uncertain.
        operation: &'static str,
        /// Original native failure, when one is available.
        source: Option<NativeStatus>,
    },
}

impl Error {
    /// Original native status with its namespace. Portable recovery decisions
    /// use [`Self::kind`]; native interop may interpret a known status domain.
    #[must_use]
    pub fn native_status(&self) -> Option<NativeStatus> {
        match self {
            Self::NativeOperation { source, .. } => NativeStatus::from_io(source),
            Self::DriverOperation { status, .. } => Some(*status),
            Self::QueueBackingMayBeLive { source, .. } => *source,
            _ => None,
        }
    }
    /// Portable recovery category without diagnostic allocation.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Capacity { .. } => ErrorKind::ResourceExhausted,
            Self::Operation { kind, .. }
            | Self::NativeOperation { kind, .. }
            | Self::DriverOperation { kind, .. } => *kind,
            Self::QueueBackingMayBeLive { .. } => ErrorKind::DriverContract,
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
            Self::DriverOperation {
                operation, status, ..
            } => {
                write!(f, "{operation} failed: {status}")
            }
            Self::QueueBackingMayBeLive { operation, source } => {
                if let Some(source) = source {
                    write!(
                        f,
                        "{operation} failed with queue backing possibly live: {source}"
                    )
                } else {
                    write!(f, "{operation} failed with queue backing possibly live")
                }
            }
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::NativeOperation { source, .. } => Some(source),
            Self::QueueBackingMayBeLive { source, .. } => {
                source.as_ref().map(|source| source as &dyn StdError)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_status_preserves_domain_bits_and_portable_category() {
        for domain in [
            NativeStatusDomain::Errno,
            NativeStatusDomain::Win32,
            NativeStatusDomain::NtStatus,
            NativeStatusDomain::Hresult,
            NativeStatusDomain::Firmware,
        ] {
            let status = NativeStatus {
                domain,
                code: 0xc000_0022,
            };
            let error = Error::DriverOperation {
                kind: ErrorKind::PermissionDenied,
                operation: "test native operation",
                status,
            };
            assert_eq!(error.kind(), ErrorKind::PermissionDenied);
            assert_eq!(error.native_status(), Some(status));
            assert!(error.to_string().contains("0xc0000022"));
            assert_eq!(status.errno_code(), None);
        }
        assert_eq!(
            NativeStatus {
                domain: NativeStatusDomain::Errno,
                code: 13
            }
            .errno_code(),
            Some(13)
        );
        assert_eq!(
            NativeStatus {
                domain: NativeStatusDomain::Win32,
                code: 13
            }
            .errno_code(),
            None
        );
    }

    #[test]
    fn io_wrapping_preserves_a_provider_status_domain() {
        let status = NativeStatus {
            domain: NativeStatusDomain::NtStatus,
            code: 0xc000_0022,
        };
        let source = io::Error::other(Error::DriverOperation {
            kind: ErrorKind::PermissionDenied,
            operation: "native query",
            status,
        });
        let error = Error::NativeOperation {
            kind: ErrorKind::Driver,
            operation: "file operation",
            source,
        };
        assert_eq!(error.native_status(), Some(status));
    }

    #[test]
    fn io_errors_use_the_host_namespace_without_inventing_codes() {
        let error = Error::NativeOperation {
            kind: ErrorKind::Driver,
            operation: "test OS operation",
            source: io::Error::from_raw_os_error(5),
        };
        assert_eq!(
            error.native_status(),
            Some(NativeStatus {
                domain: if cfg!(windows) {
                    NativeStatusDomain::Win32
                } else {
                    NativeStatusDomain::Errno
                },
                code: 5,
            })
        );
        assert!(error.source().is_some());
        assert_eq!(
            NativeStatus::from_io(&io::Error::from(io::ErrorKind::Other)),
            None
        );
    }
    #[test]
    fn uncertain_queue_backing_retains_arbitrary_native_domains() {
        let status = NativeStatus {
            domain: NativeStatusDomain::Hresult,
            code: 0x887a_0005,
        };
        let error = Error::QueueBackingMayBeLive {
            operation: "provider queue",
            source: Some(status),
        };
        assert_eq!(error.kind(), ErrorKind::DriverContract);
        assert_eq!(error.native_status(), Some(status));
        assert!(error.source().is_some());
    }
}
