//! Operating-system-specific topology extensions.
//!
//! The base [`Endpoint`](crate::topology::Endpoint) model intentionally avoids
//! native node numbers, device-file identities, and other platform transports.
//! Code that must interoperate with a platform API opts into the corresponding
//! child module explicitly.

#[cfg(target_os = "linux")]
pub mod linux;
