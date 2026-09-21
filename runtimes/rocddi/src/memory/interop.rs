//! Platform-specific memory-sharing transports.
//!
//! Interop moves memory ownership or access between rocddi and another API or
//! process. The allocation, placement, and mapping model remains platform
//! neutral; each operating system exposes its native transport in a dedicated
//! child module with the correct handle type and lifetime rules.

#[cfg(target_os = "linux")]
pub mod linux;
