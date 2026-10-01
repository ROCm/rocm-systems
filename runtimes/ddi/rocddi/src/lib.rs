//! Implementation-neutral device-interface mechanisms for heterogeneous
//! compute runtimes.
//!
//! # Early-access status
//!
//! This crate is early-access runtime infrastructure. Its Rust interfaces,
//! platform coverage, packaging, and deployment model may change while it is
//! integrated into `ROCm` Systems. It is not currently part of the repository's
//! default build or installation and does not promise a stable Rust API or ABI.
//!
//! Session creation is inert. Passive endpoint queries and explicit device
//! activation have distinct ownership and native side effects. API frontends
//! supply their own ABI validation, public ownership, and policy.
#![deny(missing_docs)]
pub mod device;
mod driver;
mod error;
#[cfg(target_os = "linux")]
mod event;
pub mod gpu;
pub mod host_storage;
mod kernel_queue;
pub mod memory;
mod profiling;
mod queue;
pub mod session;
pub mod topology;
pub use error::{Error, ErrorKind};
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod test_support;
