// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
mod cpu_cache;
pub mod device;
mod driver;
mod error;
pub mod host_storage;
pub mod memory;
mod os;
pub mod session;
pub mod topology;
pub use error::{Error, ErrorKind};
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod test_support;
