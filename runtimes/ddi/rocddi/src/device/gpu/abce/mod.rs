// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Accelerated Blit Copy Engine: host planning, SDMA encoding and ring publication.
//!
//! The raw interface projects caller-owned C records without adding hidden
//! ownership or acquiring an HSA runtime. Native GPU services use these same
//! mechanisms with native queue and memory owners.

#![allow(unsafe_code)]
#![allow(
    missing_docs,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]

pub mod abi;
mod builder;
pub use builder::*;
mod config;
pub use config::*;
mod data;
pub use data::*;
mod decode;
pub use decode::*;
mod device;
pub use device::*;
mod frame;
pub use frame::*;
mod planner;
pub use planner::*;
mod policy;
pub use policy::*;
mod ring;
pub use ring::*;
mod topology;
pub use topology::*;
pub mod native;
mod support;

pub use support::*;
