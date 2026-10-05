// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#![allow(dead_code)]
mod amdsmi_wrapper;

#[cfg(feature = "dynamic-loading")]
mod runtime;

#[macro_use]
mod utils;
mod amdsmi;

pub use utils::*;
pub use amdsmi::*;
