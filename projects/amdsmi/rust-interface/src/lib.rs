// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#![allow(dead_code)]
#[cfg(feature = "dynamic-loading")]
mod runtime;
mod amdsmi_wrapper;

#[macro_use]
mod utils;
mod amdsmi;

pub use utils::*;
pub use amdsmi::*;
