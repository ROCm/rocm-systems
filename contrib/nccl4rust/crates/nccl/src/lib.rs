// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe-ish Rust wrappers for NCCL's host API.
//!
//! This crate owns NCCL resources and gives scalar element types a checked
//! mapping to `ncclDataType_t`. Communication calls are still `unsafe`: NCCL
//! only enqueues work, so Rust cannot prove that a raw CUDA pointer is valid,
//! device-accessible, correctly sized, or kept alive until its stream finishes.

mod communicator;
mod config;
mod device;
mod error;
mod group;
mod memory;
mod types;

pub use communicator::{AsyncStatus, Communicator, CommunicatorState};
pub use config::{Config, CtaPolicy, GraphUsageMode};
pub use device::{DeviceCommRequirements, DeviceCommunicator, GinConnectionType, GinType};
pub use error::{Error, Result, Status};
pub use group::Group;
pub use memory::{NcclMemory, Window, WindowFlags};
pub use types::{
    BFloat16, CudaStream, Float8E4M3, Float8E5M2, Float16, NcclDataType, ReductionOp, UniqueId,
    Version,
};

/// Raw bindings used by this wrapper.
///
/// Prefer the owned types in this crate. This re-export is provided for APIs
/// that have not acquired a Rust wrapper yet.
pub use nccl_sys as sys;
