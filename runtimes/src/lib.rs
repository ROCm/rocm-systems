// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Link the AMDF, HSA and ABCE frontends into one shared runtime image.
//!
//! This workspace-root crate owns no frontend implementation. Cargo links their
//! frontend rlibs and their common rocddi dependency into one cdylib. The
//! linker options in build.rs retain their public C entry points.

use abce as _;
use amdf as _;
use hsa_runtime64 as _;
