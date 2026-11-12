// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Link both public ABI frontends into one Linux shared image.
//!
//! This workspace-root crate owns no frontend implementation. Cargo links both
//! frontend rlibs and their common rocddi dependency into one cdylib. The
//! linker options in build.rs retain their public C entry points.

use amdf as _;
use hsa_runtime64 as _;
