// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Source compatibility for clients of the original backing_store.h, including
// the VFIO system CTS. New clients can include backing_store_id.h for identity
// alone or retained_backing_range.h for access to existing bytes.
#include "simdojo/components/retained_backing_range.h"

namespace simdojo {
using BackingStore = BackingRangeAccess;
} // namespace simdojo
