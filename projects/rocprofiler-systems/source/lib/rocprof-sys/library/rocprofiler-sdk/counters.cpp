// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// All counter_storage<Backend> and related template implementations are defined
// in counters.hpp. This explicit instantiation stamps out the production specialisation.

#include "library/rocprofiler-sdk/counters.hpp"

namespace rocprofsys::rocprofiler_sdk
{

template struct counter_dispatch_record<backend>;
template struct counter_event<backend>;
template struct counter_storage<backend>;

}  // namespace rocprofsys::rocprofiler_sdk
