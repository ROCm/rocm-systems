// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// All roctx_client<MarkerWriterPolicy, Backend> template implementations are
// defined in roctx_client.hpp. This explicit instantiation stamps out the
// production specialisation.

#include "library/rocprofiler-sdk/roctx_client.hpp"

namespace rocprofsys::rocprofiler_sdk
{

template class roctx_client<default_marker_policy, backend>;

}  // namespace rocprofsys::rocprofiler_sdk
