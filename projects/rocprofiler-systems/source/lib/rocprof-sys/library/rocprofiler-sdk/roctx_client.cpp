// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// All roctx_client<Wrapper, MarkerWriterPolicy> template implementations are
// defined in roctx_client.hpp. This explicit instantiation stamps out the
// production specialisation.

#include "library/rocprofiler-sdk/roctx_client.hpp"

namespace rocprofsys::rocprofiler_sdk
{

template class roctx_client<backend, default_marker_policy>;

}  // namespace rocprofsys::rocprofiler_sdk
