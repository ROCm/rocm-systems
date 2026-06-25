// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "sqlite_backend_impl.hpp"

#include "sqlite_api_policy.hpp"

namespace profiler_hub::data_storage
{

template class database_backend<sqlite_api_policy>;

}  // namespace profiler_hub::data_storage
