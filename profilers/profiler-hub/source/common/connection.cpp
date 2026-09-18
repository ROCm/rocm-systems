// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "connection.hpp"

#include "profiler-hub/storage.hpp"

#include <string>

namespace profiler_hub::common
{

connection::connection(std::string_view file_path)
: m_reader{ std::make_unique<profiler_hub::reader_t>(
      std::make_unique<profiler_hub::storage_t>(std::string{ file_path }, "")) }
{}

}  // namespace profiler_hub::common
