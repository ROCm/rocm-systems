// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "connection.hpp"

#include "profiler-hub/cpp/storage.hpp"

#include <string>

namespace profiler_hub::common
{

connection::connection(std::string_view file_path)
: m_reader{ std::make_unique<profiler_hub::reader_t>(
      std::make_unique<profiler_hub::storage_t>(std::string{ file_path }, "")) }
{}

connection::connection(std::string_view                                file_path,
                       std::shared_ptr<profiler_hub::reader_catalog_t> catalog)
// std::make_unique can't invoke reader_t's private constructor -- it does
// the `new` from inside its own (non-friend) implementation, not from here.
// A raw `new` expression written directly in this friend's body works.
: m_reader{ new profiler_hub::reader_t(
      std::make_unique<profiler_hub::storage_t>(std::string{ file_path }, ""),
      std::move(catalog)) }
{}

}  // namespace profiler_hub::common
