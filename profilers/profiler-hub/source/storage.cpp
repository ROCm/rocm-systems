// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/storage.hpp"

#include "profiler-hub/version.hpp"
#include "storage_impl.hpp"

#include <memory>
#include <string>

namespace profiler_hub
{

storage_t::storage_t(const std::string& database_path, const std::string& uuid)
: m_impl(std::make_unique<impl>(database_path, uuid))
{}

storage_t::~storage_t() { m_impl.reset(); }

profiler_hub::version_t
storage_t::get_storage_version() const
{
    return m_impl->get_storage_version();
}

}  // namespace profiler_hub
