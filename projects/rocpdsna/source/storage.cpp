// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocpdsna/storage.hpp"

#include "rocpdsna/storage_types.hpp"
#include "storage_impl.hpp"

#include <memory>
#include <string>

namespace rocpdsna
{

storage_t::storage_t(const std::string& database_path, const std::string& uuid)
: m_impl(std::make_unique<impl>(database_path, uuid))
{}

storage_t::~storage_t() { m_impl.reset(); }

rocpdsna::version_t
storage_t::get_storage_version() const
{
    return m_impl->get_storage_version();
}

}  // namespace rocpdsna
