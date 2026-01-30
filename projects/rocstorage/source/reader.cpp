// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocstorage/reader.hpp"
#include "reader_impl.hpp"
#include "rocstorage/storage.hpp"

#include <memory>
#include <utility>

namespace rocstorage
{

reader_t::reader_t(std::unique_ptr<rocstorage::storage_t> storage)
: m_impl(std::make_unique<impl>(std::move(storage)))
{}

reader_t::~reader_t() = default;

data_types::node_info_list_t
reader_t::get_node_list() const
{
    return m_impl->get_node_list();
}

}  // namespace rocstorage
