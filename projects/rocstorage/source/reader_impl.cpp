// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "reader_impl.hpp"
#include "rocstorage/reader.hpp"
#include "rocstorage/storage.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace rocstorage
{

reader_t::impl::impl(std::unique_ptr<rocstorage::storage_t> storage)
: m_storage(std::move(storage))
{
    if(!m_storage)
    {
        throw std::invalid_argument("Provided pointer to a non-existing storage!");
    }
}

}  // namespace rocstorage
