// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/reader.hpp>

namespace rocstorage
{

struct reader_t::impl
{
    explicit impl(std::unique_ptr<rocstorage::storage_t> storage);

private:
    std::unique_ptr<rocstorage::storage_t> m_storage;
};

}  // namespace rocstorage
