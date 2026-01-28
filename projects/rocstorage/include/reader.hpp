// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/storage.hpp>

#include <memory>

namespace rocstorage
{

struct reader_t
{
    explicit reader_t(std::unique_ptr<rocstorage::storage_t> storage);

    virtual ~reader_t();

    reader_t()                            = delete;
    reader_t(const reader_t&)             = delete;
    reader_t& operator=(const reader_t&)  = delete;
    reader_t(const reader_t&&)            = delete;
    reader_t& operator=(const reader_t&&) = delete;

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace rocstorage
