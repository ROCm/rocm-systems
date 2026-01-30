// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/data_types.hpp>
#include <rocstorage/storage.hpp>

#include <memory>
#include <vector>

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

    [[nodiscard]] data_types::node_info_list_t get_node_list() const;

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace rocstorage
