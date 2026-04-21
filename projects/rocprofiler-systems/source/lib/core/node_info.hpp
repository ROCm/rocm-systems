// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocprofsys
{

struct node_info
{
private:
    node_info();

public:
    ~node_info()                               = default;
    node_info(const node_info&)                = default;
    node_info(node_info&&) noexcept            = default;
    node_info& operator=(const node_info&)     = default;
    node_info& operator=(node_info&&) noexcept = default;

    static node_info& get_instance();

    uint64_t    id          = 0;
    uint64_t    hash        = 0;
    std::string machine_id  = {};
    std::string system_name = {};
    std::string node_name   = {};
    std::string release     = {};
    std::string version     = {};
    std::string machine     = {};
    std::string domain_name = {};
};

const node_info&
get_node_info();
}  // namespace rocprofsys
