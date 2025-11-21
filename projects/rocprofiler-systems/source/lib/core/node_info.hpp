// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
