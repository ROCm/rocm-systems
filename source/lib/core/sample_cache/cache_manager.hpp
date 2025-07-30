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

#include "cache_storage.hpp"
#include "cache_storage_parser.hpp"
#include "core/sample_cache/rocpd_post_processing.hpp"
#include "metadata_storage.hpp"

namespace rocprofsys
{
namespace sample_cache
{

class cache_manager
{
public:
    static cache_manager& get_instance();
    cache_storage&        get_cache() { return m_storage; }
    metadata&             get_metadata() { return m_metadata; }
    void                  shutdown();
    void                  post_process();

private:
    void post_process_metadata();
    cache_manager();

    cache_storage         m_storage;
    metadata              m_metadata;
    storage_parser        m_parser;
    rocpd_post_processing m_postprocessing;
};

inline metadata&
get_cache_metadata()
{
    return cache_manager::get_instance().get_metadata();
}

inline cache_storage&
get_cache_storage()
{
    return cache_manager::get_instance().get_cache();
}

}  // namespace sample_cache
}  // namespace rocprofsys
