// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "fwd.hpp"

#include <memory>
#include <vector>

namespace rocprofiler
{
namespace tracing
{
namespace  // Anonymous namespace - private to this translation unit
{
// Internal pool implementation - not intended for direct use outside this file
class pool_impl
{
public:
    pool_impl()
    {
        // Pre-allocate some objects to reduce initial allocation overhead
        pool_.reserve(8);  // Typical depth for nested calls
    }

    ~pool_impl() = default;

    // Get a tracing_data object (reused or new)
    tracing_data* acquire()
    {
        if(!pool_.empty())
        {
            auto* obj = pool_.back().release();
            pool_.pop_back();
            return obj;
        }
        return new tracing_data{};
    }

    // Return object to pool
    void release(tracing_data* obj)
    {
        if(!obj) return;

        // Clear the object for reuse (keeps capacity allocated)
        obj->callback_contexts.clear();
        obj->buffered_contexts.clear();
        obj->external_correlation_ids.clear();

        // Prevent pool from growing unbounded
        if(pool_.size() < 16)
        {
            pool_.emplace_back(obj);
        }
        else
        {
            delete obj;
        }
    }

private:
    std::vector<std::unique_ptr<tracing_data>> pool_;
};

// Thread-local pool instance - private to this translation unit
inline pool_impl&
get_pool()
{
    thread_local pool_impl pool;
    return pool;
}

}  // anonymous namespace

// RAII wrapper for automatic return to pool - PUBLIC interface
class pooled_tracing_data
{
public:
    pooled_tracing_data()
    : data_(get_pool().acquire())
    {}

    ~pooled_tracing_data() { get_pool().release(data_); }

    // No copy, but allow move
    pooled_tracing_data(const pooled_tracing_data&) = delete;
    pooled_tracing_data& operator=(const pooled_tracing_data&) = delete;

    pooled_tracing_data(pooled_tracing_data&& other) noexcept
    : data_(other.data_)
    {
        other.data_ = nullptr;
    }

    // Access the tracing_data
    tracing_data&       operator*() { return *data_; }
    tracing_data*       operator->() { return data_; }
    tracing_data*       get() { return data_; }
    const tracing_data& operator*() const { return *data_; }
    const tracing_data* operator->() const { return data_; }
    const tracing_data* get() const { return data_; }

private:
    tracing_data* data_;
};

}  // namespace tracing
}  // namespace rocprofiler
