// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stack>
#include <stdexcept>

namespace rocprofiler
{
namespace common
{
namespace memory
{
template <size_t BlockSize, size_t ReservedBlocks = 0>
class pool
{
public:
    explicit pool(size_t size)
    : m_size(size)
    {
        for(size_t i = 0; i < ReservedBlocks; i++)
        {
            append();
        }
    }

    void* allocate()
    {
        if(m_addrs.empty())
        {
            append();
        }

        auto* ptr = m_addrs.top();
        m_addrs.pop();
        return ptr;
    }

    void deallocate(void* ptr) { m_addrs.push(ptr); }

    void rebind(size_t size)
    {
        if(!(m_addrs.empty() && m_blocks.empty()))
        {
            ROCP_FATAL << "cannot call pool::rebind() after alloc";
        }

        m_size = size;
    }

private:
    // Refill the address stack by allocating another block of memory
    void append()
    {
        auto block      = std::make_unique<uint8_t[]>(BlockSize);
        auto total_size = BlockSize % m_size == 0 ? BlockSize : BlockSize - m_size;

        // Divide the block into chunks of m_size bytes, and add their addrs
        for(size_t i = 0; i < total_size; i += m_size)
        {
            m_addrs.push(&block.get()[i]);
        }

        // Keep the memory of the block alive by adding it to our stack
        m_blocks.push(std::move(block));
    }

private:
    size_t                                 m_size   = {};
    std::stack<void*>                      m_addrs  = {};
    std::stack<std::unique_ptr<uint8_t[]>> m_blocks = {};
};

}  // namespace memory
}  // namespace common
}  // namespace rocprofiler
