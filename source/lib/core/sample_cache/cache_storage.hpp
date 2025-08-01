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

#include "cache_utility.hpp"
#include "sample_type.hpp"
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <string.h>
#include <thread>
#include <type_traits>
#include <unistd.h>

namespace rocprofsys
{
namespace sample_cache
{

class cache_manager;
class cache_storage
{
public:
    static cache_storage& get_instance();

    template <typename... T>
    void store(entry_type type, T&&... values)
    {
        auto   arg_size        = get_size(values...);
        auto   total_size      = arg_size + sizeof(type) + sizeof(size_t);
        auto*  reserved_memory = reserve_memory_space(total_size);
        size_t position        = 0;

        auto store_value = [&](const auto& val) {
            using Type  = decltype(val);
            size_t len  = 0;
            auto*  dest = reserved_memory + position;
            if constexpr(std::is_same_v<std::decay_t<Type>, const char*>)
            {
                len = strlen(val) + 1;
                std::memcpy(dest, val, len);
            }
            else
            {
                using ClearType                     = std::decay_t<decltype(val)>;
                len                                 = sizeof(ClearType);
                *reinterpret_cast<ClearType*>(dest) = val;
            }
            position += len;
        };

        store_value(type);
        store_value(arg_size);

        (store_value(values), ...);
    }

private:
    friend class cache_manager;
    cache_storage();
    void     shutdown();
    bool     is_running() const;
    void     fragment_memory();
    uint8_t* reserve_memory_space(size_t len);

    template <typename... T>
    constexpr size_t get_size(T&... val)
    {
        auto get_size_impl = [&](auto& _val) {
            using Type = std::decay_t<decltype(_val)>;
            if constexpr(std::is_same_v<Type, const char*>)
            {
                return strlen(_val) + 1;
            }
            else
            {
                return sizeof(Type);
            }
        };

        auto total_size = 0;
        ((total_size += get_size_impl(val)), ...);
        return total_size;
    }

private:
    std::mutex                      m_mutex;
    std::condition_variable         m_exit_condition;
    bool                            m_exit_finished{ false };
    bool                            m_running{ true };
    std::condition_variable         m_shutdown_condition;
    std::thread                     m_flushing_thread;
    size_t                          m_head{ 0 };
    size_t                          m_tail{ 0 };
    std::unique_ptr<buffer_array_t> m_buffer{ std::make_unique<buffer_array_t>() };
};

}  // namespace sample_cache
}  // namespace rocprofsys
