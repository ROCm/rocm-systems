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

#include "buffer_storage.hpp"
#include "library/ptl.hpp"
#include <chrono>
#include <mutex>
#include <stdexcept>

using namespace std::chrono_literals;

namespace rocprofsys
{
namespace trace_cache
{

namespace
{
constexpr auto CACHE_FILE_FLUSH_TIMEOUT = 10ms;
}  // namespace

buffer_storage::buffer_storage()
{
    tasking::general::get_task_group().exec([this]() {
        std::ofstream _ofs(filename, std::ios::binary | std::ios::out);

        if(!_ofs)
        {
            std::stringstream _ss;
            _ss << "Error opening file for writing: " << filename;
            throw std::runtime_error(_ss.str());
        }

        auto execute_flush = [&](std::ofstream& ofs, bool force = false) {
            size_t _head, _tail;
            {
                std::lock_guard guard{ m_mutex };
                _head = m_head;
                _tail = m_tail;

                if(_head == _tail)
                {
                    return;
                }

                auto used_space =
                    m_head > m_tail ? (m_head - m_tail) : (buffer_size - m_tail + m_head);
                if(!force && used_space < flush_threshold)
                {
                    return;
                }
                m_tail = m_head;
            }

            if(_head > _tail)
            {
                ofs.write(reinterpret_cast<const char*>(m_buffer->data() + _tail),
                          _head - _tail);
            }
            else
            {
                ofs.write(reinterpret_cast<const char*>(m_buffer->data() + _tail),
                          buffer_size - _tail);
                ofs.write(reinterpret_cast<const char*>(m_buffer->data()), _head);
            }
        };

        std::mutex _shutdown_condition_mutex;
        while(m_running)
        {
            execute_flush(_ofs);
            std::unique_lock _lock{ _shutdown_condition_mutex };
            m_shutdown_condition.wait_for(
                _lock, std::chrono::milliseconds(CACHE_FILE_FLUSH_TIMEOUT),
                [&]() { return !m_running; });
        }

        execute_flush(_ofs, true);
        _ofs.close();
        m_exit_finished = true;
        m_exit_condition.notify_one();
    });
}

void
buffer_storage::shutdown()
{
    m_running = false;
    m_shutdown_condition.notify_all();
    std::mutex       _exit_mutex;
    std::unique_lock _exit_lock{ _exit_mutex };
    m_exit_condition.wait(_exit_lock, [&]() { return m_exit_finished; });
}

void
buffer_storage::fragment_memory()
{
    auto* _data = m_buffer->data();
    memset(_data + m_head, 0xFFFF, buffer_size - m_head);
    *reinterpret_cast<entry_type*>(_data + m_head) = entry_type::fragmented_space;

    size_t remaining_bytes = buffer_size - m_head - minimal_fragmented_memory_size;
    *reinterpret_cast<size_t*>(_data + m_head + sizeof(entry_type)) = remaining_bytes;
    m_head                                                          = 0;
}

uint8_t*
buffer_storage::reserve_memory_space(size_t len)
{
    size_t _size;
    {
        std::lock_guard scope{ m_mutex };

        if((m_head + len + minimal_fragmented_memory_size) > buffer_size)
        {
            fragment_memory();
        }
        _size  = m_head;
        m_head = m_head + len;
    }

    auto* _result = m_buffer->data() + _size;
    memset(_result, 0, len);
    return _result;
}

bool
buffer_storage::is_running() const
{
    return m_running;
}

}  // namespace trace_cache
}  // namespace rocprofsys
