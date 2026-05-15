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

#include "core/trace_cache/cacheable.hpp"

#include "common/defines.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>

#include <unistd.h>

namespace rocprofsys
{
namespace trace_cache
{

using ofs_t             = std::basic_ostream<char>;
using worker_function_t = std::function<void(ofs_t& ofs, bool force)>;

struct worker_synchronization_t
{
    std::condition_variable is_running_condition;
    std::atomic_bool        is_running{ false };

    std::condition_variable exit_finished_condition;
    std::atomic_bool        exit_finished{ false };

    pid_t origin_pid;
};
using worker_synchronization_ptr_t = std::shared_ptr<worker_synchronization_t>;

struct flush_worker_t
{
    explicit flush_worker_t(worker_function_t            worker_function,
                            worker_synchronization_ptr_t worker_synchronization_ptr,
                            std::string                  filepath);

    void start(const pid_t& current_pid);

    void stop(const pid_t& current_pid);

private:
    worker_function_t            m_worker_function;
    worker_synchronization_ptr_t m_worker_synchronization;
    std::string                  m_filepath;
    std::ofstream                m_ofs;
    std::unique_ptr<std::thread> m_flushing_thread;
};

struct flush_worker_factory_t
{
    using worker_t = flush_worker_t;

    flush_worker_factory_t()                                    = delete;
    flush_worker_factory_t(flush_worker_factory_t&)             = delete;
    flush_worker_factory_t& operator=(flush_worker_factory_t&)  = delete;
    flush_worker_factory_t(flush_worker_factory_t&&)            = delete;
    flush_worker_factory_t& operator=(flush_worker_factory_t&&) = delete;

    static std::shared_ptr<worker_t> get_worker(
        worker_function_t                   worker_function,
        const worker_synchronization_ptr_t& worker_synchronization_ptr,
        std::string                         filepath)
    {
        return std::make_shared<worker_t>(worker_function, worker_synchronization_ptr,
                                          std::move(filepath));
    }
};

template <typename WorkerFactory, typename TypeIdentifierEnum>
class buffer_storage
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");

public:
    explicit buffer_storage(std::string filepath)
    : m_worker{ std::move(
          WorkerFactory::get_worker([this](ofs_t& ofs, bool force) { flush(ofs, force); },
                                    m_worker_synchronization, std::move(filepath))) }
    {}

    ~buffer_storage() { shutdown(); }

    void start(const pid_t& current_pid = getpid())
    {
        if(m_worker == nullptr)
        {
            throw std::runtime_error(
                "Worker is null - unable to start buffered storage.");
        }

        if(is_running())
        {
            return;
        }

        m_worker->start(current_pid);
    }

    void shutdown(const pid_t& current_pid = getpid())
    {
        if(m_worker == nullptr)
        {
            throw std::runtime_error(
                "Worker is null - unable to shutdown buffered storage.");
        }

        if(!is_running())
        {
            return;
        }

        m_worker->stop(current_pid);
    }

    template <typename Type>
    auto store(const Type& value)
    {
        if(m_worker == nullptr || !is_running())
        {
            throw std::runtime_error(
                "Trying to use buffered storage while it is not running");
        }

        type_traits::check_type<Type, TypeIdentifierEnum>();

        using TypeIdentifierEnumUderlayingType =
            std::underlying_type_t<TypeIdentifierEnum>;

        size_t sample_size      = get_size(value);
        size_t bytes_to_reserve = header_size<TypeIdentifierEnum> + sample_size;
        auto*  buf              = reserve_memory_space(bytes_to_reserve);
        size_t position         = 0;
        auto   type_identifier_value =
            static_cast<TypeIdentifierEnumUderlayingType>(Type::type_identifier);

        utility::store_value(type_identifier_value, buf, position);
        utility::store_value(sample_size, buf, position);
        serialize(buf + position, value);
    }

    ROCPROFSYS_INLINE bool is_running() const
    {
        return m_worker_synchronization != nullptr &&
               m_worker_synchronization->is_running;
    }

private:
    void flush(ofs_t& ofs, bool force)
    {
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
        if(ofs.fail())
        {
            throw std::runtime_error(
                std::string("Error flushing buffered storage to file for pid: ") +
                std::to_string(m_worker_synchronization->origin_pid) + "\n");
        }
    }

    void fragment_memory()
    {
        auto* _data = m_buffer->data();
        memset(_data + m_head, std::numeric_limits<uint8_t>::max(), buffer_size - m_head);
        *reinterpret_cast<TypeIdentifierEnum*>(_data + m_head) =
            TypeIdentifierEnum::fragmented_space;

        size_t remaining_bytes = buffer_size - m_head - header_size<TypeIdentifierEnum>;
        *reinterpret_cast<size_t*>(_data + m_head + sizeof(TypeIdentifierEnum)) =
            remaining_bytes;
        m_head = 0;
    }

    ROCPROFSYS_INLINE uint8_t* reserve_memory_space(const size_t& number_of_bytes)
    {
        size_t _size;
        {
            std::lock_guard scope{ m_mutex };

            if(__builtin_expect((m_head + number_of_bytes +
                                 header_size<TypeIdentifierEnum>) > buffer_size,
                                0))
            {
                fragment_memory();
            }
            _size = m_head;
            m_head += number_of_bytes;
        }

        return m_buffer->data() + _size;
    }

private:
    worker_synchronization_ptr_t m_worker_synchronization{
        std::make_shared<worker_synchronization_t>()
    };

    std::shared_ptr<typename WorkerFactory::worker_t> m_worker;

    std::mutex                      m_mutex;
    size_t                          m_head{ 0 };
    size_t                          m_tail{ 0 };
    std::unique_ptr<buffer_array_t> m_buffer{ std::make_unique<buffer_array_t>() };
};

}  // namespace trace_cache
}  // namespace rocprofsys
