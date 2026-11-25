// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/state.hpp"
#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/record_header.hpp"

#include "policies/thread_state_policy.hpp"

#include <atomic>
#include <cassert>
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

namespace rocprofsys::trace_cache
{

using ofs_t             = std::basic_ostream<char>;
using worker_function_t = std::function<void(ofs_t& ofs, bool force)>;

// Status flags exposed for observers (cache_manager, tests). The actual
// thread synchronization is done via std::stop_token inside flush_worker_t;
// these flags are mirrors maintained by the worker for external visibility.
struct worker_synchronization_t
{
    std::atomic_bool is_running{ false };
    std::atomic_bool exit_finished{ false };
    pid_t            origin_pid{};
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
    std::jthread                 m_flushing_thread;
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

template <typename WorkerFactory, typename TypeIdentifierEnum,
          rocprofsys::policies::thread_state_policy ThreadStatePolicy = state::thread>
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
        requires type_traits::cacheable<Type, TypeIdentifierEnum>
    auto store(const Type& value)
    {
        if(m_worker == nullptr || !is_running())
        {
            throw std::runtime_error(
                "Trying to use buffered storage while it is not running");
        }

        const record_header<TypeIdentifierEnum> header{ .type = Type::type_identifier,
                                                        .size = get_size(value) };

        // Hold the mutex for the entire reserve-and-write operation so that
        // the flush worker thread never reads a buffer region whose write is
        // still in flight.  Writers were already serialised through m_mutex
        // for position management; extending the critical section to cover the
        // actual memcpy closes the window that TSan (correctly) flags.
        //
        auto const thread_state_guard =
            ThreadStatePolicy::scoped(ThreadStatePolicy::Internal);
        std::lock_guard const scope{ m_mutex };

        auto* buf = reserve_memory_space(k_header_size<TypeIdentifierEnum> + header.size);
        write_record(buf, header, value);
    }

    [[nodiscard]] __attribute__((always_inline)) bool is_running() const
    {
        return m_worker_synchronization != nullptr &&
               m_worker_synchronization->is_running;
    }

private:
    void flush(ofs_t& ofs, bool force)
    {
        // Hold m_mutex for the full read so store() cannot write into the
        // region we are draining to the file.
        auto const thread_state_guard =
            ThreadStatePolicy::scoped(ThreadStatePolicy::Internal);
        std::lock_guard const guard{ m_mutex };

        size_t const _head = m_head;
        size_t const _tail = m_tail;

        if(_head == _tail)
        {
            return;
        }

        auto const used_space =
            _head > _tail ? (_head - _tail) : (k_buffer_size - _tail + _head);
        if(!force && used_space < k_flush_threshold)
        {
            return;
        }
        m_tail = m_head;

        if(_head > _tail)
        {
            ofs.write(reinterpret_cast<const char*>(m_buffer->data() + _tail),
                      _head - _tail);
        }
        else
        {
            ofs.write(reinterpret_cast<const char*>(m_buffer->data() + _tail),
                      k_buffer_size - _tail);
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
        memset(_data + m_head, std::numeric_limits<std::uint8_t>::max(),
               k_buffer_size - m_head);

        const record_header<TypeIdentifierEnum> header{
            .type = TypeIdentifierEnum::fragmented_space,
            .size = k_buffer_size - m_head - k_header_size<TypeIdentifierEnum>
        };
        binary_write_archive archive{ .buffer = _data + m_head, .position = 0 };
        archive(header);

        m_head = 0;
    }

    // Caller must hold m_mutex.
    [[nodiscard]] __attribute__((always_inline)) std::uint8_t* reserve_memory_space(
        const size_t& number_of_bytes)
    {
        if(__builtin_expect((m_head + number_of_bytes +
                             k_header_size<TypeIdentifierEnum>) > k_buffer_size,
                            0))
        {
            fragment_memory();
        }
        auto* ptr = m_buffer->data() + m_head;
        m_head += number_of_bytes;
        return ptr;
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

}  // namespace rocprofsys::trace_cache
