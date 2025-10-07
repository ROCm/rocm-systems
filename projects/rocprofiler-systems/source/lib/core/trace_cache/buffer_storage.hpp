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

#include "cacheable.hpp"
#include "library/runtime.hpp"

#include <PTL/PTL.hh>

#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stdint.h>
#include <string.h>
#include <type_traits>
#include <unistd.h>

namespace rocprofsys
{
namespace trace_cache
{

using ofs_t             = std::basic_ostream<char>;
using worker_function_t = std::function<void(ofs_t& ofs, bool force)>;

/**
 * @brief Synchronization structure for coordinating worker lifecycle
 *
 * This structure contains the necessary synchronization primitives to manage
 * a background worker execution state and coordinate shutdown between
 * different processes.
 */
struct worker_synchronization_t
{
    std::condition_variable is_running_condition;
    bool                    is_running{ false };

    std::condition_variable exit_finished_condition;
    bool                    exit_finished{ false };

    pid_t origin_pid;
};
using worker_synchronization_ptr_t = std::shared_ptr<worker_synchronization_t>;

/**
 * @brief Background worker that handles asynchronous buffer flushing to disk
 *
 * This class manages a background thread that periodically flushes buffered data
 * to a file. It uses a thread pool for execution and provides proper synchronization
 * for startup and shutdown operations across process boundaries.
 */
struct flush_worker_t
{
    /**
     * @brief Construct a new flush worker
     *
     * @param worker_function Function to call for performing flush operations
     * @param worker_synchronization_ptr Shared synchronization object
     * @param filepath Path to the output file where data will be flushed
     */
    explicit flush_worker_t(worker_function_t            worker_function,
                            worker_synchronization_ptr_t worker_synchronization_ptr,
                            std::string                  filepath)

    : m_worker_function(std::move(worker_function))
    , m_worker_synchronization(std::move(worker_synchronization_ptr))
    , m_filepath(std::move(filepath))
    {}

    ~flush_worker_t()
    {
        if(m_thread_pool != nullptr && m_thread_pool->is_alive())
        {
            m_thread_pool->destroy_threadpool();
            m_thread_pool.reset();
        }
        if(m_task_group != nullptr)
        {
            m_task_group.reset();
        }
    }

    /**
     * @brief Start the background flush worker thread
     *
     * Creates and initializes the thread pool, opens the output file, and starts
     * the background task that periodically calls the worker function.
     *
     * @param current_pid Process ID of the calling process
     * @throws std::runtime_error if the output file cannot be opened
     */
    void start(const pid_t& current_pid)
    {
        m_ofs = std::ofstream{ m_filepath, std::ios::binary | std::ios::out };

        if(!m_ofs.good())
        {
            std::stringstream _ss;
            _ss << "Error opening file for writing: " << m_filepath;
            throw std::runtime_error(_ss.str());
        }

        m_worker_synchronization->origin_pid = current_pid;
        m_worker_synchronization->is_running = true;

        ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        m_thread_pool = std::make_unique<PTL::ThreadPool>(NUM_OF_THREADS);
        m_thread_pool->initialize_threadpool(NUM_OF_THREADS);

        m_task_group = std::make_unique<PTL::TaskGroup<void>>(m_thread_pool.get());

        m_task_group->exec([&]() {
            std::mutex _shutdown_condition_mutex;
            while(m_worker_synchronization->is_running)
            {
                m_worker_function(m_ofs, false);
                std::unique_lock _lock{ _shutdown_condition_mutex };
                m_worker_synchronization->is_running_condition.wait_for(
                    _lock, CACHE_FILE_FLUSH_TIMEOUT,
                    [&]() { return !m_worker_synchronization->is_running; });
            }

            m_worker_function(m_ofs, true);
            m_ofs.close();
            m_worker_synchronization->exit_finished = true;
            m_worker_synchronization->exit_finished_condition.notify_one();
        });
    }

    /**
     * @brief Stop the background flush worker thread
     *
     * Signals the worker to stop, waits for it to finish if created in the same process,
     * and cleans up thread resources. Handles cross-process scenarios gracefully.
     *
     * @param current_pid Process ID of the calling process
     */
    void stop(const pid_t& current_pid)
    {
        const bool flushing_thread_exist = m_thread_pool->is_alive();
        const bool worker_is_running =
            m_worker_synchronization != nullptr && m_worker_synchronization->is_running;

        if(flushing_thread_exist && worker_is_running)
        {
            std::cout << "Buffer storage shutting down.." << std::endl;
            m_worker_synchronization->is_running = false;
            m_worker_synchronization->is_running_condition.notify_all();

            const bool thread_is_created_in_this_process =
                current_pid == m_worker_synchronization->origin_pid;
            if(!thread_is_created_in_this_process)
            {
                std::cout
                    << "Buffer storage is not created in same process as shutting down.."
                    << std::endl;
                return;
            }

            std::mutex       _exit_mutex;
            std::unique_lock _exit_lock{ _exit_mutex };
            m_worker_synchronization->exit_finished_condition.wait(
                _exit_lock, [&]() { return m_worker_synchronization->exit_finished; });

            if(m_thread_pool != nullptr && m_thread_pool->is_alive())
            {
                m_thread_pool->destroy_threadpool();
                m_thread_pool.reset();
            }
            if(m_task_group != nullptr)
            {
                m_task_group.reset();
            }
        }
    }

private:
    static constexpr auto NUM_OF_THREADS = 1;  ///< Number of background threads to use

    worker_function_t m_worker_function;  ///< Function to call for flush operations
    worker_synchronization_ptr_t
                  m_worker_synchronization;  ///< Shared synchronization object
    std::string   m_filepath;                ///< Path to output file
    std::ofstream m_ofs;                     ///< Output file stream
    std::unique_ptr<PTL::ThreadPool>
        m_thread_pool;  ///< Thread pool for background execution
    std::unique_ptr<PTL::TaskGroup<void>>
        m_task_group;  ///< Task group for managing background tasks
};

/**
 * @brief Factory for creating flush worker instances
 *
 * This factory provides a static interface for creating flush workers while
 * preventing instantiation of the factory itself.
 */
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

/**
 * @brief Thread-safe buffered storage for serializable data with asynchronous flushing
 *
 * This template class provides a high-performance buffered storage system that can
 * store typed data in memory and asynchronously flush it to disk. It uses a circular
 * buffer design and background threads for optimal performance.
 *
 * @tparam WorkerFactory Factory type for creating flush workers
 * @tparam TypeIdentifierEnum Enum class for identifying stored data types
 */
template <typename TypeIdentifierEnum, typename WorkerFactory = flush_worker_factory_t>
class buffered_storage
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");

public:
    explicit buffered_storage(std::string filepath)
    : m_worker{ std::move(WorkerFactory::get_worker(
          [this](ofs_t& ofs, bool force) { execute_flush(ofs, force); },
          m_worker_synchronization, std::move(filepath))) }
    {}

    ~buffered_storage() { shutdown(); }

    /**
     * @brief Start the buffered storage system
     *
     * Initializes the background flush worker and begins accepting data.
     *
     * @param current_pid Process ID of the calling process (defaults to current PID)
     * @throws std::runtime_error if the worker is null or cannot be started
     */
    void start(const pid_t& current_pid = getpid())
    {
        if(m_worker == nullptr)
        {
            throw std::runtime_error("Worker is null unable to start buffered storage.");
        }
        if(m_worker_synchronization && m_worker_synchronization->is_running)
        {
            return;
        }

        m_worker->start(current_pid);
    }

    /**
     * @brief Shutdown the buffered storage system
     *
     * Stops the background worker and ensures all data is flushed before returning.
     *
     * @param current_pid Process ID of the calling process (defaults to current PID)
     */
    void shutdown(const pid_t& current_pid = getpid())
    {
        if(m_worker_synchronization == nullptr || m_worker == nullptr)
        {
            return;
        }

        if(!m_worker_synchronization->is_running)
        {
            return;
        }

        m_worker->stop(current_pid);
    }

    /**
     * @brief Store a value in the buffer
     *
     * Serializes the provided value and stores it in the circular buffer with
     * proper type identification headers. The data will be asynchronously
     * flushed to disk by the background worker.
     *
     * @tparam Type Type of the value to store (must be cacheable)
     * @param value Value to store in the buffer
     * @throws std::runtime_error if storage is not running
     */
    template <typename Type>
    auto store(const Type& value)
    {
        if(!is_running())
        {
            throw std::runtime_error(
                "Trying to use buffered storage while it is not running");
            return;
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

    /**
     * @brief Check if the storage system is currently running
     *
     * @return true if the storage is running and accepting data
     * @return false if the storage is stopped
     */
    bool is_running() const { return m_worker_synchronization->is_running; }

private:
    /**
     * @brief Execute a flush operation to write buffered data to disk
     *
     * This method is called by the background worker to flush data from the
     * circular buffer to the output stream. It handles the circular buffer
     * wraparound correctly.
     *
     * @param ofs Output stream to write data to
     * @param force If true, flush all data regardless of threshold
     */
    void execute_flush(ofs_t& ofs, bool force)
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
    }

    /**
     * @brief Fragment memory when buffer wraps around
     *
     * When there's insufficient space at the end of the buffer for a new entry,
     * this method marks the remaining space as fragmented and resets the head
     * pointer to the beginning.
     */
    void fragment_memory()
    {
        auto* _data = m_buffer->data();
        memset(_data + m_head, 0xFFFF, buffer_size - m_head);
        *reinterpret_cast<TypeIdentifierEnum*>(_data + m_head) =
            TypeIdentifierEnum::fragmented_space;

        size_t remaining_bytes = buffer_size - m_head - header_size<TypeIdentifierEnum>;
        *reinterpret_cast<size_t*>(_data + m_head + sizeof(TypeIdentifierEnum)) =
            remaining_bytes;
        m_head = 0;
    }

    /**
     * @brief Reserve space in the buffer for new data
     *
     * Thread-safely reserves the requested number of bytes in the circular buffer.
     * If there's insufficient space at the end, it will fragment the remaining space
     * and wrap to the beginning.
     *
     * @param number_of_bytes Number of bytes to reserve
     * @return uint8_t* Pointer to the reserved memory space
     */
    uint8_t* reserve_memory_space(const size_t& number_of_bytes)
    {
        size_t _size;
        {
            std::lock_guard scope{ m_mutex };

            if((m_head + number_of_bytes + header_size<TypeIdentifierEnum>) > buffer_size)
            {
                fragment_memory();
            }
            _size  = m_head;
            m_head = m_head + number_of_bytes;
        }

        auto* _result = m_buffer->data() + _size;
        memset(_result, 0, number_of_bytes);
        return _result;
    }

private:
    worker_synchronization_ptr_t m_worker_synchronization{
        std::make_shared<worker_synchronization_t>()
    };  ///< Shared synchronization object for coordinating with background worker

    std::shared_ptr<typename WorkerFactory::worker_t>
        m_worker;  ///< Background flush worker

    std::mutex m_mutex;      ///< Mutex for protecting buffer state
    size_t     m_head{ 0 };  ///< Current write position in circular buffer
    size_t     m_tail{ 0 };  ///< Current read position for flushing
    std::unique_ptr<buffer_array_t> m_buffer{
        std::make_unique<buffer_array_t>()
    };  ///< Circular buffer storage
};

}  // namespace trace_cache
}  // namespace rocprofsys
