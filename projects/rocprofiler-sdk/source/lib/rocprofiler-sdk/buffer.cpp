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

#include "lib/rocprofiler-sdk/buffer.hpp"

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <atomic>
#include <exception>
#include <mutex>
#include <random>
#include <vector>

namespace rocprofiler
{
namespace buffer
{
namespace
{
using reserve_size_t = common::container::reserve_size;

auto&
get_buffers_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

uint64_t
get_buffer_offset()
{
    static uint64_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint64_t>{std::numeric_limits<uint8_t>::max(),
                                                           std::numeric_limits<uint16_t>::max()};
        return rng(gen);
    }();
    return _v;
}
}  // namespace

bool
is_valid_buffer_id(rocprofiler_buffer_id_t id)
{
    if(!get_buffers()) return false;
    auto nbuffers = get_buffers()->size();
    auto offset   = get_buffer_offset();
    return (id.handle >= offset && id.handle < (offset + nbuffers));
}

unique_buffer_vec_t*
get_buffers()
{
    static auto*& _v = common::static_object<unique_buffer_vec_t>::construct(
        reserve_size_t{unique_buffer_vec_t::chunk_size});
    return _v;
}

instance*
get_buffer(rocprofiler_buffer_id_t buffer_id)
{
    if(is_valid_buffer_id(buffer_id) && get_buffers())
    {
        for(auto& itr : *get_buffers())
        {
            if(itr && itr->buffer_id == buffer_id.handle)
            {
                return itr.get();
            }
        }
    }
    return nullptr;
}

std::optional<rocprofiler_buffer_id_t>
allocate_buffer()
{
    if(registration::get_fini_status() > 0) return std::nullopt;

    // ensure buffer has thread to handle flushing it
    static auto _init_threads_once = std::once_flag{};
    std::call_once(_init_threads_once, []() { internal_threading::initialize(); });

    // ... allocate any internal space needed to handle another context ...
    auto _lk = std::unique_lock<std::mutex>{get_buffers_mutex()};

    // initial context identifier number
    auto _idx = get_buffer_offset() + CHECK_NOTNULL(get_buffers())->size();

    // make space in registered
    CHECK_NOTNULL(get_buffers())->emplace_back(nullptr);

    // create an entry in the registered
    auto& _cfg_v = CHECK_NOTNULL(get_buffers())->back();
    _cfg_v       = std::make_unique<buffer::instance>();
    auto* _cfg   = _cfg_v.get();

    if(!_cfg) return std::nullopt;

    // set the buffer id value
    _cfg_v->buffer_id = _idx;

    return rocprofiler_buffer_id_t{_idx};
}

void
launch_flush_task(rocprofiler_buffer_id_t buffer_id)
{
    if(registration::get_fini_status() > 0)
    {
        ROCP_ERROR << "ignoring rocprofiler buffer flush task launch (handle=" << buffer_id.handle
                   << ") after finalization";
        return;
    }

    auto offset = get_buffer_offset();

    if(!is_valid_buffer_id(buffer_id)) return;

    auto* buff = get_buffer(buffer_id);
    if(!buff) return;

    // Check if we should launch a flush task (under lock)
    {
        auto lock = std::unique_lock<std::mutex>{buff->mutex};

        // Already running - no need to launch another
        if(buff->flush_running.load(std::memory_order_relaxed)) return;

        // Nothing to flush
        if(buff->flushing.empty()) return;

        // Mark as running before releasing lock
        buff->flush_running.store(true, std::memory_order_relaxed);
    }

    auto* task_group =
        internal_threading::get_task_group(rocprofiler_callback_thread_t{buff->task_group_id});

    ROCP_FATAL_IF(!task_group)
        << "buffer (" << buffer_id.handle
        << ") flush request received after the task group for handling request was destroyed";

    ROCP_INFO << fmt::format("launching buffer flush task [id={}]...", buffer_id.handle);

    auto _task = [buffer_id, offset]() {
        ROCP_ERROR_IF(registration::get_fini_status() > 0)
            << "executing buffer (" << buffer_id.handle << ") flush task during finalization!";

        auto& buff_v = CHECK_NOTNULL(get_buffers())->at(buffer_id.handle - offset);

        // Process all buffers in the flushing queue
        while(true)
        {
            instance::buffer_t buf_to_flush;

            // Pop a buffer from the flushing queue
            {
                auto lock = std::unique_lock<std::mutex>{buff_v->mutex};

                if(buff_v->flushing.empty())
                {
                    // No more buffers to flush - mark task as not running
                    buff_v->flush_running.store(false, std::memory_order_relaxed);
                    ROCP_INFO << fmt::format("flush task complete [id={}]", buffer_id.handle);
                    return;
                }

                buf_to_flush = std::move(buff_v->flushing.front());
                buff_v->flushing.pop_front();
            }

            // Process buffer without holding lock
            if(!buf_to_flush.is_empty())
            {
                constexpr auto clear_buffer_v = std::true_type{};

                auto num_processed = buf_to_flush.process_record_headers(
                    clear_buffer_v, [&buffer_id, &offset, &buff_v](auto&& _headers) {
                        try
                        {
                            if(buff_v->callback)
                            {
                                ROCP_INFO << fmt::format(
                                    "invoking buffer callback for {} records [buffer_id={}, "
                                    "offset={}]",
                                    _headers.size(),
                                    buffer_id.handle,
                                    offset);
                                buff_v->callback(rocprofiler_context_id_t{buff_v->context_id},
                                                 rocprofiler_buffer_id_t{buff_v->buffer_id},
                                                 _headers.data(),
                                                 _headers.size(),
                                                 buff_v->callback_data,
                                                 buff_v->drop_count);
                            }
                            else
                            {
                                ROCP_TRACE << fmt::format(
                                    "no buffer callback for {} records [buffer_id={}, offset={}]",
                                    _headers.size(),
                                    buffer_id.handle,
                                    offset);
                            }
                        } catch(std::exception& e)
                        {
                            ROCP_CI_LOG(ERROR) << fmt::format(
                                "buffer callback threw an exception: {} [buffer_id={}, "
                                "offset={}, context_id={}, callback={}, callback_data={}, "
                                "records={}, first_record_ptr={}, drop_count={}]",
                                e.what(),
                                buffer_id.handle,
                                offset,
                                buff_v->context_id,
                                reinterpret_cast<const void*>(buff_v->callback),
                                buff_v->callback_data,
                                _headers.size(),
                                (!_headers.empty() ? static_cast<const void*>(&_headers[0])
                                                   : nullptr),
                                buff_v->drop_count.load());
                        }
                    });

                ROCP_INFO << fmt::format(
                    "completed buffer callback for {} records [buffer_id={}, offset={}]",
                    num_processed,
                    buffer_id.handle,
                    offset);
            }

            // Return buffer to pool and notify waiting writers
            {
                auto lock = std::unique_lock<std::mutex>{buff_v->mutex};
                buff_v->buffers.push_back(std::move(buf_to_flush));
            }
            buff_v->buffer_avail.notify_all();
        }
    };

    task_group->exec(std::move(_task));
}

rocprofiler_status_t
flush(rocprofiler_buffer_id_t buffer_id, bool wait)
{
    if(registration::get_fini_status() > 0)
    {
        ROCP_ERROR << "ignoring rocprofiler buffer flush (handle=" << buffer_id.handle
                   << ") request after finalization";
        return ROCPROFILER_STATUS_ERROR_FINALIZED;
    }

    // During finalization, always wait
    if(registration::get_fini_status() < 0 && !wait) wait = true;

    if(!is_valid_buffer_id(buffer_id)) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    auto* buff = get_buffer(buffer_id);
    if(!buff) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    auto* task_group =
        internal_threading::get_task_group(rocprofiler_callback_thread_t{buff->task_group_id});

    ROCP_FATAL_IF(!task_group)
        << "buffer (" << buffer_id.handle
        << ") flush request received after the task group for handling request was destroyed";

    ROCP_INFO << fmt::format("executing buffer flush [id={}, wait={}]...", buffer_id.handle, wait);

    // Move current active buffer to flushing queue (if it has data)
    {
        auto lock = std::unique_lock<std::mutex>{buff->mutex};

        if(!buff->buffers.empty() && !buff->buffers.front().is_empty())
        {
            buff->flushing.push_back(std::move(buff->buffers.front()));
            buff->buffers.pop_front();
        }
    }

    // Launch flush task (will check if already running internally)
    launch_flush_task(buffer_id);

    if(wait)
    {
        task_group->wait();
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace buffer
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_create_buffer(rocprofiler_context_id_t        context,
                          size_t                          size,
                          size_t                          watermark,
                          rocprofiler_buffer_policy_t     action,
                          rocprofiler_buffer_tracing_cb_t callback,
                          void*                           callback_data,
                          rocprofiler_buffer_id_t*        buffer_id)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* existing_buff = rocprofiler::buffer::get_buffer(*buffer_id);
    if(existing_buff)
    {
        ROCP_ERROR << "buffer (handle=" << buffer_id->handle
                   << ") already allocated: handle=" << existing_buff->buffer_id;
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
    }

    auto opt_buff_id = rocprofiler::buffer::allocate_buffer();
    if(!opt_buff_id) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
    buffer_id->handle = opt_buff_id->handle;

    auto& buff = CHECK_NOTNULL(rocprofiler::buffer::get_buffers())
                     ->at(opt_buff_id->handle - rocprofiler::buffer::get_buffer_offset());

    // Initialize the buffer pool (default 2 buffers)
    buff->buffer_size = size;
    buff->pool_size   = 2;
    for(size_t i = 0; i < buff->pool_size; ++i)
    {
        buff->buffers.emplace_back();
        buff->buffers.back().allocate(size);
    }

    buff->watermark     = watermark;
    buff->policy        = action;
    buff->callback      = callback;
    buff->callback_data = callback_data;
    buff->context_id    = context.handle;
    buff->buffer_id     = buffer_id->handle;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_flush_buffer(rocprofiler_buffer_id_t buffer_id)
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    // Drain internal PC sampling buffers, if needed.
    auto status = rocprofiler::pc_sampling::flush_internal_agent_buffers(buffer_id);
    if(status != ROCPROFILER_STATUS_SUCCESS) return status;
#endif

    return rocprofiler::buffer::flush(buffer_id, true);
}

rocprofiler_status_t
rocprofiler_destroy_buffer(rocprofiler_buffer_id_t buffer_id)
{
    if(!rocprofiler::buffer::is_valid_buffer_id(buffer_id))
        return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    auto  offset       = rocprofiler::buffer::get_buffer_offset();
    auto* buffers_list = CHECK_NOTNULL(rocprofiler::buffer::get_buffers());
    auto& buff         = buffers_list->at(buffer_id.handle - offset);

    if(!buff) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    // Try to acquire the mutex (non-blocking)
    auto lock = std::unique_lock<std::mutex>{buff->mutex, std::try_to_lock};
    if(!lock.owns_lock())
    {
        return ROCPROFILER_STATUS_ERROR_BUFFER_BUSY;
    }

    // Check if flush is in progress
    if(buff->flush_running.load(std::memory_order_relaxed) || !buff->flushing.empty())
    {
        return ROCPROFILER_STATUS_ERROR_BUFFER_BUSY;
    }

    // Reset all buffers in the pool
    for(auto& buf : buff->buffers)
    {
        buf.reset();
    }
    buff->buffers.clear();
    buff->flushing.clear();

    lock.unlock();

    buff.reset();

    return ROCPROFILER_STATUS_SUCCESS;
}
}
