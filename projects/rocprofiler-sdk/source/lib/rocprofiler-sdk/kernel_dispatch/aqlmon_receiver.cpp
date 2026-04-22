// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/kernel_dispatch/aqlmon_receiver.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <aqlmon/aqlmon.h>

#include <hsa/hsa.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <limits>
#include <thread>
#include <unordered_map>

namespace rocprofiler
{
namespace kernel_dispatch
{
namespace
{
using namespace std::chrono_literals;

constexpr auto kSleepInterval = 100us;

struct dispatch_key_t
{
    uint32_t pid         = 0;
    uint64_t queue_id    = 0;
    uint64_t dispatch_id = 0;

    bool operator==(const dispatch_key_t& rhs) const
    {
        return pid == rhs.pid && queue_id == rhs.queue_id && dispatch_id == rhs.dispatch_id;
    }
};

struct dispatch_key_hash_t
{
    size_t operator()(const dispatch_key_t& value) const
    {
        size_t seed = static_cast<size_t>(value.pid);
        seed ^= std::hash<uint64_t>{}(value.queue_id) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
                (seed >> 2);
        seed ^= std::hash<uint64_t>{}(value.dispatch_id) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
                (seed >> 2);
        return seed;
    }
};

struct packet_info_t
{
    uint32_t pid               = 0;
    uint32_t tid               = 0;
    uint64_t queue_id          = 0;
    uint64_t dispatch_id       = 0;
    uint64_t kernel_object     = 0;
    uint64_t completion_signal = 0;
    uint64_t agent_handle      = 0;
    uint8_t  packet_bytes[AQLMON_PACKET_BYTES] = {};
};

const char*
shm_name_from_env()
{
    if(const char* value = std::getenv("ROCPROFILER_AQLMON_SHM_NAME");
       value && value[0] != '\0')
        return value;
    if(const char* value = std::getenv("AQLMONITOR_SHM_NAME"); value && value[0] != '\0')
        return value;
    return nullptr;
}

bool
legacy_queue_intercept_required()
{
    const auto queue_intercept_service_filter = [](const context::context* ctx) {
        return ctx->dispatch_counter_collection || ctx->pc_sampler || ctx->dispatch_thread_trace ||
               ctx->device_counter_collection || ctx->device_thread_trace;
    };

    return !context::get_registered_contexts(queue_intercept_service_filter).empty() ||
           !context::get_active_contexts(queue_intercept_service_filter).empty();
}

uint64_t
atomic_load_u64(const uint64_t* value, int order = __ATOMIC_ACQUIRE)
{
    return __atomic_load_n(value, order);
}

bool
has_signal_timestamps(const aqlmon_record_t& record)
{
    return (record.flags & AQLMON_FLAG_SIGNAL_TIMESTAMPS_VALID) != 0 &&
           record.dispatch_start_ns != 0 && record.dispatch_end_ns != 0 &&
           record.dispatch_start_ns <= record.dispatch_end_ns;
}

rocprofiler_kernel_dispatch_info_t
make_dispatch_info(const packet_info_t& packet)
{
    auto dispatch_info = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
    dispatch_info.queue_id    = rocprofiler_queue_id_t{.handle = packet.queue_id};
    dispatch_info.dispatch_id = packet.dispatch_id;
    dispatch_info.kernel_id   = code_object::get_kernel_id(packet.kernel_object);

    if(packet.agent_handle != 0)
    {
        if(const auto* agent =
               agent::get_rocprofiler_agent(hsa_agent_t{.handle = packet.agent_handle});
           agent != nullptr)
            dispatch_info.agent_id = agent->id;
    }

    const auto* dispatch_packet =
        reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet.packet_bytes);
    dispatch_info.private_segment_size = dispatch_packet->private_segment_size;
    dispatch_info.group_segment_size   = dispatch_packet->group_segment_size;
    dispatch_info.workgroup_size       = rocprofiler_dim3_t{dispatch_packet->workgroup_size_x,
                                                      dispatch_packet->workgroup_size_y,
                                                      dispatch_packet->workgroup_size_z};
    dispatch_info.grid_size = rocprofiler_dim3_t{dispatch_packet->grid_size_x,
                                                 dispatch_packet->grid_size_y,
                                                 dispatch_packet->grid_size_z};

    return dispatch_info;
}

void
emit_dispatch_record(const packet_info_t& packet, const aqlmon_record_t& completion)
{
    auto tracing_data = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                               tracing_data,
                               std::true_type{});
    if(tracing_data.empty()) return;

    auto dispatch_info = make_dispatch_info(packet);
    auto thread_id     = rocprofiler_thread_id_t{packet.tid};

    auto callback_data =
        common::init_public_api_struct(rocprofiler_callback_tracing_kernel_dispatch_data_t{});
    callback_data.start_timestamp = completion.dispatch_start_ns;
    callback_data.end_timestamp   = completion.dispatch_end_ns;
    callback_data.dispatch_info   = dispatch_info;

    tracing::execute_phase_none_callbacks(tracing_data.callback_contexts,
                                          thread_id,
                                          0,
                                          tracing_data.external_correlation_ids,
                                          0,
                                          ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                          ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                          callback_data);

    auto buffer_record =
        common::init_public_api_struct(rocprofiler_buffer_tracing_kernel_dispatch_record_t{});
    buffer_record.start_timestamp = completion.dispatch_start_ns;
    buffer_record.end_timestamp   = completion.dispatch_end_ns;
    buffer_record.dispatch_info   = dispatch_info;

    tracing::execute_buffer_record_emplace(tracing_data.buffered_contexts,
                                           thread_id,
                                           0,
                                           tracing_data.external_correlation_ids,
                                           0,
                                           ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                           ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                           buffer_record);
}

class aqlmon_receiver
{
public:
    ~aqlmon_receiver() { finalize(); }

    bool requested() const { return shm_name_from_env() != nullptr; }

    bool operational() const { return m_operational.load(std::memory_order_acquire); }

    void start()
    {
        if(!requested()) return;

        auto lock = std::lock_guard<std::mutex>{m_mutex};
        if(m_running) return;

        if(open_mapping())
        {
            m_operational.store(true, std::memory_order_release);
        }

        m_stop.store(false, std::memory_order_release);
        m_running = true;
        m_thread  = std::thread{[this]() { run(); }};
    }

    void finalize()
    {
        {
            auto lock = std::lock_guard<std::mutex>{m_mutex};
            if(!m_running) return;
            m_stop.store(true, std::memory_order_release);
        }

        if(m_thread.joinable()) m_thread.join();

        auto lock = std::lock_guard<std::mutex>{m_mutex};
        close_mapping();
        m_packets.clear();
        m_running = false;
    }

private:
    void run()
    {
        while(!m_stop.load(std::memory_order_acquire))
        {
            if(!open_mapping())
            {
                m_operational.store(false, std::memory_order_release);
                std::this_thread::sleep_for(kSleepInterval);
                continue;
            }

            m_operational.store(true, std::memory_order_release);

            bool progress   = false;
            auto write_seq = atomic_load_u64(&m_header->write_seq);
            if(write_seq > (m_next_seq + m_header->capacity))
            {
                m_packets.clear();
                m_next_seq = write_seq - m_header->capacity;
            }

            while(m_next_seq < write_seq)
            {
                const auto& record = m_records[m_next_seq % m_header->capacity];
                if(atomic_load_u64(&record.seq) != (m_next_seq + 1))
                {
                    ++m_next_seq;
                    continue;
                }

                process_record(record);
                ++m_next_seq;
                progress = true;
            }

            if(!progress) std::this_thread::sleep_for(kSleepInterval);
        }
    }

    bool open_mapping()
    {
        if(m_header != nullptr && m_records != nullptr) return true;

        const char* shm_name = shm_name_from_env();
        if(shm_name == nullptr) return false;

        if(m_fd < 0)
        {
            m_fd = shm_open(shm_name, O_RDONLY, 0);
            if(m_fd < 0) return false;
        }

        struct stat st
        {};
        if(fstat(m_fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(aqlmon_shm_header_t)))
            return false;

        if(m_mapping == nullptr)
        {
            m_mapping_size = static_cast<size_t>(st.st_size);
            m_mapping      = mmap(nullptr, m_mapping_size, PROT_READ, MAP_SHARED, m_fd, 0);
            if(m_mapping == MAP_FAILED)
            {
                m_mapping      = nullptr;
                m_mapping_size = 0;
                return false;
            }
        }

        m_header = static_cast<const aqlmon_shm_header_t*>(m_mapping);
        const auto magic = atomic_load_u64(&m_header->magic);
        if(magic == 0 || m_header->version == 0 || m_header->header_size == 0 ||
           m_header->record_size == 0)
        {
            close_mapping();
            return false;
        }

        if(magic != AQLMON_MAGIC || m_header->version != AQLMON_SHM_VERSION ||
           m_header->header_size != sizeof(aqlmon_shm_header_t) ||
           m_header->record_size != sizeof(aqlmon_record_t))
        {
            ROCP_WARNING << "[aqlmon_receiver] incompatible AQLMON shared-memory header";
            close_mapping();
            return false;
        }

        if(m_header->capacity == 0 ||
           m_header->record_size >
               (std::numeric_limits<size_t>::max() - m_header->header_size) / m_header->capacity)
        {
            ROCP_WARNING << "[aqlmon_receiver] invalid AQLMON shared-memory capacity";
            close_mapping();
            return false;
        }

        const auto expected_mapping_size =
            m_header->header_size + (static_cast<size_t>(m_header->capacity) * m_header->record_size);
        if(m_mapping_size < expected_mapping_size)
        {
            ROCP_WARNING << "[aqlmon_receiver] truncated AQLMON shared-memory mapping";
            close_mapping();
            return false;
        }

        m_records = reinterpret_cast<const aqlmon_record_t*>(
            static_cast<const uint8_t*>(m_mapping) + sizeof(aqlmon_shm_header_t));
        const auto write_seq = atomic_load_u64(&m_header->write_seq);
        m_next_seq           = (write_seq > m_header->capacity) ? (write_seq - m_header->capacity)
                                                                : 0;
        return true;
    }

    void close_mapping()
    {
        if(m_mapping != nullptr)
        {
            munmap(m_mapping, m_mapping_size);
            m_mapping      = nullptr;
            m_mapping_size = 0;
        }
        if(m_fd >= 0)
        {
            close(m_fd);
            m_fd = -1;
        }
        m_header  = nullptr;
        m_records = nullptr;
        m_next_seq = 0;
        m_packets.clear();
        m_operational.store(false, std::memory_order_release);
    }

    void process_record(const aqlmon_record_t& record)
    {
        if(record.kind == AQLMON_RECORD_PACKET &&
           record.packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH)
        {
            if(record.completion_signal == 0) return;

            auto packet = packet_info_t{};
            packet.pid               = record.pid;
            packet.tid               = record.tid;
            packet.queue_id          = record.queue_id;
            packet.dispatch_id       = record.dispatch_id;
            packet.kernel_object     = record.kernel_object;
            packet.completion_signal = record.completion_signal;
            packet.agent_handle      = record.agent_handle;
            std::memcpy(packet.packet_bytes, record.packet_bytes, sizeof(packet.packet_bytes));
            m_packets[{packet.pid, packet.queue_id, packet.dispatch_id}] = packet;
            return;
        }

        if(record.kind == AQLMON_RECORD_DISPATCH_COMPLETE && has_signal_timestamps(record))
        {
            const dispatch_key_t key{record.pid, record.queue_id, record.dispatch_id};
            auto                 itr = m_packets.find(key);
            if(itr == m_packets.end()) return;
            emit_dispatch_record(itr->second, record);
            m_packets.erase(itr);
            return;
        }

        if(record.kind == AQLMON_RECORD_DISPATCH_COMPLETE)
        {
            const dispatch_key_t key{record.pid, record.queue_id, record.dispatch_id};
            m_packets.erase(key);
        }
    }

    std::mutex m_mutex = {};
    std::atomic<bool> m_stop{false};
    std::atomic<bool> m_operational{false};
    bool              m_running = false;
    std::thread       m_thread  = {};

    int                       m_fd           = -1;
    size_t                    m_mapping_size = 0;
    void*                     m_mapping      = nullptr;
    const aqlmon_shm_header_t* m_header      = nullptr;
    const aqlmon_record_t*     m_records     = nullptr;
    uint64_t                  m_next_seq     = 0;

    std::unordered_map<dispatch_key_t, packet_info_t, dispatch_key_hash_t> m_packets = {};
};

aqlmon_receiver&
get_receiver()
{
    if(auto* receiver = common::static_object<aqlmon_receiver>::get(); receiver != nullptr)
        return *receiver;
    return *common::static_object<aqlmon_receiver>::construct();
}
}  // namespace

bool
aqlmon_receiver_enabled()
{
    return aqlmon_receiver_allowed() && aqlmon_receiver_operational();
}

bool
aqlmon_receiver_requested()
{
    return shm_name_from_env() != nullptr && aqlmon_receiver_allowed();
}

bool
aqlmon_receiver_operational()
{
    return get_receiver().operational();
}

bool
aqlmon_receiver_allowed()
{
    return !legacy_queue_intercept_required();
}

void
aqlmon_receiver_start()
{
    get_receiver().start();
}

void
aqlmon_receiver_finalize()
{
    if(auto* receiver = common::static_object<aqlmon_receiver>::get(); receiver != nullptr)
        receiver->finalize();
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
