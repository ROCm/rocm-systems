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

#include "generateOTF2.hpp"
#include "output_stream.hpp"
#include "timestamps.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/perfetto.hpp>

#include <fmt/format.h>

#include <otf2/OTF2_AttributeList.h>
#include <otf2/OTF2_AttributeValue.h>
#include <otf2/OTF2_Definitions.h>
#include <otf2/OTF2_GeneralDefinitions.h>
#include <otf2/OTF2_Pthread_Locks.h>
#include <otf2/otf2.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <future>
#include <map>
#include <thread>
#include <unordered_map>
#include <utility>

#define OTF2_CHECK(result)                                                                         \
    {                                                                                              \
        OTF2_ErrorCode ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                       \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != OTF2_SUCCESS)                            \
        {                                                                                          \
            auto _err_name = OTF2_Error_GetName(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));      \
            auto _err_msg =                                                                        \
                OTF2_Error_GetDescription(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));            \
            ROCP_FATAL << #result << " failed with error code " << _err_name                       \
                       << " (code=" << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__)                 \
                       << ") :: " << _err_msg;                                                     \
        }                                                                                          \
    }

namespace rocprofiler
{
namespace tool
{
namespace
{
template <typename Tp, size_t N>
struct array_hash
{
    size_t operator()(const std::array<Tp, N>& _data) const
    {
        constexpr size_t seed = 0x9e3779b9;
        size_t           _val = 0;
        for(const auto& itr : _data)
            _val ^= std::hash<Tp>{}(itr) + seed + (_val << 6) + (_val >> 2);
        return _val;
    }

    template <typename... Up>
    size_t operator()(Up... _data) const
    {
        static_assert(sizeof...(Up) == N, "Insufficient data");
        return operator()(std::array<Tp, N>{std::forward<Up>(_data)...});
    }
};

struct region_info
{
    std::string          name        = {};
    OTF2_RegionRole_enum region_role = OTF2_REGION_ROLE_FUNCTION;
    OTF2_Paradigm_enum   paradigm    = OTF2_PARADIGM_HIP;
};

OTF2_FlushType
pre_flush(void*            userData,
          OTF2_FileType    fileType,
          OTF2_LocationRef location,
          void*            callerData,
          bool             fini);

OTF2_TimeStamp
post_flush(void* userData, OTF2_FileType fileType, OTF2_LocationRef location);

template <typename... Args>
void
consume_variables(Args&&...)
{}

using event_writer_t   = OTF2_EvtWriter;
using archive_t        = OTF2_Archive;
using attribute_list_t = OTF2_AttributeList;
using hash_value_t     = size_t;
using hash_map_t       = std::unordered_map<hash_value_t, region_info>;

auto       main_tid        = common::get_tid();
archive_t* archive         = nullptr;
auto       flush_callbacks = OTF2_FlushCallbacks{pre_flush, post_flush};

enum rocprofiler_location_type_t
{
    ROCPROFILER_AGENT_NO_TYPE = 0,
    ROCPROFILER_AGENT_MEMORY_COPY_TYPE,
    ROCPROFILER_AGENT_DISPATCH_TYPE,
    ROCPROFILER_AGENT_MEMORY_ALLOC_TYPE
};

struct location_base
{
    uint64_t                    pid   = 0;
    rocprofiler_thread_id_t     tid   = 0;
    rocprofiler_agent_id_t      agent = {.handle = 0};
    rocprofiler_queue_id_t      queue = {.handle = 0};
    rocprofiler_location_type_t type  = ROCPROFILER_AGENT_NO_TYPE;

    location_base(uint64_t                    _pid,
                  rocprofiler_thread_id_t     _tid,
                  rocprofiler_agent_id_t      _agent = {.handle = 0},
                  rocprofiler_location_type_t _type  = ROCPROFILER_AGENT_NO_TYPE,
                  rocprofiler_queue_id_t      _queue = {.handle = 0})
    : pid{_pid}
    , tid{_tid}
    , agent{_agent}
    , queue{_queue}
    , type{_type}
    {}

    auto hash() const
    {
        return array_hash<uint64_t, 5>{}(pid, tid, agent.handle + 1, queue.handle + 1, type);
    }
};

bool
operator<(const location_base& lhs, const location_base& rhs)
{
    return std::tie(lhs.pid, lhs.tid, lhs.agent.handle, lhs.queue.handle, lhs.type) <
           std::tie(rhs.pid, rhs.tid, rhs.agent.handle, rhs.queue.handle, rhs.type);
}

struct location_data : location_base
{
    location_data(uint64_t                    _pid,
                  rocprofiler_thread_id_t     _tid,
                  rocprofiler_agent_id_t      _agent = {.handle = 0},
                  rocprofiler_location_type_t _type  = ROCPROFILER_AGENT_NO_TYPE,
                  rocprofiler_queue_id_t      _queue = {.handle = 0})
    : location_base{_pid, _tid, _agent, _type, _queue}
    , index{++index_counter}
    , event_writer{OTF2_Archive_GetEvtWriter(CHECK_NOTNULL(archive), index)}
    {
        CHECK_NOTNULL(event_writer);
    }

    using location_base::hash;

    static uint64_t index_counter;

    uint64_t        index        = 0;
    event_writer_t* event_writer = nullptr;

    bool operator==(const location_base& rhs) const { return (hash() == rhs.hash()); }
};

uint64_t location_data::index_counter = 0;

OTF2_TimeStamp
get_time()
{
    auto _ts = rocprofiler_timestamp_t{};
    rocprofiler_get_timestamp(&_ts);
    return static_cast<OTF2_TimeStamp>(_ts);
}

auto&
get_locations()
{
    static auto _v = std::vector<std::unique_ptr<location_data>>{};
    return _v;
}

const location_data*
get_location(const location_base& _location, bool _init = false)
{
    for(auto& itr : get_locations())
        if(*itr == _location) return itr.get();

    if(_init)
        return get_locations()
            .emplace_back(std::make_unique<location_data>(
                _location.pid, _location.tid, _location.agent, _location.type, _location.queue))
            .get();

    return nullptr;
}

event_writer_t*
get_event_writer(const location_base& _location, bool _init = false)
{
    const auto* _loc = get_location(_location, _init);
    return (_loc) ? _loc->event_writer : nullptr;
}

OTF2_FlushType
pre_flush(void*            userData,
          OTF2_FileType    fileType,
          OTF2_LocationRef location,
          void*            callerData,
          bool             fini)
{
    consume_variables(userData, fileType, location, callerData, fini);
    return OTF2_FLUSH;
}

OTF2_TimeStamp
post_flush(void* userData, OTF2_FileType fileType, OTF2_LocationRef location)
{
    consume_variables(userData, fileType, location);
    return get_time();
}

template <typename Tp>
size_t
get_hash_id(Tp&& _val)
{
    using value_type = common::mpl::unqualified_type_t<Tp>;

    if constexpr(!std::is_pointer<Tp>::value)
        return std::hash<value_type>{}(std::forward<Tp>(_val));
    else if constexpr(std::is_same<value_type, const char*>::value ||
                      std::is_same<value_type, char*>::value)
        return get_hash_id(std::string_view{_val});
    else
        return get_hash_id(*_val);
}

template <typename... Args>
auto
add_event(std::string_view             name,
          const location_base&         _location,
          rocprofiler_callback_phase_t _phase,
          OTF2_TimeStamp               _ts,
          attribute_list_t*            _attributes = nullptr)
{
    auto* evt_writer = get_event_writer(_location, true);
    auto  _hash      = get_hash_id(name);

    if(_phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        OTF2_CHECK(OTF2_EvtWriter_Enter(evt_writer, _attributes, _ts, _hash))
    else if(_phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        OTF2_CHECK(OTF2_EvtWriter_Leave(evt_writer, _attributes, _ts, _hash))
    else
        ROCP_FATAL << "otf2::add_event phase is not enter or exit";
}

void
setup(const output_config& cfg)
{
    namespace fs = common::filesystem;

    auto _filename = get_output_filename(cfg, "results", std::string_view{});
    auto _filepath = fs::path{_filename};
    auto _name     = _filepath.filename().string();
    auto _path     = _filepath.parent_path().string();

    if(fs::exists(_filepath)) fs::remove_all(_filepath);

    constexpr uint64_t evt_chunk_size = 2 * common::units::MB;
    constexpr uint64_t def_chunk_size = 8 * common::units::MB;

    archive = OTF2_Archive_Open(_path.c_str(),
                                _name.c_str(),
                                OTF2_FILEMODE_WRITE,
                                evt_chunk_size,  // event chunk size
                                def_chunk_size,  // def chunk size
                                OTF2_SUBSTRATE_POSIX,
                                OTF2_COMPRESSION_NONE);

    OTF2_CHECK(OTF2_Archive_SetFlushCallbacks(archive, &flush_callbacks, nullptr));
    OTF2_CHECK(OTF2_Archive_SetSerialCollectiveCallbacks(archive));
    OTF2_CHECK(OTF2_Pthread_Archive_SetLockingCallbacks(archive, nullptr));
    OTF2_CHECK(OTF2_Archive_OpenEvtFiles(archive));

    ROCP_ERROR << "Opened result file: " << _filename << ".otf2";
}

void
shutdown()
{
    OTF2_CHECK(OTF2_Archive_Close(archive));
}

struct event_info
{
    explicit event_info(location_base&& _loc)
    : m_location{tool::get_location(std::forward<location_base>(_loc), true)}
    {}

    auto                 id() const { return m_location->index; }
    auto                 hash() const { return m_location->hash(); }
    const location_base* get_location() const { return m_location; }

    std::string name        = {};
    uint64_t    event_count = 0;

private:
    const location_data* m_location = nullptr;
};

template <typename Tp>
attribute_list_t*
create_attribute_list()
{
    auto* _val = OTF2_AttributeList_New();

    const auto* _name = sdk::perfetto_category<Tp>::name;
    auto        _hash = get_hash_id(_name);

    auto _attr_value      = OTF2_AttributeValue{};
    _attr_value.stringRef = _hash;
    OTF2_AttributeList_AddAttribute(_val, 0, OTF2_TYPE_STRING, _attr_value);

    return _val;
}
}  // namespace

void
write_otf2(const output_config&                                          cfg,
           const metadata&                                               tool_metadata,
           uint64_t                                                      pid,
           const std::vector<agent_info>&                                agent_data,
           std::deque<tool_buffer_tracing_hip_api_ext_record_t>*         hip_api_data,
           std::deque<rocprofiler_buffer_tracing_hsa_api_record_t>*      hsa_api_data,
           std::deque<tool_buffer_tracing_kernel_dispatch_ext_record_t>* kernel_dispatch_data,
           std::deque<tool_buffer_tracing_memory_copy_ext_record_t>*     memory_copy_data,
           std::deque<rocprofiler_buffer_tracing_marker_api_record_t>*   marker_api_data,
           std::deque<rocprofiler_buffer_tracing_scratch_memory_record_t>* /*scratch_memory_data*/,
           std::deque<rocprofiler_buffer_tracing_rccl_api_ext_record_t>*   rccl_api_ext_data,
           std::deque<tool_buffer_tracing_memory_allocation_ext_record_t>* memory_allocation_data,
           std::deque<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>* rocdecode_api_data,
           std::deque<rocprofiler_buffer_tracing_rocjpeg_api_record_t>*       rocjpeg_api_data)
{
    namespace sdk = ::rocprofiler::sdk;

    setup(cfg);

    auto _app_ts    = timestamps_t{tool_metadata.process_start_ns, tool_metadata.process_end_ns};
    auto agents_map = tool_metadata.agents_map;

    const auto  kernel_sym_data = tool_metadata.get_kernel_symbols();
    const auto& buffer_names    = tool_metadata.buffer_names;
    auto        tids            = std::set<rocprofiler_thread_id_t>{};
    auto agent_thread_ids = std::map<rocprofiler_thread_id_t, std::set<rocprofiler_agent_id_t>>{};
    auto agent_thread_ids_alloc =
        std::map<rocprofiler_thread_id_t, std::set<rocprofiler_agent_id_t>>{};
    auto agent_queue_ids =
        std::map<rocprofiler_thread_id_t,
                 std::map<rocprofiler_agent_id_t, std::unordered_set<rocprofiler_queue_id_t>>>{};

    auto thread_event_info = std::map<rocprofiler_thread_id_t, event_info>{};
    auto agent_memcpy_info =
        std::map<rocprofiler_thread_id_t, std::map<rocprofiler_agent_id_t, event_info>>{};
    auto agent_memalloc_info =
        std::map<rocprofiler_thread_id_t, std::map<rocprofiler_agent_id_t, event_info>>{};
    auto agent_dispatch_info =
        std::map<rocprofiler_thread_id_t,
                 std::map<rocprofiler_agent_id_t, std::map<rocprofiler_queue_id_t, event_info>>>{};

    auto _get_agent = [&agent_data](rocprofiler_agent_id_t _id) -> const rocprofiler_agent_t* {
        for(const auto& itr : agent_data)
            if(_id == itr.id) return &itr;
        return CHECK_NOTNULL(nullptr);
    };

    auto _get_kernel_sym_data =
        [&kernel_sym_data](
            const rocprofiler_kernel_dispatch_info_t& _info) -> const kernel_symbol_info* {
        for(const auto& kitr : kernel_sym_data)
            if(kitr.kernel_id == _info.kernel_id) return &kitr;
        return CHECK_NOTNULL(nullptr);
    };

    {
        for(auto itr : *hsa_api_data)
            tids.emplace(itr.thread_id);
        for(auto itr : *hip_api_data)
            tids.emplace(itr.thread_id);
        for(auto itr : *marker_api_data)
            tids.emplace(itr.thread_id);
        for(auto itr : *rccl_api_ext_data)
            tids.emplace(itr.thread_id);
        for(auto itr : *rocdecode_api_data)
            tids.emplace(itr.thread_id);
        for(auto itr : *rocjpeg_api_data)
            tids.emplace(itr.thread_id);

        for(auto itr : *memory_copy_data)
        {
            tids.emplace(itr.thread_id);
            agent_thread_ids[itr.thread_id].emplace(itr.dst_agent_id);
        }

        for(auto itr : *memory_allocation_data)
        {
            tids.emplace(itr.thread_id);
            agent_thread_ids_alloc[itr.thread_id].emplace(itr.agent_id);
        }

        for(auto itr : *kernel_dispatch_data)
        {
            tids.emplace(itr.thread_id);
            agent_queue_ids[itr.thread_id][itr.dispatch_info.agent_id].emplace(
                itr.dispatch_info.queue_id);
        }
    }

    {
        for(auto itr : tids)
            thread_event_info.emplace(itr, location_base{pid, itr});

        for(const auto& [tid, itr] : agent_thread_ids)
            for(auto agent : itr)
                agent_memcpy_info[tid].emplace(
                    agent, location_base{pid, tid, agent, ROCPROFILER_AGENT_MEMORY_COPY_TYPE});

        for(const auto& [tid, itr] : agent_thread_ids_alloc)
            for(auto agent : itr)
                agent_memalloc_info[tid].emplace(
                    agent, location_base{pid, tid, agent, ROCPROFILER_AGENT_MEMORY_ALLOC_TYPE});

        for(const auto& [tid, itr] : agent_queue_ids)
            for(const auto& [agent, qitr] : itr)
                for(auto queue : qitr)
                    agent_dispatch_info[tid][agent].emplace(
                        queue,
                        location_base{pid, tid, agent, ROCPROFILER_AGENT_DISPATCH_TYPE, queue});
    }

    for(auto& [tid, evt] : thread_event_info)
    {
        evt.name = fmt::format("Thread {}", tid);
    }

    for(auto& [tid, itr] : agent_memcpy_info)
    {
        for(auto& [agent, evt] : itr)
        {
            const auto* _agent = _get_agent(agent);
            auto        agent_index_info =
                tool_metadata.get_agent_index(_agent->id, cfg.agent_index_value);
            evt.name = fmt::format("Thread {}, Copy to {} {}",
                                   tid,
                                   std::string{agent_index_info.type},
                                   agent_index_info.as_string("-"));
        }
    }

    for(auto& [tid, itr] : agent_memalloc_info)
    {
        for(auto& [agent, evt] : itr)
        {
            // Free functions do not track agent information. Below handles case where
            // null rocprof agent id is passed to generate OTF2
            constexpr auto             null_rocp_agent_id = rocprofiler_agent_id_t{.handle = 0};
            const rocprofiler_agent_t* _agent             = nullptr;
            if(agent != null_rocp_agent_id)
            {
                _agent = _get_agent(agent);
            }
            if(_agent)
            {
                auto agent_index_info =
                    tool_metadata.get_agent_index(_agent->id, cfg.agent_index_value);
                evt.name = fmt::format("Thread {}, Memory Operation at {} {}",
                                       tid,
                                       agent_index_info.type,
                                       agent_index_info.as_string("-"));
            }
            else
            {
                auto _type_name = std::string_view{"UNK"};
                evt.name = fmt::format("Thread {}, Memory Operation at {} {}", tid, _type_name, 0);
            }
        }
    }

    auto _queue_ids = std::map<rocprofiler_queue_id_t, uint64_t>{};
    for(auto& [tid, itr] : agent_dispatch_info)
        for(auto& [agent, qitr] : itr)
            for(auto& [queue, evt] : qitr)
                _queue_ids.emplace(queue, 0);

    {
        uint64_t _n = 0;
        for(auto& qitr : _queue_ids)
            qitr.second = _n++;
    }

    for(auto& [tid, itr] : agent_dispatch_info)
    {
        for(auto& [agent, qitr] : itr)
        {
            for(auto& [queue, evt] : qitr)
            {
                const auto* _agent = _get_agent(agent);
                auto        agent_index_info =
                    tool_metadata.get_agent_index(_agent->id, cfg.agent_index_value);
                evt.name = fmt::format("Thread {}, Compute on {} {}, Queue {}",
                                       tid,
                                       agent_index_info.type,
                                       agent_index_info.as_string("-"),
                                       _queue_ids.at(queue));
            }
        }
    }

    auto _hash_data = hash_map_t{};

    struct evt_data
    {
        rocprofiler_callback_phase_t phase      = ROCPROFILER_CALLBACK_PHASE_NONE;
        std::string_view             name       = {};
        const location_base*         location   = nullptr;
        uint64_t                     timestamp  = 0;
        OTF2_AttributeList*          attributes = nullptr;
    };

    auto _data     = std::deque<evt_data>{};
    auto _attr_str = std::unordered_map<size_t, std::string_view>{};
    auto get_attr  = [&_attr_str](auto _category) {
        using category_t = common::mpl::unqualified_type_t<decltype(_category)>;
        auto _name       = sdk::perfetto_category<category_t>::name;
        _attr_str.emplace(get_hash_id(_name), _name);
        return create_attribute_list<category_t>();
    };

    // trace events
    {
        auto callbk_name_info = sdk::get_callback_tracing_names();

        auto add_event_data = [&buffer_names,
                               &_hash_data,
                               &_data,
                               &tool_metadata,
                               &thread_event_info,
                               &get_attr](const auto* _inp, auto _attrib) {
            if(!_inp) return;
            for(auto itr : *_inp)
            {
                if(itr.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API &&
                   itr.operation == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxMarkA)
                    continue;

                using value_type = common::mpl::unqualified_type_t<decltype(itr)>;
                auto name        = buffer_names.at(itr.kind, itr.operation);
                auto paradigm    = OTF2_PARADIGM_HIP;
                if constexpr(std::is_same<value_type,
                                          rocprofiler_buffer_tracing_marker_api_record_t>::value)
                {
                    paradigm = OTF2_PARADIGM_USER;
                    if(itr.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API &&
                       itr.operation != ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxGetThreadId)
                        name = tool_metadata.get_marker_message(itr.correlation_id.internal);
                }

                _hash_data.emplace(
                    get_hash_id(name),
                    region_info{std::string{name}, OTF2_REGION_ROLE_FUNCTION, paradigm});

                auto& _evt_info = thread_event_info.at(itr.thread_id);
                _evt_info.event_count += 1;
                _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                            name,
                                            _evt_info.get_location(),
                                            itr.start_timestamp,
                                            get_attr(_attrib)});
                _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                            name,
                                            _evt_info.get_location(),
                                            itr.end_timestamp,
                                            nullptr});
            }
        };

        add_event_data(hsa_api_data, sdk::category::hsa_api{});
        add_event_data(hip_api_data, sdk::category::hip_api{});
        add_event_data(marker_api_data, sdk::category::marker_api{});
        add_event_data(rocjpeg_api_data, sdk::category::rocjpeg_api{});
    }

    for(auto itr : *rocdecode_api_data)
    {
        auto name = buffer_names.at(itr.kind, itr.operation);
        _hash_data.emplace(
            get_hash_id(name),
            region_info{std::string{name}, OTF2_REGION_ROLE_FUNCTION, OTF2_PARADIGM_USER});

        auto& _evt_info = thread_event_info.at(itr.thread_id);
        _evt_info.event_count += 1;

        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                    name,
                                    _evt_info.get_location(),
                                    itr.start_timestamp,
                                    get_attr(sdk::category::rocdecode_api{})});
        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                    name,
                                    _evt_info.get_location(),
                                    itr.end_timestamp,
                                    nullptr});
    }

    for(auto itr : *rccl_api_ext_data)
    {
        auto name = buffer_names.at(itr.kind, itr.operation);
        _hash_data.emplace(
            get_hash_id(name),
            region_info{std::string{name}, OTF2_REGION_ROLE_FUNCTION, OTF2_PARADIGM_USER});

        auto& _evt_info = thread_event_info.at(itr.thread_id);
        _evt_info.event_count += 1;

        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                    name,
                                    _evt_info.get_location(),
                                    itr.start_timestamp,
                                    get_attr(sdk::category::rccl_api{})});
        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                    name,
                                    _evt_info.get_location(),
                                    itr.end_timestamp,
                                    nullptr});
    }

    for(auto itr : *memory_copy_data)
    {
        auto name = buffer_names.at(itr.kind, itr.operation);
        _hash_data.emplace(
            get_hash_id(name),
            region_info{std::string{name}, OTF2_REGION_ROLE_DATA_TRANSFER, OTF2_PARADIGM_HIP});

        // TODO: add attributes for memory copy parameters

        auto& _evt_info = agent_memcpy_info.at(itr.thread_id).at(itr.dst_agent_id);
        _evt_info.event_count += 1;

        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                    name,
                                    _evt_info.get_location(),
                                    itr.start_timestamp,
                                    get_attr(sdk::category::memory_copy{})});
        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                    name,
                                    _evt_info.get_location(),
                                    itr.end_timestamp,
                                    nullptr});
    }

    for(auto itr : *memory_allocation_data)
    {
        auto name = buffer_names.at(itr.kind, itr.operation);
        _hash_data.emplace(
            get_hash_id(name),
            region_info{std::string{name}, OTF2_REGION_ROLE_ALLOCATE, OTF2_PARADIGM_HIP});

        // TODO: add attributes for memory allocation parameters

        auto& _evt_info = agent_memalloc_info.at(itr.thread_id).at(itr.agent_id);
        _evt_info.event_count += 1;

        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                    name,
                                    _evt_info.get_location(),
                                    itr.start_timestamp,
                                    get_attr(sdk::category::memory_allocation{})});
        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                    name,
                                    _evt_info.get_location(),
                                    itr.end_timestamp,
                                    nullptr});
    }

    for(auto itr : *kernel_dispatch_data)
    {
        const auto& info = itr.dispatch_info;
        const auto* sym  = _get_kernel_sym_data(info);
        CHECK(sym != nullptr);

        auto name =
            tool_metadata.get_kernel_name(info.kernel_id, itr.correlation_id.external.value);
        _hash_data.emplace(
            get_hash_id(name),
            region_info{std::string{name}, OTF2_REGION_ROLE_FUNCTION, OTF2_PARADIGM_HIP});

        // TODO: add attributes for kernel dispatch parameters

        auto& _evt_info = agent_dispatch_info.at(itr.thread_id).at(info.agent_id).at(info.queue_id);
        _evt_info.event_count += 1;

        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                    name,
                                    _evt_info.get_location(),
                                    itr.start_timestamp,
                                    get_attr(sdk::category::kernel_dispatch{})});
        _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                    name,
                                    _evt_info.get_location(),
                                    itr.end_timestamp,
                                    nullptr});
    }

    std::sort(_data.begin(), _data.end(), [](const evt_data& lhs, const evt_data& rhs) {
        if(lhs.timestamp != rhs.timestamp) return (lhs.timestamp < rhs.timestamp);
        if(lhs.phase != rhs.phase) return (lhs.phase > rhs.phase);
        return (*lhs.location < *rhs.location);
    });

    for(const auto& itr : _data)
    {
        add_event(itr.name, *itr.location, itr.phase, itr.timestamp, itr.attributes);
        ROCP_ERROR_IF(itr.timestamp < _app_ts.app_start_time)
            << "event found with timestamp < app start time by "
            << (_app_ts.app_start_time - itr.timestamp) << " nsec :: " << itr.name;
        ROCP_ERROR_IF(itr.timestamp > _app_ts.app_end_time)
            << "event found with timestamp > app end time by "
            << (itr.timestamp - _app_ts.app_end_time) << " nsec :: " << itr.name;
    }

    for(const auto& itr : _data)
    {
        if(itr.attributes) OTF2_AttributeList_Delete(itr.attributes);
    }

    OTF2_CHECK(OTF2_Archive_CloseEvtFiles(archive));

    OTF2_CHECK(OTF2_Archive_OpenDefFiles(archive));
    for(auto& itr : get_locations())
    {
        OTF2_DefWriter* def_writer = OTF2_Archive_GetDefWriter(archive, itr->index);
        OTF2_Archive_CloseDefWriter(archive, def_writer);
    }
    OTF2_CHECK(OTF2_Archive_CloseDefFiles(archive));

    auto _timer_resolution =
        common::get_clock_period_ns_impl(common::default_clock_id) * std::nano::den;
    auto _global_offset    = _app_ts.app_start_time;
    auto _max_trace_length = (_app_ts.app_end_time - _app_ts.app_start_time);

    OTF2_GlobalDefWriter* global_def_writer = OTF2_Archive_GetGlobalDefWriter(archive);
    OTF2_CHECK(OTF2_GlobalDefWriter_WriteClockProperties(
        global_def_writer,
        _timer_resolution,
        _global_offset,
        _max_trace_length,
        std::chrono::system_clock::now().time_since_epoch().count()));

    OTF2_CHECK(OTF2_GlobalDefWriter_WriteString(global_def_writer, 0, ""));
    for(const auto& itr : _hash_data)
    {
        if(itr.first != 0)
            OTF2_CHECK(OTF2_GlobalDefWriter_WriteString(
                global_def_writer, itr.first, itr.second.name.c_str()));
    }

    for(const auto& itr : _hash_data)
    {
        if(itr.first != 0)
            OTF2_CHECK(OTF2_GlobalDefWriter_WriteRegion(global_def_writer,
                                                        itr.first,
                                                        itr.first,
                                                        0,
                                                        0,
                                                        itr.second.region_role,
                                                        itr.second.paradigm,
                                                        OTF2_REGION_FLAG_NONE,
                                                        0,
                                                        0,
                                                        0));
    }

    auto add_write_string = [&global_def_writer](size_t _hash, std::string_view _name) {
        static auto _existing = std::unordered_set<size_t>{};
        if(_hash > 0 && _existing.count(_hash) == 0)
        {
            OTF2_CHECK(OTF2_GlobalDefWriter_WriteString(global_def_writer, _hash, _name.data()));
            _existing.emplace(_hash);
        }
    };

    auto add_write_string_val = [&add_write_string](std::string_view _name_v) {
        auto _hash_v = get_hash_id(_name_v);
        add_write_string(_hash_v, _name_v);
        return _hash_v;
    };

    auto _attr_name = std::string_view{"category"};
    auto _attr_desc = std::string_view{"tracing category"};

    auto _attr_name_hash = add_write_string_val(_attr_name);
    auto _attr_desc_hash = add_write_string_val(_attr_desc);

    OTF2_CHECK(OTF2_GlobalDefWriter_WriteAttribute(
        global_def_writer, 0, _attr_name_hash, _attr_desc_hash, OTF2_TYPE_STRING));

    for(const auto& itr : _attr_str)
        add_write_string(itr.first, itr.second);

    auto _cmdline  = common::read_command_line(pid);
    auto _exe_name = (_cmdline.empty()) ? std::string{"??"} : _cmdline.at(0);
    auto _exe_hash = get_hash_id(_exe_name);
    add_write_string(_exe_hash, _exe_name);

    auto _node_name = std::string{"node"};
    {
        char _hostname_c[PATH_MAX];
        if(::gethostname(_hostname_c, PATH_MAX) == 0 && ::strnlen(_hostname_c, PATH_MAX) < PATH_MAX)
            _node_name = std::string{_hostname_c};
    }
    auto _node_hash = get_hash_id(_node_name);
    add_write_string(_node_hash, _node_name);

    OTF2_CHECK(OTF2_GlobalDefWriter_WriteSystemTreeNode(
        global_def_writer, 0, _exe_hash, _node_hash, OTF2_UNDEFINED_SYSTEM_TREE_NODE));

    // Process
    OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocationGroup(global_def_writer,
                                                       0,
                                                       _exe_hash,
                                                       OTF2_LOCATION_GROUP_TYPE_PROCESS,
                                                       0,
                                                       OTF2_UNDEFINED_LOCATION_GROUP));

    // Accelerators
    for(const auto& agent_v : agent_data)
    {
        const auto* _name = agent_v.name;
        auto        _hash = get_hash_id(_name);

        add_write_string(_hash, _name);
        OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocationGroup(global_def_writer,
                                                           agent_v.id.handle,
                                                           _hash,
                                                           OTF2_LOCATION_GROUP_TYPE_ACCELERATOR,
                                                           0,
                                                           OTF2_UNDEFINED_LOCATION_GROUP));
    }

    // Thread Events
    for(auto& [tid, evt] : thread_event_info)
    {
        auto _hash = get_hash_id(evt.name);

        add_write_string(_hash, evt.name);
        OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocation(global_def_writer,
                                                      evt.id(),  // id
                                                      _hash,
                                                      OTF2_LOCATION_TYPE_CPU_THREAD,
                                                      2 * evt.event_count,  // # events
                                                      0                     // location group
                                                      ));
    }

    // Memcpy Events
    for(auto& [tid, itr] : agent_memcpy_info)
    {
        for(auto& [agent, evt] : itr)
        {
            auto _hash = get_hash_id(evt.name);

            add_write_string(_hash, evt.name);
            OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocation(global_def_writer,
                                                          evt.id(),  // id
                                                          _hash,
                                                          OTF2_LOCATION_TYPE_ACCELERATOR_STREAM,
                                                          2 * evt.event_count,  // # events
                                                          agent.handle          // location group
                                                          ));
        }
    }

    // Memalloc Events
    for(auto& [tid, itr] : agent_memalloc_info)
    {
        for(auto& [agent, evt] : itr)
        {
            auto _hash = get_hash_id(evt.name);
            // Using max numeric limits results in an out-of-bound runtime error for OTF2
            // and perfetto for agent ids. Setting handle to 0 for free functions.
            constexpr auto null_rocp_agent_id = rocprofiler_agent_id_t{.handle = 0};
            auto           handle             = agent.handle;
            if(agent == null_rocp_agent_id) handle = 0;

            add_write_string(_hash, evt.name);
            OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocation(global_def_writer,
                                                          evt.id(),  // id
                                                          _hash,
                                                          OTF2_LOCATION_TYPE_ACCELERATOR_STREAM,
                                                          2 * evt.event_count,  // # events
                                                          handle                // location group
                                                          ));
        }
    }

    // Dispatch Events
    for(auto& [tid, itr] : agent_dispatch_info)
    {
        for(auto& [agent, qitr] : itr)
        {
            for(auto& [queue, evt] : qitr)
            {
                auto _hash = get_hash_id(evt.name);

                add_write_string(_hash, evt.name);
                OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocation(global_def_writer,
                                                              evt.id(),  // id
                                                              _hash,
                                                              OTF2_LOCATION_TYPE_ACCELERATOR_STREAM,
                                                              2 * evt.event_count,  // # events
                                                              agent.handle  // location group
                                                              ));
            }
        }
    }

    shutdown();
}

}  // namespace tool
}  // namespace rocprofiler
