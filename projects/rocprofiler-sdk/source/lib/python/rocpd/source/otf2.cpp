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

#include "lib/python/rocpd/source/otf2.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/hasher.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/node_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/output_stream.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/output/timestamps.hpp"
#include "lib/rocprofiler-sdk-tool/config.hpp"

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
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

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

namespace rocpd
{
namespace output
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

auto                  main_tid          = rocprofiler::common::get_tid();
archive_t*            archive           = nullptr;
auto                  flush_callbacks   = OTF2_FlushCallbacks{pre_flush, post_flush};
OTF2_GlobalDefWriter* global_def_writer = nullptr;  // shared between data bases  (processes)

enum rocprofiler_location_type_t
{
    ROCPROFILER_AGENT_NO_TYPE = 0,
    ROCPROFILER_AGENT_MEMORY_COPY_TYPE,
    ROCPROFILER_AGENT_DISPATCH_TYPE,
    ROCPROFILER_AGENT_MEMORY_ALLOC_TYPE,
    ROCPROFILER_AGENT_MEMORY_DEALLOC_TYPE
};

struct location_base
{
    uint64_t                    pid          = 0;
    uint64_t                    tid          = 0;
    uint64_t                    agent_handle = 0;
    uint64_t                    queue_id     = 0;
    rocprofiler_location_type_t type         = ROCPROFILER_AGENT_NO_TYPE;

    location_base(pid_t                       _pid,
                  pid_t                       _tid,
                  uint64_t                    _agent_handle = 0,
                  rocprofiler_location_type_t _type         = ROCPROFILER_AGENT_NO_TYPE,
                  uint64_t                    _queue_id     = 0)
    : pid{static_cast<uint64_t>(_pid)}
    , tid{static_cast<uint64_t>(_tid)}
    , agent_handle{_agent_handle}
    , queue_id{_queue_id}
    , type{_type}
    {}

    auto hash() const
    {
        return array_hash<uint64_t, 5>{}(pid, tid, agent_handle + 1, queue_id + 1, type);
    }
};

bool
operator<(const location_base& lhs, const location_base& rhs)
{
    return std::tie(lhs.pid, lhs.tid, lhs.agent_handle, lhs.queue_id, lhs.type) <
           std::tie(rhs.pid, rhs.tid, rhs.agent_handle, rhs.queue_id, rhs.type);
}

struct location_data : location_base
{
    location_data(pid_t                       _pid,
                  pid_t                       _tid,
                  uint64_t                    _agent_handle = 0,
                  rocprofiler_location_type_t _type         = ROCPROFILER_AGENT_NO_TYPE,
                  uint64_t                    _queue_id     = 0)
    : location_base{_pid, _tid, _agent_handle, _type, _queue_id}
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
            .emplace_back(std::make_unique<location_data>(_location.pid,
                                                          _location.tid,
                                                          _location.agent_handle,
                                                          _location.type,
                                                          _location.queue_id))
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
    using value_type = rocprofiler::common::mpl::unqualified_type_t<Tp>;

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
setup(const rocprofiler::tool::output_config& cfg, uint64_t min_start, uint64_t max_fini)
{
    namespace fs = rocprofiler::common::filesystem;

    auto _filename = rocprofiler::tool::get_output_filename(cfg, "results", std::string_view{});
    auto _filepath = fs::path{_filename};
    auto _name     = _filepath.filename().string();
    auto _path     = _filepath.parent_path().string();

    if(fs::exists(_filepath)) fs::remove_all(_filepath);

    constexpr uint64_t evt_chunk_size = 2 * rocprofiler::common::units::MB;
    constexpr uint64_t def_chunk_size = 8 * rocprofiler::common::units::MB;

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

    auto _timer_resolution =
        rocprofiler::common::get_clock_period_ns_impl(rocprofiler::common::default_clock_id) *
        std::nano::den;
    auto _global_offset    = min_start;
    auto _max_trace_length = (max_fini - min_start);

    global_def_writer = OTF2_Archive_GetGlobalDefWriter(archive);
    OTF2_CHECK(OTF2_GlobalDefWriter_WriteClockProperties(
        global_def_writer,
        _timer_resolution,
        _global_offset,
        _max_trace_length,
        std::chrono::system_clock::now().time_since_epoch().count()));

    OTF2_CHECK(OTF2_GlobalDefWriter_WriteString(global_def_writer, 0, ""));

    auto add_write_string = [](size_t _hash, std::string_view _name_strv) {
        static auto _existing = std::unordered_set<size_t>{};
        if(_hash > 0 && _existing.count(_hash) == 0)
        {
            OTF2_CHECK(
                OTF2_GlobalDefWriter_WriteString(global_def_writer, _hash, _name_strv.data()));
            _existing.emplace(_hash);
        }
    };

    auto add_write_string_val = [&add_write_string](std::string_view _name_v) {
        auto _hash_v = get_hash_id(_name_v);
        add_write_string(_hash_v, _name_v);
        return _hash_v;
    };

    //(must be shared between processes)
    auto _attr_name = std::string_view{"category"};
    auto _attr_desc = std::string_view{"tracing category"};

    auto _attr_name_hash = add_write_string_val(_attr_name);
    auto _attr_desc_hash = add_write_string_val(_attr_desc);

    OTF2_CHECK(OTF2_GlobalDefWriter_WriteAttribute(
        global_def_writer, 0, _attr_name_hash, _attr_desc_hash, OTF2_TYPE_STRING));
}

void
shutdown()
{
    OTF2_CHECK(OTF2_Archive_Close(archive));
}

struct event_info
{
    explicit event_info(location_base&& _loc)
    : m_location{output::get_location(std::forward<location_base>(_loc), true)}
    {}

    auto                 id() const { return m_location->index; }
    auto                 hash() const { return m_location->hash(); }
    const location_base* get_location() const { return m_location; }

    std::string name        = {};
    uint64_t    event_count = 0;

private:
    const location_data* m_location = nullptr;
};

attribute_list_t*
create_attribute_list_for_name(const char* name)
{
    auto* _val            = OTF2_AttributeList_New();
    auto  _hash           = get_hash_id(name);
    auto  _attr_value     = OTF2_AttributeValue{};
    _attr_value.stringRef = _hash;
    OTF2_AttributeList_AddAttribute(_val, 0, OTF2_TYPE_STRING, _attr_value);
    return _val;
}
}  // namespace

OTF2Session::OTF2Session(const tool::output_config& output_cfg,
                         uint64_t                   min_start,
                         uint64_t                   max_fini)
: config{output_cfg}
{
    setup(output_cfg, min_start, max_fini);
}

OTF2Session::~OTF2Session() { shutdown(); }

void
write_otf2(const OTF2Session&                                  otf2_session,
           const types::process&                               process,
           const uint16_t                                      tree_node_id,
           const std::unordered_map<uint64_t, extended_agent>& agent_data,
           const tool::generator<types::thread>&               thread_gen,
           const tool::generator<types::region>&               api_gen,
           const tool::generator<types::kernel_dispatch>&      kernel_dispatch_gen,
           const tool::generator<types::memory_copies>&        memory_copy_gen,
           const tool::generator<types::memory_allocation>&    memory_allocation_gen)
{
    const uint64_t _no_agent_handle = 0;
    // std::numeric_limits<uint64_t>::max() - 1;
    const auto& ocfg = otf2_session.config;

    auto _app_ts = rocprofiler::tool::timestamps_t{process.start, process.fini};

    auto thread_event_info = std::map<pid_t, event_info>{};
    auto agent_memcpy_info =
        std::map<pid_t, std::map<uint64_t, event_info>>{};  // tid -> agent_handle ->evt
    auto agent_memalloc_info =
        std::map<pid_t, std::map<uint64_t, event_info>>{};  // // tid -> agent_handle  ->evt
    auto agent_dispatch_info =
        std::map<pid_t,
                 std::map<uint64_t, std::map<uint64_t, event_info>>>{};  // tid -> agent_handle
                                                                         // -> quieue_id -> evt
    auto _get_alloc_level_type_name = [](const std::string& level,
                                         const std::string& type) -> std::string {
        static const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
            name_map = {
                {"REAL",
                 {{"ALLOC", "MEMORY_ALLOCATION_ALLOCATE"}, {"FREE", "MEMORY_ALLOCATION_FREE"}}},
                {"VIRTUAL",
                 {{"ALLOC", "MEMORY_ALLOCATION_VMEM_ALLOCATE"},
                  {"FREE", "MEMORY_ALLOCATION_VMEM_FREE"}}},
                {"SCRATCH", {{"ALLOC", "SCRATCH_MEMORY_ALLOC"}, {"FREE", "SCRATCH_MEMORY_FREE"}}}};

        if((name_map.count(level) != 0u) && (name_map.at(level).count(type) != 0u))
            return name_map.at(level).at(type);

        return level == "SCRATCH" || level == "REAL" || level == "VIRTUAL" ? level + "_MEMORY_NONE"
                                                                           : "UNKNOWN_LEVEL";
    };

    for(auto ditr : thread_gen)
        for(const auto& itr : thread_gen.get(ditr))
        {
            auto _evt_info = event_info{location_base{process.pid, itr.tid}};
            _evt_info.name = fmt::format("Thread {}", itr.tid);
            thread_event_info.emplace(itr.tid, _evt_info);
        }

    auto _hash_data = hash_map_t{};

    struct evt_data
    {
        rocprofiler_callback_phase_t phase      = ROCPROFILER_CALLBACK_PHASE_NONE;
        std::string                  name       = {};
        const location_base*         location   = nullptr;
        uint64_t                     timestamp  = 0;
        OTF2_AttributeList*          attributes = nullptr;
    };

    auto _data     = std::deque<evt_data>{};
    auto _attr_str = std::unordered_map<size_t, std::string_view>{};

    // copypatse from perfetto. TODO: Move to a common place?
    auto get_category_string = [](std::string_view _category) {
        static auto buffer_names  = rocprofiler::sdk::get_buffer_tracing_names();
        auto        _category_idx = ROCPROFILER_BUFFER_TRACING_NONE;
        for(const auto& citr : buffer_names)
        {
            if(_category == citr.name) _category_idx = citr.value;
        }
        return rocprofiler::sdk::get_perfetto_category(_category_idx);
    };

    auto get_category_attribute = [&get_category_string,
                                   &_attr_str](const std::string& category) -> OTF2_AttributeList* {
        const auto* _perfetto_category = get_category_string(category);
        _attr_str.emplace(get_hash_id(_perfetto_category), _perfetto_category);
        return create_attribute_list_for_name(_perfetto_category);
    };

    for(auto ditr : api_gen)
        for(const auto& itr : api_gen.get(ditr))
        {
            std::string _name = itr.name;
            _hash_data.emplace(get_hash_id(_name),
                               region_info{_name, OTF2_REGION_ROLE_FUNCTION, OTF2_PARADIGM_HIP});

            auto& _evt_info = thread_event_info.at(itr.tid);
            _evt_info.event_count += 1;

            auto* attributes = get_category_attribute(itr.category);
            _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                        _name,
                                        _evt_info.get_location(),
                                        itr.start,
                                        attributes});

            if(!attributes)
            {
                ROCP_FATAL << "Undefined attributes for api call " << itr.name;
            }

            _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                        _name,
                                        _evt_info.get_location(),
                                        itr.end,
                                        nullptr});
        }

    for(auto ditr : memory_copy_gen)
        for(const auto& itr : memory_copy_gen.get(ditr))
        {
            std::string _name = itr.name;
            _hash_data.emplace(
                get_hash_id(_name),
                region_info{_name, OTF2_REGION_ROLE_DATA_TRANSFER, OTF2_PARADIGM_HIP});

            auto _extended_agent = agent_data.at(itr.dst_agent_abs_index);
            auto _agent_handle   = _extended_agent.types_agent.id.handle;
            auto _evt_info       = event_info{location_base{
                process.pid, itr.tid, _agent_handle, ROCPROFILER_AGENT_MEMORY_COPY_TYPE}};

            auto agent_index_info = _extended_agent.agent_index;
            _evt_info.name        = fmt::format("Thread {}, Copy to {} {}",
                                         itr.tid,
                                         agent_index_info.type,
                                         agent_index_info.as_string("-"));

            _evt_info.event_count += 1;

            agent_memcpy_info[itr.tid].emplace(_agent_handle, _evt_info);

            const auto* _perfetto_name =
                rocprofiler::sdk::perfetto_category<rocprofiler::sdk::category::memory_copy>::name;

            _attr_str.emplace(get_hash_id(_perfetto_name), _perfetto_name);
            auto* _attrs = create_attribute_list_for_name(_perfetto_name);
            _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                        _name,
                                        _evt_info.get_location(),
                                        itr.start,
                                        _attrs});
            _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                        _name,
                                        _evt_info.get_location(),
                                        itr.end,
                                        nullptr});
        };

    for(auto ditr : memory_allocation_gen)
        for(const auto& itr : memory_allocation_gen.get(ditr))
        {
            std::string _alloc_operation = _get_alloc_level_type_name(itr.level, itr.type);

            const auto* _perfetto_name = rocprofiler::sdk::perfetto_category<
                rocprofiler::sdk::category::memory_allocation>::name;

            _attr_str.emplace(get_hash_id(_perfetto_name), _perfetto_name);
            auto* _attrs = create_attribute_list_for_name(_perfetto_name);

            if(itr.type == "ALLOC")
            {
                _hash_data.emplace(
                    get_hash_id(_alloc_operation),
                    region_info{_alloc_operation, OTF2_REGION_ROLE_ALLOCATE, OTF2_PARADIGM_HIP});

                auto _extended_agent = agent_data.at(itr.agent_abs_index);
                auto _handle         = _extended_agent.types_agent.id.handle;

                auto _evt_info = event_info{location_base{
                    process.pid, itr.tid, _handle, ROCPROFILER_AGENT_MEMORY_ALLOC_TYPE}};

                auto agent_index_info = _extended_agent.agent_index;
                _evt_info.name        = fmt::format("Thread {}, Memory Allocate at {} {}",
                                             itr.tid,
                                             agent_index_info.type,
                                             agent_index_info.as_string("-"));

                agent_memalloc_info[itr.tid].emplace(_handle, _evt_info);
                _evt_info.event_count += 1;

                _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                            _alloc_operation,
                                            _evt_info.get_location(),
                                            itr.start,
                                            _attrs});
                _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                            _alloc_operation,
                                            _evt_info.get_location(),
                                            itr.end,
                                            nullptr});
            }
            else if(itr.type == "FREE")  //
            {
                _hash_data.emplace(
                    get_hash_id(_alloc_operation),
                    region_info{_alloc_operation, OTF2_REGION_ROLE_DEALLOCATE, OTF2_PARADIGM_HIP});

                auto _evt_info = event_info{location_base{
                    process.pid, itr.tid, _no_agent_handle, ROCPROFILER_AGENT_MEMORY_DEALLOC_TYPE}};
                _evt_info.name = fmt::format("Thread {}, Memory Deallocate (Free)", itr.tid);

                agent_memalloc_info[itr.tid].emplace(_no_agent_handle, _evt_info);

                _evt_info.event_count += 1;

                _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                            _alloc_operation,
                                            _evt_info.get_location(),
                                            itr.start,
                                            _attrs});
                _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                            _alloc_operation,
                                            _evt_info.get_location(),
                                            itr.end,
                                            nullptr});
            }
            else
            {
                auto _evt_info = event_info{location_base{process.pid, itr.tid}};
                _evt_info.name = fmt::format("Thread {}, Memory Operation UNK", itr.tid);
                _evt_info.event_count += 1;
                agent_memalloc_info[itr.tid].emplace(_no_agent_handle, _evt_info);
                _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                            _alloc_operation,
                                            _evt_info.get_location(),
                                            itr.start,
                                            _attrs});
                _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                            _alloc_operation,
                                            _evt_info.get_location(),
                                            itr.end,
                                            nullptr});
            }
        }

    auto _queue_ids = std::map<uint64_t, uint64_t>{};
    for(auto ditr : kernel_dispatch_gen)
        for(const auto& itr : kernel_dispatch_gen.get(ditr))
        {
            auto _name = fmt::format(
                "{}", (ocfg.kernel_rename && !itr.region.empty()) ? itr.region : itr.name);
            _hash_data.emplace(get_hash_id(_name),
                               region_info{_name, OTF2_REGION_ROLE_FUNCTION, OTF2_PARADIGM_HIP});

            const auto* _perfetto_name = rocprofiler::sdk::perfetto_category<
                rocprofiler::sdk::category::kernel_dispatch>::name;

            _attr_str.emplace(get_hash_id(_perfetto_name), _perfetto_name);
            auto* _attrs = create_attribute_list_for_name(_perfetto_name);

            auto _extended_agent  = agent_data.at(itr.agent_abs_index);
            auto _handle          = _extended_agent.types_agent.id.handle;
            auto agent_index_info = _extended_agent.agent_index;

            auto _evt_info = event_info{location_base{
                process.pid, itr.tid, _handle, ROCPROFILER_AGENT_DISPATCH_TYPE, itr.queue_id}};

            if(_queue_ids.count(itr.queue_id) == 0)
            {
                _queue_ids.emplace(itr.queue_id, _queue_ids.size());
            }

            _evt_info.name = fmt::format("Thread {}, Compute on {} {}, Queue {}",

                                         itr.tid,
                                         agent_index_info.type,
                                         agent_index_info.as_string("-"),
                                         _queue_ids.at(itr.queue_id));

            _evt_info.event_count += 1;
            agent_dispatch_info[itr.tid][_handle].emplace(itr.queue_id, _evt_info);

            _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_ENTER,
                                        _name,
                                        _evt_info.get_location(),
                                        itr.start,
                                        _attrs});
            _data.emplace_back(evt_data{ROCPROFILER_CALLBACK_PHASE_EXIT,
                                        _name,
                                        _evt_info.get_location(),
                                        itr.end,
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

    auto add_write_string = [](size_t _hash, std::string_view _name) {
        static auto _existing = std::unordered_set<size_t>{};
        if(_hash > 0 && _existing.count(_hash) == 0)
        {
            OTF2_CHECK(OTF2_GlobalDefWriter_WriteString(global_def_writer, _hash, _name.data()));
            _existing.emplace(_hash);
        }
    };

    for(const auto& itr : _attr_str)
        add_write_string(itr.first, itr.second);

    std::istringstream command_line(process.command);
    std::string        _exe_name;
    command_line >> _exe_name;  // Extracts characters until whitespace
    _exe_name      = fmt::format("{}", _exe_name);
    auto _exe_hash = get_hash_id(_exe_name);
    add_write_string(_exe_hash, _exe_name);

    auto _node_name = std::string{"node"};
    {
        if(!process.hostname.empty())
        {
            if(process.hostname.length() >= PATH_MAX)
            {
                _node_name = process.hostname.substr(0, PATH_MAX - 1);
            }
            else
            {
                _node_name = process.hostname;
            }
        }
    }
    _node_name      = fmt::format("{}", _node_name);
    auto _node_hash = get_hash_id(_node_name);
    add_write_string(_node_hash, _node_name);

    // debug
    OTF2_CHECK(OTF2_GlobalDefWriter_WriteSystemTreeNode(
        global_def_writer, tree_node_id, _exe_hash, _node_hash, OTF2_UNDEFINED_SYSTEM_TREE_NODE));

    // Process
    OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocationGroup(global_def_writer,
                                                       tree_node_id,
                                                       _exe_hash,
                                                       OTF2_LOCATION_GROUP_TYPE_PROCESS,
                                                       tree_node_id,
                                                       OTF2_UNDEFINED_LOCATION_GROUP));

    // Accelerators (must be shared between the processes)
    for(const auto& [abs_idx, extended_agent] : agent_data)
    {
        auto       _handle = extended_agent.types_agent.id.handle;
        const auto _name   = std::string_view{extended_agent.labeled_name};
        auto       _hash   = get_hash_id(_name);

        add_write_string(_hash, _name);
        OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocationGroup(global_def_writer,
                                                           _handle,
                                                           _hash,
                                                           OTF2_LOCATION_GROUP_TYPE_ACCELERATOR,
                                                           tree_node_id,
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
                                                      tree_node_id          // location group
                                                      ));
    }

    // Memcpy Events
    for(auto& [tid, itr] : agent_memcpy_info)
    {
        for(auto& [agent_handle, evt] : itr)
        {
            auto _hash = get_hash_id(evt.name);

            add_write_string(_hash, evt.name);
            OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocation(global_def_writer,
                                                          evt.id(),  // id
                                                          _hash,
                                                          OTF2_LOCATION_TYPE_ACCELERATOR_STREAM,
                                                          2 * evt.event_count,  // # events
                                                          agent_handle          // location group
                                                          ));
        }
    }

    // Memalloc Events
    for(auto& [tid, itr] : agent_memalloc_info)
    {
        for(auto& [agent_handle, evt] : itr)
        {
            auto _hash = get_hash_id(evt.name);

            add_write_string(_hash, evt.name);

            OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocation(global_def_writer,
                                                          evt.id(),  // id
                                                          _hash,
                                                          OTF2_LOCATION_TYPE_ACCELERATOR_STREAM,
                                                          2 * evt.event_count,  // # events
                                                          agent_handle          // location group
                                                          ));
        }
    }

    // Dispatch Events
    for(auto& [tid, itr] : agent_dispatch_info)
    {
        for(auto& [agent_handle, qitr] : itr)
        {
            for(auto& [queue_id, evt] : qitr)
            {
                auto _hash = get_hash_id(evt.name);

                add_write_string(_hash, evt.name);

                OTF2_CHECK(OTF2_GlobalDefWriter_WriteLocation(global_def_writer,
                                                              evt.id(),  // id
                                                              _hash,
                                                              OTF2_LOCATION_TYPE_ACCELERATOR_STREAM,
                                                              2 * evt.event_count,  // # events
                                                              agent_handle  // location group
                                                              ));
            }
        }
    }
}

}  // namespace output
}  // namespace rocpd
