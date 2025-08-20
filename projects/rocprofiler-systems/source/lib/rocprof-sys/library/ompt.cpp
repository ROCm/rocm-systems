// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "api.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/trace_cache/buffer_storage.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <timemory/defines.h>

#if defined(ROCPROFSYS_USE_OMPT) && ROCPROFSYS_USE_OMPT > 0

#    include "binary/link_map.hpp"
#    include "core/components/fwd.hpp"
#    include "core/rocpd/data_processor.hpp"
#    include "core/rocpd/json.hpp"
#    include "core/trace_cache/cache_manager.hpp"
#    include "library/components/category_region.hpp"
#    include "library/tracing.hpp"

#    include <timemory/components/ompt.hpp>
#    include <timemory/components/ompt/backends.hpp>
#    include <timemory/components/ompt/context.hpp>
#    include <timemory/components/ompt/context_handler.hpp>
#    include <timemory/components/ompt/extern.hpp>
#    include <timemory/components/ompt/tool.hpp>
#    include <timemory/mpl/type_traits.hpp>
#    include <timemory/timemory.hpp>
#    include <timemory/units.hpp>
#    include <timemory/unwind/addr2line.hpp>
#    include <timemory/utility/demangle.hpp>
#    include <timemory/utility/join.hpp>
#    include <timemory/utility/types.hpp>

#    include <dlfcn.h>
#    include <memory>
#    include <sys/mman.h>
#    include <sys/types.h>

using api_t = tim::project::rocprofsys;

namespace rocprofsys
{
namespace
{
void
cache_category()
{
    trace_cache::get_metadata_registry().add_string(trait::name<category::ompt>::value);
}

void
cache_thread_info(size_t tid)
{
    const auto& _thread_info = thread_info::get(tid, SequentTID);
    ROCPROFSYS_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", tid);
    if(!_thread_info)
    {
        ROCPROFSYS_WARNING(3, "No valid thread info for tid=%li\n", tid);
        trace_cache::get_metadata_registry().add_thread_info(
            { getppid(), getpid(), tid, 0, 0, "{}" });
        return;
    }

    trace_cache::get_metadata_registry().add_thread_info(
        { getppid(), getpid(),
          static_cast<size_t>(_thread_info->index_data->system_value),
          static_cast<uint32_t>(_thread_info->get_start()),
          static_cast<uint32_t>(_thread_info->get_stop()), "{}" });
}

void
cache_track(int64_t tid, const std::string& track_name)
{
    const auto& _thread_info = thread_info::get(tid, SequentTID);
    ROCPROFSYS_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", tid);
    if(!_thread_info) return;

    size_t thread_id = _thread_info->index_data->system_value;

    trace_cache::get_metadata_registry().add_track({ track_name, thread_id, "{}" });
}

const std::string
format_event_args(const tim::openmp::argument_array_t& args)
{
    std::string args_str;
    if(args.empty())
    {
        return args_str;
    }

    auto pos{ 0 };
    std::for_each(args.begin(), args.end(),
                  [&](const tim::openmp::labeled_argument& arg) {
                      const auto*       delimiter = ";;";
                      std::stringstream ss;
                      ss << pos++ << delimiter << arg.type << delimiter
                         << std::string(arg.label) << delimiter << arg.value << delimiter;
                      args_str.append(ss.str());
                  });

    return args_str;
}

void
cache_ompt_region(std::string name, uint64_t thread_id, uint64_t correlation_id,
                  uint64_t start_timestamp, uint64_t end_timestamp,
                  std::string call_stack, std::string args_str, std::string extdata)

{
    trace_cache::get_buffer_storage().store(trace_cache::entry_type::ompt, name.c_str(),
                                            thread_id, correlation_id, start_timestamp,
                                            end_timestamp, call_stack.c_str(),
                                            args_str.c_str(), extdata.c_str());
}

const std::string
get_call_stack(const tim::openmp::context_info& _ctx_info)
{
    auto call_stack = ::rocpd::json::create();

    call_stack->set("file", _ctx_info.file);
    call_stack->set("line_info", _ctx_info.line);

    return call_stack->to_string();
}

}  // namespace
namespace component
{
struct ompt : comp::base<ompt, void>
{
    using value_type     = void;
    using base_type      = comp::base<ompt, void>;
    using context_info_t = tim::openmp::context_info;

    static std::string label() { return "ompt"; }
    static std::string description() { return "OpenMP tools tracing"; }

    ompt()                = default;
    ~ompt()               = default;
    ompt(const ompt&)     = default;
    ompt(ompt&&) noexcept = default;

    ompt& operator=(const ompt&)     = default;
    ompt& operator=(ompt&&) noexcept = default;

    template <typename... Args>
    void start(const context_info_t& _ctx_info, Args&&...) const
    {
        category_region<category::ompt>::start<tim::quirk::timemory>(m_prefix);

        auto _ts = tracing::now();
        if(get_use_rocpd())
        {
            auto ctx_hash                 = compute_context_hash(_ctx_info);
            get_timestamp_map()[ctx_hash] = _ts;
        }
        uint64_t _cid =
            (_ctx_info.target_arguments) ? _ctx_info.target_arguments->host_op_id : 0;
        auto _annotate = [&](::perfetto::EventContext ctx) {
            if(config::get_perfetto_annotations())
            {
                tracing::add_perfetto_annotation(ctx, "begin_ns", _ts);
                for(const auto& itr : _ctx_info.arguments)
                    tracing::add_perfetto_annotation(ctx, itr.label, itr.value);
            }
        };

        if(_cid > 0)
        {
            category_region<category::ompt>::start<tim::quirk::perfetto>(
                (_ctx_info.func.empty()) ? m_prefix : _ctx_info.func, _ts,
                ::perfetto::Flow::ProcessScoped(_cid), std::move(_annotate));
        }
        else
        {
            category_region<category::ompt>::start<tim::quirk::perfetto>(
                (_ctx_info.func.empty()) ? m_prefix : _ctx_info.func, _ts,
                std::move(_annotate));
        }
    }

    template <typename... Args>
    void stop(const context_info_t& _ctx_info, Args&&...) const
    {
        category_region<category::ompt>::stop<tim::quirk::timemory>(m_prefix);

        auto _ts = tracing::now();

        uint64_t _cid =
            (_ctx_info.target_arguments) ? _ctx_info.target_arguments->host_op_id : 0;

        if(get_use_rocpd())
        {
            auto  ctx_hash = compute_context_hash(_ctx_info);
            auto& _ctx_map = get_timestamp_map();
            auto  _it      = _ctx_map.find(ctx_hash);

            if(_it != _ctx_map.end())
            {
                auto        _start_ts = _it->second;
                auto        _thrd_id  = extract_thread_num_from_args(_ctx_info.arguments);
                const auto& _thread_info = thread_info::get(_thrd_id, SequentTID);
                auto        _call_stack  = get_call_stack(_ctx_info);
                auto        _args        = format_event_args(_ctx_info.arguments);

                cache_thread_info(_thrd_id);
                cache_ompt_region(
                    ((_ctx_info.func.empty()) ? std::string(m_prefix) : _ctx_info.func),
                    static_cast<size_t>(_thread_info->index_data->system_value), _cid,
                    _start_ts, _ts, _call_stack, _args, "{}");
            }
        }

        auto _annotate = [&](::perfetto::EventContext ctx) {
            if(config::get_perfetto_annotations())
            {
                tracing::add_perfetto_annotation(ctx, "end_ns", _ts);
                for(const auto& itr : _ctx_info.arguments)
                    tracing::add_perfetto_annotation(ctx, itr.label, itr.value);
            }
        };

        if(_cid > 0)
        {
            category_region<category::ompt>::stop<tim::quirk::perfetto>(
                (_ctx_info.func.empty()) ? m_prefix : _ctx_info.func, _ts,
                std::move(_annotate));
        }
        else
        {
            category_region<category::ompt>::stop<tim::quirk::perfetto>(
                (_ctx_info.func.empty()) ? m_prefix : _ctx_info.func, _ts,
                std::move(_annotate));
        }
    }

    template <typename... Args>
    void store(const context_info_t& _ctx_info, Args&&... _args) const
    {
        start(_ctx_info, std::forward<Args>(_args)...);
        stop(_ctx_info, std::forward<Args>(_args)...);
    }

    static void record(std::string_view name, ompt_id_t id, uint64_t beg_time,
                       uint64_t end_time, uint64_t thrd_id, uint64_t targ_id,
                       const context_info_t& common)
    {
        (void) thrd_id;
        (void) targ_id;

        auto _annotate = [&](::perfetto::EventContext ctx) {
            if(config::get_perfetto_annotations())
            {
                for(const auto& itr : common.arguments)
                    tracing::add_perfetto_annotation(ctx, itr.label, itr.value);
            }
        };

        auto _track = tracing::get_perfetto_track(
            category::ompt{},
            [](auto) -> std::string { return "OpenMP Target Offloads"; }, 0);

        category_region<category::ompt>::start<tim::quirk::perfetto>(
            name, _track, beg_time, ::perfetto::Flow::ProcessScoped(id),
            std::move(_annotate));

        category_region<category::ompt>::stop<tim::quirk::perfetto>(name, _track,
                                                                    end_time);
    }

    void set_prefix(std::string_view _v) { m_prefix = _v; }

private:
    std::string_view m_prefix = {};

    std::unordered_map<uint64_t, uint64_t>& get_timestamp_map() const
    {
        static thread_local std::unordered_map<uint64_t, uint64_t> map;
        return map;
    }

    uint64_t compute_context_hash(const context_info_t& _ctx_info) const
    {
        std::hash<std::string> string_hasher;
        std::hash<uint64_t>    uint64_hasher;

        uint64_t           hash       = 0;
        constexpr uint64_t hash_magic = 0x9e3779b9;

        auto combine_hash = [&hash](uint64_t value) {
            hash ^= value + hash_magic + (hash << 6) + (hash >> 2);
        };

        if(_ctx_info.target_arguments)
        {
            combine_hash(uint64_hasher(_ctx_info.target_arguments->host_op_id));
            combine_hash(uint64_hasher(_ctx_info.target_arguments->target_id));
        }

        if(!_ctx_info.label.empty())
        {
            combine_hash(string_hasher(std::string(_ctx_info.label)));
        }

        if(!_ctx_info.func.empty())
        {
            combine_hash(string_hasher(_ctx_info.func));
        }

        uint64_t thread_id = extract_thread_num_from_args(_ctx_info.arguments);
        if(thread_id != 0)
        {
            combine_hash(uint64_hasher(thread_id));
        }

        if(!m_prefix.empty())
        {
            combine_hash(string_hasher(std::string(m_prefix)));
        }

        return hash;
    }

    uint64_t extract_thread_num_from_args(const tim::openmp::argument_array_t& args) const
    {
        for(const auto& arg : args)
        {
            if(arg.label == "thread_num")
            {
                return static_cast<uint64_t>(std::stoull(arg.value));
            }
        }
        return 0;
    }
};
}  // namespace component
}  // namespace rocprofsys

namespace tim
{
namespace trait
{
template <>
struct ompt_handle<api_t>
{
    using type = component_tuple<::rocprofsys::component::ompt>;
};
}  // namespace trait
}  // namespace tim

namespace rocprofsys
{
namespace ompt
{
namespace
{
using ompt_handle_t  = tim::component::ompt_handle<api_t>;
using ompt_context_t = tim::openmp::context_handler<api_t>;
using ompt_toolset_t = typename ompt_handle_t::toolset_type;
using ompt_bundle_t  = tim::component_tuple<ompt_handle_t>;

std::unique_ptr<ompt_bundle_t> f_bundle = {};
bool _init_toolset_off = (trait::runtime_enabled<ompt_toolset_t>::set(false),
                          trait::runtime_enabled<ompt_context_t>::set(false), true);
tim::ompt::finalize_tool_func_t f_finalize = nullptr;
}  // namespace

void
setup()
{
    if(!tim::settings::enabled()) return;
    trait::runtime_enabled<ompt_toolset_t>::set(true);
    trait::runtime_enabled<ompt_context_t>::set(true);
    tim::auto_lock_t lk{ tim::type_mutex<ompt_handle_t>() };
    f_bundle = std::make_unique<ompt_bundle_t>("rocprofsys/ompt",
                                               quirk::config<quirk::auto_start>{});

    if(get_use_rocpd())
    {
        cache_category();
    }
}

void
shutdown()
{
    static bool _protect = false;
    if(_protect) return;
    _protect = true;
    if(f_bundle)
    {
        if(tim::manager::instance()) tim::manager::instance()->cleanup("rocprofsys-ompt");
        f_bundle->stop();
        ompt_context_t::cleanup();
        trait::runtime_enabled<ompt_toolset_t>::set(false);
        trait::runtime_enabled<ompt_context_t>::set(false);
        pthread_gotcha::shutdown();
        // call the OMPT finalize callback
        if(f_finalize)
        {
            for(const auto& itr : tim::openmp::get_ompt_device_functions<api_t>())
                if(itr.second.stop_trace) itr.second.stop_trace(itr.second.device);
            (*f_finalize)();
            f_finalize = nullptr;
        }
    }
    f_bundle.reset();
    _protect = false;
}

namespace
{
bool&
use_tool()
{
    static bool _v = false;
    return _v;
}

int
tool_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                ompt_data_t* tool_data)
{
    if(!rocprofsys::settings_are_configured())
    {
        ROCPROFSYS_BASIC_WARNING_F(
            0,
            "[%s] invoked before rocprof-sys was initialized. In instrumentation mode, "
            "settings exported to the environment have not been propagated yet...\n",
            __FUNCTION__);
        use_tool() = get_env("ROCPROFSYS_USE_OMPT", true, false);
    }
    else
    {
        use_tool() = rocprofsys::config::get_use_ompt();
    }

    if(use_tool())
    {
        ROCPROFSYS_BASIC_VERBOSE_F(
            2, "OpenMP-tools configuring for initial device %i\n\n", initial_device_num);

        static auto _generate_key = [](std::string_view                       _key_v,
                                       const ::tim::openmp::argument_array_t& _args_v) {
            return std::string{ _key_v };
            (void) _args_v;
        };

        tim::openmp::get_codeptr_ra_resolver<api_t>() =
            [](tim::openmp::context_info& _ctx_info) {
                const auto& _key       = _ctx_info.label;
                const auto* codeptr_ra = _ctx_info.codeptr_ra;
                auto&       _args      = _ctx_info.arguments;

                ROCPROFSYS_BASIC_VERBOSE(2, "resolving codeptr return address for %s\n",
                                         _key.data());

                if(!codeptr_ra) return _generate_key(_key, _args);

                static thread_local auto _once = std::once_flag{};
                std::call_once(_once, []() { ::tim::unwind::update_file_maps(); });

                auto _info = ::rocprofsys::binary::lookup_ipaddr_entry<false>(
                    reinterpret_cast<uintptr_t>(codeptr_ra));

                if(_info)
                {
                    _ctx_info.func = tim::demangle(_info->name);
                    if(_info->lineno > 0)
                    {
                        auto _linfo = _info->lineinfo.rget([](const auto& _v) -> bool {
                            return (_v && !_v.location.empty() && _v.line > 0);
                        });

                        if(_linfo)
                        {
                            _ctx_info.file = _linfo.location;
                            _ctx_info.line = _linfo.line;
                            _args.emplace_back("file", _ctx_info.file);
                            _args.emplace_back("lineinfo",
                                               ::timemory::join::join("@", _ctx_info.file,
                                                                      _ctx_info.line));
                        }
                        else
                        {
                            _ctx_info.file = _info->location;
                            _args.emplace_back("file", _ctx_info.file);
                        }

                        return _generate_key(
                            ::timemory::join::join(" @ ", _key, _ctx_info.func), _args);
                    }
                    else
                    {
                        return _generate_key(
                            ::timemory::join::join(" @ ", _key, _ctx_info.func), _args);
                    }
                }
                else
                {
                    auto _dl_info = Dl_info{ nullptr, nullptr, nullptr, nullptr };
                    if(dladdr(codeptr_ra, &_dl_info) != 0)
                    {
                        _ctx_info.file = _dl_info.dli_fname;
                        _ctx_info.func = tim::demangle(_dl_info.dli_sname);
                        _args.emplace_back("file", _ctx_info.file);
                        return _generate_key(
                            ::timemory::join::join(
                                " @ ", _key,
                                ::timemory::join::join("", _ctx_info.func, " [",
                                                       _ctx_info.file, "]")),
                            _args);
                    }
                }

                // since no line info could be deduced, include the codeptr return address
                auto _args_codeptr_v = _args;
                _args_codeptr_v.emplace_back("codeptr_ra", codeptr_ra);
                return _generate_key(_key, _args_codeptr_v);
            };

        tim::openmp::get_function_lookup_callback<
            api_t>() = [](ompt_function_lookup_t,
                          const std::optional<tim::openmp::function_lookup_params>&
                              params) {
            if(!params) return;

            ROCPROFSYS_VERBOSE(3, "[ompt] configuring device %i...\n",
                               params->device_num);

            auto& device_funcs =
                tim::openmp::get_ompt_device_functions<api_t>().at(params->device_num);

            device_funcs.set_trace_ompt(params->device, 1, ompt_callback_target_data_op);
            device_funcs.set_trace_ompt(params->device, 1, ompt_callback_target_submit);

            static ompt_callback_buffer_request_t request =
                [](int device_num, ompt_buffer_t** buffer, size_t* bytes) {
                    ROCPROFSYS_VERBOSE(3, "[ompt] buffer request...\n");
                    *bytes  = ::tim::units::get_page_size();
                    *buffer = mmap(nullptr, *bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                    (void) device_num;
                };

            static ompt_callback_buffer_complete_t complete = [](int device_num,
                                                                 ompt_buffer_t* buffer,
                                                                 size_t         bytes,
                                                                 ompt_buffer_cursor_t
                                                                     begin,
                                                                 int buffer_owned) {
                ROCPROFSYS_VERBOSE(3, "[ompt] buffer complete...\n");
                tim::consume_parameters(device_num, buffer, bytes, begin, buffer_owned);

                auto _funcs =
                    tim::openmp::get_ompt_device_functions<api_t>().at(device_num);
                auto _skew = rocprofsys::tracing::get_clock_skew(
                    [&_funcs]() { return _funcs.get_device_time(_funcs.device); });

                ompt_buffer_cursor_t _cursor   = begin;
                size_t               _nrecords = 0;
                do
                {
                    if(_cursor == 0) break;
                    ++_nrecords;
                    auto* _record = _funcs.get_record_ompt(buffer, _cursor);
                    if(_record)
                    {
                        const char* _type    = tim::openmp::get_enum_label(_record->type);
                        auto        _thrd_id = _record->thread_id;
                        auto        _targ_id = _record->target_id;

                        unsigned long beg_time = _record->time + _skew;
                        unsigned long end_time = 0;
                        ompt_id_t     id       = 0;
                        const char*   _name = tim::openmp::get_enum_label(_record->type);

                        if(_record->type == ompt_callback_target_submit)
                        {
                            auto& _data = _record->record.target_kernel;
                            end_time    = _data.end_time + _skew;
                            id          = _data.host_op_id;

                            auto _ctx_info = tim::openmp::argument_array_t{
                                { "begin_ns", beg_time },
                                { "end_ns", end_time },
                                { "type", _type },
                                { "thread_id", _thrd_id },
                                { "target_id", _targ_id },
                                { "host_op_id", id },
                                { "requested_num_teams", _data.requested_num_teams },
                                { "granted_num_teams", _data.granted_num_teams }
                            };

                            component::ompt::record(
                                _name, id, beg_time, end_time, _thrd_id, _targ_id,
                                tim::openmp::context_info{ _name, nullptr, _ctx_info });

                            if(get_use_rocpd())
                            {
                                cache_thread_info(_thrd_id);
                                cache_track(_thrd_id,
                                            JOIN("", "OMPT Kernel Submit - Thread [ ",
                                                 _thrd_id, " ]")
                                                .c_str());
                                cache_ompt_region(_name, _thrd_id, id, beg_time, end_time,
                                                  "", format_event_args(_ctx_info), "");
                            }
                        }
                        else if(_record->type == ompt_callback_target_data_op)
                        {
                            auto& _data = _record->record.target_data_op;
                            end_time    = _data.end_time + _skew;
                            id          = _data.host_op_id;
                            const auto* _opname =
                                tim::openmp::get_enum_label(_data.optype);

                            auto _ctx_info = tim::openmp::argument_array_t{
                                { "begin_ns", beg_time },
                                { "end_ns", end_time },
                                { "type", _type },
                                { "thread_id", _thrd_id },
                                { "target_id", _targ_id },
                                { "host_op_id", id },
                                { "optype", _opname },
                                { "src_addr", reinterpret_cast<void*>(_data.src_addr) },
                                { "dst_addr", reinterpret_cast<void*>(_data.dest_addr) },
                                { "src_device_num", _data.src_device_num },
                                { "dst_device_num", _data.dest_device_num },
                                { "bytes", _data.bytes },
                            };

                            component::ompt::record(
                                _opname, id, beg_time, end_time, _thrd_id, _targ_id,
                                tim::openmp::context_info{ _name, nullptr, _ctx_info });

                            if(get_use_rocpd())
                            {
                                cache_thread_info(_thrd_id);
                                cache_track(_thrd_id,
                                            JOIN("", "OMPT Target Data Op - Thread [ ",
                                                 _thrd_id, " ]")
                                                .c_str());
                                cache_ompt_region(_name, _thrd_id, id, beg_time, end_time,
                                                  "", format_event_args(_ctx_info), "");
                            }
                        }

                        ROCPROFSYS_VERBOSE(
                            3,
                            "type=%i, type_name=%s, start=%lu, end=%lu, delta=%lu, "
                            "tid=%lu, target_id=%lu, host_id=%lu\n",
                            _record->type, tim::openmp::get_enum_label(_record->type),
                            beg_time, end_time, (end_time - beg_time), _record->thread_id,
                            _record->target_id, id);
                    }

                    _funcs.advance_buffer_cursor(_funcs.device, buffer, bytes, _cursor,
                                                 &_cursor);
                } while(_cursor != 0);

                ROCPROFSYS_VERBOSE(3, "[ompt] number of records: %zu\n", _nrecords);

                if(buffer_owned == 1)
                {
                    ::munmap(buffer, bytes);
                }
            };

            device_funcs.start_trace(params->device, request, complete);
        };

        f_finalize = tim::ompt::configure<api_t>(lookup, initial_device_num, tool_data);
    }
    return 1;  // success
}

void
tool_finalize(ompt_data_t*)
{
    shutdown();
}
}  // namespace
}  // namespace ompt
}  // namespace rocprofsys

extern "C"
{
    ompt_start_tool_result_t* ompt_start_tool(unsigned int,
                                              const char*) ROCPROFSYS_PUBLIC_API;

    ompt_start_tool_result_t* ompt_start_tool(unsigned int omp_version,
                                              const char*  runtime_version)
    {
        ROCPROFSYS_BASIC_VERBOSE_F(0, "OpenMP version: %u, runtime version: %s\n",
                                   omp_version, runtime_version);
        ROCPROFSYS_METADATA("OMP_VERSION", omp_version);
        ROCPROFSYS_METADATA("OMP_RUNTIME_VERSION", runtime_version);
        static auto* data = new ompt_start_tool_result_t{
            &rocprofsys::ompt::tool_initialize, &rocprofsys::ompt::tool_finalize, { 0 }
        };
        return data;
    }
}

#else
namespace rocprofsys
{
namespace ompt
{
void
setup()
{}

void
shutdown()
{}
}  // namespace ompt
}  // namespace rocprofsys

#endif
