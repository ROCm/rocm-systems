// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "library.hpp"
#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/thread_data.hpp"
#include <string_view>

namespace
{
std::vector<bool>&
get_sample_data()
{
    static thread_local auto _v = std::vector<bool>{};
    return _v;
}

void
setup_gotchas()
{
    static bool _initialized = false;
    if(_initialized) return;
    _initialized = true;

    HOSTTRACE_DEBUG(
        "[%s] Configuring gotcha wrapper around fork, MPI_Init, and MPI_Init_thread\n",
        __FUNCTION__);

    fork_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(fork_gotcha_t, 0, fork);
    };

    mpi_gotcha_t::get_initializer() = []() {
        mpi_gotcha_t::template configure<0, int, int*, char***>("MPI_Init");
        mpi_gotcha_t::template configure<1, int, int*, char***, int, int*>(
            "MPI_Init_thread");
        mpi_gotcha_t::template configure<3, int>("MPI_Finalize");
    };
}

auto
ensure_finalization(bool _static_init = false)
{
    if(!_static_init)
    {
        HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);
    }
    return scope::destructor{ []() { hosttrace_trace_finalize(); } };
}

auto&
get_trace_session()
{
    static std::unique_ptr<perfetto::TracingSession> _session{};
    return _session;
}

auto
is_system_backend()
{
    // if get_backend() returns 'system' or 'all', this is true
    return (get_backend() != "inprocess");
}

auto&
get_instrumentation_bundles()
{
    static thread_local auto& _v =
        instrumentation_bundles::instances().at(threading::get_id());
    return _v;
}

auto&
get_functors()
{
    using functor_t = std::function<void(const char*)>;
    static auto _v =
        std::pair<functor_t, functor_t>{ [](const char*) {}, [](const char*) {} };
    return _v;
}

auto&
get_cpu_cid_parents()
{
    static thread_local auto _v =
        std::unordered_map<uint64_t, std::tuple<uint64_t, uint16_t>>{};
    return _v;
}

using Device = critical_trace::Device;
using Phase  = critical_trace::Phase;

bool
hosttrace_init_tooling()
{
    static bool _once = false;
    if(get_state() != State::PreInit || _once) return false;
    _once = true;

    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);

    if(!get_use_timemory() && !get_use_perfetto())
    {
        get_state() = State::Finalized;
        HOSTTRACE_DEBUG("[%s] Both perfetto and timemory are disabled. Setting the state "
                        "to finalized\n",
                        __FUNCTION__);
        return false;
    }

    int _threadpool_verbose = (get_debug()) ? 4 : -1;
    tasking::get_roctracer_thread_pool().set_verbose(_threadpool_verbose);
    tasking::get_critical_trace_thread_pool().set_verbose(_threadpool_verbose);

    // below will effectively do:
    //      get_cpu_cid_stack(0)->emplace_back(-1);
    // plus query some env variables
    add_critical_trace<Device::CPU, Phase::NONE>(0, -1, 0, 0, 0, 0, 0, 0);

    // configure the settings
    configure_settings();

    if(get_sample_rate() < 1) get_sample_rate() = 1;
    get_sample_data().reserve(512);

    if(get_use_timemory())
    {
        comp::user_global_bundle::global_init();
        std::set<int> _comps{};
        // convert string into set of enumerations
        for(auto&& itr : tim::delimit(tim::settings::global_components()))
            _comps.emplace(tim::runtime::enumerate(itr));
        if(_comps.size() == 1 && _comps.find(TIMEMORY_WALL_CLOCK) != _comps.end())
        {
            // using wall_clock directly is lower overhead than using it via user_bundle
            instrumentation_bundle_t::get_initializer() =
                [](instrumentation_bundle_t& _bundle) {
                    _bundle.initialize<comp::wall_clock>();
                };
        }
        else if(!_comps.empty())
        {
            // use user_bundle for other than wall-clock
            instrumentation_bundle_t::get_initializer() =
                [](instrumentation_bundle_t& _bundle) {
                    _bundle.initialize<comp::user_global_bundle>();
                };
        }
        else
        {
            tim::trait::runtime_enabled<hosttrace>::set(false);
        }
    }

    // always activate gotcha wrappers
    auto& _main_bundle = get_main_bundle();
    _main_bundle->start();
    assert(_main_bundle->get<mpi_gotcha_t>()->get_is_running());
#if defined(HOSTTRACE_USE_ROCTRACER)
    assert(_main_bundle->get<comp::roctracer>() != nullptr);
    assert(_main_bundle->get<comp::roctracer>()->get_is_running());
#endif

    perfetto::TracingInitArgs               args{};
    perfetto::TraceConfig                   cfg{};
    perfetto::protos::gen::TrackEventConfig track_event_cfg{};

    // perfetto initialization
    if(get_use_perfetto())
    {
        // environment settings
        auto shmem_size_hint = get_perfetto_shmem_size_hint();
        auto buffer_size     = get_perfetto_buffer_size();

        auto* buffer_config = cfg.add_buffers();
        buffer_config->set_size_kb(buffer_size);
        buffer_config->set_fill_policy(
            perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD);

        auto* ds_cfg = cfg.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");
        ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

        args.shmem_size_hint_kb = shmem_size_hint;

        if(get_backend() != "inprocess") args.backends |= perfetto::kSystemBackend;
        if(get_backend() != "system") args.backends |= perfetto::kInProcessBackend;

        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();

        (void) get_perfetto_output_filename();
    }

    auto        _exe         = get_exe_name();
    static auto _thread_init = [_exe]() {
        hosttrace_thread_data<hosttrace_thread_bundle_t>::construct(
            TIMEMORY_JOIN("", _exe, "/thread-", threading::get_id()),
            quirk::config<quirk::auto_start>{});
        static thread_local auto _dtor = scope::destructor{ []() {
            hosttrace_thread_data<hosttrace_thread_bundle_t>::instance()->stop();
        } };
        (void) _dtor;
    };

    // functors for starting and stopping timemory
    static auto _push_timemory = [](const char* name) {
        _thread_init();
        auto& _data = get_instrumentation_bundles();
        // this generates a hash for the raw string array
        auto  _hash   = tim::add_hash_id(tim::string_view_t{ name });
        auto* _bundle = _data.allocator.allocate(1);
        _data.bundles.emplace_back(_bundle);
        _data.allocator.construct(_bundle, _hash);
        _bundle->start();
    };

    static auto _push_perfetto = [](const char* name) {
        _thread_init();
        TRACE_EVENT_BEGIN("host", perfetto::StaticString(name),
                          [&](perfetto::EventContext ctx) {
                              // compile-time check
                              IF_CONSTEXPR(trait::is_available<papi_tot_ins>::value)
                              {
                                  ctx.event()->set_thread_instruction_count_absolute(
                                      papi_tot_ins::record().at(0));
                              }
                          });
    };

    static auto _pop_timemory = [](const char* name) {
        auto& _data = get_instrumentation_bundles();
        if(_data.bundles.empty())
        {
            HOSTTRACE_DEBUG("[%s] skipped %s :: empty bundle stack\n",
                            "hosttrace_pop_trace", name);
            return;
        }
        _data.bundles.back()->stop();
        _data.allocator.destroy(_data.bundles.back());
        _data.allocator.deallocate(_data.bundles.back(), 1);
        _data.bundles.pop_back();
    };

    static auto _pop_perfetto = [](const char*) {
        TRACE_EVENT_END("host", [&](perfetto::EventContext ctx) {
            IF_CONSTEXPR(trait::is_available<papi_tot_ins>::value)
            {
                ctx.event()->set_thread_instruction_count_absolute(
                    papi_tot_ins::record().at(0));
            }
        });
    };

    if(get_use_perfetto() && get_use_timemory())
    {
        get_functors().first = [](const char* name) {
            _push_perfetto(name);
            _push_timemory(name);
        };
        get_functors().second = [](const char* name) {
            _pop_timemory(name);
            _pop_perfetto(name);
        };
    }
    else if(get_use_perfetto())
    {
        get_functors().first  = _push_perfetto;
        get_functors().second = _pop_perfetto;
    }
    else if(get_use_timemory())
    {
        get_functors().first  = _push_timemory;
        get_functors().second = _pop_timemory;
    }

    if(dmp::rank() == 0)
    {
        // generic filter for filtering relevant options
        auto _is_hosttrace_option = [](const auto& _v) {
#if !defined(HOSTTRACE_USE_ROCTRACER)
            if(_v.find("HOSTTRACE_ROCTRACER_") == 0) return false;
#endif
            if(!get_use_critical_trace() && _v.find("HOSTTRACE_CRITICAL_TRACE_") == 0)
                return false;
            return (_v.find("HOSTTRACE_") == 0) ||
                   ((_v.find("TIMEMORY_") != 0) && (_v.find("SIGNAL_") != 0));
        };

        tim::print_env(std::cerr, [_is_hosttrace_option](const std::string& _v) {
            return _is_hosttrace_option(_v);
        });

        print_config_settings(std::cerr, _is_hosttrace_option);
    }

    if(get_use_perfetto() && !is_system_backend())
    {
#if defined(CUSTOM_DATA_SOURCE)
        // Add the following:
        perfetto::DataSourceDescriptor dsd{};
        dsd.set_name("com.example.custom_data_source");
        CustomDataSource::Register(dsd);
        auto* ds_cfg = cfg.add_data_sources()->mutable_config();
        ds_cfg->set_name("com.example.custom_data_source");
        CustomDataSource::Trace([](CustomDataSource::TraceContext ctx) {
            auto packet = ctx.NewTracePacket();
            packet->set_timestamp(perfetto::TrackEvent::GetTraceTimeNs());
            packet->set_for_testing()->set_str("Hello world!");
            PRINT_HERE("%s", "Trace");
        });
#endif
        auto& tracing_session = get_trace_session();
        tracing_session       = perfetto::Tracing::NewTrace();
        tracing_session->Setup(cfg);
        tracing_session->StartBlocking();
    }

    get_state() = State::Active;

    // if static objects are destroyed in the inverse order of when they are
    // created this should ensure that finalization is called before perfetto
    // ends the tracing session
    static auto _ensure_finalization = ensure_finalization();

    if(dmp::rank() == 0) puts("");
    return true;
}
}  // namespace

//--------------------------------------------------------------------------------------//

extern "C"
{
    void hosttrace_push_trace(const char* name)
    {
        // return if not active
        if(get_state() == State::Finalized) return;

        if(get_state() != State::Active && !hosttrace_init_tooling())
        {
            HOSTTRACE_DEBUG("[%s] %s :: not active and perfetto not initialized\n",
                            __FUNCTION__, name);
            return;
        }
        else
        {
            HOSTTRACE_DEBUG("[%s] %s\n", __FUNCTION__, name);
        }

        static auto                _sample_rate = std::max<size_t>(get_sample_rate(), 1);
        static thread_local size_t _sample_idx  = 0;
        auto                       _enabled     = (_sample_idx++ % _sample_rate == 0);
        get_sample_data().emplace_back(_enabled);
        if(_enabled) get_functors().first(name);
        if(get_use_critical_trace())
        {
            auto     _ts         = comp::wall_clock::record();
            auto     _cid        = get_cpu_cid()++;
            uint16_t _depth      = (get_cpu_cid_stack()->empty())
                                       ? get_cpu_cid_stack(0)->size()
                                       : get_cpu_cid_stack()->size() - 1;
            auto     _parent_cid = (get_cpu_cid_stack()->empty())
                                       ? get_cpu_cid_stack(0)->back()
                                       : get_cpu_cid_stack()->back();
            get_cpu_cid_parents().emplace(_cid, std::make_tuple(_parent_cid, _depth));
            add_critical_trace<Device::CPU, Phase::BEGIN>(
                threading::get_id(), _cid, 0, _parent_cid, _ts, 0,
                critical_trace::add_hash_id(name), _depth);
        }
    }

    void hosttrace_pop_trace(const char* name)
    {
        if(get_state() == State::Active)
        {
            HOSTTRACE_DEBUG("[%s] %s\n", __FUNCTION__, name);
            auto& _sample_data = get_sample_data();
            if(!_sample_data.empty())
            {
                if(_sample_data.back()) get_functors().second(name);
                _sample_data.pop_back();
            }
            if(get_use_critical_trace())
            {
                if(get_cpu_cid_stack() && !get_cpu_cid_stack()->empty())
                {
                    auto     _ts                  = comp::wall_clock::record();
                    auto     _cid                 = get_cpu_cid_stack()->back();
                    uint64_t _parent_cid          = 0;
                    uint16_t _depth               = 0;
                    std::tie(_parent_cid, _depth) = get_cpu_cid_parents().at(_cid);
                    add_critical_trace<Device::CPU, Phase::END>(
                        threading::get_id(), _cid, 0, _parent_cid, _ts, _ts,
                        critical_trace::add_hash_id(name), _depth);
                }
            }
        }
        else
        {
            HOSTTRACE_DEBUG("[%s] %s :: not active\n", __FUNCTION__, name);
        }
    }

    void hosttrace_trace_init(const char*, bool, const char*)
    {
        HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);
        hosttrace_init_tooling();
    }

    void hosttrace_trace_finalize(void)
    {
        // return if not active
        if(get_state() != State::Active) return;

        HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);

        if(dmp::rank() == 0) puts("");

        get_state() = State::Finalized;

#if defined(HOSTTRACE_USE_ROCTRACER)
        // ensure that threads running roctracer callbacks shutdown
        comp::roctracer::tear_down();
#endif

        // join extra thread(s) used by roctracer
        HOSTTRACE_DEBUG("[%s] waiting for all roctracer tasks to complete...\n",
                        __FUNCTION__);
        tasking::get_roctracer_task_group().join();

        // stop the main bundle and report the high-level metrics
        if(get_main_bundle())
        {
            get_main_bundle()->stop();
            std::string _msg = JOIN("", *get_main_bundle());
            auto        _pos = _msg.find(">>>  ");
            if(_pos != std::string::npos) _msg = _msg.substr(_pos + 5);
            HOSTTRACE_PRINT("%s\n", _msg.c_str());
            get_main_bundle().reset();
        }

        // print out thread-data if they are not still running
        // if they are still running (e.g. thread-pool still alive), the
        // thread-specific data will be wrong if try to stop them from
        // the main thread.
        for(auto& itr : hosttrace_thread_data<hosttrace_thread_bundle_t>::instances())
        {
            if(itr && itr->get<comp::wall_clock>() &&
               !itr->get<comp::wall_clock>()->get_is_running())
            {
                std::string _msg = JOIN("", *itr);
                auto        _pos = _msg.find(">>>  ");
                if(_pos != std::string::npos) _msg = _msg.substr(_pos + 5);
                HOSTTRACE_PRINT("%s\n", _msg.c_str());
            }
        }

        // ensure that all the MT instances are flushed
        for(auto& itr : instrumentation_bundles::instances())
        {
            while(!itr.bundles.empty())
            {
                itr.bundles.back()->stop();
                itr.bundles.back()->pop();
                itr.allocator.destroy(itr.bundles.back());
                itr.allocator.deallocate(itr.bundles.back(), 1);
                itr.bundles.pop_back();
            }
        }

        if(get_use_critical_trace())
        {
            // increase the thread-pool size
            tasking::get_critical_trace_thread_pool().initialize_threadpool(
                get_critical_trace_num_threads());

            for(size_t i = 0; i < max_supported_threads; ++i)
            {
                using critical_trace_hash_data =
                    hosttrace_thread_data<critical_trace::hash_ids, critical_trace::id>;

                if(critical_trace_hash_data::instances().at(i))
                    critical_trace::add_hash_id(
                        *critical_trace_hash_data::instances().at(i));
            }

            for(size_t i = 0; i < max_supported_threads; ++i)
            {
                using critical_trace_chain_data =
                    hosttrace_thread_data<critical_trace::call_chain>;

                if(critical_trace_chain_data::instances().at(i))
                    critical_trace::update(i);  // launch update task
            }

            // make sure outstanding hash tasks completed before compute
            HOSTTRACE_PRINT("[%s] waiting for all critical trace tasks to complete...\n",
                            __FUNCTION__);
            tasking::get_critical_trace_task_group().join();

            // launch compute task
            HOSTTRACE_PRINT("[%s] launching critical trace compute task...\n",
                            __FUNCTION__);
            critical_trace::compute();
        }

        tasking::get_critical_trace_task_group().join();

        bool _perfetto_output_error = false;
        if(get_use_perfetto() && !is_system_backend())
        {
            // Make sure the last event is closed for this example.
            perfetto::TrackEvent::Flush();

            auto& tracing_session = get_trace_session();
            tracing_session->StopBlocking();

            std::vector<char> trace_data{ tracing_session->ReadTraceBlocking() };

            if(trace_data.empty())
            {
                fprintf(stderr,
                        "[%s]> trace data is empty. File '%s' will not be written...\n",
                        __FUNCTION__, get_perfetto_output_filename().c_str());
                return;
            }
            // Write the trace into a file.
            fprintf(stderr,
                    "[%s]> Outputting '%s'. Trace data: %lu B (%.2f KB / %.2f MB / %.2f "
                    "GB)...\n",
                    __FUNCTION__, get_perfetto_output_filename().c_str(),
                    (unsigned long) trace_data.size(),
                    static_cast<double>(trace_data.size()) / units::KB,
                    static_cast<double>(trace_data.size()) / units::MB,
                    static_cast<double>(trace_data.size()) / units::GB);
            std::ofstream ofs{};
            if(!tim::filepath::open(ofs, get_perfetto_output_filename(),
                                    std::ios::out | std::ios::binary))
            {
                fprintf(stderr, "[%s]> Error opening '%s'...\n", __FUNCTION__,
                        get_perfetto_output_filename().c_str());
                _perfetto_output_error = true;
            }
            else
                ofs.write(&trace_data[0], trace_data.size());
            ofs.close();
        }

        // these should be destroyed before timemory is finalized, especially the
        // roctracer thread-pool
        tasking::get_roctracer_thread_pool().destroy_threadpool();
        tasking::get_critical_trace_thread_pool().destroy_threadpool();

        HOSTTRACE_DEBUG("Finalizing timemory...\n");
        tim::timemory_finalize();
        HOSTTRACE_DEBUG("Finalizing timemory... Done\n");

        if(_perfetto_output_error)
            throw std::runtime_error("Unable to create perfetto output file");
    }

    void hosttrace_trace_set_env(const char* env_name, const char* env_val)
    {
        HOSTTRACE_DEBUG("[%s] Setting env: %s=%s\n", __FUNCTION__, env_name, env_val);

        tim::set_env(env_name, env_val, 0);
    }

    void hosttrace_trace_set_mpi(bool use, bool attached)
    {
        HOSTTRACE_DEBUG("[%s] use: %s, attached: %s\n", __FUNCTION__, (use) ? "y" : "n",
                        (attached) ? "y" : "n");
        if(use && !attached)
        {
            auto& _main_bundle = get_main_bundle();
            _main_bundle->start();
            get_use_pid() = true;
            get_state()   = State::DelayedInit;
        }
    }
}

std::unique_ptr<main_bundle_t>&
get_main_bundle()
{
    static auto _v =
        (setup_gotchas(), std::make_unique<main_bundle_t>(
                              "hosttrace", quirk::config<quirk::auto_start>{}));
    return _v;
}

namespace
{
// if static objects are destroyed randomly (relatively uncommon behavior)
// this might call finalization before perfetto ends the tracing session
// but static variable in hosttrace_init_tooling is more likely
auto _ensure_finalization = ensure_finalization(true);
}  // namespace
