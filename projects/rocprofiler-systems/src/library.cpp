// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/components/mpi_gotcha.hpp"
#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/gpu.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/timemory.hpp"

#include <mutex>
#include <string_view>

using namespace omnitrace;

namespace
{
std::vector<bool>&
get_interval_data()
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

    OMNITRACE_CONDITIONAL_PRINT(
        get_debug_env(),
        "[%s] Configuring gotcha wrapper around fork, MPI_Init, and MPI_Init_thread\n",
        __FUNCTION__);

    mpi_gotcha::configure();
    fork_gotcha::configure();
    pthread_gotcha::configure();
}

auto
ensure_finalization(bool _static_init = false)
{
    auto _main_tid = threading::get_id();
    (void) _main_tid;
    if(!_static_init)
    {
        OMNITRACE_DEBUG("[%s]\n", __FUNCTION__);
    }
    else
    {
        OMNITRACE_CONDITIONAL_PRINT(get_debug_env(), "[%s]\n", __FUNCTION__);
    }
    return scope::destructor{ []() { omnitrace_trace_finalize(); } };
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
omnitrace_init_tooling()
{
    static bool _once = false;
    if(get_state() != State::PreInit || _once) return false;
    _once = true;

    auto _tid = threading::get_id();
    (void) _tid;

    auto _mode = tim::get_env<std::string>("OMNITRACE_MODE", "");
    OMNITRACE_CONDITIONAL_BASIC_PRINT(true, "Instrumentation mode: %s\n", _mode.c_str());

    // configure the settings
    configure_settings();

    if(gpu::device_count() == 0)
    {
        OMNITRACE_DEBUG("No HIP devices were found: disabling roctracer...\n");
        get_use_roctracer() = false;
    }

    if(_mode == "sampling")
    {
        OMNITRACE_PRINT(
            "Disabling perfetto, timemory, and critical trace in %s mode...\n",
            _mode.c_str());
        get_use_sampling()       = true;
        get_use_timemory()       = false;
        get_use_perfetto()       = false;
        get_use_roctracer()      = false;
        get_use_critical_trace() = false;
    }

    auto _dtor = scope::destructor{ []() {
        if(get_use_sampling())
        {
            pthread_gotcha::enable_sampling_on_child_threads() = false;
            sampling::setup();
            pthread_gotcha::enable_sampling_on_child_threads() = true;
            sampling::unblock_signals();
        }
    } };

    if(get_use_sampling())
    {
        pthread_gotcha::enable_sampling_on_child_threads() = false;
        sampling::block_signals();
    }

    OMNITRACE_DEBUG("[%s]\n", __FUNCTION__);

    if(!get_use_timemory() && !get_use_perfetto() && !get_use_sampling())
    {
        get_state() = State::Finalized;
        OMNITRACE_DEBUG("[%s] Both perfetto and timemory are disabled. Setting the state "
                        "to finalized\n",
                        __FUNCTION__);
        return false;
    }

    // below will effectively do:
    //      get_cpu_cid_stack(0)->emplace_back(-1);
    // plus query some env variables
    add_critical_trace<Device::CPU, Phase::NONE>(0, -1, 0, 0, 0, 0, 0, 0);

    tim::trait::runtime_enabled<comp::roctracer>::set(get_use_roctracer());

    if(get_instrumentation_interval() < 1) get_instrumentation_interval() = 1;
    get_interval_data().reserve(512);

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
            tim::trait::runtime_enabled<api::omnitrace>::set(false);
        }
    }

    // always activate gotcha wrappers
    auto& _main_bundle = get_main_bundle();
    _main_bundle->start();
    assert(_main_bundle->get<mpi_gotcha_t>()->get_is_running());

#if defined(OMNITRACE_USE_ROCTRACER)
    if(get_use_roctracer())
    {
        assert(_main_bundle->get<comp::roctracer>() != nullptr);
        assert(_main_bundle->get<comp::roctracer>()->get_is_running());
    }
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
        omnitrace_thread_data<omnitrace_thread_bundle_t>::construct(
            TIMEMORY_JOIN("", _exe, "/thread-", threading::get_id()),
            quirk::config<quirk::auto_start>{});
        if(get_use_sampling())
        {
            static thread_local auto _once = std::once_flag{};
            std::call_once(_once, sampling::setup);
        }
        static thread_local auto _dtor = scope::destructor{ []() {
            if(get_use_sampling()) sampling::shutdown();
            omnitrace_thread_data<omnitrace_thread_bundle_t>::instance()->stop();
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
        TRACE_EVENT_BEGIN("host", perfetto::StaticString(name));
    };

    static auto _pop_timemory = [](const char* name) {
        auto& _data = get_instrumentation_bundles();
        if(_data.bundles.empty())
        {
            OMNITRACE_DEBUG("[%s] skipped %s :: empty bundle stack\n",
                            "omnitrace_pop_trace", name);
            return;
        }
        _data.bundles.back()->stop();
        _data.allocator.destroy(_data.bundles.back());
        _data.allocator.deallocate(_data.bundles.back(), 1);
        _data.bundles.pop_back();
    };

    static auto _pop_perfetto = [](const char*) { TRACE_EVENT_END("host"); };

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
        static std::set<tim::string_view_t> _sample_options = {
            "OMNITRACE_SAMPLING_FREQ", "OMNITRACE_SAMPLING_DELAY",
            "OMNITRACE_FLAT_SAMPLING", "OMNITRACE_TIMELINE_SAMPLING",
            "OMNITRACE_FLAT_SAMPLING", "OMNITRACE_TIMELINE_SAMPLING",
        };
        static std::set<tim::string_view_t> _perfetto_options = {
            "OMNITRACE_OUTPUT_FILE",
            "OMNITRACE_BACKEND",
            "OMNITRACE_SHMEM_SIZE_HINT_KB",
            "OMNITRACE_BUFFER_SIZE_KB",
        };
        static std::set<tim::string_view_t> _timemory_options = {
            "OMNITRACE_ROCTRACER_FLAT_PROFILE", "OMNITRACE_ROCTRACER_TIMELINE_PROFILE"
        };
        // generic filter for filtering relevant options
        auto _is_omnitrace_option = [](const auto& _v, const auto& _c) {
            if(!get_use_roctracer() && _v.find("OMNITRACE_ROCTRACER_") == 0) return false;
            if(!get_use_critical_trace() && _v.find("OMNITRACE_CRITICAL_TRACE_") == 0)
                return false;
            if(!get_use_perfetto() && _perfetto_options.count(_v) > 0) return false;
            if(!get_use_timemory() && _timemory_options.count(_v) > 0) return false;
            if(!get_use_sampling() && _sample_options.count(_v) > 0) return false;
            const auto npos = std::string::npos;
            if(_v.find("WIDTH") != npos || _v.find("SEPARATOR_FREQ") != npos ||
               _v.find("AUTO_OUTPUT") != npos || _v.find("DART_OUTPUT") != npos ||
               _v.find("FILE_OUTPUT") != npos || _v.find("PLOT_OUTPUT") != npos ||
               _v.find("FLAMEGRAPH_OUTPUT") != npos)
                return false;
            if(!_c.empty())
            {
                if(_c.find("omnitrace") != _c.end()) return true;
                if(_c.find("debugging") != _c.end() && _v.find("DEBUG") != npos)
                    return true;
                if(_c.find("config") != _c.end()) return true;
                if(_c.find("dart") != _c.end()) return false;
                if(_c.find("io") != _c.end() && _v.find("_OUTPUT") != npos) return true;
                if(_c.find("format") != _c.end()) return true;
                return false;
            }
            return (_v.find("OMNITRACE_") == 0);
        };

        if(tim::settings::verbose() > 0)
            tim::print_env(std::cerr, [_is_omnitrace_option](const std::string& _v) {
                return _is_omnitrace_option(_v, std::set<std::string>{});
            });

        print_config_settings(std::cerr, _is_omnitrace_option);
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
    void omnitrace_push_trace(const char* name)
    {
        // return if not active
        if(get_state() == State::Finalized) return;

        if(get_state() != State::Active && !omnitrace_init_tooling())
        {
            OMNITRACE_DEBUG("[%s] %s :: not active and perfetto not initialized\n",
                            __FUNCTION__, name);
            return;
        }
        else
        {
            OMNITRACE_DEBUG("[%s] %s\n", __FUNCTION__, name);
        }

        static auto _sample_rate = std::max<size_t>(get_instrumentation_interval(), 1);
        static thread_local size_t _sample_idx = 0;
        auto                       _enabled    = (_sample_idx++ % _sample_rate == 0);
        get_interval_data().emplace_back(_enabled);
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

    void omnitrace_pop_trace(const char* name)
    {
        if(get_state() == State::Active)
        {
            OMNITRACE_DEBUG("[%s] %s\n", __FUNCTION__, name);
            auto& _interval_data = get_interval_data();
            if(!_interval_data.empty())
            {
                if(_interval_data.back()) get_functors().second(name);
                _interval_data.pop_back();
            }
            if(get_use_critical_trace())
            {
                if(get_cpu_cid_stack() && !get_cpu_cid_stack()->empty())
                {
                    auto _cid = get_cpu_cid_stack()->back();
                    if(get_cpu_cid_parents().find(_cid) != get_cpu_cid_parents().end())
                    {
                        uint64_t _parent_cid          = 0;
                        uint16_t _depth               = 0;
                        auto     _ts                  = comp::wall_clock::record();
                        std::tie(_parent_cid, _depth) = get_cpu_cid_parents().at(_cid);
                        add_critical_trace<Device::CPU, Phase::END>(
                            threading::get_id(), _cid, 0, _parent_cid, _ts, _ts,
                            critical_trace::add_hash_id(name), _depth);
                    }
                }
            }
        }
        else
        {
            OMNITRACE_DEBUG("[%s] %s :: not active\n", __FUNCTION__, name);
        }
    }

    void omnitrace_trace_init(const char* _info, bool _b, const char* _extra)
    {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug_env(), "[%s] %s | %s | %s\n",
                                          __FUNCTION__, _info, (_b) ? "y" : "n", _extra);
        omnitrace_init_tooling();
    }

    void omnitrace_trace_finalize(void)
    {
        // return if not active
        if(get_state() != State::Active) return;

        OMNITRACE_DEBUG("[%s]\n", __FUNCTION__);

        if(get_use_sampling())
        {
            OMNITRACE_DEBUG("[%s] Shutting down sampling...\n", __FUNCTION__);
            pthread_gotcha::enable_sampling_on_child_threads() = false;
            sampling::shutdown();
            sampling::block_signals();
        }

        int _threadpool_verbose = (get_debug()) ? 4 : -1;
        tasking::get_roctracer_thread_pool().set_verbose(_threadpool_verbose);
        tasking::get_critical_trace_thread_pool().set_verbose(_threadpool_verbose);

        if(dmp::rank() == 0) puts("");

        get_state() = State::Finalized;

#if defined(OMNITRACE_USE_ROCTRACER)
        OMNITRACE_DEBUG("[%s] Shutting down roctracer...\n", __FUNCTION__);
        // ensure that threads running roctracer callbacks shutdown
        if(get_use_roctracer()) comp::roctracer::tear_down();
#endif

        // join extra thread(s) used by roctracer
        OMNITRACE_DEBUG("[%s] waiting for all roctracer tasks to complete...\n",
                        __FUNCTION__);
        tasking::get_roctracer_task_group().join();

        OMNITRACE_DEBUG("[%s] Stopping main bundle...\n", __FUNCTION__);
        // stop the main bundle and report the high-level metrics
        if(get_main_bundle())
        {
            get_main_bundle()->stop();
            std::string _msg = JOIN("", *get_main_bundle());
            auto        _pos = _msg.find(">>>  ");
            if(_pos != std::string::npos) _msg = _msg.substr(_pos + 5);
            OMNITRACE_PRINT("%s\n", _msg.c_str());
            get_main_bundle().reset();
        }

        // print out thread-data if they are not still running
        // if they are still running (e.g. thread-pool still alive), the
        // thread-specific data will be wrong if try to stop them from
        // the main thread.
        OMNITRACE_DEBUG("[%s] Destroying thread bundle data...\n", __FUNCTION__);
        for(auto& itr : omnitrace_thread_data<omnitrace_thread_bundle_t>::instances())
        {
            if(itr && itr->get<comp::wall_clock>() &&
               !itr->get<comp::wall_clock>()->get_is_running())
            {
                std::string _msg = JOIN("", *itr);
                auto        _pos = _msg.find(">>>  ");
                if(_pos != std::string::npos) _msg = _msg.substr(_pos + 5);
                OMNITRACE_PRINT("%s\n", _msg.c_str());
            }
        }

        // ensure that all the MT instances are flushed
        OMNITRACE_DEBUG("[%s] Stopping and destroying instrumentation bundles...\n",
                        __FUNCTION__);
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

        // ensure that all the MT instances are flushed
        if(get_use_sampling())
        {
            OMNITRACE_DEBUG("[%s] Post-processing the sampling backtraces...\n",
                            __FUNCTION__);
            for(size_t i = 0; i < max_supported_threads; ++i)
            {
                sampling::backtrace::post_process(i);
                sampling::get_sampler(i).reset();
            }
        }

        if(get_use_critical_trace())
        {
            OMNITRACE_DEBUG("[%s] Generating the critical trace...\n", __FUNCTION__);
            // increase the thread-pool size
            tasking::get_critical_trace_thread_pool().initialize_threadpool(
                get_critical_trace_num_threads());

            for(size_t i = 0; i < max_supported_threads; ++i)
            {
                using critical_trace_hash_data =
                    omnitrace_thread_data<critical_trace::hash_ids, critical_trace::id>;

                if(critical_trace_hash_data::instances().at(i))
                    critical_trace::add_hash_id(
                        *critical_trace_hash_data::instances().at(i));
            }

            for(size_t i = 0; i < max_supported_threads; ++i)
            {
                using critical_trace_chain_data =
                    omnitrace_thread_data<critical_trace::call_chain>;

                if(critical_trace_chain_data::instances().at(i))
                    critical_trace::update(i);  // launch update task
            }

            // make sure outstanding hash tasks completed before compute
            OMNITRACE_PRINT("[%s] waiting for all critical trace tasks to complete...\n",
                            __FUNCTION__);
            tasking::get_critical_trace_task_group().join();

            // launch compute task
            OMNITRACE_PRINT("[%s] launching critical trace compute task...\n",
                            __FUNCTION__);
            critical_trace::compute();
        }

        tasking::get_critical_trace_task_group().join();

        bool _perfetto_output_error = false;
        if(get_use_perfetto() && !is_system_backend())
        {
            OMNITRACE_DEBUG("[%s] Flushing perfetto...\n", __FUNCTION__);
            // Make sure the last event is closed for this example.
            perfetto::TrackEvent::Flush();

            auto& tracing_session = get_trace_session();
            OMNITRACE_DEBUG("[%s] Stopping the blocking perfetto trace sessions...\n",
                            __FUNCTION__);
            tracing_session->StopBlocking();

            OMNITRACE_DEBUG("[%s] Getting the trace data...\n", __FUNCTION__);
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
                    "GB)... ",
                    __FUNCTION__, get_perfetto_output_filename().c_str(),
                    (unsigned long) trace_data.size(),
                    static_cast<double>(trace_data.size()) / units::KB,
                    static_cast<double>(trace_data.size()) / units::MB,
                    static_cast<double>(trace_data.size()) / units::GB);
            std::ofstream ofs{};
            if(!tim::filepath::open(ofs, get_perfetto_output_filename(),
                                    std::ios::out | std::ios::binary))
            {
                fprintf(stderr, "\n[%s]> Error opening '%s'...\n", __FUNCTION__,
                        get_perfetto_output_filename().c_str());
                _perfetto_output_error = true;
            }
            else
            {
                // Write the trace into a file.
                fprintf(stderr, "Done\n");
                ofs.write(&trace_data[0], trace_data.size());
            }
            ofs.close();
        }

        // these should be destroyed before timemory is finalized, especially the
        // roctracer thread-pool
        OMNITRACE_DEBUG("[%s] Destroing the thread pools...\n", __FUNCTION__);
        tasking::get_roctracer_thread_pool().destroy_threadpool();
        tasking::get_critical_trace_thread_pool().destroy_threadpool();

        if(get_use_sampling())
            static_cast<tim::tsettings<bool>*>(
                tim::settings::instance()->find("OMNITRACE_DEBUG")->second.get())
                ->set(false);

        OMNITRACE_DEBUG("[%s] Finalizing timemory...\n", __FUNCTION__);
        tim::timemory_finalize();
        OMNITRACE_DEBUG("[%s] Finalizing timemory... Done\n", __FUNCTION__);

        if(_perfetto_output_error)
            throw std::runtime_error("Unable to create perfetto output file");
    }

    void omnitrace_trace_set_env(const char* env_name, const char* env_val)
    {
        // just search env to avoid initializing the settings
        OMNITRACE_CONDITIONAL_PRINT(get_debug_env(), "[%s] Setting env: %s=%s\n",
                                    __FUNCTION__, env_name, env_val);

        tim::set_env(env_name, env_val, 0);
    }

    void omnitrace_trace_set_mpi(bool use, bool attached)
    {
        // just search env to avoid initializing the settings
        OMNITRACE_CONDITIONAL_PRINT(get_debug_env(), "[%s] use: %s, attached: %s\n",
                                    __FUNCTION__, (use) ? "y" : "n",
                                    (attached) ? "y" : "n");
        if(use && !attached &&
           (get_state() == State::PreInit || get_state() == State::DelayedInit))
        {
            auto& _main_bundle = get_main_bundle();
            _main_bundle->start();
            tim::set_env("OMNITRACE_USE_PID", "ON", 1);
            get_state() = State::DelayedInit;
        }
    }
}

namespace omnitrace
{
std::unique_ptr<main_bundle_t>&
get_main_bundle()
{
    static auto _v =
        (setup_gotchas(), std::make_unique<main_bundle_t>(
                              "omnitrace", quirk::config<quirk::auto_start>{}));
    return _v;
}
}  // namespace omnitrace

namespace
{
// if static objects are destroyed randomly (relatively uncommon behavior)
// this might call finalization before perfetto ends the tracing session
// but static variable in omnitrace_init_tooling is more likely
auto _ensure_finalization = ensure_finalization(true);
}  // namespace
