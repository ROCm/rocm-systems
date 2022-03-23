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
#include "library/api.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/components/functors.hpp"
#include "library/components/fwd.hpp"
#include "library/components/mpi_gotcha.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/gpu.hpp"
#include "library/ompt.hpp"
#include "library/ptl.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/thread_sampler.hpp"
#include "library/timemory.hpp"

#include <atomic>
#include <mutex>
#include <string_view>

using namespace omnitrace;

//======================================================================================//

namespace
{
struct omni_regions
{};
struct user_regions
{};
}  // namespace

using omni_functors = omnitrace::component::functors<omni_regions>;
using user_functors = omnitrace::component::functors<user_regions>;

// TIMEMORY_DEFINE_NAME_TRAIT("host", omni_functors);
// TIMEMORY_DEFINE_NAME_TRAIT("user", user_functors);

TIMEMORY_INVOKE_PREINIT(omni_functors)
TIMEMORY_INVOKE_PREINIT(user_functors)

//======================================================================================//

extern "C" bool
omnitrace_init_tooling_hidden() OMNITRACE_HIDDEN_API;

namespace
{
using interval_data_instances = thread_data<std::vector<bool>>;

auto&
get_interval_data(int64_t _tid = threading::get_id())
{
    static auto& _v =
        interval_data_instances::instances(interval_data_instances::construct_on_init{});
    return _v.at(_tid);
}

auto&
get_timemory_hash_ids(int64_t _tid = threading::get_id())
{
    static auto _v = std::array<tim::hash_map_ptr_t, omnitrace::max_supported_threads>{};
    return _v.at(_tid);
}

auto&
get_timemory_hash_aliases(int64_t _tid = threading::get_id())
{
    static auto _v =
        std::array<tim::hash_alias_ptr_t, omnitrace::max_supported_threads>{};
    return _v.at(_tid);
}

auto
ensure_finalization(bool _static_init = false)
{
    auto _main_tid = threading::get_id();
    (void) _main_tid;
    if(!_static_init)
    {
        OMNITRACE_DEBUG_F("\n");
    }
    else
    {
        OMNITRACE_CONDITIONAL_PRINT_F(get_debug_env(), "\n");
        // This environment variable forces the ROCR-Runtime to use polling to wait
        // for signals rather than interrupts. We set this variable to avoid issues with
        // rocm/roctracer hanging when interrupted by the sampler
        //
        // see:
        // https://github.com/ROCm-Developer-Tools/roctracer/issues/22#issuecomment-572814465
        tim::set_env("HSA_ENABLE_INTERRUPT", "0", 0);
    }
    return scope::destructor{ []() { omnitrace_finalize_hidden(); } };
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
get_instrumentation_bundles(int64_t _tid = threading::get_id())
{
    static thread_local auto& _v = instrumentation_bundles::instances().at(_tid);
    return _v;
}

auto&
get_finalization_functions()
{
    static auto _v = std::vector<std::function<void()>>{};
    return _v;
}

using Device = critical_trace::Device;
using Phase  = critical_trace::Phase;
}  // namespace

//======================================================================================//
///
///
///
//======================================================================================//
namespace
{
auto&
push_count()
{
    static std::atomic<size_t> _v{ 0 };
    return _v;
}
auto&
pop_count()
{
    static std::atomic<size_t> _v{ 0 };
    return _v;
}

auto _debug_push = tim::get_env("OMNITRACE_DEBUG_PUSH", false) && !get_debug_env();
auto _debug_pop  = tim::get_env("OMNITRACE_DEBUG_POP", false) && !get_debug_env();
auto _debug_user =
    tim::get_env("OMNITRACE_DEBUG_USER_REGIONS", false) && !get_debug_env();
}  // namespace

//======================================================================================//

extern "C" void
omnitrace_push_trace_hidden(const char* name)
{
    ++push_count();

    // unconditionally return if finalized
    if(get_state() == State::Finalized)
    {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug_push, "omnitrace_push_trace(%s) called during finalization\n", name);
        return;
    }

    OMNITRACE_CONDITIONAL_BASIC_PRINT(_debug_push, "omnitrace_push_trace(%s)\n", name);

    // the expectation here is that if the state is not active then the call
    // to omnitrace_init_tooling_hidden will activate all the appropriate
    // tooling one time and as it exits set it to active and return true.
    if(get_state() != State::Active && !omnitrace_init_tooling_hidden())
    {
        static auto _debug = get_debug_env() || get_debug_init();
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug, "omnitrace_push_trace(%s) ignored :: not active. state = %s\n", name,
            std::to_string(get_state()).c_str());
        return;
    }

    OMNITRACE_DEBUG("omnitrace_push_trace(%s)\n", name);

    static auto _sample_rate = std::max<size_t>(get_instrumentation_interval(), 1);
    static thread_local size_t _sample_idx = 0;
    auto&                      _interval   = get_interval_data();
    auto                       _enabled    = (_sample_idx++ % _sample_rate == 0);

    _interval->emplace_back(_enabled);
    if(_enabled) omni_functors::start(name);
    if(get_use_critical_trace())
    {
        uint64_t _cid                       = 0;
        uint64_t _parent_cid                = 0;
        uint16_t _depth                     = 0;
        std::tie(_cid, _parent_cid, _depth) = create_cpu_cid_entry();
        auto _ts                            = comp::wall_clock::record();
        add_critical_trace<Device::CPU, Phase::BEGIN>(
            threading::get_id(), _cid, 0, _parent_cid, _ts, 0,
            critical_trace::add_hash_id(name), _depth);
    }
}

//======================================================================================//
///
///
///
//======================================================================================//

extern "C" void
omnitrace_pop_trace_hidden(const char* name)
{
    ++pop_count();

    OMNITRACE_CONDITIONAL_BASIC_PRINT(_debug_pop, "omnitrace_pop_trace(%s)\n", name);

    // only execute when active
    if(get_state() == State::Active)
    {
        OMNITRACE_DEBUG("omnitrace_pop_trace(%s)\n", name);

        auto& _interval_data = get_interval_data();
        if(!_interval_data->empty())
        {
            if(_interval_data->back()) omni_functors::stop(name);
            _interval_data->pop_back();
        }

        if(get_use_critical_trace())
        {
            if(get_cpu_cid_stack() && !get_cpu_cid_stack()->empty())
            {
                auto _cid = get_cpu_cid_stack()->back();
                if(get_cpu_cid_parents()->find(_cid) != get_cpu_cid_parents()->end())
                {
                    uint64_t _parent_cid          = 0;
                    uint16_t _depth               = 0;
                    auto     _ts                  = comp::wall_clock::record();
                    std::tie(_parent_cid, _depth) = get_cpu_cid_parents()->at(_cid);
                    add_critical_trace<Device::CPU, Phase::END>(
                        threading::get_id(), _cid, 0, _parent_cid, _ts, _ts,
                        critical_trace::add_hash_id(name), _depth);
                }
            }
        }
    }
    else
    {
        static auto _debug = get_debug_env();
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug, "omnitrace_pop_trace(%s) ignored :: state = %s\n", name,
            std::to_string(get_state()).c_str());
    }
}

//======================================================================================//
///
///
///
//======================================================================================//

extern "C" void
omnitrace_push_region_hidden(const char* name)
{
    // unconditionally return if finalized
    if(get_state() == State::Finalized)
    {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug_user, "omnitrace_push_region(%s) called during finalization\n", name);
        return;
    }

    OMNITRACE_CONDITIONAL_BASIC_PRINT(_debug_push, "omnitrace_push_region(%s)\n", name);

    // the expectation here is that if the state is not active then the call
    // to omnitrace_init_tooling_hidden will activate all the appropriate
    // tooling one time and as it exits set it to active and return true.
    if(get_state() != State::Active && !omnitrace_init_tooling_hidden())
    {
        static auto _debug = get_debug_env() || get_debug_init();
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug, "omnitrace_push_region(%s) ignored :: not active. state = %s\n", name,
            std::to_string(get_state()).c_str());
        return;
    }

    OMNITRACE_DEBUG("omnitrace_push_region(%s)\n", name);
    user_functors::start(name);
}

//======================================================================================//

extern "C" void
omnitrace_pop_region_hidden(const char* name)
{
    // only execute when active
    if(get_state() == State::Active)
    {
        OMNITRACE_DEBUG("omnitrace_pop_region(%s)\n", name);
        user_functors::stop(name);
    }
    else
    {
        static auto _debug = get_debug_env();
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug, "omnitrace_pop_region(%s) ignored :: state = %s\n", name,
            std::to_string(get_state()).c_str());
    }
}

//======================================================================================//
///
///
///
//======================================================================================//

extern "C" void
omnitrace_set_env_hidden(const char* env_name, const char* env_val)
{
    // just search env to avoid initializing the settings
    OMNITRACE_CONDITIONAL_PRINT_F(get_debug_init() || get_verbose_env() > 2,
                                  "Setting env: %s=%s\n", env_name, env_val);

    tim::set_env(env_name, env_val, 0);

    OMNITRACE_CONDITIONAL_THROW(
        get_state() >= State::Init &&
            (config::get_is_continuous_integration() || get_debug_init()),
        "omnitrace_set_env(\"%s\", \"%s\") called after omnitrace was initialized. state "
        "= %s",
        env_name, env_val, std::to_string(get_state()).c_str());
}

//======================================================================================//
///
///
///
//======================================================================================//

namespace
{
bool                  _set_mpi_called        = false;
std::function<void()> _start_gotcha_callback = []() {};
}  // namespace

extern "C" void
omnitrace_set_mpi_hidden(bool use, bool attached)
{
    // just search env to avoid initializing the settings
    OMNITRACE_CONDITIONAL_PRINT_F(get_debug_init() || get_verbose_env() > 2,
                                  "use: %s, attached: %s\n", (use) ? "y" : "n",
                                  (attached) ? "y" : "n");

    _set_mpi_called       = true;
    config::is_attached() = attached;

    if(use && !attached &&
       (get_state() == State::PreInit || get_state() == State::DelayedInit))
    {
        tim::set_env("OMNITRACE_USE_PID", "ON", 1);
    }
    else if(!use)
    {
        trait::runtime_enabled<mpi_gotcha_t>::set(false);
    }

    OMNITRACE_CONDITIONAL_THROW(
        get_state() >= State::Init &&
            (config::get_is_continuous_integration() || get_debug_init()),
        "omnitrace_set_mpi(use=%s, attached=%s) called after omnitrace was initialized. "
        "state = %s",
        std::to_string(use).c_str(), std::to_string(attached).c_str(),
        std::to_string(get_state()).c_str());

    _start_gotcha_callback();
}

//======================================================================================//

extern "C" void
omnitrace_init_library_hidden()
{
    auto _tid = threading::get_id();
    (void) _tid;

    static bool _once       = false;
    auto        _mode       = get_mode();
    auto        _debug_init = get_debug_init();

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "State is %s...\n",
                                        std::to_string(get_state()).c_str());

    OMNITRACE_CI_THROW(get_state() != State::PreInit, "State is not PreInit :: %s",
                       std::to_string(get_state()).c_str());

    if(get_state() != State::PreInit || get_state() == State::Init || _once) return;
    _once = true;

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "State is %s. Setting to %s...\n",
                                        std::to_string(get_state()).c_str(),
                                        std::to_string(State::Init).c_str());

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(
        _debug_init, "Calling backtrace once so that the one-time call of malloc in "
                     "glibc's backtrace() occurs...\n");
    {
        std::stringstream _ss{};
        tim::print_backtrace<64>(_ss);
        (void) _ss;
    }

    set_state(State::Init);

    OMNITRACE_CI_THROW(get_state() != State::Init,
                       "set_state(State::Init) failed. state is %s",
                       std::to_string(get_state()).c_str());

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "Configuring settings...\n");

    // configure the settings
    configure_settings();

    auto _debug_value = get_debug();
    if(_debug_init) config::set_setting_value("OMNITRACE_DEBUG", true);
    scope::destructor _debug_dtor{ [_debug_value, _debug_init]() {
        if(_debug_init) config::set_setting_value("OMNITRACE_DEBUG", _debug_value);
    } };

    OMNITRACE_DEBUG_F("\n");

    // below will effectively do:
    //      get_cpu_cid_stack(0)->emplace_back(-1);
    // plus query some env variables
    add_critical_trace<Device::CPU, Phase::NONE>(0, -1, 0, 0, 0, 0, 0, 0);

    if(gpu::device_count() == 0 && get_state() != State::Active)
    {
        OMNITRACE_DEBUG_F(
            "No HIP devices were found: disabling roctracer and rocm_smi...\n");
        get_use_roctracer() = false;
        get_use_rocm_smi()  = false;
    }

    if(_mode == Mode::Sampling)
    {
        OMNITRACE_CONDITIONAL_PRINT_F(get_verbose() >= 0,
                                      "Disabling critical trace in %s mode...\n",
                                      std::to_string(_mode).c_str());
        get_use_sampling()       = tim::get_env("OMNITRACE_USE_SAMPLING", true);
        get_use_critical_trace() = false;
    }

    tim::trait::runtime_enabled<comp::roctracer>::set(get_use_roctracer());
    tim::trait::runtime_enabled<comp::roctracer_data>::set(get_use_roctracer());

    if(get_instrumentation_interval() < 1) get_instrumentation_interval() = 1;

    if(get_use_kokkosp())
    {
        auto _force               = 0;
        auto _current_kokkosp_lib = tim::get_env<std::string>("KOKKOS_PROFILE_LIBRARY");
        if(std::regex_search(_current_kokkosp_lib, std::regex{ "libtimemory\\." }))
            _force = 1;
        tim::set_env("KOKKOS_PROFILE_LIBRARY", "libomnitrace.so", _force);
    }

#if defined(OMNITRACE_USE_ROCTRACER)
    tim::set_env("HSA_TOOLS_LIB", "libomnitrace.so", 0);
#endif
}

//======================================================================================//

extern "C" bool
omnitrace_init_tooling_hidden()
{
    static bool _once       = false;
    static auto _debug_init = get_debug_init();

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "State is %s...\n",
                                        std::to_string(get_state()).c_str());

    if(get_state() != State::PreInit || get_state() == State::Init || _once) return false;
    _once = true;

    OMNITRACE_CONDITIONAL_THROW(
        get_state() == State::Init,
        "%s called after omnitrace_init_library() was explicitly called",
        OMNITRACE_FUNCTION);

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_verbose_env() >= 0,
                                        "Instrumentation mode: %s\n",
                                        std::to_string(config::get_mode()).c_str());

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "Printing banner...\n");

    if(get_verbose_env() >= 0) print_banner();

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init,
                                        "Calling omnitrace_init_library()...\n");

    omnitrace_init_library_hidden();

    OMNITRACE_DEBUG_F("\n");

    auto _dtor = scope::destructor{ []() {
        if(get_use_sampling())
        {
            pthread_gotcha::push_enable_sampling_on_child_threads(false);
            thread_sampler::setup();
            sampling::setup();
            pthread_gotcha::pop_enable_sampling_on_child_threads();
            pthread_gotcha::push_enable_sampling_on_child_threads(get_use_sampling());
            sampling::unblock_signals();
        }
        get_main_bundle()->start();
        get_state()= State::Active;  // set to active as very last operation
    } };

    if(get_use_sampling())
    {
        pthread_gotcha::push_enable_sampling_on_child_threads(false);
        sampling::block_signals();
    }

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

    auto _exe          = get_exe_name();
    auto _hash_ids     = tim::get_hash_ids();
    auto _hash_aliases = tim::get_hash_aliases();
    auto _thread_init  = [_exe, _hash_ids, _hash_aliases]() {
        static thread_local auto _thread_setup = [_exe]() {
            if(threading::get_id() > 0)
                threading::set_thread_name(
                    TIMEMORY_JOIN(" ", "Thread", threading::get_id()).c_str());
            thread_data<omnitrace_thread_bundle_t>::construct(
                TIMEMORY_JOIN("", _exe, "/thread-", threading::get_id()),
                quirk::config<quirk::auto_start>{});
            get_interval_data()->reserve(512);
            // save the hash maps
            get_timemory_hash_ids()     = tim::get_hash_ids();
            get_timemory_hash_aliases() = tim::get_hash_aliases();
            return true;
        }();
        static thread_local auto _dtor = scope::destructor{ []() {
            if(get_state() != State::Finalized)
            {
                if(get_use_sampling()) sampling::shutdown();
                if(thread_data<omnitrace_thread_bundle_t>::instance())
                    thread_data<omnitrace_thread_bundle_t>::instance()->stop();
            }
        } };
        (void) _thread_setup;
        (void) _dtor;
    };

    // separate from _thread_init so that it can be called after the first
    // instrumentation on the thread
    auto _setup_thread_sampling = []() {
        static thread_local auto _v = []() {
            auto _use_sampling = get_use_sampling();
            if(_use_sampling) sampling::setup();
            return _use_sampling;
        }();
        (void) _v;
    };

    // functors for starting and stopping timemory omni functors
    auto _push_timemory = [](const char* name) {
        auto& _data = get_instrumentation_bundles();
        // this generates a hash for the raw string array
        auto  _hash   = tim::add_hash_id(tim::string_view_t{ name });
        auto* _bundle = _data.allocator.allocate(1);
        _data.bundles.emplace_back(_bundle);
        _data.allocator.construct(_bundle, _hash);
        _bundle->start();
    };

    auto _pop_timemory = [](const char* name) {
        auto  _hash = tim::hash::get_hash_id(tim::string_view_t{ name });
        auto& _data = get_instrumentation_bundles();
        if(_data.bundles.empty())
        {
            OMNITRACE_DEBUG("[%s] skipped %s :: empty bundle stack\n",
                            "omnitrace_pop_trace", name);
            return;
        }
        for(size_t i = _data.bundles.size(); i > 0; --i)
        {
            auto*& _v = _data.bundles.at(i - 1);
            if(_v->get_hash() == _hash)
            {
                _v->stop();
                _data.allocator.destroy(_v);
                _data.allocator.deallocate(_v, 1);
                _data.bundles.erase(_data.bundles.begin() + (i - 1));
            }
        }
    };

    // functors for starting and stopping perfetto omni functors
    auto _push_perfetto = [](const char* name) {
        uint64_t _ts = comp::wall_clock::record();
        TRACE_EVENT_BEGIN("host", perfetto::StaticString(name), _ts);
    };

    auto _pop_perfetto = [](const char*) {
        uint64_t _ts = comp::wall_clock::record();
        TRACE_EVENT_END("host", _ts);
    };

    // functors for starting and stopping perfetto user functors
    auto _push_user_perfetto = [](const char* name) {
        uint64_t _ts = comp::wall_clock::record();
        TRACE_EVENT_BEGIN("user", nullptr, _ts, [name](perfetto::EventContext& _ctx) {
            _ctx.event()->set_name(name);
        });
    };

    auto _pop_user_perfetto = [](const char*) {
        uint64_t _ts = comp::wall_clock::record();
        TRACE_EVENT_END("user", _ts);
    };

    if(get_use_perfetto() && get_use_timemory())
    {
        omni_functors::configure(
            [=](const char* name) {
                _thread_init();
                _push_perfetto(name);
                _push_timemory(name);
                _setup_thread_sampling();
            },
            [=](const char* name) {
                _pop_timemory(name);
                _pop_perfetto(name);
            });
        user_functors::configure(
            [=](const char* name) {
                _thread_init();
                _push_user_perfetto(name);
                _push_timemory(name);
            },
            [=](const char* name) {
                _pop_timemory(name);
                _pop_user_perfetto(name);
            });
    }
    else if(get_use_perfetto())
    {
        omni_functors::configure(
            [=](const char* name) {
                _thread_init();
                _push_perfetto(name);
                _setup_thread_sampling();
            },
            [=](const char* name) { _pop_perfetto(name); });
        user_functors::configure(
            [=](const char* name) {
                _thread_init();
                _push_user_perfetto(name);
                _setup_thread_sampling();
            },
            [=](const char* name) { _pop_user_perfetto(name); });
    }
    else if(get_use_timemory())
    {
        omni_functors::configure(
            [=](const char* name) {
                _thread_init();
                _push_timemory(name);
                _setup_thread_sampling();
            },
            [=](const char* name) { _pop_timemory(name); });
        user_functors::configure(
            [=](const char* name) {
                _thread_init();
                _push_timemory(name);
                _setup_thread_sampling();
            },
            [=](const char* name) { _pop_timemory(name); });
    }

    ompt::setup();

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

    // if static objects are destroyed in the inverse order of when they are
    // created this should ensure that finalization is called before perfetto
    // ends the tracing session
    static auto _ensure_finalization = ensure_finalization();

    if(dmp::rank() == 0 && get_verbose() >= 0) fprintf(stderr, "\n");

    return true;
}

//======================================================================================//

extern "C" void
omnitrace_init_hidden(const char* _mode, bool _is_binary_rewrite, const char* _argv0)
{
    // always the first
    (void) get_state();
    (void) push_count();
    (void) pop_count();

    get_finalization_functions().emplace_back([_argv0]() {
        OMNITRACE_CI_THROW(get_state() != State::Active,
                           "Finalizer function for popping main invoked in non-active "
                           "state :: state = %s\n",
                           std::to_string(get_state()).c_str());
        if(get_state() == State::Active)
        {
            // if main hasn't been popped yet, pop it
            OMNITRACE_BASIC_VERBOSE(2, "Running omnitrace_pop_trace(%s)...\n", _argv0);
            omnitrace_pop_trace_hidden(_argv0);
        }
    });

    std::atexit([]() {
        // if not already finalized then we should finalize
        if(get_state() != State::Finalized) omnitrace_finalize_hidden();
    });

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(
        get_debug_env() || get_verbose_env() > 2,
        "mode: %s | is binary rewrite: %s | command: %s\n", _mode,
        (_is_binary_rewrite) ? "y" : "n", _argv0);

    tim::set_env("OMNITRACE_MODE", _mode, 0);
    config::is_binary_rewrite() = _is_binary_rewrite;

    // set OMNITRACE_USE_SAMPLING to ON by default if mode is sampling
    tim::set_env("OMNITRACE_USE_SAMPLING", (get_mode() == Mode::Sampling) ? "ON" : "OFF",
                 0);

    // default to KokkosP enabled when sampling, otherwise default to off
    tim::set_env("OMNITRACE_USE_KOKKOSP", (get_mode() == Mode::Sampling) ? "ON" : "OFF",
                 0);

    if(!_set_mpi_called)
    {
        _start_gotcha_callback = []() { get_gotcha_bundle()->start(); };
    }
    else
    {
        get_gotcha_bundle()->start();
    }
}

//======================================================================================//

extern "C" void
omnitrace_finalize_hidden(void)
{
    // return if not active
    if(get_state() != State::Active)
    {
        OMNITRACE_BASIC_DEBUG_F("State = %s. Finalization skipped\n",
                                std::to_string(get_state()).c_str());
        return;
    }

    // some functions called during finalization may alter the push/pop count so we need
    // to save them here
    auto _push_count = push_count().load();
    auto _pop_count  = pop_count().load();

    // e.g. omnitrace_pop_trace("main");
    if(_push_count > _pop_count)
    {
        for(auto& itr : get_finalization_functions())
        {
            itr();
            ++_pop_count;
        }
    }

    set_state(State::Finalized);

    omni_functors::configure([](const char*) {}, [](const char*) {});
    user_functors::configure([](const char*) {}, [](const char*) {});

    pthread_gotcha::push_enable_sampling_on_child_threads(false);
    pthread_gotcha::set_sampling_on_all_future_threads(false);

    auto _debug_init  = get_debug_finalize();
    auto _debug_value = get_debug();
    if(_debug_init) config::set_setting_value("OMNITRACE_DEBUG", true);
    scope::destructor _debug_dtor{ [_debug_value, _debug_init]() {
        if(_debug_init) config::set_setting_value("OMNITRACE_DEBUG", _debug_value);
    } };

    OMNITRACE_DEBUG_F("\n");

    auto& _thread_bundle = thread_data<omnitrace_thread_bundle_t>::instance();
    if(_thread_bundle) _thread_bundle->stop();

    if(dmp::rank() == 0 && get_verbose() >= 0) fprintf(stderr, "\n");
    if(get_verbose() > 0 || get_debug()) config::print_settings();

    OMNITRACE_VERBOSE_F(1, "omnitrace_push_trace :: called %zux\n", _push_count);
    OMNITRACE_VERBOSE_F(1, "omnitrace_pop_trace  :: called %zux\n", _pop_count);

    OMNITRACE_DEBUG_F("Copying over all timemory hash information to main thread...\n");
    // copy these over so that all hashes are known
    auto& _hzero = get_timemory_hash_ids(0);
    auto& _azero = get_timemory_hash_aliases(0);
    for(size_t i = 1; i < max_supported_threads; ++i)
    {
        auto& _hitr = get_timemory_hash_ids(i);
        auto& _aitr = get_timemory_hash_aliases(i);
        if(_hzero && _hitr)
        {
            for(auto itr : *_hitr)
                _hzero->emplace(itr.first, itr.second);
        }
        if(_azero && _aitr)
        {
            for(auto itr : *_aitr)
                _azero->emplace(itr.first, itr.second);
        }
    }

    ompt::shutdown();

    OMNITRACE_DEBUG_F("Stopping and destroying instrumentation bundles...\n");
    for(auto& itr : instrumentation_bundles::instances())
    {
        while(!itr.bundles.empty())
        {
            OMNITRACE_VERBOSE_F(1,
                                "Warning! instrumentation bundle on thread %li with "
                                "label '%s' was not stopped.\n",
                                itr.bundles.back()->tid(),
                                itr.bundles.back()->key().c_str());
            itr.bundles.back()->stop();
            itr.bundles.back()->pop();
            itr.allocator.destroy(itr.bundles.back());
            itr.allocator.deallocate(itr.bundles.back(), 1);
            itr.bundles.pop_back();
        }
    }

    if(get_use_sampling())
    {
        OMNITRACE_DEBUG_F("Shutting down sampling...\n");
        sampling::shutdown();
        sampling::block_signals();
    }

    OMNITRACE_DEBUG_F("Stopping gotcha bundle...\n");
    // stop the gotcha bundle
    if(get_gotcha_bundle())
    {
        get_gotcha_bundle()->stop();
        get_gotcha_bundle().reset();
    }

    pthread_gotcha::shutdown();
    thread_sampler::shutdown();

    OMNITRACE_DEBUG_F("Shutting down roctracer...\n");
    // ensure that threads running roctracer callbacks shutdown
    comp::roctracer::shutdown();

    if(dmp::rank() == 0) fprintf(stderr, "\n");

    OMNITRACE_DEBUG_F("Stopping main bundle...\n");
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

    int _threadpool_verbose = (get_debug()) ? 4 : -1;
    tasking::get_roctracer_thread_pool().set_verbose(_threadpool_verbose);
    if(get_use_critical_trace())
        tasking::get_critical_trace_thread_pool().set_verbose(_threadpool_verbose);

    // join extra thread(s) used by roctracer
    OMNITRACE_DEBUG_F("waiting for all roctracer tasks to complete...\n");
    tasking::get_roctracer_task_group().join();

    // print out thread-data if they are not still running
    // if they are still running (e.g. thread-pool still alive), the
    // thread-specific data will be wrong if try to stop them from
    // the main thread.
    OMNITRACE_DEBUG_F("Destroying thread bundle data...\n");
    for(auto& itr : thread_data<omnitrace_thread_bundle_t>::instances())
    {
        if(itr && itr->get<comp::wall_clock>() &&
           !itr->get<comp::wall_clock>()->get_is_running())
        {
            std::string _msg = JOIN("", *itr);
            auto        _pos = _msg.find(">>>  ");
            if(_pos != std::string::npos) _msg = _msg.substr(_pos + 5);
            OMNITRACE_CONDITIONAL_PRINT(get_verbose() >= 0, "%s\n", _msg.c_str());
        }
    }

    // ensure that all the MT instances are flushed
    OMNITRACE_DEBUG_F("Stopping and destroying instrumentation bundles...\n");
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
        OMNITRACE_DEBUG_F("Post-processing the sampling backtraces...\n");
        for(size_t i = 0; i < max_supported_threads; ++i)
        {
            sampling::backtrace::post_process(i);
            sampling::get_sampler(i).reset();
        }
    }

    if(get_use_critical_trace() || (get_use_rocm_smi() && get_use_roctracer()))
    {
        OMNITRACE_DEBUG_F("Generating the critical trace...\n");
        // increase the thread-pool size
        tasking::get_critical_trace_thread_pool().initialize_threadpool(
            get_critical_trace_num_threads());

        for(size_t i = 0; i < max_supported_threads; ++i)
        {
            using critical_trace_hash_data =
                thread_data<critical_trace::hash_ids, critical_trace::id>;

            if(critical_trace_hash_data::instances().at(i))
                critical_trace::add_hash_id(*critical_trace_hash_data::instances().at(i));
        }

        for(size_t i = 0; i < max_supported_threads; ++i)
        {
            using critical_trace_chain_data = thread_data<critical_trace::call_chain>;

            if(critical_trace_chain_data::instances().at(i))
                critical_trace::update(i);  // launch update task
        }
    }

    OMNITRACE_DEBUG_F("Post-processing the system-level samples...\n");
    thread_sampler::post_process();

    if(get_use_critical_trace())
    {
        // make sure outstanding hash tasks completed before compute
        OMNITRACE_PRINT_F("waiting for all critical trace tasks to complete...\n");
        tasking::get_critical_trace_task_group().join();

        // launch compute task
        OMNITRACE_PRINT_F("launching critical trace compute task...\n");
        critical_trace::compute();
    }

    tasking::get_critical_trace_task_group().join();

    bool _perfetto_output_error = false;
    if(get_use_perfetto() && !is_system_backend())
    {
        if(get_verbose() >= 0) fprintf(stderr, "\n");
        if(get_verbose() >= 0 || get_debug())
            fprintf(stderr, "[%s]|%i> Flushing perfetto...\n", OMNITRACE_FUNCTION,
                    dmp::rank());

        // Make sure the last event is closed for this example.
        perfetto::TrackEvent::Flush();

        auto& tracing_session = get_trace_session();
        OMNITRACE_VERBOSE_F(3, "Stopping the blocking perfetto trace sessions...\n");
        tracing_session->StopBlocking();

        OMNITRACE_VERBOSE_F(3, "Getting the trace data...\n");
        std::vector<char> trace_data{ tracing_session->ReadTraceBlocking() };

        if(trace_data.empty())
        {
            fprintf(stderr,
                    "[%s]> trace data is empty. File '%s' will not be written...\n",
                    OMNITRACE_FUNCTION, get_perfetto_output_filename().c_str());
            return;
        }
        // Write the trace into a file.
        if(get_verbose() >= 0)
            fprintf(stderr, "[%s]|%i> Outputting '%s' (%.2f KB / %.2f MB / %.2f GB)... ",
                    OMNITRACE_FUNCTION, dmp::rank(),
                    get_perfetto_output_filename().c_str(),
                    static_cast<double>(trace_data.size()) / units::KB,
                    static_cast<double>(trace_data.size()) / units::MB,
                    static_cast<double>(trace_data.size()) / units::GB);
        std::ofstream ofs{};
        if(!tim::filepath::open(ofs, get_perfetto_output_filename(),
                                std::ios::out | std::ios::binary))
        {
            fprintf(stderr, "\n[%s]> Error opening '%s'...\n", OMNITRACE_FUNCTION,
                    get_perfetto_output_filename().c_str());
            _perfetto_output_error = true;
        }
        else
        {
            // Write the trace into a file.
            if(get_verbose() >= 0) fprintf(stderr, "Done\n");
            ofs.write(&trace_data[0], trace_data.size());
        }
        ofs.close();
        if(get_verbose() >= 0) fprintf(stderr, "\n");
    }

    // these should be destroyed before timemory is finalized, especially the
    // roctracer thread-pool
    OMNITRACE_DEBUG_F("Destroying the roctracer thread pool...\n");
    {
        std::unique_lock<std::mutex> _lk{ tasking::get_roctracer_mutex() };
        tasking::get_roctracer_task_group().join();
        tasking::get_roctracer_task_group().clear();
        tasking::get_roctracer_task_group().set_pool(nullptr);
        tasking::get_roctracer_thread_pool().destroy_threadpool();
    }

    OMNITRACE_DEBUG_F("Destroying the critical trace thread pool...\n");
    {
        std::unique_lock<std::mutex> _lk{ tasking::get_critical_trace_mutex() };
        tasking::get_critical_trace_task_group().join();
        tasking::get_critical_trace_task_group().clear();
        tasking::get_critical_trace_task_group().set_pool(nullptr);
        tasking::get_critical_trace_thread_pool().destroy_threadpool();
    }

    OMNITRACE_DEBUG_F("Finalizing timemory...\n");
    tim::timemory_finalize();

    OMNITRACE_DEBUG_F("Disabling signal handling...\n");
    tim::disable_signal_detection();

    OMNITRACE_DEBUG_F("Finalized\n");

    if(_perfetto_output_error)
    {
        OMNITRACE_THROW("Error opening perfetto output file: %s",
                        get_perfetto_output_filename().c_str());
    }

    OMNITRACE_CONDITIONAL_THROW(
        get_is_continuous_integration() && _push_count > _pop_count, "%s",
        TIMEMORY_JOIN(" ",
                      "omnitrace_push_trace was called more times than "
                      "omnitrace_pop_trace. The inverse is fine but the current state "
                      "means not every measurement was ended :: pushed:",
                      _push_count, "vs. popped:", _pop_count)
            .c_str());
}

//======================================================================================//

namespace
{
// if static objects are destroyed randomly (relatively uncommon behavior)
// this might call finalization before perfetto ends the tracing session
// but static variable in omnitrace_init_tooling_hidden is more likely
auto _ensure_finalization = ensure_finalization(true);
}  // namespace
