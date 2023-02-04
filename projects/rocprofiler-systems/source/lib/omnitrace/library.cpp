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

#include <timemory/log/color.hpp>
//
//  above should always be included first
//
#include "api.hpp"
#include "common/setup.hpp"
#include "core/categories.hpp"
#include "core/components/fwd.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/constraint.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/gpu.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/causal/data.hpp"
#include "library/causal/experiment.hpp"
#include "library/causal/sampling.hpp"
#include "library/components/exit_gotcha.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/components/mpi_gotcha.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/components/rocprofiler.hpp"
#include "library/coverage.hpp"
#include "library/critical_trace.hpp"
#include "library/ompt.hpp"
#include "library/process_sampler.hpp"
#include "library/ptl.hpp"
#include "library/rcclp.hpp"
#include "library/rocprofiler.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "omnitrace/categories.h"  // in omnitrace-user

#include <timemory/hash/types.hpp>
#include <timemory/manager/manager.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/operations/types/file_output_message.hpp>
#include <timemory/process/threading.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/signals/signal_handlers.hpp>
#include <timemory/signals/signal_mask.hpp>
#include <timemory/signals/types.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>

using namespace omnitrace;

//======================================================================================//

namespace omnitrace
{
namespace perfetto
{
// declare this here bc it has a tendency to cause namespace ambiguities
void
setup();

void
start();
}  // namespace perfetto
}  // namespace omnitrace

//======================================================================================//

namespace
{
auto _timemory_manager  = tim::manager::instance();
auto _timemory_settings = tim::settings::shared_instance();

bool
ensure_initialization(bool _offset, int64_t _glob_n, int64_t _offset_n)
{
    auto _exit_info = component::exit_gotcha::get_exit_info();
    if(_exit_info.is_known && _exit_info.exit_code != EXIT_SUCCESS) return _offset;

    auto _tid         = utility::get_thread_index();
    auto _max_threads = grow_data(_tid + 1);

    if(_tid > 0 && _tid < _max_threads)
    {
        const auto& _info = thread_info::get();
        OMNITRACE_BASIC_VERBOSE_F(3,
                                  "thread info: %s, offset: %s, global counter: %li, "
                                  "offset counter: %li, max threads: %li\n",
                                  std::to_string(static_cast<bool>(_info)).c_str(),
                                  std::to_string(_offset).c_str(), _glob_n, _offset_n,
                                  _max_threads);
    }

    return _offset;
}

void
finalization_handler()
{
    if(get_state() == State::Active) omnitrace_finalize();
}

auto
ensure_finalization(bool _static_init = false)
{
    if(config::set_signal_handler(nullptr) == nullptr)
        config::set_signal_handler(&finalization_handler);

    if(_static_init)
    {
        auto _idx = threading::add_callback(&ensure_initialization);
        if(_idx < 0)
            throw exception<std::runtime_error>("failure adding threading callback");
    }

    OMNITRACE_CI_BASIC_THROW(
        config::set_signal_handler(nullptr) != &finalization_handler,
        "Assignment of signal handler failed. signal handler is %s, expected %s\n",
        as_hex(reinterpret_cast<void*>(config::set_signal_handler(nullptr))).c_str(),
        as_hex(reinterpret_cast<void*>(&finalization_handler)).c_str());

    const auto& _info = thread_info::init();
    const auto& _tid  = _info->index_data;
    if(_tid)
    {
        OMNITRACE_CI_THROW(_tid->sequent_value != threading::get_id(),
                           "Error! internal tid != %li :: %li", threading::get_id(),
                           _tid->sequent_value);
        OMNITRACE_CI_THROW(_tid->system_value != threading::get_sys_tid(),
                           "Error! system tid != %li :: %li", threading::get_sys_tid(),
                           _tid->system_value);
    }

    if(get_env("OMNITRACE_MONOCHROME", false)) tim::log::monochrome() = true;

    (void) tim::manager::instance();
    (void) tim::settings::shared_instance();

    if(!tim::get_shared_ptr_pair_callback())
    {
        tim::get_shared_ptr_pair_callback() =
            new tim::shared_ptr_pair_callback_t{ [](int64_t _n) {
                if(_n == 0) omnitrace_finalize_hidden();
            } };
    }

    if(_static_init)
    {
        OMNITRACE_BASIC_DEBUG_F("\n");
        auto _verbose =
            get_verbose_env() + ((get_debug_env() || get_debug_init()) ? 16 : 0);
        auto _search_paths = JOIN(':', tim::get_env<std::string>("OMNITRACE_PATH", ""),
                                  tim::get_env<std::string>("PWD"), ".",
                                  tim::get_env<std::string>("LD_LIBRARY_PATH", ""),
                                  tim::get_env<std::string>("LIBRARY_PATH", ""),
                                  tim::get_env<std::string>("PATH", ""));
        common::setup_environ(_verbose, _search_paths);
    }
    else
    {
        OMNITRACE_DEBUG_F("\n");
    }

    return scope::destructor{ []() { omnitrace_finalize_hidden(); } };
}

auto
is_system_backend()
{
    // if get_backend() returns 'system' or 'all', this is true
    return (get_backend() != "inprocess");
}

using Device = critical_trace::Device;
using Phase  = critical_trace::Phase;

template <typename... Tp>
struct fini_bundle
{
    using data_type = std::tuple<Tp...>;

    OMNITRACE_DEFAULT_OBJECT(fini_bundle)

    fini_bundle(std::string_view _label)
    : m_label{ _label }
    {}

    template <typename... Args>
    void start(Args&&... _args)
    {
        TIMEMORY_FOLD_EXPRESSION(tim::operation::start<Tp>{}(
            std::get<Tp>(m_data), std::forward<Args>(_args)...));
    }

    template <typename... Args>
    void stop(Args&&... _args)
    {
        TIMEMORY_FOLD_EXPRESSION(tim::operation::stop<Tp>{}(
            std::get<Tp>(m_data), std::forward<Args>(_args)...));
    }

    std::string as_string(bool _print_prefix = true) const
    {
        std::stringstream _ss;
        if(_print_prefix && m_label.length() > 0) _ss << m_label << " : ";
        _ss << timemory::join::join(", ", std::get<Tp>(m_data)...);
        return _ss.str();
    }

    std::string_view m_label = {};
    data_type        m_data  = {};
};

template <typename... Tp>
struct fini_bundle<tim::lightweight_tuple<Tp...>>
{
    using base_type = fini_bundle<Tp...>;
};

using fini_bundle_t = typename fini_bundle<main_bundle_t>::base_type;
}  // namespace

//======================================================================================//
///
///
///
//======================================================================================//

namespace
{
struct set_env_s  // NOLINT
{};
}  // namespace

extern "C" void
omnitrace_set_env_hidden(const char* env_name, const char* env_val)
{
    tim::auto_lock_t _lk{ tim::type_mutex<set_env_s>() };

    static auto _set_envs = std::set<std::string_view>{};
    bool        _success  = _set_envs.emplace(env_name).second;

    // just search env to avoid initializing the settings
    OMNITRACE_CONDITIONAL_PRINT_F(get_debug_init() || get_verbose_env() > 2,
                                  "Setting env: %s=%s\n", env_name, env_val);

    tim::set_env(env_name, env_val, 0);

    if(_success && get_state() >= State::Init)
    {
        OMNITRACE_WARNING_F(
            0,
            "omnitrace_set_env(\"%s\", \"%s\") called after omnitrace was initialized. "
            "state = %s. This environment variable will have no effect\n",
            env_name, env_val, std::to_string(get_state()).c_str());
    }
}

//======================================================================================//
///
///
///
//======================================================================================//

namespace
{
bool                  _set_mpi_called   = false;
std::function<void()> _preinit_callback = []() { get_preinit_bundle()->start(); };

void
omnitrace_preinit_hidden()
{
    // run once and discard
    _preinit_callback();
    _preinit_callback = []() {};
}
}  // namespace

extern "C" void
omnitrace_set_mpi_hidden(bool use, bool attached)
{
    static bool _once = false;
    static auto _args = std::make_pair(use, attached);

    // this function may be called multiple times if multiple libraries are instrumented
    // we want to guard against multiple calls which with different arguments
    if(_once && std::tie(_args.first, _args.second) == std::tie(use, attached)) return;
    _once = true;

    // just search env to avoid initializing the settings
    OMNITRACE_CONDITIONAL_PRINT_F(get_debug_init() || get_verbose_env() > 2,
                                  "use: %s, attached: %s\n", (use) ? "y" : "n",
                                  (attached) ? "y" : "n");

    _set_mpi_called       = true;
    config::is_attached() = attached;

    if(use && !attached && get_state() == State::PreInit)
    {
        tim::set_env("OMNITRACE_USE_PID", "ON", 1);
    }
    else if(!use)
    {
        trait::runtime_enabled<mpi_gotcha_t>::set(false);
    }

    if(get_state() >= State::Init)
    {
        OMNITRACE_WARNING_F(
            0,
            "omnitrace_set_mpi(use=%s, attached=%s) called after omnitrace was "
            "initialized. state = %s. MPI support may not be properly initialized. Use "
            "OMNITRACE_USE_MPIP=ON and OMNITRACE_USE_PID=ON to ensure full support\n",
            std::to_string(use).c_str(), std::to_string(attached).c_str(),
            std::to_string(get_state()).c_str());
    }

    omnitrace_preinit_hidden();
}

//======================================================================================//

extern "C" void
omnitrace_init_library_hidden()
{
    auto _tid = threading::get_id();
    (void) _tid;

    static bool _once       = false;
    auto        _debug_init = get_debug_init();

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "State is %s...\n",
                                        std::to_string(get_state()).c_str());

    OMNITRACE_CI_THROW(get_state() != State::PreInit, "State is not PreInit :: %s",
                       std::to_string(get_state()).c_str());

    if(get_state() != State::PreInit || get_state() == State::Init || _once) return;
    _once = true;

    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "State is %s. Setting to %s...\n",
                                        std::to_string(get_state()).c_str(),
                                        std::to_string(State::Init).c_str());

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(
        _debug_init, "Calling backtrace once so that the one-time call of malloc in "
                     "glibc's backtrace() occurs...\n");
    {
        std::stringstream _ss{};
        timemory_print_backtrace<16>(_ss);
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

    // below will effectively do:
    //      get_cpu_cid_stack(0)->emplace_back(-1);
    // plus query some env variables
    add_critical_trace<Device::CPU, Phase::NONE>(0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0);

    tim::trait::runtime_enabled<comp::roctracer>::set(get_use_roctracer());
    tim::trait::runtime_enabled<comp::roctracer_data>::set(get_use_roctracer() &&
                                                           get_use_timemory());

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "\n");
}

//======================================================================================//

extern "C" bool
omnitrace_init_tooling_hidden()
{
    if(get_env("OMNITRACE_MONOCHROME", false, false)) tim::log::monochrome() = true;

    if(!tim::get_env("OMNITRACE_INIT_TOOLING", true))
    {
        omnitrace_init_library_hidden();
        return false;
    }

    static bool _once       = false;
    static auto _debug_init = get_debug_init();

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(_debug_init, "State is %s...\n",
                                        std::to_string(get_state()).c_str());

    if(get_state() != State::PreInit || get_state() == State::Init || _once) return false;
    _once = true;

    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

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
        // if set to finalized, don't continue
        if(get_state() > State::Active) return;
        if(get_use_process_sampling())
        {
            OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
            process_sampler::setup();
        }
        if(get_use_causal())
        {
            {
                OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
                causal::sampling::setup();
            }
            push_enable_sampling_on_child_threads(get_use_causal());
            sampling::unblock_signals();
        }
        else if(get_use_sampling())
        {
            {
                OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
                sampling::setup();
            }
            push_enable_sampling_on_child_threads(get_use_sampling());
            sampling::unblock_signals();
        }
        get_main_bundle()->start();
        OMNITRACE_DEBUG_F("State: %s -> State::Active\n",
                          std::to_string(get_state()).c_str());
        set_state(State::Active);  // set to active as very last operation
    } };

    OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    // perfetto initialization
    if(get_use_perfetto())
    {
        OMNITRACE_VERBOSE_F(1, "Setting up Perfetto...\n");
        omnitrace::perfetto::setup();
    }

    // ideally these have already been started
    omnitrace_preinit_hidden();

    // start these gotchas once settings have been initialized
    if(get_init_bundle()) get_init_bundle()->start();

    if(get_use_sampling()) sampling::block_signals();

    tasking::setup();

    if(get_use_causal()) causal::start_experimenting();

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
            tim::trait::runtime_enabled<project::omnitrace>::set(false);
        }
    }

    if(get_use_ompt())
    {
        OMNITRACE_VERBOSE_F(1, "Setting up OMPT...\n");
        ompt::setup();
    }

    if(get_use_rcclp())
    {
        OMNITRACE_VERBOSE_F(1, "Setting up RCCLP...\n");
        rcclp::setup();
    }

    if(get_use_perfetto() && !is_system_backend())
    {
        omnitrace::perfetto::start();
    }

    categories::setup();

    // if static objects are destroyed in the inverse order of when they are
    // created this should ensure that finalization is called before perfetto
    // ends the tracing session
    static auto _ensure_finalization = ensure_finalization();

    return true;
}

//======================================================================================//

extern "C" void
omnitrace_init_hidden(const char* _mode, bool _is_binary_rewrite, const char* _argv0_c)
{
    static int  _total_count = 0;
    static auto _args = std::make_pair(std::string_view{ _mode }, _is_binary_rewrite);

    auto _count   = _total_count++;
    auto _mode_sv = std::string_view{ _mode };
    auto _argv0   = (_argv0_c) ? std::string{ _argv0_c } : config::get_exe_name();
    // this function may be called multiple times if multiple libraries are instrumented
    // we want to guard against multiple calls which with different arguments
    if(_count > 0 &&
       std::tie(_args.first, _args.second) == std::tie(_mode_sv, _is_binary_rewrite))
        return;

    OMNITRACE_CONDITIONAL_THROW(
        _count > 0 &&
            std::tie(_args.first, _args.second) != std::tie(_mode_sv, _is_binary_rewrite),
        "\nomnitrace_init(...) called multiple times with different arguments for mode "
        "and/or is_binary_rewrite:"
        "\n    Invocation #1: omnitrace_init(mode=%-8s, is_binary_rewrite=%-5s, ...)"
        "\n    Invocation #%i: omnitrace_init(mode=%-8s, is_binary_rewrite=%-5s, ...)",
        _args.first.data(), std::to_string(_args.second).c_str(), _count + 1, _mode,
        std::to_string(_is_binary_rewrite).c_str());

    // always the first
    (void) get_state();
    (void) tracing::push_count();
    (void) tracing::pop_count();

    if(get_state() >= State::Init)
    {
        if(std::string_view{ _mode } != "trace" && std::string_view{ _mode } != "Trace")
        {
            OMNITRACE_WARNING_F(
                0,
                "omnitrace_init(mode=%s, is_binary_rewrite=%s, argv0=%s) "
                "called after omnitrace was initialized. state = %s. Mode-based settings "
                "(via -M <MODE> passed to omnitrace exe) may not be properly "
                "configured.\n",
                _mode, std::to_string(_is_binary_rewrite).c_str(), _argv0.c_str(),
                std::to_string(get_state()).c_str());
        }
    }

    tracing::get_finalization_functions().emplace_back([_argv0_c]() {
        OMNITRACE_CI_THROW(get_state() != State::Active,
                           "Finalizer function for popping main invoked in non-active "
                           "state :: state = %s\n",
                           std::to_string(get_state()).c_str());
        if(get_state() == State::Active)
        {
            auto _name = (_argv0_c) ? std::string{ _argv0_c } : config::get_exe_name();
            // if main hasn't been popped yet, pop it
            OMNITRACE_BASIC_VERBOSE(2, "Running omnitrace_pop_trace(%s)...\n",
                                    _name.c_str());
            omnitrace_pop_trace_hidden(_name.c_str());
        }
    });

    std::atexit([]() {
        // if active (not already finalized) then we should finalize
        if(get_state() == State::Active) omnitrace_finalize_hidden();
    });

    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(
        get_debug_env() || get_verbose_env() > 2,
        "mode: %s | is binary rewrite: %s | command: %s\n", _mode,
        (_is_binary_rewrite) ? "y" : "n", _argv0.c_str());

    tim::set_env("OMNITRACE_MODE", _mode, 0);
    config::is_binary_rewrite() = _is_binary_rewrite;

    if(_set_mpi_called)
    {
        omnitrace_preinit_hidden();
    }
}

//======================================================================================//

extern "C" void
omnitrace_reset_preload_hidden(void)
{
    tim::set_env("OMNITRACE_PRELOAD", "0", 1);
    auto&& _preload_libs = get_env("LD_PRELOAD", std::string{});
    if(_preload_libs.find("libomnitrace") != std::string::npos)
    {
        auto _modified_preload = std::string{};
        for(const auto& itr : delimit(_preload_libs, ":"))
        {
            if(itr.find("libomnitrace") != std::string::npos) continue;
            _modified_preload += common::join("", ":", itr);
        }
        if(!_modified_preload.empty() && _modified_preload.find(':') == 0)
            _modified_preload = _modified_preload.substr(1);

        tim::set_env("LD_PRELOAD", _modified_preload, 1);
    }
}

//======================================================================================//

extern "C" void
omnitrace_finalize_hidden(void)
{
    // disable thread id recycling during finalization
    threading::recycle_ids() = false;
    // disable initialization callback
    threading::remove_callback(&ensure_initialization);

    set_thread_state(ThreadState::Completed);

    // return if not active
    if(get_state() != State::Active)
    {
        OMNITRACE_BASIC_DEBUG_F("State = %s. Finalization skipped\n",
                                std::to_string(get_state()).c_str());
        return;
    }

    if(get_verbose() >= 0 || get_debug()) fprintf(stderr, "\n");
    OMNITRACE_VERBOSE_F(0, "finalizing...\n");

    sampling::block_samples();

    thread_info::set_stop(comp::wall_clock::record());

    tim::signals::block_signals(get_sampling_signals(),
                                tim::signals::sigmask_scope::process);

    omnitrace_reset_preload_hidden();

    // some functions called during finalization may alter the push/pop count so we need
    // to save them here
    auto _push_count = tracing::push_count().load();
    auto _pop_count  = tracing::pop_count().load();

    // e.g. omnitrace_pop_trace("main");
    if(_push_count > _pop_count)
    {
        for(auto& itr : tracing::get_finalization_functions())
        {
            itr();
            ++_pop_count;
        }
    }

    set_state(State::Finalized);

    push_enable_sampling_on_child_threads(false);
    set_sampling_on_all_future_threads(false);

    // if the categories are not enabled, it can/will suppress generating output for data
    // in category
    categories::enable_categories();

    auto _debug_init  = get_debug_finalize();
    auto _debug_value = get_debug();
    if(_debug_init) config::set_setting_value("OMNITRACE_DEBUG", true);
    scope::destructor _debug_dtor{ [_debug_value, _debug_init]() {
        if(_debug_init) config::set_setting_value("OMNITRACE_DEBUG", _debug_value);
    } };

    auto& _thread_bundle = thread_data<thread_bundle_t>::instance();
    if(_thread_bundle) _thread_bundle->stop();

    if(get_verbose() >= 1 || get_debug())
    {
        if(dmp::rank() == 0)
        {
            OMNITRACE_PRINT_F("\n");
            config::print_settings(get_env<bool>("OMNITRACE_PRINT_ENV", get_debug()));
        }
    }

    OMNITRACE_VERBOSE_F(1, "omnitrace_push_trace :: called %zux\n", _push_count);
    OMNITRACE_VERBOSE_F(1, "omnitrace_pop_trace  :: called %zux\n", _pop_count);

    tim::signals::enable_signal_detection({ tim::signals::sys_signal::Interrupt },
                                          [](int) {});

    OMNITRACE_DEBUG_F("Copying over all timemory hash information to main thread...\n");
    // copy these over so that all hashes are known
    auto& _hzero = tracing::get_timemory_hash_ids(0);
    auto& _azero = tracing::get_timemory_hash_aliases(0);
    for(size_t i = 1; i < max_supported_threads; ++i)
    {
        auto& _hitr = tracing::get_timemory_hash_ids(i);
        auto& _aitr = tracing::get_timemory_hash_aliases(i);
        if(_hzero && _hitr)
        {
            for(const auto& itr : *_hitr)
                _hzero->emplace(itr.first, itr.second);
        }
        if(_azero && _aitr)
        {
            for(auto itr : *_aitr)
                _azero->emplace(itr.first, itr.second);
        }
    }

    // stop the main bundle which has stats for run
    if(get_main_bundle())
    {
        OMNITRACE_DEBUG_F("Stopping main bundle...\n");
        get_main_bundle()->stop();
    }

    fini_bundle_t _finalization{};
    _finalization.start();

    if(get_use_rcclp())
    {
        OMNITRACE_VERBOSE_F(1, "Shutting down RCCLP...\n");
        rcclp::shutdown();
    }

    if(get_use_ompt())
    {
        OMNITRACE_VERBOSE_F(1, "Shutting down OMPT...\n");
        ompt::shutdown();
    }

    OMNITRACE_DEBUG_F("Stopping and destroying instrumentation bundles...\n");
    for(size_t i = 0; i < max_supported_threads; ++i)
    {
        auto&       itr   = instrumentation_bundles::instances().at(i);
        const auto& _info = thread_info::get(i, SequentTID);
        while(!itr.bundles.empty())
        {
            int _lvl = 1;
            if(_info->is_offset)
            {
                ++_pop_count;
                _lvl = 4;
            }
            OMNITRACE_VERBOSE_F(_lvl,
                                "Warning! instrumentation bundle on thread %zu (TID=%li) "
                                "with label '%s' was not stopped.\n",
                                i, itr.bundles.back()->tid(),
                                itr.bundles.back()->key().c_str());

            itr.bundles.back()->stop();
            itr.bundles.back()->pop();
            itr.allocator.destroy(itr.bundles.back());
            itr.allocator.deallocate(itr.bundles.back(), 1);
            itr.bundles.pop_back();
        }
    }

    // stop the main gotcha which shuts down the pthread gotchas
    if(get_init_bundle())
    {
        OMNITRACE_DEBUG_F("Stopping main gotcha...\n");
        get_init_bundle()->stop();
    }

    // stop the gotcha bundle
    if(get_preinit_bundle())
    {
        OMNITRACE_VERBOSE_F(1, "Shutting down miscellaneous gotchas...\n");
        get_preinit_bundle()->stop();
        component::mpi_gotcha::shutdown();
    }

    if(get_use_process_sampling())
    {
        OMNITRACE_VERBOSE_F(1, "Shutting down background sampler...\n");
        process_sampler::shutdown();
    }

    if(get_use_roctracer())
    {
        OMNITRACE_VERBOSE_F(1, "Shutting down roctracer...\n");
        // ensure that threads running roctracer callbacks shutdown
        comp::roctracer::shutdown();

        // join extra thread(s) used by roctracer
        OMNITRACE_VERBOSE_F(2, "Waiting on roctracer tasks...\n");
        tasking::join();
    }

    if(get_use_rocprofiler())
    {
        OMNITRACE_VERBOSE_F(1, "Shutting down rocprofiler...\n");
        rocprofiler::post_process();
        rocprofiler::rocm_cleanup();
    }

    if(get_use_causal())
    {
        OMNITRACE_VERBOSE_F(1, "Shutting down causal sampling...\n");
        causal::sampling::shutdown();
    }

    if(get_use_sampling())
    {
        OMNITRACE_VERBOSE_F(1, "Shutting down sampling...\n");
        sampling::shutdown();
    }

    OMNITRACE_VERBOSE_F(3, "Reporting the process- and thread-level metrics...\n");
    // report the high-level metrics for the process
    if(get_main_bundle())
    {
        OMNITRACE_VERBOSE_F(0, "\n");
        std::string _msg = JOIN("", *get_main_bundle());
        auto        _pos = _msg.find(">>>  ");
        if(_pos != std::string::npos) _msg = _msg.substr(_pos + 5);
        OMNITRACE_VERBOSE_F(0, "%s\n", _msg.c_str());
        OMNITRACE_DEBUG_F("Resetting main bundle...\n");
        get_main_bundle()->reset();
    }

    // print out thread-data if they are not still running
    // if they are still running (e.g. thread-pool still alive), the
    // thread-specific data will be wrong if try to stop them from
    // the main thread.
    auto _thr_verbose = (config::get_use_causal()) ? 1 : 0;
    for(auto& itr : thread_data<thread_bundle_t>::instances())
    {
        if(itr && itr->get<comp::wall_clock>() &&
           !itr->get<comp::wall_clock>()->get_is_running())
        {
            std::string _msg = JOIN("", *itr);
            auto        _pos = _msg.find(">>>  ");
            if(_pos != std::string::npos) _msg = _msg.substr(_pos + 5);
            OMNITRACE_VERBOSE_F(_thr_verbose, "%s\n", _msg.c_str());
        }
    }

    OMNITRACE_VERBOSE_F(0, "\n");

    // ensure that all the MT instances are flushed
    if(get_use_sampling())
    {
        OMNITRACE_VERBOSE_F(1, "Post-processing the sampling backtraces...\n");
        sampling::post_process();
    }

    if(get_use_causal())
    {
        OMNITRACE_VERBOSE_F(1, "Finishing the causal experiments...\n");
        causal::finish_experimenting();
    }

    if(get_use_critical_trace() || (get_use_rocm_smi() && get_use_roctracer()))
    {
        OMNITRACE_VERBOSE_F(1, "Generating the critical trace...\n");
        for(size_t i = 0; i < max_supported_threads; ++i)
        {
            using critical_trace_hash_data =
                thread_data<critical_trace::hash_ids, critical_trace::id>;

            if(critical_trace_hash_data::instances().at(i))
            {
                OMNITRACE_DEBUG_F("Copying the hash id data for thread %zu...\n", i);
                critical_trace::add_hash_id(*critical_trace_hash_data::instances().at(i));
            }
        }

        for(size_t i = 0; i < max_supported_threads; ++i)
        {
            using critical_trace_chain_data = thread_data<critical_trace::call_chain>;

            if(critical_trace_chain_data::instances().at(i))
            {
                OMNITRACE_DEBUG_F(
                    "Updating the critical trace call-chains for thread %zu...\n", i);
                critical_trace::update(i);  // launch update task
            }
        }

        OMNITRACE_VERBOSE_F(1, "Waiting on critical trace updates...\n");
        tasking::join();
    }

    if(get_use_process_sampling())
    {
        OMNITRACE_VERBOSE_F(1, "Post-processing the system-level samples...\n");
        process_sampler::post_process();
    }

    if(get_use_critical_trace())
    {
        // launch compute task
        OMNITRACE_VERBOSE_F(1, "Launching critical trace compute task...\n");
        critical_trace::compute();

        OMNITRACE_VERBOSE_F(1, "Waiting on critical trace computation...\n");
        tasking::join();
    }

    // shutdown tasking before timemory is finalized, especially the roctracer thread-pool
    OMNITRACE_VERBOSE_F(1, "Shutting down thread-pools...\n");
    tasking::shutdown();

    if(get_use_code_coverage())
    {
        OMNITRACE_VERBOSE_F(1, "Post-processing the code coverage...\n");
        coverage::post_process();
    }

    bool _perfetto_output_error = false;
    if(get_use_perfetto() && !is_system_backend())
    {
        auto& tracing_session = get_perfetto_session();

        OMNITRACE_CI_THROW(tracing_session == nullptr,
                           "Null pointer to the tracing session");

        OMNITRACE_VERBOSE_F(0, "Finalizing perfetto...\n");

        // Make sure the last event is closed for this example.
        ::perfetto::TrackEvent::Flush();
        tracing_session->FlushBlocking();

        OMNITRACE_VERBOSE_F(3, "Stopping the blocking perfetto trace session...\n");
        tracing_session->StopBlocking();

        using char_vec_t = std::vector<char>;
        OMNITRACE_VERBOSE_F(3, "Getting the trace data...\n");

        auto trace_data = char_vec_t{};
#if defined(TIMEMORY_USE_MPI) && TIMEMORY_USE_MPI > 0
        if(get_perfetto_combined_traces())
        {
            using perfetto_mpi_get_t =
                tim::operation::finalize::mpi_get<char_vec_t, true>;

            char_vec_t              _trace_data{ tracing_session->ReadTraceBlocking() };
            std::vector<char_vec_t> _rank_data = {};
            auto _combine = [](char_vec_t& _dst, const char_vec_t& _src) -> char_vec_t& {
                _dst.reserve(_dst.size() + _src.size());
                for(auto&& itr : _src)
                    _dst.emplace_back(itr);
                return _dst;
            };

            perfetto_mpi_get_t{ get_perfetto_combined_traces(), settings::node_count() }(
                _rank_data, _trace_data, _combine);
            for(auto& itr : _rank_data)
                trace_data =
                    (trace_data.empty()) ? std::move(itr) : _combine(trace_data, itr);
        }
        else
        {
            trace_data = tracing_session->ReadTraceBlocking();
        }
#else
        trace_data = tracing_session->ReadTraceBlocking();
#endif

        if(!trace_data.empty())
        {
            operation::file_output_message<tim::project::omnitrace> _fom{};
            // Write the trace into a file.
            if(get_verbose() >= 0)
                _fom(get_perfetto_output_filename(), std::string{ "perfetto" },
                     " (%.2f KB / %.2f MB / %.2f GB)... ",
                     static_cast<double>(trace_data.size()) / units::KB,
                     static_cast<double>(trace_data.size()) / units::MB,
                     static_cast<double>(trace_data.size()) / units::GB);
            std::ofstream ofs{};
            if(!tim::filepath::open(ofs, get_perfetto_output_filename(),
                                    std::ios::out | std::ios::binary))
            {
                _fom.append("Error opening '%s'...",
                            get_perfetto_output_filename().c_str());
                _perfetto_output_error = true;
            }
            else
            {
                // Write the trace into a file.
                ofs.write(&trace_data[0], trace_data.size());
                if(get_verbose() >= 0) _fom.append("%s", "Done");  // NOLINT
                if(_timemory_manager)
                    _timemory_manager->add_file_output("protobuf", "perfetto",
                                                       get_perfetto_output_filename());
            }
            ofs.close();
        }
        else if(dmp::rank() == 0)
        {
            OMNITRACE_VERBOSE_F(0,
                                "trace data is empty. File '%s' will not be written...\n",
                                get_perfetto_output_filename().c_str());
        }
    }

    if(_timemory_manager && _timemory_manager != nullptr)
    {
        _timemory_manager->add_metadata([](auto& ar) {
            auto _maps = tim::procfs::read_maps(process::get_id());
            auto _libs = std::set<std::string>{};
            for(auto& itr : _maps)
            {
                auto&& _path = itr.pathname;
                if(!_path.empty() && _path.at(0) != '[' && filepath::exists(_path))
                    _libs.emplace(_path);
            }
            ar(tim::cereal::make_nvp("memory_maps_files", _libs),
               tim::cereal::make_nvp("memory_maps", _maps));
        });

        _timemory_manager->set_write_metadata(-1);

        OMNITRACE_VERBOSE_F(1, "Finalizing timemory...\n");
        tim::timemory_finalize(_timemory_manager.get());

        auto _cfg       = settings::compose_filename_config{};
        _cfg.use_suffix = config::get_use_pid();
        _timemory_manager->write_metadata(settings::get_global_output_prefix(),
                                          "omnitrace", _cfg);
    }

    categories::shutdown();

    _finalization.stop();

    if(_perfetto_output_error)
    {
        OMNITRACE_THROW("Error opening perfetto output file: %s",
                        get_perfetto_output_filename().c_str());
    }

    OMNITRACE_CI_THROW(
        _push_count > _pop_count, "%s",
        TIMEMORY_JOIN(" ",
                      "omnitrace_push_trace was called more times than "
                      "omnitrace_pop_trace. The inverse is fine but the current state "
                      "means not every measurement was ended :: pushed:",
                      _push_count, "vs. popped:", _pop_count)
            .c_str());

    config::finalize();

    OMNITRACE_VERBOSE_F(0, "Finalized: %s\n", _finalization.as_string().c_str());
}

//======================================================================================//

namespace
{
// if static objects are destroyed randomly (relatively uncommon behavior)
// this might call finalization before perfetto ends the tracing session
// but static variable in omnitrace_init_tooling_hidden is more likely
auto _ensure_finalization = ensure_finalization(true);
auto _manager             = tim::manager::instance();
auto _settings            = tim::settings::shared_instance();
}  // namespace
