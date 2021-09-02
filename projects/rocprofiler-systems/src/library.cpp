
#include "library.hpp"

bool
get_debug()
{
    static bool _v = tim::get_env("HOSTTRACE_DEBUG", false);
    return _v;
}

State&
get_state()
{
    static State _v{ State::PreInit };
    return _v;
}

//--------------------------------------------------------------------------------------//

namespace
{
auto
get_use_perfetto()
{
    // if using timemory, default to perfetto being off
    static auto _default_v = !tim::get_env<bool>("HOSTTRACE_USE_TIMEMORY", false, false);
    // explicit env control for using perfetto
    static auto _v = tim::get_env<bool>("HOSTTRACE_USE_PERFETTO", _default_v);
    return _v;
}

auto
get_use_timemory()
{
    // default to opposite of whether perfetto setting
    // to use both timemory and perfetto, both HOSTTRACE_USE_TIMEMORY and
    // HOSTTRACE_USE_PERFETTO must be true
    static auto _v = tim::get_env<bool>("HOSTTRACE_USE_TIMEMORY", !get_use_perfetto());
    return _v;
}

bool&
get_use_mpi()
{
    // this does not enable anything particularly useful when not using timemory
    static bool _v = tim::get_env("HOSTTRACE_USE_MPI", false, get_use_timemory());
    return _v;
}

void
setup_gotchas()
{
    static bool _initialized = false;
    if(_initialized)
        return;
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
    };
}

auto
ensure_finalization()
{
    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);
    return tim::scope::destructor{ []() { hosttrace_trace_finalize(); } };
}

auto&
get_trace_session()
{
    static std::unique_ptr<perfetto::TracingSession> _session{};
    return _session;
}

auto
get_perfetto_output_filename()
{
    static auto _v = []() {
        // default name: perfetto-trace.<pid>.proto or perfetto-trace.<rank>.proto
        auto _default_fname = tim::settings::compose_output_filename(
            JOIN('.', "perfetto-trace", (get_use_mpi()) ? "%rank%" : "%pid%"), "proto");
        // have the default display the full path to the output file
        return tim::get_env<std::string>(
            "HOSTTRACE_OUTPUT_FILE",
            JOIN('/', tim::get_env<std::string>("PWD", ".", false), _default_fname));
    }();

    auto _tmp     = _v;
    auto _replace = [&_tmp](const std::string& _key, auto&& _val) {
        auto _pos = _tmp.find(_key);
        if(_pos != std::string::npos)
            _tmp.replace(_pos, _key.length(), std::to_string(_val()));
    };
    _replace("%pid%", []() { return tim::process::get_id(); });
    _replace("%rank%", []() { return tim::mpi::rank(); });
    // backwards compatibility
    _replace("%p", []() { return tim::process::get_id(); });
    return _tmp;
}

auto&
get_backend()
{
    // select inprocess, system, or both (i.e. all)
    static auto _v = tim::get_env_choice<std::string>(
        "HOSTTRACE_BACKEND",
        tim::get_env("HOSTTRACE_BACKEND_SYSTEM", false, false)
            ? "system"      // if HOSTTRACE_BACKEND_SYSTEM is true, default to system.
            : "inprocess",  // Otherwise, default to inprocess
        { "inprocess", "system", "all" });
    return _v;
}

auto
is_system_backend()
{
    // if get_backend() returns 'system' or 'all', this is true
    return (get_backend() != "inprocess");
}

auto&
get_timemory_data()
{
    static thread_local auto& _v =
        hosttrace_timemory_data::instances().at(threading::get_id());
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

bool
hosttrace_init_perfetto()
{
    if(get_state() != State::PreInit)
        return false;

    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);

    // always initialize timemory because gotcha wrappers are always used
    tim::settings::flamegraph_output()     = false;
    tim::settings::cout_output()           = false;
    tim::settings::file_output()           = true;
    tim::settings::enable_signal_handler() = true;
    tim::settings::collapse_processes()    = false;
    tim::settings::collapse_threads()      = false;
    tim::settings::max_thread_bookmarks()  = 1;
    tim::settings::global_components()     = tim::get_env<std::string>(
        "HOSTTRACE_COMPONENTS", "wall_clock", get_use_timemory());

    // enable timestamp directories when perfetto + mpi is activated
    if(get_use_perfetto() && get_use_mpi())
        tim::settings::time_output() = true;

    auto _cmd = tim::read_command_line(tim::process::get_id());
    auto _exe = (_cmd.empty()) ? "hosttrace" : _cmd.front();
    auto _pos = _exe.find_last_of('/');
    if(_pos < _exe.length() - 1)
        _exe = _exe.substr(_pos + 1);

    tim::timemory_init({ _exe }, "hosttrace-");

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
            bundle_t::get_initializer() = [](bundle_t& _bundle) {
                _bundle.initialize<comp::wall_clock>();
            };
        }
        else if(!_comps.empty())
        {
            // use user_bundle for other than wall-clock
            bundle_t::get_initializer() = [](bundle_t& _bundle) {
                _bundle.initialize<comp::user_global_bundle>();
            };
        }
        else
        {
            tim::trait::runtime_enabled<hosttrace>::set(false);
        }
    }

    // always activate gotcha wrappers
    auto& _fork_gotcha = get_main_bundle();
    _fork_gotcha->start();
    assert(_fork_gotcha->get<mpi_gotcha_t>()->get_is_running());

    perfetto::TracingInitArgs               args{};
    perfetto::TraceConfig                   cfg{};
    perfetto::protos::gen::TrackEventConfig track_event_cfg{};

    // perfetto initialization
    if(get_use_perfetto())
    {
        // environment settings
        auto shmem_size_hint =
            tim::get_env<size_t>("HOSTTRACE_SHMEM_SIZE_HINT_KB", 40960);
        auto buffer_size = tim::get_env<size_t>("HOSTTRACE_BUFFER_SIZE_KB", 1024000);

        auto* buffer_config = cfg.add_buffers();
        buffer_config->set_size_kb(buffer_size);
        buffer_config->set_fill_policy(
            perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD);

        auto* ds_cfg = cfg.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");
        ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

        args.shmem_size_hint_kb = shmem_size_hint;

        if(get_backend() != "inprocess")
            args.backends |= perfetto::kSystemBackend;
        if(get_backend() != "system")
            args.backends |= perfetto::kInProcessBackend;

        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();

        (void) get_perfetto_output_filename();
    }

    // functors for starting and stopping timemory
    static auto _push_timemory = [](const char* name) {
        auto& _data = get_timemory_data();
        // this generates a hash for the raw string array
        auto  _hash   = tim::add_hash_id(tim::string_view_t{ name });
        auto* _bundle = _data.allocator.allocate(1);
        _data.bundles.emplace_back(_bundle);
        _data.allocator.construct(_bundle, _hash);
        _bundle->start();
    };

    static auto _pop_timemory = [](const char* name) {
        auto& _data = get_timemory_data();
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

    if(get_use_perfetto() && get_use_timemory())
    {
        // if both are used, then use perfetto overload for calling lambda to launch
        // timemory
        get_functors().first = [](const char* name) {
            TRACE_EVENT_BEGIN("hosttrace", perfetto::StaticString(name),
                              [&](perfetto::EventContext) { _push_timemory(name); });
        };
        get_functors().second = [](const char* name) {
            TRACE_EVENT_END("hosttrace",
                            [&](perfetto::EventContext) { _pop_timemory(name); });
        };
    }
    else if(get_use_perfetto())
    {
        get_functors().first = [](const char* name) {
            TRACE_EVENT_BEGIN("hosttrace", perfetto::StaticString(name));
        };
        get_functors().second = [](const char*) { TRACE_EVENT_END("hosttrace"); };
    }
    else if(get_use_timemory())
    {
        get_functors().first  = _push_timemory;
        get_functors().second = _pop_timemory;
    }

    if(tim::dmp::rank() == 0)
    {
        tim::print_env(std::cerr,
                       [](const std::string& _v) { return _v.find("HOSTTRACE_") == 0; });
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

    if(tim::dmp::rank() == 0)
        puts("");
    return true;
}
}  // namespace

//--------------------------------------------------------------------------------------//

extern "C"
{
    void hosttrace_push_trace(const char* name)
    {
        // return if not active
        if(get_state() == State::Finalized)
            return;

        if(get_state() != State::Active && !hosttrace_init_perfetto())
        {
            HOSTTRACE_DEBUG("[%s] %s :: not active and perfetto not initialized\n",
                            __FUNCTION__, name);
            return;
        }
        else
        {
            HOSTTRACE_DEBUG("[%s] %s\n", __FUNCTION__, name);
        }

        get_functors().first(name);
    }

    void hosttrace_pop_trace(const char* name)
    {
        if(get_state() == State::Active)
        {
            HOSTTRACE_DEBUG("[%s] %s\n", __FUNCTION__, name);
            get_functors().second(name);
        }
        else
        {
            HOSTTRACE_DEBUG("[%s] %s :: not active\n", __FUNCTION__, name);
        }
    }

    void hosttrace_trace_init(const char*, bool, const char*)
    {
        HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);
        hosttrace_init_perfetto();
    }

    void hosttrace_trace_finalize(void)
    {
        // return if not active
        if(get_state() != State::Active)
            return;

        HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);

        if(tim::dmp::rank() == 0)
            puts("");

        get_state() = State::Finalized;

        if(get_main_bundle())
        {
            get_main_bundle()->stop();
            int64_t _id = (get_use_mpi()) ? tim::dmp::rank() : tim::process::get_id();
            std::stringstream _ss{};
            _ss << "[" << __FUNCTION__ << "][" << _id << "] " << *get_main_bundle()
                << "\n";
            std::cout << _ss.str();
            get_main_bundle().reset();
        }

        // ensure that all the MT instances are flushed
        for(auto& itr : hosttrace_timemory_data::instances())
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
            fprintf(stderr, "[%s]> Outputting '%s'. Trace data: %lu bytes...\n",
                    __FUNCTION__, get_perfetto_output_filename().c_str(),
                    (unsigned long) trace_data.size());
            std::ofstream output{};
            output.open(get_perfetto_output_filename(), std::ios::out | std::ios::binary);
            if(!output)
                fprintf(stderr, "[%s]> Error opening '%s'...\n", __FUNCTION__,
                        get_perfetto_output_filename().c_str());
            else
                output.write(&trace_data[0], trace_data.size());
            output.close();
        }

        tim::timemory_finalize();
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
            auto& _fork_gotcha = get_main_bundle();
            _fork_gotcha->start();
            tim::set_env("HOSTTRACE_USE_MPI", "ON", 1);
            get_use_mpi() = true;
            get_state()   = State::DelayedInit;
        }
    }
}

std::unique_ptr<hosttrace_bundle_t>&
get_main_bundle()
{
    static auto _v =
        (setup_gotchas(), std::make_unique<hosttrace_bundle_t>(
                              "hosttrace", quirk::config<quirk::auto_start>{}));
    return _v;
}

namespace
{
// if static objects are destroyed randomly (relatively uncommon behavior)
// this might call finalization before perfetto ends the tracing session
// but static variable in hosttrace_init_perfetto is more likely
auto _ensure_finalization = ensure_finalization();
}  // namespace
