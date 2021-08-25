
#include <perfetto.h>

#if defined(NDEBUG)
#    undef NDEBUG
#endif

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "timemory/api.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/components.hpp"
#include "timemory/config.hpp"
#include "timemory/environment.hpp"
#include "timemory/manager.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/operations.hpp"
#include "timemory/settings.hpp"
#include "timemory/storage.hpp"
#include "timemory/variadic.hpp"

#if !defined(JOIN)
#    define JOIN(...) tim::mpl::apply<std::string>::join(__VA_ARGS__)
#endif

namespace audit = tim::audit;
namespace comp  = tim::component;
namespace quirk = tim::quirk;

struct fork_gotcha : tim::component::base<fork_gotcha, void>
{
    using gotcha_data_t = tim::component::gotcha_data;

    TIMEMORY_DEFAULT_OBJECT(fork_gotcha)

    void audit(const gotcha_data_t& _data, audit::incoming);
    void audit(const gotcha_data_t& _data, audit::outgoing, pid_t _pid);
};

struct fork_gotcha_api : tim::concepts::api
{};

using fork_gotcha_t =
    tim::component::gotcha<4, tim::component_tuple<fork_gotcha>, fork_gotcha_api>;
using fork_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::cpu_clock,
                           comp::cpu_util, fork_gotcha_t>;

//--------------------------------------------------------------------------------------//

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("hosttrace").SetDescription("Function trace"));

#if defined(CUSTOM_DATA_SOURCE)
class CustomDataSource : public perfetto::DataSource<CustomDataSource>
{
public:
    void OnSetup(const SetupArgs&) override
    {
        // Use this callback to apply any custom configuration to your data source
        // based on the TraceConfig in SetupArgs.
        PRINT_HERE("%s", "setup");
    }

    void OnStart(const StartArgs&) override
    {
        // This notification can be used to initialize the GPU driver, enable
        // counters, etc. StartArgs will contains the DataSourceDescriptor,
        // which can be extended.
        PRINT_HERE("%s", "start");
    }

    void OnStop(const StopArgs&) override
    {
        // Undo any initialization done in OnStart.
        PRINT_HERE("%s", "stop");
    }

    // Data sources can also have per-instance state.
    int my_custom_state = 0;
};

PERFETTO_DECLARE_DATA_SOURCE_STATIC_MEMBERS(CustomDataSource);
#endif

extern "C" void
hosttrace_trace_finalize();

namespace
{
bool
get_debug()
{
    static bool _v = tim::get_env("HOSTTRACE_DEBUG", false);
    return _v;
}

void
setup_fork_gotcha()
{
    CONDITIONAL_PRINT_HERE(get_debug(), "%s", "configuring gotcha wrapper around fork");

    fork_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(fork_gotcha_t, 0, fork);
    };
}

auto&
get_fork_gotcha()
{
    static auto _v =
        (setup_fork_gotcha(), std::make_unique<fork_bundle_t>(
                                  "hosttrace", quirk::config<quirk::auto_start>{}));
    return _v;
}

auto
ensure_finalization()
{
    if(get_debug())
        fprintf(stderr, "[%s]\n", __FUNCTION__);
    return tim::scope::destructor{ []() { hosttrace_trace_finalize(); } };
}

auto&
get_trace_session()
{
    static std::unique_ptr<perfetto::TracingSession> _session{};
    return _session;
}

enum class State : unsigned short
{
    PreInit = 0,
    Active,
    Finalized
};

auto&
get_state()
{
    static State _v{ State::PreInit };
    return _v;
}

auto&
get_output_filename()
{
    static auto _v = []() {
        auto _tmp = tim::get_env<std::string>(
            "HOSTTRACE_OUTPUT_FILE",
            JOIN('/', tim::get_env<std::string>("PWD", ".", false),
                 "hosttrace.perfetto-trace-%pid%"));
        auto _replace = [&_tmp](const std::string& _key, auto _val) {
            auto _pos = _tmp.find(_key);
            if(_pos != std::string::npos)
                _tmp.replace(_pos, _key.length(), std::to_string(_val));
        };
        _replace("%pid%", tim::process::get_id());
        _replace("%rank%", tim::mpi::rank());
        // backwards compatibility
        _replace("%p", tim::process::get_id());
        return _tmp;
    }();
    return _v;
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

bool
hosttrace_init_perfetto()
{
    if(get_debug())
        fprintf(stderr, "[%s]\n", __FUNCTION__);

    if(get_state() != State::PreInit)
        return false;

    tim::settings::flamegraph_output()     = false;
    tim::settings::file_output()           = false;
    tim::settings::enable_signal_handler() = true;
    tim::timemory_init({ "hosttrace" });

    auto& _fork_gotcha = get_fork_gotcha();
    _fork_gotcha->start();
    assert(_fork_gotcha->get<fork_gotcha_t>()->get_is_running());

    // environment settings
    auto shmem_size_hint = tim::get_env<size_t>("HOSTTRACE_SHMEM_SIZE_HINT_KB", 40960);
    auto buffer_size     = tim::get_env<size_t>("HOSTTRACE_BUFFER_SIZE_KB", 1024000);

    perfetto::TracingInitArgs               args{};
    perfetto::TraceConfig                   cfg{};
    perfetto::protos::gen::TrackEventConfig track_event_cfg{};

    auto *buffer_config = cfg.add_buffers();
    buffer_config->set_size_kb(buffer_size);
    buffer_config->set_fill_policy(perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD);

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

    (void) get_output_filename();

    tim::print_env(std::cerr,
                   [](const std::string& _v) { return _v.find("HOSTTRACE_") == 0; });

    if(!is_system_backend())
    {
#if defined(CUSTOM_DATA_SOURCE)
        // Add the following:
        perfetto::DataSourceDescriptor dsd{};
        dsd.set_name("com.example.custom_data_source");
        CustomDataSource::Register(dsd);
        ds_cfg = cfg.add_data_sources()->mutable_config();
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

    puts("");
    return true;
}
}  // namespace

extern "C"
{
    void hosttrace_push_trace(const char* name)
    {
        if(get_debug())
            fprintf(stderr, "[%s] %s\n", __FUNCTION__, name);
        // return if not active
        if(get_state() != State::Active && !hosttrace_init_perfetto())
            return;
        // TRACE_EVENT_BEGIN(
        //    "hosttrace", perfetto::StaticString(name),
        //    [&](perfetto::EventContext ctx) { PRINT_HERE("executing %s", name); });
        TRACE_EVENT_BEGIN("hosttrace", perfetto::StaticString(name));
    }

    void hosttrace_pop_trace(const char* name)
    {
        if(get_debug())
            fprintf(stderr, "[%s] %s\n", __FUNCTION__, name);
        // return if not active
        if(get_state() != State::Active)
            return;
        // TRACE_EVENT_END("hosttrace",
        //   [&](perfetto::EventContext ctx) { PRINT_HERE("executing %s", name); });
        TRACE_EVENT_END("hosttrace");
    }

    void hosttrace_trace_init(const char*, bool, const char*)
    {
        if(get_debug())
            fprintf(stderr, "[%s]\n", __FUNCTION__);
        hosttrace_init_perfetto();
    }

    void hosttrace_trace_finalize(void)
    {
        if(get_debug())
            fprintf(stderr, "[%s]\n", __FUNCTION__);
        if(get_state() != State::Active)
            return;

        puts("");
        get_state() = State::Finalized;

        if(get_fork_gotcha())
        {
            get_fork_gotcha()->stop();
            std::cout << *get_fork_gotcha() << std::endl;
            get_fork_gotcha().reset();
        }

        if(!is_system_backend())
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
                        __FUNCTION__, get_output_filename().c_str());
                return;
            }
            // Write the trace into a file.
            fprintf(stderr, "[%s]> Outputting '%s'. Trace data: %lu bytes...\n",
                    __FUNCTION__, get_output_filename().c_str(),
                    (unsigned long) trace_data.size());
            std::ofstream output{};
            output.open(get_output_filename(), std::ios::out | std::ios::binary);
            if(!output)
                fprintf(stderr, "[%s]> Error opening '%s'...\n", __FUNCTION__,
                        get_output_filename().c_str());
            else
                output.write(&trace_data[0], trace_data.size());
            output.close();
        }

        tim::timemory_finalize();
    }

    void hosttrace_trace_set_env(const char* env_name, const char* env_val)
    {
        if(get_debug())
            fprintf(stderr, "[%s] Setting env: %s=%s\n", __FUNCTION__, env_name, env_val);

        tim::set_env(env_name, env_val, 0);
    }
}

void
fork_gotcha::audit(const gotcha_data_t& _data, audit::incoming)
{
    PRINT_HERE("%s",
               "Warning! Calling fork() within an OpenMPI application using libfabric "
               "may result is segmentation fault");
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(get_debug(), 16);
}

void
fork_gotcha::audit(const gotcha_data_t& _data, audit::outgoing, pid_t _pid)
{
    PRINT_HERE("%s() return PID %i", _data.tool_id.c_str(), (int) _pid);
}

namespace
{
// if static objects are destroyed randomly (relatively uncommon behavior)
// this might call finalization before perfetto ends the tracing session
// but static variable in hosttrace_init_perfetto is more likely
auto _ensure_finalization = ensure_finalization();
}  // namespace

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
TIMEMORY_INITIALIZE_STORAGE(fork_gotcha)

#if defined(CUSTOM_DATA_SOURCE)
PERFETTO_DEFINE_DATA_SOURCE_STATIC_MEMBERS(CustomDataSource);
#endif
