
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <perfetto.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "timemory/backends/process.hpp"
#include "timemory/environment.hpp"
#include "timemory/mpl/apply.hpp"

#if !defined(JOIN)
#    define JOIN(...) tim::mpl::apply<std::string>::join(__VA_ARGS__)
#endif

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("hosttrace").SetDescription("Function trace"));

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
                 "hosttrace.perfetto-trace-%p"));
        auto _pos = _tmp.find("%p");
        if(_pos != std::string::npos)
            _tmp.replace(_pos, 2, std::to_string(tim::process::get_id()));
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

    // environment settings
    auto shmem_size_hint = tim::get_env<size_t>("HOSTTRACE_SHMEM_SIZE_HINT_KB", 40960);
    auto buffer_size     = tim::get_env<size_t>("HOSTTRACE_BUFFER_SIZE_KB", 1024000);

    perfetto::TracingInitArgs               args{};
    perfetto::TraceConfig                   cfg{};
    perfetto::protos::gen::TrackEventConfig track_event_cfg{};

    cfg.add_buffers()->set_size_kb(buffer_size);
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

    tim::print_env(std::cerr);

    if(!is_system_backend())
    {
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
        TRACE_EVENT_BEGIN("hosttrace", perfetto::StaticString(name));
    }

    void hosttrace_pop_trace(const char* name)
    {
        if(get_debug())
            fprintf(stderr, "[%s] %s\n", __FUNCTION__, name);
        // return if not active
        if(get_state() != State::Active)
            return;
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

        get_state() = State::Finalized;

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
    }

    void hosttrace_trace_set_env(const char* env_name, const char* env_val)
    {
        if(get_debug())
            fprintf(stderr, "[%s] Setting env: %s=%s\n", __FUNCTION__, env_name, env_val);

        tim::set_env(env_name, env_val, 0);
    }
}

namespace
{
// if static objects are destroyed randomly (relatively uncommon behavior)
// this might call finalization before perfetto ends the tracing session
// but static variable in hosttrace_init_perfetto is more likely
auto _ensure_finalization = ensure_finalization();
}  // namespace

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
