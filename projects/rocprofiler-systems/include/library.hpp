
#pragma once

#if !defined(TIMEMORY_USE_PERFETTO)
#    include <perfetto.h>
#    define PERFETTO_CATEGORIES                                                          \
        perfetto::Category("hosttrace").SetDescription("Function trace")
#else
#    define PERFETTO_CATEGORIES                                                          \
        perfetto::Category("hosttrace").SetDescription("Function trace"),                \
            perfetto::Category("timemory")                                               \
                .SetDescription("Events from the timemory API")
#    define TIMEMORY_PERFETTO_CATEGORIES PERFETTO_CATEGORIES
#endif

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "timemory/api.hpp"
#include "timemory/backends/mpi.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/components.hpp"
#include "timemory/components/gotcha/mpip.hpp"
#include "timemory/config.hpp"
#include "timemory/environment.hpp"
#include "timemory/manager.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/operations.hpp"
#include "timemory/runtime.hpp"
#include "timemory/settings.hpp"
#include "timemory/storage.hpp"
#include "timemory/variadic.hpp"

// forward decl of the API
extern "C"
{
    void hosttrace_push_trace(const char* name) TIMEMORY_VISIBILITY("default");
    void hosttrace_pop_trace(const char* name) TIMEMORY_VISIBILITY("default");
    void hosttrace_trace_init(const char*, bool, const char*)
        TIMEMORY_VISIBILITY("default");
    void hosttrace_trace_finalize(void) TIMEMORY_VISIBILITY("default");
    void hosttrace_trace_set_env(const char* env_name, const char* env_val)
        TIMEMORY_VISIBILITY("default");
    void hosttrace_trace_set_mpi(bool use, bool attached) TIMEMORY_VISIBILITY("default");
}

//--------------------------------------------------------------------------------------//

// same sort of functionality as python's " ".join([...])
#if !defined(JOIN)
#    define JOIN(...) tim::mpl::apply<std::string>::join(__VA_ARGS__)
#endif

#define HOSTTRACE_DEBUG(...)                                                             \
    if(get_debug())                                                                      \
    {                                                                                    \
        fprintf(stderr, __VA_ARGS__);                                                    \
    }

//--------------------------------------------------------------------------------------//

namespace audit     = tim::audit;
namespace comp      = tim::component;
namespace quirk     = tim::quirk;
namespace threading = tim::threading;

// this is used to wrap fork()
struct fork_gotcha : comp::base<fork_gotcha, void>
{
    using gotcha_data_t = comp::gotcha_data;

    TIMEMORY_DEFAULT_OBJECT(fork_gotcha)

    // this will get called right before fork
    void audit(const gotcha_data_t& _data, audit::incoming);

    // this will get called right after fork with the return value
    void audit(const gotcha_data_t& _data, audit::outgoing, pid_t _pid);
};

// this is used to wrap MPI_Init and MPI_Init_thread
struct mpi_gotcha : comp::base<mpi_gotcha, void>
{
    using gotcha_data_t = comp::gotcha_data;

    TIMEMORY_DEFAULT_OBJECT(mpi_gotcha)

    // this will get called right before MPI_Init with that functions arguments
    void audit(const gotcha_data_t& _data, audit::incoming, int*, char***);

    // this will get called right before MPI_Init_thread with that functions arguments
    void audit(const gotcha_data_t& _data, audit::incoming, int*, char***, int, int*);

    // this will get called right after MPI_Init and MPI_Init_thread with the return value
    void audit(const gotcha_data_t& _data, audit::outgoing, int _retval);
};

// timemory api struct
struct hosttrace : tim::concepts::api
{};

// timemory component which calls hosttrace functions
// (used in gotcha wrappers)
struct hosttrace_component : tim::component::base<hosttrace_component, void>
{
    void start();
    void stop();
    void set_prefix(const char*);

private:
    const char* m_prefix = nullptr;
};

using fork_gotcha_t = comp::gotcha<4, tim::component_tuple<fork_gotcha>, hosttrace>;
using mpi_gotcha_t  = comp::gotcha<4, tim::component_tuple<mpi_gotcha>, hosttrace>;
using hosttrace_bundle_t =
    tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::cpu_clock,
                           comp::cpu_util, comp::user_global_bundle, fork_gotcha_t,
                           mpi_gotcha_t>;
using bundle_t =
    tim::component_bundle<hosttrace, comp::wall_clock*, comp::user_global_bundle*>;
using bundle_allocator_t = tim::data::ring_buffer_allocator<bundle_t>;

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_USE_PERFETTO)
PERFETTO_DEFINE_CATEGORIES(PERFETTO_CATEGORIES);
#endif

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

//--------------------------------------------------------------------------------------//

// used for specifying the state of hosttrace
enum class State : unsigned short
{
    DelayedInit = 0,
    PreInit,
    Active,
    Finalized
};

bool
get_debug();

State&
get_state();

std::unique_ptr<hosttrace_bundle_t>&
get_main_bundle();

//--------------------------------------------------------------------------------------//

// there are currently some strange things that happen with vector<bundle_t> so using
// vector<bundle_t*> and timemory's ring_buffer_allocator to create contiguous memory-page
// aligned instances of the bundle
struct hosttrace_timemory_data
{
    static constexpr size_t max_supported_threads = 1024;
    using instance_array_t = std::array<hosttrace_timemory_data, max_supported_threads>;

    bundle_allocator_t     allocator{};
    std::vector<bundle_t*> bundles{};

    static instance_array_t& instances();
};

//--------------------------------------------------------------------------------------//
