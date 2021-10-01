#include "library.hpp"

//
//  This file contains miscellaneous function definitions related to timemory
//  placed in separate file so that, during development, the long compile-times
//  arising from compiling timemory's gotcha wrappers are reduced
//

namespace
{
uint64_t mpip_index = std::numeric_limits<uint64_t>::max();

// this ensures hosttrace_trace_finalize is called before MPI_Finalize
void
hosttrace_mpi_set_attr()
{
#if defined(TIMEMORY_USE_MPI)
    static auto _mpi_copy = [](MPI_Comm, int, void*, void*, void*, int*) {
        return MPI_SUCCESS;
    };
    static auto _mpi_fini = [](MPI_Comm, int, void*, void*) {
        if(mpip_index != std::numeric_limits<uint64_t>::max())
            comp::deactivate_mpip<tim::component_tuple<hosttrace_component>, hosttrace>(
                mpip_index);
        hosttrace_pop_trace("MPI_Finalize()");
        hosttrace_trace_finalize();
        return MPI_SUCCESS;
    };
    using copy_func_t = int (*)(MPI_Comm, int, void*, void*, void*, int*);
    using fini_func_t = int (*)(MPI_Comm, int, void*, void*);
    int _comm_key     = -1;
    if(PMPI_Comm_create_keyval(static_cast<copy_func_t>(_mpi_copy),
                               static_cast<fini_func_t>(_mpi_fini), &_comm_key,
                               nullptr) == MPI_SUCCESS)
        PMPI_Comm_set_attr(MPI_COMM_SELF, _comm_key, nullptr);
#endif
}
}  // namespace

void
fork_gotcha::audit(const gotcha_data_t&, audit::incoming)
{
    HOSTTRACE_DEBUG(
        "Warning! Calling fork() within an OpenMPI application using libfabric "
        "may result is segmentation fault\n");
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(get_debug(), 16);
}

void
fork_gotcha::audit(const gotcha_data_t& _data, audit::outgoing, pid_t _pid)
{
    HOSTTRACE_DEBUG("%s() return PID %i\n", _data.tool_id.c_str(), (int) _pid);
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***)
{
    HOSTTRACE_DEBUG("[%s] %s(int*, char***)\n", __FUNCTION__, _data.tool_id.c_str());
    if(get_state() == ::State::DelayedInit) get_state() = ::State::PreInit;
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***, int, int*)
{
    HOSTTRACE_DEBUG("[%s] %s(int*, char***, int, int*)\n", __FUNCTION__,
                    _data.tool_id.c_str());
    if(get_state() == ::State::DelayedInit) get_state() = ::State::PreInit;
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::outgoing, int _retval)
{
    HOSTTRACE_DEBUG("[%s] %s() returned %i\n", __FUNCTION__, _data.tool_id.c_str(),
                    (int) _retval);
    if(_retval == tim::mpi::success_v && get_state() == ::State::PreInit)
    {
        hosttrace_mpi_set_attr();
        // hosttrace will set this environement variable to true in binary rewrite mode
        // when it detects MPI. Hides this env variable from the user to avoid this
        // being activated unwaringly during runtime instrumentation because that
        // will result in double instrumenting the MPI functions (unless the MPI functions
        // were excluded via a regex expression)
        if(tim::get_env("HOSTTRACE_USE_MPIP", false, false))
        {
            HOSTTRACE_DEBUG("[%s] Activating MPI wrappers...\n", __FUNCTION__);
            comp::configure_mpip<tim::component_tuple<hosttrace_component>, hosttrace>();
            mpip_index = comp::activate_mpip<tim::component_tuple<hosttrace_component>,
                                             hosttrace>();
        }
        hosttrace_push_trace(_data.tool_id.c_str());
    }
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming)
{
    HOSTTRACE_DEBUG("[%s] %s()\n", __FUNCTION__, _data.tool_id.c_str());
    if(mpip_index != std::numeric_limits<uint64_t>::max())
        comp::deactivate_mpip<tim::component_tuple<hosttrace_component>, hosttrace>(
            mpip_index);
    hosttrace_pop_trace("MPI_Finalize()");
    hosttrace_trace_finalize();
}

void
hosttrace_component::start()
{
    if(m_prefix) hosttrace_push_trace(m_prefix);
}

void
hosttrace_component::stop()
{
    if(m_prefix) hosttrace_pop_trace(m_prefix);
}

void
hosttrace_component::set_prefix(const char* _prefix)
{
    m_prefix = _prefix;
}

hosttrace_timemory_data::instance_array_t&
hosttrace_timemory_data::instances()
{
    static auto _v = instance_array_t{};
    return _v;
}

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
TIMEMORY_INITIALIZE_STORAGE(fork_gotcha, mpi_gotcha, comp::wall_clock,
                            comp::user_global_bundle)

#if defined(CUSTOM_DATA_SOURCE)
PERFETTO_DEFINE_DATA_SOURCE_STATIC_MEMBERS(CustomDataSource);
#endif
