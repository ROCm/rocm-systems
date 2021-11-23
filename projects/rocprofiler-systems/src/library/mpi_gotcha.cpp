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

#include "library/mpi_gotcha.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/hosttrace_component.hpp"

namespace
{
uint64_t    mpip_index      = std::numeric_limits<uint64_t>::max();
std::string mpi_init_string = {};

// this ensures hosttrace_trace_finalize is called before MPI_Finalize
void
hosttrace_mpi_set_attr()
{
#if defined(TIMEMORY_USE_MPI)
    static auto _mpi_copy = [](MPI_Comm, int, void*, void*, void*, int*) {
        return MPI_SUCCESS;
    };
    static auto _mpi_fini = [](MPI_Comm, int, void*, void*) {
        HOSTTRACE_DEBUG("MPI Comm attribute finalize\n");
        if(mpip_index != std::numeric_limits<uint64_t>::max())
            comp::deactivate_mpip<tim::component_tuple<hosttrace_component>, hosttrace>(
                mpip_index);
        if(!mpi_init_string.empty()) hosttrace_pop_trace(mpi_init_string.c_str());
        mpi_init_string = {};
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
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***)
{
    HOSTTRACE_DEBUG("[%s] %s(int*, char***)\n", __FUNCTION__, _data.tool_id.c_str());
    if(get_state() == ::State::DelayedInit)
    {
        get_state()     = ::State::PreInit;
        mpi_init_string = _data.tool_id;
    }
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***, int, int*)
{
    HOSTTRACE_DEBUG("[%s] %s(int*, char***, int, int*)\n", __FUNCTION__,
                    _data.tool_id.c_str());
    if(get_state() == ::State::DelayedInit)
    {
        get_state()     = ::State::PreInit;
        mpi_init_string = _data.tool_id;
    }
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
        if(get_use_mpip())
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
    if(!mpi_init_string.empty()) hosttrace_pop_trace(mpi_init_string.c_str());
    mpi_init_string = {};
    hosttrace_trace_finalize();
}

TIMEMORY_INITIALIZE_STORAGE(mpi_gotcha)
