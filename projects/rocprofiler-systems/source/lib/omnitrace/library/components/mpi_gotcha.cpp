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

#include "library/components/mpi_gotcha.hpp"
#include "library/api.hpp"
#include "library/components/omnitrace.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"

#include <thread>
#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>

namespace omnitrace
{
namespace
{
uint64_t    mpip_index      = std::numeric_limits<uint64_t>::max();
std::string mpi_init_string = {};

// this ensures omnitrace_finalize is called before MPI_Finalize
void
omnitrace_mpi_set_attr()
{
#if defined(TIMEMORY_USE_MPI)
    static auto _mpi_copy = [](MPI_Comm, int, void*, void*, void*, int*) {
        return MPI_SUCCESS;
    };
    static auto _mpi_fini = [](MPI_Comm, int, void*, void*) {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug_env(),
                                          "MPI Comm attribute finalize\n");
        if(mpip_index != std::numeric_limits<uint64_t>::max())
            comp::deactivate_mpip<tim::component_tuple<omnitrace::component::omnitrace>,
                                  api::omnitrace>(mpip_index);
        omnitrace_finalize_hidden();
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
mpi_gotcha::configure()
{
    mpi_gotcha_t::get_initializer() = []() {
        mpi_gotcha_t::template configure<0, int, int*, char***>("MPI_Init");
        mpi_gotcha_t::template configure<1, int, int*, char***, int, int*>(
            "MPI_Init_thread");
        mpi_gotcha_t::template configure<2, int>("MPI_Finalize");
#if defined(OMNITRACE_USE_MPI_HEADERS) && OMNITRACE_USE_MPI_HEADERS > 0
        mpi_gotcha_t::template configure<3, int, comm_t, int*>("MPI_Comm_rank");
        mpi_gotcha_t::template configure<4, int, comm_t, int*>("MPI_Comm_size");
#endif
    };
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_debug_env(), "%s(int*, char***)\n",
                                        _data.tool_id.c_str());

    if(get_state() < ::omnitrace::State::Init) set_state(::omnitrace::State::PreInit);

    omnitrace_push_trace_hidden(_data.tool_id.c_str());
#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    tim::mpi::is_initialized_callback() = []() { return true; };
    tim::mpi::is_finalized()            = false;
#endif
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***, int, int*)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_debug_env(), "%s(int*, char***, int, int*)\n",
                                        _data.tool_id.c_str());

    if(get_state() < ::omnitrace::State::Init) set_state(::omnitrace::State::PreInit);

    omnitrace_push_trace_hidden(_data.tool_id.c_str());
#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    tim::mpi::is_initialized_callback() = []() { return true; };
    tim::mpi::is_finalized()            = false;
#endif
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_debug_env(), "%s()\n", _data.tool_id.c_str());

    if(mpip_index != std::numeric_limits<uint64_t>::max())
        comp::deactivate_mpip<tim::component_tuple<omnitrace::component::omnitrace>,
                              api::omnitrace>(mpip_index);

#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    tim::mpi::is_initialized_callback() = []() { return false; };
    tim::mpi::is_finalized()            = true;
#else
    omnitrace_finalize_hidden();
#endif
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, comm_t, int* _val)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_debug_env(), "%s()\n", _data.tool_id.c_str());

    omnitrace_push_trace_hidden(_data.tool_id.c_str());
    if(_data.tool_id == "MPI_Comm_rank")
    {
        m_rank_ptr = _val;
    }
    else if(_data.tool_id == "MPI_Comm_size")
    {
        m_size_ptr = _val;
    }
    else
    {
        OMNITRACE_BASIC_PRINT_F("%s(<comm>, %p) :: unexpected function wrapper\n",
                                _data.tool_id.c_str(), _val);
    }
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::outgoing, int _retval)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_debug_env(), "%s() returned %i\n",
                                        _data.tool_id.c_str(), (int) _retval);

    if(_retval == tim::mpi::success_v && _data.tool_id.find("MPI_Init") == 0)
    {
        omnitrace_mpi_set_attr();
        // omnitrace will set this environement variable to true in binary rewrite mode
        // when it detects MPI. Hides this env variable from the user to avoid this
        // being activated unwaringly during runtime instrumentation because that
        // will result in double instrumenting the MPI functions (unless the MPI functions
        // were excluded via a regex expression)
        if(get_use_mpip())
        {
            OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_debug_env() || get_verbose_env() > 0,
                                                "Activating MPI wrappers...\n");

            // use env vars OMNITRACE_MPIP_PERMIT_LIST and OMNITRACE_MPIP_REJECT_LIST
            // to control the gotcha bindings at runtime
            comp::configure_mpip<tim::component_tuple<omnitrace::component::omnitrace>,
                                 api::omnitrace>();
            mpip_index =
                comp::activate_mpip<tim::component_tuple<omnitrace::component::omnitrace>,
                                    api::omnitrace>();
        }
    }
    else if(_retval == tim::mpi::success_v && _data.tool_id.find("MPI_Comm_") == 0)
    {
        if(_data.tool_id == "MPI_Comm_rank")
        {
            if(m_rank_ptr)
            {
                m_rank = std::max<int>(*m_rank_ptr, m_rank);
                tim::mpi::set_rank(m_rank);
                tim::settings::default_process_suffix() = m_rank;
                get_perfetto_output_filename().clear();
                OMNITRACE_CONDITIONAL_BASIC_PRINT(
                    get_debug() || get_verbose() > 0, "[pid=%i] MPI rank: %i (%i)\n",
                    process::get_id(), tim::mpi::rank(), m_rank);
            }
            else
            {
                OMNITRACE_BASIC_PRINT_F("%s() returned %i :: nullptr to rank\n",
                                        _data.tool_id.c_str(), (int) _retval);
            }
        }
        else if(_data.tool_id == "MPI_Comm_size")
        {
            if(m_size_ptr)
            {
                m_size = std::max<int>(*m_size_ptr, m_size);
                tim::mpi::set_size(m_size);
                OMNITRACE_CONDITIONAL_BASIC_PRINT(
                    get_debug() || get_verbose() > 0, "[pid=%i] MPI size: %i (%i)\n",
                    process::get_id(), tim::mpi::size(), m_size);
            }
            else
            {
                OMNITRACE_BASIC_PRINT_F("%s() returned %i :: nullptr to size\n",
                                        _data.tool_id.c_str(), (int) _retval);
            }
        }
        else
        {
            OMNITRACE_BASIC_PRINT_F("%s() returned %i :: unexpected function wrapper\n",
                                    _data.tool_id.c_str(), (int) _retval);
        }
    }
    omnitrace_pop_trace_hidden(_data.tool_id.c_str());
}
}  // namespace omnitrace

TIMEMORY_INITIALIZE_STORAGE(omnitrace::mpi_gotcha)
