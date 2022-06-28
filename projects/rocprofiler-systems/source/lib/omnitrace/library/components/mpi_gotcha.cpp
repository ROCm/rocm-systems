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
#include "library/components/category_region.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/mproc.hpp"

#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/utility/locking.hpp>

#include <cstdint>
#include <limits>
#include <thread>
#include <unistd.h>

namespace omnitrace
{
namespace
{
struct comm_rank_data
{
    int       rank = -1;
    int       size = -1;
    uintptr_t comm = mpi_gotcha::null_comm();

    auto updated() const
    {
        return comm != mpi_gotcha::null_comm() && rank >= 0 && size > 0;
    };

    friend bool operator==(const comm_rank_data& _lhs, const comm_rank_data& _rhs)
    {
        auto _lupd = _lhs.updated();
        auto _rupd = _rhs.updated();
        return std::tie(_lupd, _lhs.rank, _lhs.size, _lhs.comm) ==
               std::tie(_rupd, _rhs.rank, _rhs.size, _rhs.comm);
    }

    friend bool operator!=(const comm_rank_data& _lhs, const comm_rank_data& _rhs)
    {
        return !(_lhs == _rhs);
    }

    friend bool operator>(const comm_rank_data& _lhs, const comm_rank_data& _rhs)
    {
        OMNITRACE_CI_THROW(!_lhs.updated() && !_rhs.updated(),
                           "Error! comparing rank data that is not updated");

        if(_lhs.updated() && !_rhs.updated()) return true;
        if(!_lhs.updated() && _rhs.updated()) return false;

        if(_lhs.size != _rhs.size) return _lhs.size > _rhs.size;
        if(_lhs.rank != _rhs.rank) return _lhs.rank > _rhs.rank;

        // lesser comm is greater
        return _lhs.comm < _rhs.comm;
    }

    friend bool operator<(const comm_rank_data& _lhs, const comm_rank_data& _rhs)
    {
        return (_lhs != _rhs && !(_lhs > _rhs));
    }
};

uint64_t mpip_index        = std::numeric_limits<uint64_t>::max();
auto     last_comm_record  = comm_rank_data{};
auto     mproc_comm_record = comm_rank_data{};
auto     mpi_comm_records  = std::map<uintptr_t, comm_rank_data>{};

using tim::auto_lock_t;
using tim::type_mutex;

// this ensures omnitrace_finalize is called before MPI_Finalize
void
omnitrace_mpi_set_attr()
{
#if defined(TIMEMORY_USE_MPI)
    static auto _mpi_copy = [](MPI_Comm, int, void*, void*, void*, int*) {
        return MPI_SUCCESS;
    };
    static auto _mpi_fini = [](MPI_Comm, int, void*, void*) {
        OMNITRACE_DEBUG("MPI Comm attribute finalize\n");
        if(mpip_index != std::numeric_limits<uint64_t>::max())
            comp::deactivate_mpip<
                tim::component_tuple<
                    omnitrace::component::category_region<category::mpi>>,
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
mpi_gotcha::stop()
{
    OMNITRACE_BASIC_VERBOSE(0, "[pid=%i] Stopping MPI gotcha...\n", process::get_id());
    if(mpip_index == std::numeric_limits<uint64_t>::max()) return;
    update();
}

void
mpi_gotcha::update()
{
    auto_lock_t _lk{ type_mutex<mpi_gotcha>(), std::defer_lock };
    if(!_lk.owns_lock()) _lk.lock();

    comm_rank_data _rank_data = mproc_comm_record;
    for(const auto& itr : mpi_comm_records)
    {
        // skip null comms
        if(itr.first == null_comm()) continue;
        // if currently have null comm, replace
        else if(_rank_data.comm == null_comm())
            _rank_data = itr.second;
        // if
        else if(itr.second > _rank_data)
            _rank_data = itr.second;
    }

    if(_rank_data.updated() && _rank_data != last_comm_record)
    {
        auto _rank = _rank_data.rank;
        auto _size = _rank_data.size;

        tim::mpi::set_rank(_rank);
        tim::mpi::set_size(_size);
        tim::settings::default_process_suffix() = _rank;
        get_perfetto_output_filename().clear();

        OMNITRACE_BASIC_VERBOSE(0, "[pid=%i] MPI rank: %i (%i)\n", process::get_id(),
                                tim::mpi::rank(), _rank);
        OMNITRACE_BASIC_VERBOSE(0, "[pid=%i] MPI size: %i (%i)\n", process::get_id(),
                                tim::mpi::size(), _size);
        last_comm_record      = _rank_data;
        config::get_use_pid() = true;
    }
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***)
{
    OMNITRACE_BASIC_DEBUG_F("%s(int*, char***)\n", _data.tool_id.c_str());

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
    OMNITRACE_BASIC_DEBUG_F("%s(int*, char***, int, int*)\n", _data.tool_id.c_str());

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
    OMNITRACE_BASIC_DEBUG_F("%s()\n", _data.tool_id.c_str());

    if(mpip_index != std::numeric_limits<uint64_t>::max())
        comp::deactivate_mpip<
            tim::component_tuple<omnitrace::component::category_region<category::mpi>>,
            api::omnitrace>(mpip_index);

#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    tim::mpi::is_initialized_callback() = []() { return false; };
    tim::mpi::is_finalized()            = true;
#else
    omnitrace_finalize_hidden();
#endif
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, comm_t _comm, int* _val)
{
    OMNITRACE_BASIC_DEBUG_F("%s()\n", _data.tool_id.c_str());

    omnitrace_push_trace_hidden(_data.tool_id.c_str());
    if(_data.tool_id == "MPI_Comm_rank")
    {
        m_comm_val = (uintptr_t) _comm;  // NOLINT
        m_rank_ptr = _val;
    }
    else if(_data.tool_id == "MPI_Comm_size")
    {
        m_comm_val = (uintptr_t) _comm;  // NOLINT
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
    OMNITRACE_BASIC_DEBUG_F("%s() returned %i\n", _data.tool_id.c_str(), (int) _retval);

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
            OMNITRACE_BASIC_VERBOSE_F(2, "Activating MPI wrappers...\n");

            // use env vars OMNITRACE_MPIP_PERMIT_LIST and OMNITRACE_MPIP_REJECT_LIST
            // to control the gotcha bindings at runtime
            comp::configure_mpip<
                tim::component_tuple<
                    omnitrace::component::category_region<category::mpi>>,
                api::omnitrace>();
            mpip_index = comp::activate_mpip<
                tim::component_tuple<
                    omnitrace::component::category_region<category::mpi>>,
                api::omnitrace>();
        }

        auto_lock_t _lk{ type_mutex<mpi_gotcha>() };
        if(!mproc_comm_record.updated())
        {
            auto _pid  = getpid();
            auto _ppid = getppid();
            auto _size = mproc::get_concurrent_processes(_ppid).size();
            if(_size > 0)
            {
                mproc_comm_record.comm = _ppid;
                mproc_comm_record.size = m_size = _size;
                auto _rank                      = mproc::get_process_index(_pid, _ppid);
                if(_rank >= 0) mproc_comm_record.rank = m_rank = _rank;
            }
        }
    }
    else if(_retval == tim::mpi::success_v && _data.tool_id.find("MPI_Comm_") == 0)
    {
        auto_lock_t _lk{ type_mutex<mpi_gotcha>() };
        if(m_comm_val != null_comm())
        {
            auto& _comm_entry = mpi_comm_records[m_comm_val];
            _comm_entry.comm  = m_comm_val;

            auto _get_rank = [&]() {
                return (m_rank_ptr) ? std::max<int>(*m_rank_ptr, m_rank) : m_rank;
            };

            auto _get_size = [&]() {
                return (m_size_ptr) ? std::max<int>(*m_size_ptr, m_size)
                                    : std::max<int>(m_size, _get_rank() + 1);
            };

            if(_data.tool_id == "MPI_Comm_rank" || _data.tool_id == "MPI_Comm_size")
            {
                _comm_entry.rank = m_rank = std::max<int>(_comm_entry.rank, _get_rank());
                _comm_entry.size = m_size = std::max<int>(_comm_entry.size, _get_size());
            }
            else
            {
                OMNITRACE_BASIC_VERBOSE(
                    0, "%s() returned %i :: unexpected function wrapper\n",
                    _data.tool_id.c_str(), (int) _retval);
            }

            if(_comm_entry.updated()) update();
        }
    }
    omnitrace_pop_trace_hidden(_data.tool_id.c_str());
}
}  // namespace omnitrace

TIMEMORY_INITIALIZE_STORAGE(omnitrace::mpi_gotcha)
