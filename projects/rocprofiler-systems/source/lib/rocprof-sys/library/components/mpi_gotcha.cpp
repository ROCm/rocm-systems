// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "api.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/mpi.hpp"
#include "core/mproc.hpp"
#include "library/components/category_region.hpp"
#include "library/components/comm_data.hpp"
#include "mpip.hpp"

#include <timemory/backends/process.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/signals/signal_mask.hpp>
#include <timemory/utility/locking.hpp>

#include <cstdint>
#include <limits>
#include <thread>
#include <unistd.h>

namespace rocprofsys
{
namespace component
{
namespace
{
using mpip_bundle_t = tim::component_tuple<category_region<category::mpi>, comm_data>;

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
        ROCPROFSYS_CI_THROW(!_lhs.updated() && !_rhs.updated(),
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

#if defined(ROCPROFSYS_USE_MPI)
int
rocprofsys_mpi_copy(MPI_Comm, int, void*, void*, void*, int*)
{
    return MPI_SUCCESS;
}

int
rocprofsys_mpi_fini(MPI_Comm, int, void*, void*)
{
    ROCPROFSYS_DEBUG("MPI Comm attribute finalize\n");
    auto _blocked = get_sampling_signals();
    if(!_blocked.empty())
        tim::signals::block_signals(_blocked, tim::signals::sigmask_scope::process);
    if(mpip_index != std::numeric_limits<uint64_t>::max())
        deactivate_mpip<mpip_bundle_t, project::rocprofsys>(mpip_index);
    if(is_root_process()) rocprofsys_finalize_hidden();
    return MPI_SUCCESS;
}
#endif

// this ensures rocprofsys_finalize is called before MPI_Finalize
void
rocprofsys_mpi_set_attr()
{
#if defined(ROCPROFSYS_USE_MPI)
    auto _blocked = get_sampling_signals();
    if(!_blocked.empty())
        tim::signals::block_signals(_blocked, tim::signals::sigmask_scope::process);

    int _comm_key = -1;
    if(PMPI_Comm_create_keyval(&rocprofsys_mpi_copy, &rocprofsys_mpi_fini, &_comm_key,
                               nullptr) == MPI_SUCCESS)
        PMPI_Comm_set_attr(MPI_COMM_SELF, _comm_key, nullptr);

    if(!_blocked.empty())
        tim::signals::unblock_signals(_blocked, tim::signals::sigmask_scope::process);
#endif
}

using strset_t       = std::set<std::string>;
auto permit_bindings = strset_t{};
auto reject_bindings = strset_t{};
}  // namespace

void
mpi_gotcha::configure()
{
    // don't emit warnings for missing MPI functions unless debug or verbosity >= 3
    if(get_verbose_env() < 3 && !get_debug_env())
    {
        for(size_t i = 0; i < mpi_gotcha_t::capacity(); ++i)
        {
            auto* itr = mpi_gotcha_t::at(i);
            if(itr) itr->verbose = -1;
        }
    }

    mpi_gotcha_t::get_initializer() = []() {
        mpi_gotcha_t::template configure<0, int, int*, char***>("MPI_Init");
        mpi_gotcha_t::template configure<1, int, int*, char***>("PMPI_Init");
        mpi_gotcha_t::template configure<2, int, int*, char***, int, int*>(
            "MPI_Init_thread");
        mpi_gotcha_t::template configure<3, int, int*, char***, int, int*>(
            "PMPI_Init_thread");
        mpi_gotcha_t::template configure<4, int>("MPI_Finalize");
        mpi_gotcha_t::template configure<5, int>("PMPI_Finalize");
        reject_bindings.emplace("MPI_Init");
        reject_bindings.emplace("PMPI_Init");
        reject_bindings.emplace("MPI_Init_thread");
        reject_bindings.emplace("PMPI_Init_thread");
        reject_bindings.emplace("MPI_Finalize");
        reject_bindings.emplace("PMPI_Finalize");
#if defined(ROCPROFSYS_USE_MPI_HEADERS) && ROCPROFSYS_USE_MPI_HEADERS > 0
        mpi_gotcha_t::template configure<6, int, comm_t, int*>("MPI_Comm_rank");
        mpi_gotcha_t::template configure<7, int, comm_t, int*>("PMPI_Comm_rank");
        mpi_gotcha_t::template configure<8, int, comm_t, int*>("MPI_Comm_size");
        mpi_gotcha_t::template configure<9, int, comm_t, int*>("PMPI_Comm_size");
        reject_bindings.emplace("MPI_Comm_rank");
        reject_bindings.emplace("PMPI_Comm_rank");
        reject_bindings.emplace("MPI_Comm_size");
        reject_bindings.emplace("PMPI_Comm_size");
#endif
    };
}

void
mpi_gotcha::shutdown()
{
    update();
}

bool
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

        rocprofsys::mpi::set_rank(_rank);
        rocprofsys::mpi::set_size(_size);
        rocprofsys::settings::default_process_suffix() = _rank;

        ROCPROFSYS_BASIC_VERBOSE(0, "[pid=%i] MPI rank: %i (%i), MPI size: %i (%i)\n",
                                 process::get_id(), rocprofsys::mpi::rank(), _rank,
                                 rocprofsys::mpi::size(), _size);
        last_comm_record      = _rank_data;
        config::get_use_pid() = true;
        return true;
    }
    return false;
}

void
mpi_gotcha::disable_comm_intercept()
{
#if defined(ROCPROFSYS_USE_MPI_HEADERS) && ROCPROFSYS_USE_MPI_HEADERS > 0
    mpi_gotcha_t::revert<3>();
    mpi_gotcha_t::revert<4>();
#endif
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***)
{
    ROCPROFSYS_BASIC_DEBUG_F("%s(int*, char***)\n", _data.tool_id.c_str());

    rocprofsys_push_trace_hidden(_data.tool_id.c_str());
#if !defined(ROCPROFSYS_USE_MPI) && defined(ROCPROFSYS_USE_MPI_HEADERS)
    rocprofsys::mpi::is_initialized_callback() = []() { return true; };
    rocprofsys::mpi::is_finalized()            = false;
#endif
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, int*, char***, int, int*)
{
    ROCPROFSYS_BASIC_DEBUG_F("%s(int*, char***, int, int*)\n", _data.tool_id.c_str());

    rocprofsys_push_trace_hidden(_data.tool_id.c_str());
#if !defined(ROCPROFSYS_USE_MPI) && defined(ROCPROFSYS_USE_MPI_HEADERS)
    rocprofsys::mpi::is_initialized_callback() = []() { return true; };
    rocprofsys::mpi::is_finalized()            = false;
#endif
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming)
{
    ROCPROFSYS_BASIC_DEBUG_F("%s()\n", _data.tool_id.c_str());

    auto _blocked = get_sampling_signals();
    if(!_blocked.empty())
        tim::signals::block_signals(_blocked, tim::signals::sigmask_scope::process);

    if(mpip_index != std::numeric_limits<uint64_t>::max())
        deactivate_mpip<mpip_bundle_t, project::rocprofsys>(mpip_index);

#if !defined(ROCPROFSYS_USE_MPI) && defined(ROCPROFSYS_USE_MPI_HEADERS)
    rocprofsys::mpi::is_initialized_callback() = []() { return false; };
    rocprofsys::mpi::is_finalized()            = true;
#else
    if(is_root_process() && rocprofsys::get_state() < rocprofsys::State::Finalized)
        rocprofsys_finalize_hidden();
#endif
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::incoming, comm_t _comm, int* _val)
{
    ROCPROFSYS_BASIC_DEBUG_F("%s(comm_t _comm, int* _val)\n", _data.tool_id.c_str());

    rocprofsys_push_trace_hidden(_data.tool_id.c_str());
    if(_data.tool_id.find("MPI_Comm_rank") == 0 ||
       _data.tool_id.find("PMPI_Comm_rank") == 0)
    {
        m_comm_val = (uintptr_t) _comm;  // NOLINT
        m_rank_ptr = _val;
    }
    else if(_data.tool_id.find("MPI_Comm_size") == 0 ||
            _data.tool_id.find("PMPI_Comm_size") == 0)
    {
        m_comm_val = (uintptr_t) _comm;  // NOLINT
        m_size_ptr = _val;
    }
    else
    {
        ROCPROFSYS_BASIC_PRINT_F("%s(<comm>, %p) :: unexpected function wrapper\n",
                                 _data.tool_id.c_str(), static_cast<void*>(_val));
    }
}

void
mpi_gotcha::audit(const gotcha_data_t& _data, audit::outgoing, int _retval)
{
    ROCPROFSYS_BASIC_DEBUG_F("%s() returned %i\n", _data.tool_id.c_str(), (int) _retval);

    if(!settings::use_output_suffix()) settings::use_output_suffix() = true;

    if(_retval == rocprofsys::mpi::success_v &&
       (_data.tool_id.find("MPI_Init") == 0 || _data.tool_id.find("PMPI_Init") == 0))
    {
        rocprofsys_mpi_set_attr();
        // rocprof-sys will set this environement variable to true in binary rewrite mode
        // when it detects MPI. Hides this env variable from the user to avoid this
        // being activated unwaringly during runtime instrumentation because that
        // will result in double instrumenting the MPI functions (unless the MPI functions
        // were excluded via a regex expression)
        if(get_use_mpip())
        {
            ROCPROFSYS_BASIC_VERBOSE_F(2, "Activating MPI wrappers...\n");

            // use env vars ROCPROFSYS_MPIP_PERMIT_LIST and ROCPROFSYS_MPIP_REJECT_LIST
            // to control the gotcha bindings at runtime
            configure_mpip<mpip_bundle_t, project::rocprofsys>(permit_bindings,
                                                               reject_bindings);
            mpip_index = activate_mpip<mpip_bundle_t, project::rocprofsys>();
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
    else if(_retval == rocprofsys::mpi::success_v &&
            (_data.tool_id.find("MPI_Comm_") == 0 ||
             _data.tool_id.find("PMPI_Comm_") == 0))
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

            if(_data.tool_id == "MPI_Comm_rank" || _data.tool_id == "MPI_Comm_size" ||
               _data.tool_id == "PMPI_Comm_rank" || _data.tool_id == "PMPI_Comm_size")
            {
                _comm_entry.rank = m_rank = std::max<int>(_comm_entry.rank, _get_rank());
                _comm_entry.size = m_size = std::max<int>(_comm_entry.size, _get_size());
            }
            else
            {
                ROCPROFSYS_BASIC_VERBOSE(
                    0, "%s() returned %i :: unexpected function wrapper\n",
                    _data.tool_id.c_str(), (int) _retval);
            }

            if(_comm_entry.updated())
            {
                static thread_local int _num_updates = 0;
                static int              _disable_after =
                    tim::get_env<int>("ROCPROFSYS_MPI_MAX_COMM_UPDATES", 4);
                if(_num_updates++ < _disable_after) update();
            }
        }
    }
    rocprofsys_pop_trace_hidden(_data.tool_id.c_str());
}
}  // namespace component
}  // namespace rocprofsys

TIMEMORY_INITIALIZE_STORAGE(rocprofsys::component::mpi_gotcha)
