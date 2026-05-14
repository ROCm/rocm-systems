// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/mpi.hpp"
#include "core/timemory.hpp"

#include <cstdint>
#include <mutex>

namespace rocprofsys
{
namespace component
{
// this is used to wrap MPI_Init and MPI_Init_thread
struct mpi_gotcha : comp::base<mpi_gotcha, void>
{
    using comm_t        = rocprofsys::mpi::comm_t;
    using gotcha_data_t = comp::gotcha_data;

    // string id for component
    static std::string label() { return "mpi_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static void pause();
    static void resume();

    // called right before MPI_Init with that functions arguments
    static void audit(const gotcha_data_t& _data, audit::incoming, int*, char***);

    // called right before MPI_Init_thread with that functions arguments
    static void audit(const gotcha_data_t& _data, audit::incoming, int*, char***, int,
                      int*);

    // called right before MPI_Finalize
    static void audit(const gotcha_data_t& _data, audit::incoming);

    // called right before MPI_Comm_{rank,size} with that functions arguments
    void audit(const gotcha_data_t& _data, audit::incoming, comm_t, int*);

    // called right after MPI_{Init,Init_thread,Comm_rank,Comm_size} with the return value
    void audit(const gotcha_data_t& _data, audit::outgoing, int _retval);

    // without these you will get a verbosity level 1 warning
    static void start() {}
    static void stop() {}

    static bool      update();
    static uintptr_t null_comm() { return std::numeric_limits<uintptr_t>::max(); }
    static void      disable_comm_intercept();

    static void subscribe_to_init_event(
        const std::function<void(int rank, int size)>& _callback);

private:
    int       m_rank     = 0;
    int       m_size     = 1;
    int*      m_rank_ptr = nullptr;
    int*      m_size_ptr = nullptr;
    uintptr_t m_comm_val = null_comm();

    void        populate_rank_and_size();
    static void publish_rank_and_size(int rank, int size);

    static std::mutex                                           s_on_init_callbacks_mutex;
    static std::vector<std::function<void(int rank, int size)>> s_on_init_callbacks;

    static std::mutex s_mutex;
};
}  // namespace component

using mpi_gotcha_t =
    comp::gotcha<10, tim::component_tuple<component::mpi_gotcha>, project::rocprofsys>;
}  // namespace rocprofsys
