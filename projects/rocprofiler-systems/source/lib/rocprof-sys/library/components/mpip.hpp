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

#pragma once

#include "core/timemory.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>
#include <timemory/components/gotcha/types.hpp>
#include <timemory/manager/declaration.hpp>
#include <timemory/mpl/apply.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/variadic/types.hpp>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#if !defined(ROCPROFSYS_USE_MPI) && defined(ROCPROFSYS_USE_MPI_HEADERS) &&               \
    !defined(OMPI_SKIP_MPICXX)
#    define ROCPROFSYS_UNDEFINE_OMPI_SKIP_MPICXX 1
#    define OMPI_SKIP_MPICXX                     1
#endif

#if defined(ROCPROFSYS_USE_MPI) || defined(ROCPROFSYS_USE_MPI_HEADERS)
#    include <mpi.h>
#endif

#if !defined(NUM_ROCPROFSYS_MPIP_WRAPPERS)
#    define NUM_ROCPROFSYS_MPIP_WRAPPERS 500
#endif

namespace rocprofsys
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Toolset, typename Tag>
TIMEMORY_VISIBILITY("default")
TIMEMORY_NOINLINE void configure_mpip(const std::set<std::string>& permit = {},
                                      const std::set<std::string>& reject = {});
//
//--------------------------------------------------------------------------------------//
//
template <typename Toolset, typename Tag>
TIMEMORY_VISIBILITY("default")
TIMEMORY_NOINLINE uint64_t activate_mpip();
//
//--------------------------------------------------------------------------------------//
//
template <typename Toolset, typename Tag>
TIMEMORY_VISIBILITY("default")
TIMEMORY_NOINLINE uint64_t deactivate_mpip(uint64_t);
//
//--------------------------------------------------------------------------------------//
//
template <typename Toolset, typename Tag>
struct mpip_handle : base<mpip_handle<Toolset, Tag>, void>
{
    static constexpr size_t mpip_wrapper_count = NUM_ROCPROFSYS_MPIP_WRAPPERS;

    using value_type = void;
    using this_type  = mpip_handle<Toolset, Tag>;
    using base_type  = base<this_type, value_type>;

    using mpi_toolset_t = Toolset;
    using mpip_gotcha_t = tim::component::gotcha<mpip_wrapper_count, mpi_toolset_t, Tag>;
    using mpip_tuple_t  = tim::component_tuple<mpip_gotcha_t>;
    using toolset_ptr_t = std::shared_ptr<mpip_tuple_t>;

    static std::string label() { return "mpip_handle"; }
    static std::string description() { return "Handle for activating MPI wrappers"; }

    void get() {}

    void start()
    {
        if(get_tool_count()++ == 0)
        {
            get_tool_instance() = std::make_shared<mpip_tuple_t>("rocprofsys_mpip");
            get_tool_instance()->start();
        }
    }

    void stop()
    {
        auto idx = --get_tool_count();
        if(get_tool_instance().get())
        {
            get_tool_instance()->stop();
            if(idx == 0) get_tool_instance().reset();
        }
    }

    int get_count() { return get_tool_count().load(); }

private:
    struct persistent_data
    {
        std::atomic<short>   m_configured;
        std::atomic<int64_t> m_count;
        toolset_ptr_t        m_tool;
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }

    static std::atomic<short>& get_configured()
    {
        return get_persistent_data().m_configured;
    }

    static toolset_ptr_t& get_tool_instance() { return get_persistent_data().m_tool; }

    static std::atomic<int64_t>& get_tool_count()
    {
        return get_persistent_data().m_count;
    }
};
//
//======================================================================================//
//
}  // namespace component
}  // namespace rocprofsys
//
//======================================================================================//
//
#include <timemory/timemory.hpp>
//
//======================================================================================//
//
/// \fn uint64_t rocprofsys::component::activate_mpip()
/// \brief The thread that first activates mpip will be the thread that turns it off.
/// Function returns the number of new mpip handles
///
template <typename Toolset, typename Tag>
uint64_t
rocprofsys::component::activate_mpip()
{
    using handle_t = rocprofsys::component::mpip_handle<Toolset, Tag>;

    static std::shared_ptr<handle_t> _handle;

    if(!_handle.get())
    {
        _handle = std::make_shared<handle_t>();
        _handle->start();

        auto cleanup_functor = [=]() {
            if(_handle)
            {
                _handle->stop();
                _handle.reset();
            }
        };

        static std::string _label = []() {
            std::stringstream ss;
            ss << "rocprofsys-mpip-" << demangle<Toolset>() << "-" << demangle<Tag>();
            return ss.str();
        }();
        ROCPROFSYS_BASIC_DEBUG_F("Adding cleanup for %s", _label.c_str());
        tim::manager::instance()->add_cleanup(_label, cleanup_functor);
        return 1;
    }
    return 0;
}
//
//======================================================================================//
//
/// \fn uint64_t rocprofsys::component::deactivate_mpip(uint64_t id)
/// \brief The thread that created the initial mpip handle will turn off. Returns
/// the number of handles active
///
template <typename Toolset, typename Tag>
uint64_t
rocprofsys::component::deactivate_mpip(uint64_t id)
{
    if(id > 0)
    {
        static std::string _label = []() {
            std::stringstream ss;
            ss << "rocprofsys-mpip-" << demangle<Toolset>() << "-" << demangle<Tag>();
            return ss.str();
        }();
        ROCPROFSYS_BASIC_DEBUG_F("Removing cleanup for %s", _label.c_str());
        tim::manager::instance()->cleanup(_label);
        return 0;
    }
    return 1;
}
//
//======================================================================================//
//
#if !defined(TIMEMORY_USE_GOTCHA) ||                                                     \
    (!defined(ROCPROFSYS_USE_MPI) && !defined(ROCPROFSYS_USE_MPI_HEADERS))
//
template <typename Toolset, typename Tag>
void
rocprofsys::component::configure_mpip(const std::set<std::string>&,
                                      const std::set<std::string>&)
{}
//
#else
//
template <typename Toolset, typename Tag>
void
rocprofsys::component::configure_mpip(const std::set<std::string>& permit,
                                      const std::set<std::string>& reject)
{
    static constexpr size_t mpip_wrapper_count = NUM_ROCPROFSYS_MPIP_WRAPPERS;
    static bool             is_initialized     = false;

    using mpip_gotcha_t = tim::component::gotcha<mpip_wrapper_count, Toolset, Tag>;

    if(!is_initialized)
    {
        // generate the gotcha wrappers
        mpip_gotcha_t::get_initializer() = []() {
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 0, MPI_Accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 1, MPI_Add_error_class);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 2, MPI_Add_error_code);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 3, MPI_Add_error_string);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 4, MPI_Aint_add);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 5, MPI_Aint_diff);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 6, MPI_Allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 7, MPI_Allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 8, MPI_Alloc_mem);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 9, MPI_Allreduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 10, MPI_Alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 11, MPI_Alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 12, MPI_Alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 13, MPI_Barrier);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 14, MPI_Bcast);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 15, MPI_Bsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 16, MPI_Bsend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 17, MPI_Buffer_attach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 18, MPI_Buffer_detach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 19, MPI_Cancel);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 20, MPI_Cart_coords);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 21, MPI_Cart_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 22, MPI_Cart_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 23, MPI_Cart_map);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 24, MPI_Cart_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 25, MPI_Cart_shift);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 26, MPI_Cart_sub);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 27, MPI_Cartdim_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 28, MPI_Close_port);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 29, MPI_Comm_accept);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 30, MPI_Comm_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 31, MPI_Comm_compare);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 32, MPI_Comm_connect);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 33, MPI_Comm_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 34, MPI_Comm_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 35, MPI_Comm_create_group);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 36, MPI_Comm_create_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 37, MPI_Comm_delete_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 38, MPI_Comm_disconnect);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 39, MPI_Comm_dup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 40, MPI_Comm_dup_with_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 41, MPI_Comm_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 42, MPI_Comm_free_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 43, MPI_Comm_get_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 44, MPI_Comm_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 45, MPI_Comm_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 46, MPI_Comm_get_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 47, MPI_Comm_get_parent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 48, MPI_Comm_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 49, MPI_Comm_idup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 50, MPI_Comm_join);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 51, MPI_Comm_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 52, MPI_Comm_remote_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 53, MPI_Comm_remote_size);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 54, MPI_Comm_set_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 55, MPI_Comm_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 56, MPI_Comm_set_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 57, MPI_Comm_set_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 58, MPI_Comm_size);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 59, MPI_Comm_spawn);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 60, MPI_Comm_spawn_multiple);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 61, MPI_Comm_split);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 62, MPI_Comm_split_type);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 63, MPI_Comm_test_inter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 64, MPI_Compare_and_swap);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 65, MPI_Dims_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 66, MPI_Dist_graph_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 67, MPI_Dist_graph_create_adjacent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 68, MPI_Dist_graph_neighbors);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 69, MPI_Dist_graph_neighbors_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 70, MPI_Error_class);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 71, MPI_Error_string);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 72, MPI_Exscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 73, MPI_Fetch_and_op);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 74, MPI_File_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 75, MPI_File_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 76, MPI_File_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 77, MPI_File_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 78, MPI_Free_mem);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 79, MPI_Gather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 80, MPI_Gatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 81, MPI_Get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 82, MPI_Get_accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 83, MPI_Get_address);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 84, MPI_Get_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 85, MPI_Get_elements);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 86, MPI_Get_elements_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 87, MPI_Get_library_version);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 88, MPI_Get_processor_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 89, MPI_Get_version);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 90, MPI_Graph_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 91, MPI_Graph_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 92, MPI_Graph_map);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 93, MPI_Graph_neighbors);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 94, MPI_Graph_neighbors_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 95, MPI_Graphdims_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 96, MPI_Grequest_complete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 97, MPI_Grequest_start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 98, MPI_Group_compare);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 99, MPI_Group_difference);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 100, MPI_Group_excl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 101, MPI_Group_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 102, MPI_Group_incl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 103, MPI_Group_intersection);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 104, MPI_Group_range_excl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 105, MPI_Group_range_incl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 106, MPI_Group_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 107, MPI_Group_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 108, MPI_Group_translate_ranks);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 109, MPI_Group_union);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 110, MPI_Iallgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 111, MPI_Iallgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 112, MPI_Iallreduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 113, MPI_Ialltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 114, MPI_Ialltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 115, MPI_Ialltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 116, MPI_Ibarrier);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 117, MPI_Ibcast);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 118, MPI_Ibsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 119, MPI_Iexscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 120, MPI_Igather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 121, MPI_Igatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 122, MPI_Improbe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 123, MPI_Imrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 124, MPI_Ineighbor_allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 125, MPI_Ineighbor_allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 126, MPI_Ineighbor_alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 127, MPI_Ineighbor_alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 128, MPI_Ineighbor_alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 129, MPI_Info_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 130, MPI_Info_delete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 131, MPI_Info_dup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 132, MPI_Info_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 133, MPI_Info_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 134, MPI_Info_get_nkeys);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 135, MPI_Info_get_nthkey);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 136, MPI_Info_get_valuelen);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 137, MPI_Info_set);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 138, MPI_Intercomm_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 139, MPI_Intercomm_merge);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 140, MPI_Iprobe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 141, MPI_Irecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 142, MPI_Ireduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 143, MPI_Ireduce_scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 144, MPI_Ireduce_scatter_block);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 145, MPI_Irsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 146, MPI_Is_thread_main);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 147, MPI_Iscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 148, MPI_Iscatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 149, MPI_Iscatterv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 150, MPI_Isend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 151, MPI_Issend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 152, MPI_Lookup_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 153, MPI_Mprobe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 154, MPI_Mrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 155, MPI_Neighbor_allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 156, MPI_Neighbor_allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 157, MPI_Neighbor_alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 158, MPI_Neighbor_alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 159, MPI_Neighbor_alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 160, MPI_Op_commutative);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 161, MPI_Op_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 162, MPI_Op_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 163, MPI_Open_port);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 164, MPI_Pack);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 165, MPI_Pack_external);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 166, MPI_Pack_external_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 167, MPI_Pack_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 168, MPI_Probe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 169, MPI_Publish_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 170, MPI_Put);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 171, MPI_Query_thread);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 172, MPI_Raccumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 173, MPI_Recv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 174, MPI_Recv_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 175, MPI_Reduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 176, MPI_Reduce_local);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 177, MPI_Reduce_scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 178, MPI_Reduce_scatter_block);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 179, MPI_Request_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 180, MPI_Request_get_status);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 181, MPI_Rget);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 182, MPI_Rget_accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 183, MPI_Rput);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 184, MPI_Rsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 185, MPI_Rsend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 186, MPI_Scan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 187, MPI_Scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 188, MPI_Scatterv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 189, MPI_Send);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 190, MPI_Send_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 191, MPI_Sendrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 192, MPI_Sendrecv_replace);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 193, MPI_Ssend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 194, MPI_Ssend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 195, MPI_Start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 196, MPI_Startall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 197, MPI_Status_f2c);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 198, MPI_Status_set_cancelled);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 199, MPI_Status_set_elements);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 200, MPI_Status_set_elements_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 201, MPI_Topo_test);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 202, MPI_Unpack);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 203, MPI_Unpack_external);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 204, MPI_Unpublish_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 205, MPI_Wait);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 206, MPI_Waitall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 207, MPI_Waitany);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 208, MPI_Waitsome);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 209, MPI_Win_allocate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 210, MPI_Win_allocate_shared);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 211, MPI_Win_attach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 212, MPI_Win_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 213, MPI_Win_complete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 214, MPI_Win_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 215, MPI_Win_create_dynamic);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 216, MPI_Win_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 217, MPI_Win_create_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 218, MPI_Win_delete_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 219, MPI_Win_detach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 220, MPI_Win_fence);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 221, MPI_Win_flush);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 222, MPI_Win_flush_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 223, MPI_Win_flush_local);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 224, MPI_Win_flush_local_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 225, MPI_Win_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 226, MPI_Win_free_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 227, MPI_Win_get_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 228, MPI_Win_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 229, MPI_Win_get_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 230, MPI_Win_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 231, MPI_Win_get_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 232, MPI_Win_lock);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 233, MPI_Win_lock_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 234, MPI_Win_post);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 235, MPI_Win_set_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 236, MPI_Win_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 237, MPI_Win_set_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 238, MPI_Win_set_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 239, MPI_Win_shared_query);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 240, MPI_Win_start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 241, MPI_Win_sync);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 242, MPI_Win_test);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 243, MPI_Win_unlock);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 244, MPI_Win_unlock_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 245, MPI_Win_wait);

            // MPI profiling interface wrappers
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 250, PMPI_Accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 251, PMPI_Add_error_class);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 252, PMPI_Add_error_code);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 253, PMPI_Add_error_string);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 254, PMPI_Aint_add);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 255, PMPI_Aint_diff);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 256, PMPI_Allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 257, PMPI_Allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 258, PMPI_Alloc_mem);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 259, PMPI_Allreduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 260, PMPI_Alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 261, PMPI_Alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 262, PMPI_Alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 263, PMPI_Barrier);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 264, PMPI_Bcast);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 265, PMPI_Bsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 266, PMPI_Bsend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 267, PMPI_Buffer_attach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 268, PMPI_Buffer_detach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 269, PMPI_Cancel);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 270, PMPI_Cart_coords);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 271, PMPI_Cart_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 272, PMPI_Cart_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 273, PMPI_Cart_map);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 274, PMPI_Cart_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 275, PMPI_Cart_shift);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 276, PMPI_Cart_sub);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 277, PMPI_Cartdim_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 278, PMPI_Close_port);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 279, PMPI_Comm_accept);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 280, PMPI_Comm_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 281, PMPI_Comm_compare);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 282, PMPI_Comm_connect);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 283, PMPI_Comm_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 284, PMPI_Comm_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 285, PMPI_Comm_create_group);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 286, PMPI_Comm_create_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 287, PMPI_Comm_delete_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 288, PMPI_Comm_disconnect);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 289, PMPI_Comm_dup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 290, PMPI_Comm_dup_with_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 291, PMPI_Comm_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 292, PMPI_Comm_free_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 293, PMPI_Comm_get_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 294, PMPI_Comm_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 295, PMPI_Comm_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 296, PMPI_Comm_get_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 297, PMPI_Comm_get_parent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 298, PMPI_Comm_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 299, PMPI_Comm_idup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 300, PMPI_Comm_join);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 301, PMPI_Comm_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 302, PMPI_Comm_remote_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 303, PMPI_Comm_remote_size);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 304, PMPI_Comm_set_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 305, PMPI_Comm_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 306, PMPI_Comm_set_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 307, PMPI_Comm_set_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 308, PMPI_Comm_size);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 309, PMPI_Comm_spawn);
            // TIMEMORY_C_GOTCHA(mpip_gotcha_t, 310, PMPI_Comm_spawn_multiple);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 311, PMPI_Comm_split);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 312, PMPI_Comm_split_type);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 313, PMPI_Comm_test_inter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 314, PMPI_Compare_and_swap);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 315, PMPI_Dims_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 316, PMPI_Dist_graph_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 317, PMPI_Dist_graph_create_adjacent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 318, PMPI_Dist_graph_neighbors);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 319, PMPI_Dist_graph_neighbors_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 320, PMPI_Error_class);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 321, PMPI_Error_string);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 322, PMPI_Exscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 323, PMPI_Fetch_and_op);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 324, PMPI_File_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 325, PMPI_File_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 326, PMPI_File_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 327, PMPI_File_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 328, PMPI_Free_mem);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 329, PMPI_Gather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 330, PMPI_Gatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 331, PMPI_Get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 332, PMPI_Get_accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 333, PMPI_Get_address);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 334, PMPI_Get_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 335, PMPI_Get_elements);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 336, PMPI_Get_elements_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 337, PMPI_Get_library_version);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 338, PMPI_Get_processor_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 339, PMPI_Get_version);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 340, PMPI_Graph_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 341, PMPI_Graph_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 342, PMPI_Graph_map);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 343, PMPI_Graph_neighbors);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 344, PMPI_Graph_neighbors_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 345, PMPI_Graphdims_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 346, PMPI_Grequest_complete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 347, PMPI_Grequest_start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 348, PMPI_Group_compare);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 349, PMPI_Group_difference);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 350, PMPI_Group_excl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 351, PMPI_Group_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 352, PMPI_Group_incl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 353, PMPI_Group_intersection);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 354, PMPI_Group_range_excl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 355, PMPI_Group_range_incl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 356, PMPI_Group_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 357, PMPI_Group_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 358, PMPI_Group_translate_ranks);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 359, PMPI_Group_union);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 360, PMPI_Iallgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 361, PMPI_Iallgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 362, PMPI_Iallreduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 363, PMPI_Ialltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 364, PMPI_Ialltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 365, PMPI_Ialltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 366, PMPI_Ibarrier);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 367, PMPI_Ibcast);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 368, PMPI_Ibsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 369, PMPI_Iexscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 370, PMPI_Igather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 371, PMPI_Igatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 372, PMPI_Improbe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 373, PMPI_Imrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 374, PMPI_Ineighbor_allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 375, PMPI_Ineighbor_allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 376, PMPI_Ineighbor_alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 377, PMPI_Ineighbor_alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 378, PMPI_Ineighbor_alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 379, PMPI_Info_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 380, PMPI_Info_delete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 381, PMPI_Info_dup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 382, PMPI_Info_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 383, PMPI_Info_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 384, PMPI_Info_get_nkeys);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 385, PMPI_Info_get_nthkey);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 386, PMPI_Info_get_valuelen);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 387, PMPI_Info_set);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 388, PMPI_Intercomm_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 389, PMPI_Intercomm_merge);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 390, PMPI_Iprobe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 391, PMPI_Irecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 392, PMPI_Ireduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 393, PMPI_Ireduce_scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 394, PMPI_Ireduce_scatter_block);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 395, PMPI_Irsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 396, PMPI_Is_thread_main);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 397, PMPI_Iscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 398, PMPI_Iscatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 399, PMPI_Iscatterv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 400, PMPI_Isend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 401, PMPI_Issend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 402, PMPI_Lookup_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 403, PMPI_Mprobe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 404, PMPI_Mrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 405, PMPI_Neighbor_allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 406, PMPI_Neighbor_allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 407, PMPI_Neighbor_alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 408, PMPI_Neighbor_alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 409, PMPI_Neighbor_alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 410, PMPI_Op_commutative);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 411, PMPI_Op_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 412, PMPI_Op_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 413, PMPI_Open_port);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 414, PMPI_Pack);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 415, PMPI_Pack_external);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 416, PMPI_Pack_external_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 417, PMPI_Pack_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 418, PMPI_Probe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 419, PMPI_Publish_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 420, PMPI_Put);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 421, PMPI_Query_thread);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 422, PMPI_Raccumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 423, PMPI_Recv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 424, PMPI_Recv_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 425, PMPI_Reduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 426, PMPI_Reduce_local);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 427, PMPI_Reduce_scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 428, PMPI_Reduce_scatter_block);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 429, PMPI_Request_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 430, PMPI_Request_get_status);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 431, PMPI_Rget);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 432, PMPI_Rget_accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 433, PMPI_Rput);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 434, PMPI_Rsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 435, PMPI_Rsend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 436, PMPI_Scan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 437, PMPI_Scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 438, PMPI_Scatterv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 439, PMPI_Send);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 440, PMPI_Send_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 441, PMPI_Sendrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 442, PMPI_Sendrecv_replace);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 443, PMPI_Ssend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 444, PMPI_Ssend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 445, PMPI_Start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 446, PMPI_Startall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 447, PMPI_Status_f2c);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 448, PMPI_Status_set_cancelled);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 449, PMPI_Status_set_elements);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 450, PMPI_Status_set_elements_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 451, PMPI_Topo_test);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 452, PMPI_Unpack);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 453, PMPI_Unpack_external);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 454, PMPI_Unpublish_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 455, PMPI_Wait);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 456, PMPI_Waitall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 457, PMPI_Waitany);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 458, PMPI_Waitsome);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 459, PMPI_Win_allocate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 460, PMPI_Win_allocate_shared);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 461, PMPI_Win_attach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 462, PMPI_Win_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 463, PMPI_Win_complete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 464, PMPI_Win_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 465, PMPI_Win_create_dynamic);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 466, PMPI_Win_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 467, PMPI_Win_create_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 468, PMPI_Win_delete_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 469, PMPI_Win_detach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 470, PMPI_Win_fence);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 471, PMPI_Win_flush);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 472, PMPI_Win_flush_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 473, PMPI_Win_flush_local);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 474, PMPI_Win_flush_local_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 475, PMPI_Win_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 476, PMPI_Win_free_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 477, PMPI_Win_get_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 478, PMPI_Win_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 479, PMPI_Win_get_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 480, PMPI_Win_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 481, PMPI_Win_get_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 482, PMPI_Win_lock);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 483, PMPI_Win_lock_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 484, PMPI_Win_post);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 485, PMPI_Win_set_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 486, PMPI_Win_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 487, PMPI_Win_set_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 488, PMPI_Win_set_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 489, PMPI_Win_shared_query);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 490, PMPI_Win_start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 491, PMPI_Win_sync);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 492, PMPI_Win_test);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 493, PMPI_Win_unlock);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 494, PMPI_Win_unlock_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 495, PMPI_Win_wait);
        };

        // provide environment variable for suppressing wrappers
        mpip_gotcha_t::get_reject_list() = [reject]() {
            auto _reject = reject;
            // check environment
            auto reject_list = tim::get_env<std::string>(
                TIMEMORY_SETTINGS_PREFIX "ROCPROFSYS_MPIP_REJECT_LIST", "");
            // add environment setting
            for(const auto& itr : tim::delimit(reject_list))
                _reject.insert(itr);
            return _reject;
        };

        // provide environment variable for selecting wrappers
        mpip_gotcha_t::get_permit_list() = [permit]() {
            auto _permit = permit;
            // check environment
            auto permit_list = tim::get_env<std::string>(
                TIMEMORY_SETTINGS_PREFIX "ROCPROFSYS_MPIP_PERMIT_LIST", "");
            // add environment setting
            for(const auto& itr : tim::delimit(permit_list))
                _permit.insert(itr);
            return _permit;
        };

        is_initialized = true;
    }
}
//
#endif
//
//======================================================================================//
//

#if defined(ROCPROFSYS_UNDEFINE_OMPI_SKIP_MPICXX) && ROCPROFSYS_UNDEFINE_OMPI_SKIP_MPICXX
#    undef OMPI_SKIP_MPICXX
#endif
