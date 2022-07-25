// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "library/common.hpp"
#include "library/components/category_region.hpp"
#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/timemory.hpp"

#include <timemory/api/macros.hpp>
#include <timemory/components/macros.hpp>

#if OMNITRACE_HIP_VERSION == 0 || OMNITRACE_HIP_VERSION >= 50200
#    include <rccl/rccl.h>
#else
#    include <rccl.h>
#endif

#include <atomic>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>

#if !defined(OMNITRACE_NUM_RCCLP_WRAPPERS)
#    define OMNITRACE_NUM_RCCLP_WRAPPERS 25
#endif

TIMEMORY_COMPONENT_ALIAS(
    rccl_toolset_t,
    component_bundle<rccl_api_t, omnitrace::component::category_region<category::rccl>,
                     rccl_comm_data*>)
TIMEMORY_COMPONENT_ALIAS(rcclp_gotcha_t,
                         gotcha<OMNITRACE_NUM_RCCLP_WRAPPERS, rccl_toolset_t, rccl_api_t>)

#if !defined(OMNITRACE_USE_RCCL)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::rcclp_gotcha_t, false_type)
#endif

TIMEMORY_STATISTICS_TYPE(component::rccl_data_tracker_t, float)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::rccl_data_tracker_t,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::rccl_data_tracker_t,
                               true_type)

namespace tim
{
namespace component
{
uint64_t
activate_rcclp();

uint64_t
deactivate_rcclp(uint64_t id);

void
configure_rcclp(const std::set<std::string>& permit = {},
                const std::set<std::string>& reject = {});

struct rcclp_handle : base<rcclp_handle, void>
{
    static constexpr size_t rcclp_wrapper_count = OMNITRACE_NUM_RCCLP_WRAPPERS;

    using value_type = void;
    using this_type  = rcclp_handle;
    using base_type  = base<this_type, value_type>;

    using rcclp_tuple_t = tim::component_tuple<rcclp_gotcha_t>;
    using toolset_ptr_t = std::shared_ptr<rcclp_tuple_t>;

    static std::string label() { return "rcclp_handle"; }
    static std::string description() { return "Handle for activating NCCL wrappers"; }
    static void        get() {}
    static void        start();
    static void        stop();
    static int         get_count() { return get_tool_count().load(); }

private:
    struct persistent_data
    {
        std::atomic<short>   m_configured{ 0 };
        std::atomic<int64_t> m_count{ 0 };
        toolset_ptr_t        m_tool = toolset_ptr_t{};
    };

    static persistent_data&      get_persistent_data();
    static std::atomic<short>&   get_configured();
    static toolset_ptr_t&        get_tool_instance();
    static std::atomic<int64_t>& get_tool_count();
};

struct rccl_comm_data : base<rccl_comm_data, void>
{
    using value_type = void;
    using this_type  = rccl_comm_data;
    using base_type  = base<this_type, value_type>;
    using tracker_t  = tim::auto_tuple<rccl_data_tracker_t>;
    using data_type  = float;

    TIMEMORY_DEFAULT_OBJECT(rccl_comm_data)

    static void preinit();
    static void start() {}
    static void stop() {}

    static auto rccl_type_size(ncclDataType_t datatype)
    {
        switch(datatype)
        {
            case ncclInt8:
            case ncclUint8: return 1;
            case ncclFloat16: return 2;
            case ncclInt32:
            case ncclUint32:
            case ncclFloat32: return 4;
            case ncclInt64:
            case ncclUint64:
            case ncclFloat64: return 8;
            default: return 0;
        };
    }

    // ncclReduce
    static void audit(const gotcha_data& _data, audit::incoming, const void*, void*,
                      size_t count, ncclDataType_t datatype, ncclRedOp_t, int root,
                      ncclComm_t, hipStream_t);

    // ncclSend
    static void audit(const gotcha_data& _data, audit::incoming, const void*,
                      size_t count, ncclDataType_t datatype, int peer, ncclComm_t,
                      hipStream_t);

    // ncclBcast
    // ncclRecv
    static void audit(const gotcha_data& _data, audit::incoming, void*, size_t count,
                      ncclDataType_t datatype, int root, ncclComm_t, hipStream_t);

    // ncclBroadcast
    static void audit(const gotcha_data& _data, audit::incoming, const void*, void*,
                      size_t count, ncclDataType_t datatype, int root, ncclComm_t,
                      hipStream_t);

    // ncclAllReduce
    // ncclReduceScatter
    static void audit(const gotcha_data& _data, audit::incoming, const void*, void*,
                      size_t count, ncclDataType_t datatype, ncclRedOp_t, ncclComm_t,
                      hipStream_t);

    // ncclAllGather
    static void audit(const gotcha_data& _data, audit::incoming, const void*, void*,
                      size_t count, ncclDataType_t datatype, ncclComm_t, hipStream_t);

private:
    template <typename... Args>
    static void add(tracker_t& _t, data_type value, Args&&... args)
    {
        _t.store(std::plus<data_type>{}, value);
        TIMEMORY_FOLD_EXPRESSION(add_secondary(_t, std::forward<Args>(args), value));
    }

    template <typename... Args>
    static void add(const gotcha_data& _data, data_type value, Args&&... args)
    {
        tracker_t _t{ std::string_view{ _data.tool_id.c_str() } };
        add(_t, value, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void add_secondary(tracker_t&, const gotcha_data& _data, data_type value,
                              Args&&... args)
    {
        // if(tim::settings::add_secondary())
        {
            tracker_t _s{ std::string_view{ _data.tool_id.c_str() } };
            add(_s, _data, value, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static void add(std::string_view _name, data_type value, Args&&... args)
    {
        tracker_t _t{ _name };
        add(_t, value, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void add_secondary(tracker_t&, std::string_view _name, data_type value,
                              Args&&... args)
    {
        // if(tim::settings::add_secondary())
        {
            tracker_t _s{ _name };
            add(_s, value, std::forward<Args>(args)...);
        }
    }
};
}  // namespace component
}  // namespace tim
