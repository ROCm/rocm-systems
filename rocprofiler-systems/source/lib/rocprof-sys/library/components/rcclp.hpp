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

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "core/rccl.hpp"
#include "core/timemory.hpp"
#include "library/components/category_region.hpp"
#include "library/components/comm_data.hpp"

#include <timemory/api/macros.hpp>
#include <timemory/components/macros.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>

#if !defined(ROCPROFSYS_NUM_RCCLP_WRAPPERS)
#    define ROCPROFSYS_NUM_RCCLP_WRAPPERS 25
#endif

ROCPROFSYS_COMPONENT_ALIAS(
    rccl_toolset_t,
    ::tim::component_bundle<category::rocm_rccl,
                            rocprofsys::component::category_region<category::rocm_rccl>,
                            comm_data>)
ROCPROFSYS_COMPONENT_ALIAS(rcclp_gotcha_t,
                           ::tim::component::gotcha<ROCPROFSYS_NUM_RCCLP_WRAPPERS,
                                                    rccl_toolset_t, category::rocm_rccl>)

#if !defined(ROCPROFSYS_USE_RCCL)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::rcclp_gotcha_t, false_type)
#endif

namespace rocprofsys
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
    static constexpr size_t rcclp_wrapper_count = ROCPROFSYS_NUM_RCCLP_WRAPPERS;

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
}  // namespace component
}  // namespace rocprofsys
