// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/components/rcclp.hpp"
#include "library/rcclp.hpp"

#include <timemory/manager.hpp>

std::ostream&
operator<<(std::ostream& _os, const ncclUniqueId& _v)
{
    for(auto itr : _v.internal)
        _os << itr;
    return _os;
}

namespace rocprofsys
{
namespace component
{
uint64_t
activate_rcclp()
{
    using handle_t = tim::component::rcclp_handle;

    static auto _handle = std::shared_ptr<handle_t>{};

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

        std::stringstream ss;
        ss << "timemory-rcclp-" << demangle<rccl_toolset_t>() << "-"
           << demangle<category::rocm_rccl>();
        tim::manager::instance()->add_cleanup(ss.str(), cleanup_functor);
        return 1;
    }
    return 0;
}
//
//======================================================================================//
//
uint64_t
deactivate_rcclp(uint64_t id)
{
    if(id > 0)
    {
        std::stringstream ss;
        ss << "timemory-rcclp-" << demangle<rccl_toolset_t>() << "-"
           << demangle<category::rocm_rccl>();
        tim::manager::instance()->cleanup(ss.str());
        return 0;
    }
    return 1;
}
//
//======================================================================================//
//
void
configure_rcclp(const std::set<std::string>& permit, const std::set<std::string>& reject)
{
    static bool is_initialized = false;
    if(!is_initialized)
    {
        // generate the gotcha wrappers
        rcclp_gotcha_t::get_initializer() = []() {
            // TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 0, ncclGetVersion);
            // TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 1, ncclGetUniqueId);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 2, ncclCommInitRank);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 3, ncclCommInitAll);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 4, ncclCommDestroy);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 5, ncclCommCount);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 6, ncclCommCuDevice);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 7, ncclCommUserRank);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 8, ncclReduce);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 9, ncclBcast);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 10, ncclBroadcast);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 11, ncclAllReduce);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 12, ncclReduceScatter);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 13, ncclAllGather);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 14, ncclGroupStart);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 15, ncclGroupEnd);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 16, ncclSend);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 17, ncclRecv);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 18, ncclGather);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 19, ncclScatter);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 20, ncclAllToAll);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 21, ncclAllToAllv);
            // TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 22, ncclRedOpCreatePreMulSum);
            // TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 23, ncclRedOpDestroy);
        };

        // provide environment variable for suppressing wrappers
        rcclp_gotcha_t::get_reject_list() = [reject]() {
            auto _reject = reject;
            // check environment
            auto reject_list =
                tim::get_env<std::string>("ROCPROFSYS_RCCLP_REJECT_LIST", "");
            // add environment setting
            for(const auto& itr : tim::delimit(reject_list))
                _reject.insert(itr);
            return _reject;
        };

        // provide environment variable for selecting wrappers
        rcclp_gotcha_t::get_permit_list() = [permit]() {
            auto _permit = permit;
            // check environment
            auto permit_list =
                tim::get_env<std::string>("ROCPROFSYS_RCCLP_PERMIT_LIST", "");
            // add environment setting
            for(const auto& itr : tim::delimit(permit_list))
                _permit.insert(itr);
            return _permit;
        };

        is_initialized = true;
    }
}

void
rcclp_handle::start()
{
    if(get_tool_count()++ == 0)
    {
        get_tool_instance() = std::make_shared<rcclp_tuple_t>("timemory_rcclp");
        get_tool_instance()->start();
    }
}

void
rcclp_handle::stop()
{
    auto idx = --get_tool_count();
    if(get_tool_instance().get())
    {
        get_tool_instance()->stop();
        if(idx == 0) get_tool_instance().reset();
    }
}

rcclp_handle::persistent_data&
rcclp_handle::get_persistent_data()
{
    static persistent_data _instance;
    return _instance;
}

std::atomic<short>&
rcclp_handle::get_configured()
{
    return get_persistent_data().m_configured;
}

rcclp_handle::toolset_ptr_t&
rcclp_handle::get_tool_instance()
{
    return get_persistent_data().m_tool;
}

std::atomic<int64_t>&
rcclp_handle::get_tool_count()
{
    return get_persistent_data().m_count;
}
}  // namespace component
}  // namespace rocprofsys
