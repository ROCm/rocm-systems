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

#include "core/common.hpp"
#include "core/defines.hpp"
#include "library/components/comm_data.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>
#include <timemory/utility/types.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace rocprofsys
{
namespace component
{

namespace traits
{
template <typename Policy, typename = void>
struct has_comm_data : std::false_type
{};

template <typename Policy>
struct has_comm_data<Policy, std::void_t<typename Policy::comm_data>> : std::true_type
{};

template <typename Policy, typename = void>
struct has_gotcha_data : std::false_type
{};

template <typename Policy>
struct has_gotcha_data<Policy, std::void_t<typename Policy::gotcha_data>> : std::true_type
{};

template <typename Policy, typename = void>
struct has_category_region : std::false_type
{};

template <typename Policy>
struct has_category_region<Policy, std::void_t<typename Policy::category_region>>
: std::true_type
{};

template <typename Policy, typename = void>
struct has_ucx_bundle_t : std::false_type
{};

template <typename Policy>
struct has_ucx_bundle_t<Policy, std::void_t<typename Policy::ucx_bundle_t>>
: std::true_type
{};

template <typename Policy, typename = void>
struct has_ucx_gotcha_t : std::false_type
{};

template <typename Policy>
struct has_ucx_gotcha_t<Policy, std::void_t<typename Policy::ucx_gotcha_t>>
: std::true_type
{};

}  // namespace traits

template <typename UCXPolicy>
struct ucx_gotcha : tim::component::base<ucx_gotcha<UCXPolicy>, void>
{
    static_assert(traits::has_comm_data<UCXPolicy>::value,
                  "UCXPolicy must have a comm_data type");

    static_assert(traits::has_gotcha_data<UCXPolicy>::value,
                  "UCXPolicy must have a gotcha_data type");

    static_assert(traits::has_category_region<UCXPolicy>::value,
                  "UCXPolicy must have a category_region type");

    static constexpr size_t gotcha_capacity = 100;

    ROCPROFSYS_DEFAULT_OBJECT(ucx_gotcha<UCXPolicy>)

    static std::string label() { return "ucx_gotcha"; }

    static void configure();
    static void shutdown();

    static void start();
    static void stop();

    template <typename... Args>
    static void audit(const typename UCXPolicy::gotcha_data& _data, audit::incoming,
                      Args...)
    {
        UCXPolicy::category_region::start(std::string_view{ _data.tool_id });
    }

public:
    // ucp_tag_send_nbx
    static void audit(const typename UCXPolicy::gotcha_data&, audit::incoming, void*,
                      const void*, size_t, uint64_t, const void*);
    // ucp_tag_recv_nbx
    static void audit(const typename UCXPolicy::gotcha_data&, audit::incoming, void*,
                      void*, size_t, uint64_t, uint64_t, const void*);
    // ucp_put_nbx
    static void audit(const typename UCXPolicy::gotcha_data&, audit::incoming, void*,
                      const void*, size_t, uint64_t, void*, const void*);
    // ucp_get_nbx
    static void audit(const typename UCXPolicy::gotcha_data&, audit::incoming, void*,
                      void*, size_t, uint64_t, void*, const void*);
    // ucp_am_send_nbx
    static void audit(const typename UCXPolicy::gotcha_data&, audit::incoming, void*,
                      unsigned, const void*, size_t, const void*, size_t, const void*);
    // ucp_stream_send_nbx
    static void audit(const typename UCXPolicy::gotcha_data&, audit::incoming, void*,
                      const void*, size_t, const void*);
    // ucp_stream_recv_nbx
    static void audit(const typename UCXPolicy::gotcha_data&, audit::incoming, void*,
                      void*, size_t, size_t*, const void*);
    // outgoing
    static void audit(const typename UCXPolicy::gotcha_data&, audit::outgoing, void*);
    static void audit(const typename UCXPolicy::gotcha_data&, audit::outgoing, int);
};

namespace detail
{
template <typename UCXPolicy>
auto&
get_ucx_gotcha()
{
    static auto _v = tim::lightweight_tuple<typename UCXPolicy::ucx_gotcha_t>{};
    return _v;
}
}  // namespace detail

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::configure()
{
    static_assert(traits::has_ucx_bundle_t<UCXPolicy>::value,
                  "UCXPolicy must have a ucx_bundle_t type");
    static_assert(traits::has_ucx_gotcha_t<UCXPolicy>::value,
                  "UCXPolicy must have a ucx_gotcha_t type");

    using ucx_gotcha_t = typename UCXPolicy::ucx_gotcha_t;

    for(size_t i = 0; i < ucx_gotcha_t::capacity(); ++i)
    {
        auto* itr = ucx_gotcha_t::at(i);
        if(itr) itr->verbose = -1;
    }

    ucx_gotcha_t::get_initializer() = []() {
        // Active Message
        ucx_gotcha_t::template configure<0, void*, void*, unsigned, void*, size_t, void*,
                                         size_t, unsigned, void*>("ucp_am_send_nb");
        ucx_gotcha_t::template configure<1, void*, void*, unsigned, const void*, size_t,
                                         const void*, size_t, const void*>(
            "ucp_am_send_nbx");
        ucx_gotcha_t::template configure<2, void*, void*, void*, size_t, void*>(
            "ucp_am_recv_data_nbx");
        ucx_gotcha_t::template configure<3, void, void*, void*>("ucp_am_data_release");

        // Atomic operations
        ucx_gotcha_t::template configure<4, void*, void*, uint32_t, uint64_t, void*>(
            "ucp_atomic_add32");
        ucx_gotcha_t::template configure<5, void*, void*, uint64_t, uint64_t, void*>(
            "ucp_atomic_add64");
        ucx_gotcha_t::template configure<6, void*, void*, uint32_t, uint32_t, uint64_t,
                                         void*>("ucp_atomic_cswap32");
        ucx_gotcha_t::template configure<7, void*, void*, uint64_t, uint64_t, uint64_t,
                                         void*>("ucp_atomic_cswap64");
        ucx_gotcha_t::template configure<8, void*, void*, uint32_t, uint64_t, void*,
                                         void*>("ucp_atomic_fadd32");
        ucx_gotcha_t::template configure<9, void*, void*, uint64_t, uint64_t, void*,
                                         void*>("ucp_atomic_fadd64");
        ucx_gotcha_t::template configure<10, void*, void*, uint32_t, uint64_t, void*,
                                         void*>("ucp_atomic_swap32");
        ucx_gotcha_t::template configure<11, void*, void*, uint64_t, uint64_t, void*,
                                         void*>("ucp_atomic_swap64");
        ucx_gotcha_t::template configure<12, int, void*, int, uint64_t, const void*,
                                         size_t, void*>("ucp_atomic_post");
        ucx_gotcha_t::template configure<13, void*, void*, int, uint64_t, void*, size_t,
                                         void*, void*>("ucp_atomic_fetch_nb");
        ucx_gotcha_t::template configure<14, void*, void*, unsigned, void*, void*, size_t,
                                         uint64_t, void*>("ucp_atomic_op_nbx");

        // Cleanup and config
        ucx_gotcha_t::template configure<15, void, void*>("ucp_cleanup");
        ucx_gotcha_t::template configure<16, int, void*, const char*, const char*,
                                         const char*>("ucp_config_modify");
        ucx_gotcha_t::template configure<17, int, const char*, const char*, void**>(
            "ucp_config_read");
        ucx_gotcha_t::template configure<18, void, void*>("ucp_config_release");

        // Connection management
        ucx_gotcha_t::template configure<19, void*, void*, unsigned>("ucp_disconnect_nb");

        // Datatype
        ucx_gotcha_t::template configure<20, int, void*, void**>("ucp_dt_create_generic");
        ucx_gotcha_t::template configure<21, void, void*>("ucp_dt_destroy");

        // Endpoint
        ucx_gotcha_t::template configure<22, int, void*, const void*, void**>(
            "ucp_ep_create");
        ucx_gotcha_t::template configure<23, void, void*>("ucp_ep_destroy");
        ucx_gotcha_t::template configure<24, void*, void*, const void*>(
            "ucp_ep_modify_nb");
        ucx_gotcha_t::template configure<25, void*, void*, const void*>(
            "ucp_ep_close_nbx");
        ucx_gotcha_t::template configure<26, int, void*>("ucp_ep_flush");
        ucx_gotcha_t::template configure<27, void*, void*, unsigned, void*>(
            "ucp_ep_flush_nb");
        ucx_gotcha_t::template configure<28, void*, void*, const void*>(
            "ucp_ep_flush_nbx");

        // Listener
        ucx_gotcha_t::template configure<29, int, void*, const void*, void**>(
            "ucp_listener_create");
        ucx_gotcha_t::template configure<30, void, void*>("ucp_listener_destroy");
        ucx_gotcha_t::template configure<31, int, void*, void*>("ucp_listener_query");
        ucx_gotcha_t::template configure<32, int, void*, void*>("ucp_listener_reject");

        // Memory
        ucx_gotcha_t::template configure<33, int, void*, void*, size_t, int>(
            "ucp_mem_advise");
        ucx_gotcha_t::template configure<34, int, void*, const void*, void**>(
            "ucp_mem_map");
        ucx_gotcha_t::template configure<35, int, void*, void*>("ucp_mem_unmap");
        ucx_gotcha_t::template configure<36, int, void*, void*>("ucp_mem_query");

        // Put/Get operations
        ucx_gotcha_t::template configure<37, int, void*, const void*, size_t, uint64_t,
                                         void*>("ucp_put");
        ucx_gotcha_t::template configure<38, int, void*, void*, size_t, uint64_t, void*>(
            "ucp_get");
        ucx_gotcha_t::template configure<39, int, void*, const void*, size_t, uint64_t,
                                         void*>("ucp_put_nbi");
        ucx_gotcha_t::template configure<40, int, void*, void*, size_t, uint64_t, void*>(
            "ucp_get_nbi");
        ucx_gotcha_t::template configure<41, void*, void*, const void*, size_t, uint64_t,
                                         void*, void*>("ucp_put_nb");
        ucx_gotcha_t::template configure<42, void*, void*, void*, size_t, uint64_t, void*,
                                         void*>("ucp_get_nb");
        ucx_gotcha_t::template configure<43, void*, void*, const void*, size_t, uint64_t,
                                         void*, const void*>("ucp_put_nbx");
        ucx_gotcha_t::template configure<44, void*, void*, void*, size_t, uint64_t, void*,
                                         const void*>("ucp_get_nbx");

        // Request
        ucx_gotcha_t::template configure<45, void*, void*>("ucp_request_alloc");
        ucx_gotcha_t::template configure<46, void, void*, void*>("ucp_request_cancel");
        ucx_gotcha_t::template configure<47, int, void*>("ucp_request_is_completed");

        // Remote key
        ucx_gotcha_t::template configure<48, void, void*>("ucp_rkey_buffer_release");
        ucx_gotcha_t::template configure<49, void, void*>("ucp_rkey_destroy");
        ucx_gotcha_t::template configure<50, int, void*, void*, void**, size_t*>(
            "ucp_rkey_pack");
        ucx_gotcha_t::template configure<51, int, void*, void*, void**>("ucp_rkey_ptr");

        // Stream
        ucx_gotcha_t::template configure<52, void, void*, void*>(
            "ucp_stream_data_release");
        ucx_gotcha_t::template configure<53, void*, void*, void*, size_t, size_t*,
                                         unsigned, void*>("ucp_stream_recv_data_nb");
        ucx_gotcha_t::template configure<54, void*, void*, const void*, size_t, void*>(
            "ucp_stream_send_nb");
        ucx_gotcha_t::template configure<55, void*, void*, void*, size_t, size_t*, void*>(
            "ucp_stream_recv_nb");
        ucx_gotcha_t::template configure<56, void*, void*, const void*, size_t,
                                         const void*>("ucp_stream_send_nbx");
        ucx_gotcha_t::template configure<57, void*, void*, void*, size_t, size_t*,
                                         const void*>("ucp_stream_recv_nbx");
        ucx_gotcha_t::template configure<58, void*, void*>("ucp_stream_worker_poll");

        // Tag matching
        ucx_gotcha_t::template configure<59, void*, void*, void*, void*, size_t, void*,
                                         void*>("ucp_tag_msg_recv_nb");
        ucx_gotcha_t::template configure<60, void*, void*, void*, void*, size_t,
                                         const void*>("ucp_tag_msg_recv_nbx");
        ucx_gotcha_t::template configure<61, void*, void*, const void*, size_t, void*,
                                         void*>("ucp_tag_send_nbr");
        ucx_gotcha_t::template configure<62, void*, void*, void*, size_t, void*, void*,
                                         void*>("ucp_tag_recv_nbr");
        ucx_gotcha_t::template configure<63, void*, void*, const void*, size_t, void*,
                                         void*>("ucp_tag_send_nb");
        ucx_gotcha_t::template configure<64, void*, void*, void*, size_t, void*, void*,
                                         void*>("ucp_tag_recv_nb");
        ucx_gotcha_t::template configure<65, void*, void*, const void*, size_t, uint64_t,
                                         const void*>("ucp_tag_send_nbx");
        ucx_gotcha_t::template configure<66, void*, void*, void*, size_t, uint64_t,
                                         uint64_t, const void*>("ucp_tag_recv_nbx");
        ucx_gotcha_t::template configure<67, void*, void*, const void*, size_t, uint64_t,
                                         void*>("ucp_tag_send_sync_nb");
        ucx_gotcha_t::template configure<68, void*, void*, const void*, size_t, uint64_t,
                                         const void*>("ucp_tag_send_sync_nbx");

        // Worker
        ucx_gotcha_t::template configure<69, int, void*, const void*, void**>(
            "ucp_worker_create");
        ucx_gotcha_t::template configure<70, void, void*>("ucp_worker_destroy");
        ucx_gotcha_t::template configure<71, int, void*, void**, size_t*>(
            "ucp_worker_get_address");
        ucx_gotcha_t::template configure<72, int, void*, int*>("ucp_worker_get_efd");
        ucx_gotcha_t::template configure<73, int, void*>("ucp_worker_arm");
        ucx_gotcha_t::template configure<74, int, void*>("ucp_worker_fence");
        ucx_gotcha_t::template configure<75, int, void*>("ucp_worker_wait");
        ucx_gotcha_t::template configure<76, int, void*>("ucp_worker_signal");
        ucx_gotcha_t::template configure<77, int, void*, void*, size_t, void*>(
            "ucp_worker_wait_mem");
        ucx_gotcha_t::template configure<78, int, void*>("ucp_worker_flush");
        ucx_gotcha_t::template configure<79, void*, void*, unsigned, void*>(
            "ucp_worker_flush_nb");
        ucx_gotcha_t::template configure<80, void*, void*, unsigned, void*>(
            "ucp_worker_flush_nbx");
        ucx_gotcha_t::template configure<81, int, void*, unsigned, void*, void*, void*>(
            "ucp_worker_set_am_handler");
        ucx_gotcha_t::template configure<82, int, void*, const void*>(
            "ucp_worker_set_am_recv_handler");
        ucx_gotcha_t::template configure<83, unsigned, void*>("ucp_worker_progress");

        // UCT Active Message (low-level transport)
        ucx_gotcha_t::template configure<84, ssize_t, void*, unsigned, void*, void*>(
            "uct_ep_am_bcopy");
        ucx_gotcha_t::template configure<85, ssize_t, void*, unsigned, const void*,
                                         unsigned, const void*, size_t, void*>(
            "uct_ep_am_zcopy");
        ucx_gotcha_t::template configure<86, ssize_t, void*, unsigned, uint64_t,
                                         const void*, unsigned>("uct_ep_am_short");
        ucx_gotcha_t::template configure<87, unsigned, void*>("uct_iface_progress");
        ucx_gotcha_t::template configure<88, int, void*, unsigned, void*, void*,
                                         unsigned>("uct_iface_set_am_handler");
    };
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::shutdown()
{
    using ucx_gotcha_t = typename UCXPolicy::ucx_gotcha_t;
    ucx_gotcha_t::disable();
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::start()
{
    using ucx_gotcha_t = typename UCXPolicy::ucx_gotcha_t;

    if(!detail::get_ucx_gotcha<UCXPolicy>()
            .template get<ucx_gotcha_t>()
            ->get_is_running())
    {
        configure();
        detail::get_ucx_gotcha<UCXPolicy>().template get<ucx_gotcha_t>()->start();
    }
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::stop()
{}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::incoming, void* arg1, const void* arg2, size_t arg3,
                             uint64_t arg4, const void* arg5)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "tag", arg4, "param",
                                      arg5);
    UCXPolicy::comm_data::audit(_data, audit::incoming{}, arg1, arg2, arg3, arg4, arg5);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::incoming, void* arg1, void* arg2, size_t arg3,
                             uint64_t arg4, uint64_t arg5, const void* arg6)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "worker", arg1,
                                      "buffer", arg2, "count", arg3, "tag", arg4,
                                      "tag_mask", arg5, "param", arg6);
    UCXPolicy::comm_data::audit(_data, audit::incoming{}, arg1, arg2, arg3, arg4, arg5,
                                arg6);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::incoming, void* arg1, const void* arg2, size_t arg3,
                             uint64_t arg4, void* arg5, const void* arg6)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "remote_addr", arg4,
                                      "rkey", arg5, "param", arg6);
    UCXPolicy::comm_data::audit(_data, audit::incoming{}, arg1, arg2, arg3, arg4, arg5,
                                arg6);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::incoming, void* arg1, void* arg2, size_t arg3,
                             uint64_t arg4, void* arg5, const void* arg6)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "remote_addr", arg4,
                                      "rkey", arg5, "param", arg6);
    UCXPolicy::comm_data::audit(_data, audit::incoming{}, arg1, arg2, arg3, arg4, arg5,
                                arg6);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::incoming, void* arg1, unsigned arg2, const void* arg3,
                             size_t arg4, const void* arg5, size_t arg6, const void* arg7)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1, "id",
                                      arg2, "header", arg3, "header_length", arg4,
                                      "buffer", arg5, "count", arg6, "param", arg7);
    UCXPolicy::comm_data::audit(_data, audit::incoming{}, arg1, arg2, arg3, arg4, arg5,
                                arg6, arg7);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::incoming, void* arg1, const void* arg2, size_t arg3,
                             const void* arg4)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "param", arg4);
    UCXPolicy::comm_data::audit(_data, audit::incoming{}, arg1, arg2, arg3, arg4);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::incoming, void* arg1, void* arg2, size_t arg3,
                             size_t* arg4, const void* arg5)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "length", arg4,
                                      "param", arg5);
    UCXPolicy::comm_data::audit(_data, audit::incoming{}, arg1, arg2, arg3, arg4, arg5);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::outgoing, void* ret)
{
    UCXPolicy::category_region::stop(std::string_view{ _data.tool_id }, "return", ret);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const typename UCXPolicy::gotcha_data& _data,
                             audit::outgoing, int ret)
{
    UCXPolicy::category_region::stop(std::string_view{ _data.tool_id }, "return", ret);
}

}  // namespace component

struct DefaultUCXPolicy
{
    using comm_data       = component::comm_data;
    using gotcha_data     = tim::component::gotcha_data;
    using category_region = component::category_region<category::ucx>;

    using component_t = component::ucx_gotcha<DefaultUCXPolicy>;

    using ucx_bundle_t = tim::component_bundle<category::ucx, component_t, comm_data>;
    using ucx_gotcha_t =
        tim::component::gotcha<component_t::gotcha_capacity, ucx_bundle_t, category::ucx>;
};

}  // namespace rocprofsys
