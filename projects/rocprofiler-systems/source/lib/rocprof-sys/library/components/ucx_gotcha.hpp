// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <timemory/components/base/declaration.hpp>
#include <timemory/utility/types.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <utility>

namespace rocprofsys::component::ucx_concepts
{
template <typename Policy>
concept HasCommData = requires { typename Policy::comm_data; };

template <typename Policy>
concept HasGotchaData = requires { typename Policy::gotcha_data; };

template <typename Policy>
concept HasCategoryRegion = requires { typename Policy::category_region; };

template <typename Policy>
concept HasUcxBundle = requires { typename Policy::ucx_bundle_t; };

template <typename Policy>
concept HasUcxGotcha = requires { typename Policy::ucx_gotcha_t; };
}  // namespace rocprofsys::component::ucx_concepts

namespace rocprofsys
{
namespace component
{
template <typename UCXPolicy>
struct ucx_gotcha : tim::component::base<ucx_gotcha<UCXPolicy>, void>
{
    static_assert(ucx_concepts::HasCommData<UCXPolicy>,
                  "UCXPolicy must have a comm_data type");
    static_assert(ucx_concepts::HasGotchaData<UCXPolicy>,
                  "UCXPolicy must have a gotcha_data type");
    static_assert(ucx_concepts::HasCategoryRegion<UCXPolicy>,
                  "UCXPolicy must have a category_region type");

    static constexpr size_t gotcha_capacity = 100;

    using gotcha_data = typename UCXPolicy::gotcha_data;

    ucx_gotcha()                                 = default;
    ucx_gotcha(const ucx_gotcha&)                = default;
    ucx_gotcha& operator=(const ucx_gotcha&)     = default;
    ucx_gotcha(ucx_gotcha&&) noexcept            = default;
    ucx_gotcha& operator=(ucx_gotcha&&) noexcept = default;
    ~ucx_gotcha() noexcept                       = default;

    // string id for component
    static std::string label() { return "ucx_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static void start();
    static void stop();

    static void pause();
    static void resume();

    // Generic template audit function for UCX operations with void* parameters
    template <typename... Args>
    static void audit(const gotcha_data& _data, tim::audit::incoming, Args...)
    {
        UCXPolicy::category_region::start(std::string_view{ _data.tool_id });
    }

    // Specific audit functions for tag operations (with std::uint64_t tags)
    // ucp_tag_send_nbx: (void* ep, const void* buffer, size_t count, std::uint64_t tag,
    // const void* param)
    static void audit(const gotcha_data&, tim::audit::incoming, void*, const void*,
                      size_t, std::uint64_t, const void*);
    // ucp_tag_recv_nbx: (void* worker, void* buffer, size_t count, std::uint64_t tag,
    // std::uint64_t tag_mask, const void* param)
    static void audit(const gotcha_data&, tim::audit::incoming, void*, void*, size_t,
                      std::uint64_t, std::uint64_t, const void*);

    // RMA operations
    // ucp_put_nbx: (void* ep, const void* buffer, size_t count, std::uint64_t
    // remote_addr, void* rkey, const void* param)
    static void audit(const gotcha_data&, tim::audit::incoming, void*, const void*,
                      size_t, std::uint64_t, void*, const void*);
    // ucp_get_nbx: (void* ep, void* buffer, size_t count, std::uint64_t remote_addr,
    // void* rkey, const void* param)
    static void audit(const gotcha_data&, tim::audit::incoming, void*, void*, size_t,
                      std::uint64_t, void*, const void*);

    // Active message send
    // ucp_am_send_nbx: (void* ep, unsigned id, const void* header, size_t header_length,
    // const void* buffer, size_t count, const void* param)
    static void audit(const gotcha_data&, tim::audit::incoming, void*, unsigned,
                      const void*, size_t, const void*, size_t, const void*);

    // Stream operations
    // ucp_stream_send_nbx: (void* ep, const void* buffer, size_t count, const void*
    // param)
    static void audit(const gotcha_data&, tim::audit::incoming, void*, const void*,
                      size_t, const void*);
    // ucp_stream_recv_nbx: (void* ep, void* buffer, size_t count, size_t* length, const
    // void* param)
    static void audit(const gotcha_data&, tim::audit::incoming, void*, void*, size_t,
                      size_t*, const void*);

    // Outgoing audit. These overloads must match the return types of the wrapped APIs
    static void audit(const gotcha_data&, tim::audit::outgoing);
    static void audit(const gotcha_data&, tim::audit::outgoing, void*);
    static void audit(const gotcha_data&, tim::audit::outgoing, int);
    static void audit(const gotcha_data&, tim::audit::outgoing, unsigned);
    static void audit(const gotcha_data&, tim::audit::outgoing, long);

private:
    static std::mutex s_mutex;
};

namespace detail
{
template <typename UCXPolicy>
auto&
get_ucx_gotcha()
{
    static auto _v = typename UCXPolicy::gotcha_bundle_t{};
    return _v;
}
}  // namespace detail

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::configure()
{
    static_assert(ucx_concepts::HasUcxBundle<UCXPolicy>,
                  "UCXPolicy must have a ucx_bundle_t type");
    static_assert(ucx_concepts::HasUcxGotcha<UCXPolicy>,
                  "UCXPolicy must have a ucx_gotcha_t type");

    using ucx_gotcha_t  = typename UCXPolicy::ucx_gotcha_t;
    using gotcha_data_t = typename UCXPolicy::gotcha_data;

    for(size_t i = 0; i < ucx_gotcha_t::capacity(); ++i)
    {
        auto* itr = static_cast<gotcha_data_t*>(ucx_gotcha_t::at(i));
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
        ucx_gotcha_t::template configure<4, void*, void*, std::uint32_t, std::uint64_t,
                                         void*>("ucp_atomic_add32");
        ucx_gotcha_t::template configure<5, void*, void*, std::uint64_t, std::uint64_t,
                                         void*>("ucp_atomic_add64");
        ucx_gotcha_t::template configure<6, void*, void*, std::uint32_t, std::uint32_t,
                                         std::uint64_t, void*>("ucp_atomic_cswap32");
        ucx_gotcha_t::template configure<7, void*, void*, std::uint64_t, std::uint64_t,
                                         std::uint64_t, void*>("ucp_atomic_cswap64");
        ucx_gotcha_t::template configure<8, void*, void*, std::uint32_t, std::uint64_t,
                                         void*, void*>("ucp_atomic_fadd32");
        ucx_gotcha_t::template configure<9, void*, void*, std::uint64_t, std::uint64_t,
                                         void*, void*>("ucp_atomic_fadd64");
        ucx_gotcha_t::template configure<10, void*, void*, std::uint32_t, std::uint64_t,
                                         void*, void*>("ucp_atomic_swap32");
        ucx_gotcha_t::template configure<11, void*, void*, std::uint64_t, std::uint64_t,
                                         void*, void*>("ucp_atomic_swap64");
        ucx_gotcha_t::template configure<12, int, void*, int, std::uint64_t, const void*,
                                         size_t, void*>("ucp_atomic_post");
        ucx_gotcha_t::template configure<13, void*, void*, int, std::uint64_t, void*,
                                         size_t, void*, void*>("ucp_atomic_fetch_nb");
        ucx_gotcha_t::template configure<14, void*, void*, unsigned, void*, void*, size_t,
                                         std::uint64_t, void*>("ucp_atomic_op_nbx");

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
        ucx_gotcha_t::template configure<37, int, void*, const void*, size_t,
                                         std::uint64_t, void*>("ucp_put");
        ucx_gotcha_t::template configure<38, int, void*, void*, size_t, std::uint64_t,
                                         void*>("ucp_get");
        ucx_gotcha_t::template configure<39, int, void*, const void*, size_t,
                                         std::uint64_t, void*>("ucp_put_nbi");
        ucx_gotcha_t::template configure<40, int, void*, void*, size_t, std::uint64_t,
                                         void*>("ucp_get_nbi");
        ucx_gotcha_t::template configure<41, void*, void*, const void*, size_t,
                                         std::uint64_t, void*, void*>("ucp_put_nb");
        ucx_gotcha_t::template configure<42, void*, void*, void*, size_t, std::uint64_t,
                                         void*, void*>("ucp_get_nb");
        ucx_gotcha_t::template configure<43, void*, void*, const void*, size_t,
                                         std::uint64_t, void*, const void*>(
            "ucp_put_nbx");
        ucx_gotcha_t::template configure<44, void*, void*, void*, size_t, std::uint64_t,
                                         void*, const void*>("ucp_get_nbx");

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
        ucx_gotcha_t::template configure<65, void*, void*, const void*, size_t,
                                         std::uint64_t, const void*>("ucp_tag_send_nbx");
        ucx_gotcha_t::template configure<66, void*, void*, void*, size_t, std::uint64_t,
                                         std::uint64_t, const void*>("ucp_tag_recv_nbx");
        ucx_gotcha_t::template configure<67, void*, void*, const void*, size_t,
                                         std::uint64_t, void*>("ucp_tag_send_sync_nb");
        ucx_gotcha_t::template configure<68, void*, void*, const void*, size_t,
                                         std::uint64_t, const void*>(
            "ucp_tag_send_sync_nbx");

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
        ucx_gotcha_t::template configure<86, ssize_t, void*, unsigned, std::uint64_t,
                                         const void*, unsigned>("uct_ep_am_short");
        ucx_gotcha_t::template configure<87, unsigned, void*>("uct_iface_progress");
        ucx_gotcha_t::template configure<88, int, void*, unsigned, void*, void*,
                                         unsigned>("uct_iface_set_am_handler");

        // Legacy UCX function variants that might be used on older systems
        ucx_gotcha_t::template configure<89, void*, void*, const void*, size_t, void*>(
            "ucp_tag_send");
        ucx_gotcha_t::template configure<90, void*, void*, void*, size_t, void*, void*>(
            "ucp_tag_recv");
        ucx_gotcha_t::template configure<91, void*, void*, const void*, size_t, int, int,
                                         void*>("ucp_send");
        ucx_gotcha_t::template configure<92, void*, void*, void*, size_t, int, int,
                                         void*>("ucp_recv");
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
        // Initializing comm_data metadata
        UCXPolicy::comm_data::start();
        detail::get_ucx_gotcha<UCXPolicy>().template get<ucx_gotcha_t>()->start();
    }
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::stop()
{}

template <typename UCXPolicy>
std::mutex ucx_gotcha<UCXPolicy>::s_mutex = {};

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::pause()
{
    std::scoped_lock<std::mutex> _lk{ s_mutex };
    using ucx_gotcha_t = typename UCXPolicy::ucx_gotcha_t;
    ucx_gotcha_t::set_ready(false);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::resume()
{
    std::scoped_lock<std::mutex> _lk{ s_mutex };
    using ucx_gotcha_t = typename UCXPolicy::ucx_gotcha_t;
    ucx_gotcha_t::set_ready(true);
}

// Specific audit functions for tag operations
template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::incoming, void* arg1,
                             const void* arg2, size_t arg3, std::uint64_t arg4,
                             const void* arg5)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "tag", arg4, "param",
                                      arg5);

    // Also trigger communication data tracking
    UCXPolicy::comm_data::audit(_data, tim::audit::incoming{}, arg1, arg2, arg3, arg4,
                                arg5);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::incoming, void* arg1,
                             void* arg2, size_t arg3, std::uint64_t arg4,
                             std::uint64_t arg5, const void* arg6)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "worker", arg1,
                                      "buffer", arg2, "count", arg3, "tag", arg4,
                                      "tag_mask", arg5, "param", arg6);

    // Also trigger communication data tracking
    UCXPolicy::comm_data::audit(_data, tim::audit::incoming{}, arg1, arg2, arg3, arg4,
                                arg5, arg6);
}

// RMA operations
template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::incoming, void* arg1,
                             const void* arg2, size_t arg3, std::uint64_t arg4,
                             void* arg5, const void* arg6)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "remote_addr", arg4,
                                      "rkey", arg5, "param", arg6);

    // Also trigger communication data tracking
    UCXPolicy::comm_data::audit(_data, tim::audit::incoming{}, arg1, arg2, arg3, arg4,
                                arg5, arg6);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::incoming, void* arg1,
                             void* arg2, size_t arg3, std::uint64_t arg4, void* arg5,
                             const void* arg6)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "remote_addr", arg4,
                                      "rkey", arg5, "param", arg6);

    // Also trigger communication data tracking
    UCXPolicy::comm_data::audit(_data, tim::audit::incoming{}, arg1, arg2, arg3, arg4,
                                arg5, arg6);
}

// Active message send
template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::incoming, void* arg1,
                             unsigned arg2, const void* arg3, size_t arg4,
                             const void* arg5, size_t arg6, const void* arg7)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1, "id",
                                      arg2, "header", arg3, "header_length", arg4,
                                      "buffer", arg5, "count", arg6, "param", arg7);

    // Also trigger communication data tracking
    UCXPolicy::comm_data::audit(_data, tim::audit::incoming{}, arg1, arg2, arg3, arg4,
                                arg5, arg6, arg7);
}

// Stream operations
template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::incoming, void* arg1,
                             const void* arg2, size_t arg3, const void* arg4)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "param", arg4);

    // Also trigger communication data tracking
    UCXPolicy::comm_data::audit(_data, tim::audit::incoming{}, arg1, arg2, arg3, arg4);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::incoming, void* arg1,
                             void* arg2, size_t arg3, size_t* arg4, const void* arg5)
{
    UCXPolicy::category_region::start(std::string_view{ _data.tool_id }, "ep", arg1,
                                      "buffer", arg2, "count", arg3, "length", arg4,
                                      "param", arg5);

    // Also trigger communication data tracking
    UCXPolicy::comm_data::audit(_data, tim::audit::incoming{}, arg1, arg2, arg3, arg4,
                                arg5);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::outgoing)
{
    UCXPolicy::category_region::stop(std::string_view{ _data.tool_id });
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::outgoing, void* ret)
{
    UCXPolicy::category_region::stop(std::string_view{ _data.tool_id }, "return", ret);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::outgoing, int ret)
{
    UCXPolicy::category_region::stop(std::string_view{ _data.tool_id }, "return", ret);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::outgoing, unsigned ret)
{
    UCXPolicy::category_region::stop(std::string_view{ _data.tool_id }, "return", ret);
}

template <typename UCXPolicy>
void
ucx_gotcha<UCXPolicy>::audit(const gotcha_data& _data, tim::audit::outgoing, long ret)
{
    UCXPolicy::category_region::stop(std::string_view{ _data.tool_id }, "return", ret);
}

}  // namespace component
}  // namespace rocprofsys
