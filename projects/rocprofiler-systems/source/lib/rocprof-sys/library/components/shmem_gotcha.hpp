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
#include "core/config.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"
#include "library/components/category_region.hpp"
#include "library/components/comm_data.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>
#include <timemory/utility/types.hpp>

#include <cstddef>
#include <cstdlib>
#include <set>
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
struct has_shmem_bundle_t : std::false_type
{};

template <typename Policy>
struct has_shmem_bundle_t<Policy, std::void_t<typename Policy::shmem_bundle_t>>
: std::true_type
{};

template <typename Policy, typename = void>
struct has_shmem_gotcha_t : std::false_type
{};

template <typename Policy>
struct has_shmem_gotcha_t<Policy, std::void_t<typename Policy::shmem_gotcha_t>>
: std::true_type
{};

}  // namespace traits

template <typename SHMEMPolicy>
struct shmem_gotcha : tim::component::base<shmem_gotcha<SHMEMPolicy>, void>
{
    static_assert(traits::has_comm_data<SHMEMPolicy>::value,
                  "SHMEMPolicy must have a comm_data type");
    static_assert(traits::has_gotcha_data<SHMEMPolicy>::value,
                  "SHMEMPolicy must have a gotcha_data type");
    static_assert(traits::has_category_region<SHMEMPolicy>::value,
                  "SHMEMPolicy must have a category_region type");

    static constexpr size_t gotcha_capacity = 180;

    ROCPROFSYS_DEFAULT_OBJECT(shmem_gotcha<SHMEMPolicy>)

    static std::string label() { return "shmem_gotcha"; }

    static void configure();
    static void shutdown();

    static void start();
    static void stop();

    template <typename... Args>
    static void audit(const typename SHMEMPolicy::gotcha_data& _data, audit::incoming,
                      Args...)
    {
        SHMEMPolicy::category_region::start(std::string_view{ _data.tool_id });
    }

    static void audit(const typename SHMEMPolicy::gotcha_data&, audit::outgoing, void*);
    static void audit(const typename SHMEMPolicy::gotcha_data&, audit::outgoing, int);
};

namespace detail
{
template <typename SHMEMPolicy>
auto&
get_shmem_gotcha()
{
    static auto _v = tim::lightweight_tuple<typename SHMEMPolicy::shmem_gotcha_t>{};
    return _v;
}
}  // namespace detail

template <typename SHMEMPolicy>
void
shmem_gotcha<SHMEMPolicy>::configure()
{
    static_assert(traits::has_shmem_bundle_t<SHMEMPolicy>::value,
                  "SHMEMPolicy must have a shmem_bundle_t type");
    static_assert(traits::has_shmem_gotcha_t<SHMEMPolicy>::value,
                  "SHMEMPolicy must have a shmem_gotcha_t type");

    using shmem_gotcha_t = typename SHMEMPolicy::shmem_gotcha_t;

    shmem_gotcha_t::get_initializer() = []() {
        shmem_gotcha_t::template configure<0, void>("shmem_init");
        shmem_gotcha_t::template configure<1, void>("shmem_finalize");
        shmem_gotcha_t::template configure<2, void, int>("start_pes");
        shmem_gotcha_t::template configure<3, int>("shmem_my_pe");
        shmem_gotcha_t::template configure<4, int>("shmem_n_pes");
        shmem_gotcha_t::template configure<5, int>("shmem_num_pes");

        shmem_gotcha_t::template configure<6, void>("shmem_quiet");
        shmem_gotcha_t::template configure<7, void>("shmem_fence");
        shmem_gotcha_t::template configure<8, void>("shmem_barrier_all");
        shmem_gotcha_t::template configure<9, void, int, int, int>("shmem_barrier");

        shmem_gotcha_t::template configure<10, void, void*, const void*, size_t, int>(
            "shmem_putmem");
        shmem_gotcha_t::template configure<11, void, void*, const void*, size_t, int>(
            "shmem_getmem");
        shmem_gotcha_t::template configure<12, void, void*, const void*, size_t, int>(
            "shmem_put_nbi");
        shmem_gotcha_t::template configure<13, void, void*, const void*, size_t, int>(
            "shmem_get_nbi");

        shmem_gotcha_t::template configure<14, void, void*, const void*, size_t, int>(
            "shmem_put32");
        shmem_gotcha_t::template configure<15, void, void*, const void*, size_t, int>(
            "shmem_get32");
        shmem_gotcha_t::template configure<16, void, void*, const void*, size_t, int>(
            "shmem_put64");
        shmem_gotcha_t::template configure<17, void, void*, const void*, size_t, int>(
            "shmem_get64");
        shmem_gotcha_t::template configure<18, void, void*, const void*, size_t, int>(
            "shmem_put128");
        shmem_gotcha_t::template configure<19, void, void*, const void*, size_t, int>(
            "shmem_get128");
        shmem_gotcha_t::template configure<20, void, void*, const void*, size_t, int>(
            "shmem_put8");
        shmem_gotcha_t::template configure<21, void, void*, const void*, size_t, int>(
            "shmem_get8");
        shmem_gotcha_t::template configure<22, void, void*, const void*, size_t, int>(
            "shmem_put16");
        shmem_gotcha_t::template configure<23, void, void*, const void*, size_t, int>(
            "shmem_get16");

        shmem_gotcha_t::template configure<24, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iput32");
        shmem_gotcha_t::template configure<25, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iget32");
        shmem_gotcha_t::template configure<26, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iput64");
        shmem_gotcha_t::template configure<27, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iget64");
        shmem_gotcha_t::template configure<28, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iput128");
        shmem_gotcha_t::template configure<29, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iget128");
        shmem_gotcha_t::template configure<30, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iput8");
        shmem_gotcha_t::template configure<31, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iget8");
        shmem_gotcha_t::template configure<32, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iput16");
        shmem_gotcha_t::template configure<33, void, void*, const void*, ptrdiff_t,
                                           ptrdiff_t, size_t, int>("shmem_iget16");

        shmem_gotcha_t::template configure<34, void, void*, const void*, size_t, int, int,
                                           int, int, long*>("shmem_broadcast32");
        shmem_gotcha_t::template configure<35, void, void*, const void*, size_t, int, int,
                                           int, int, long*>("shmem_broadcast64");
        shmem_gotcha_t::template configure<36, void, void*, const void*, size_t, int, int,
                                           int, int, long*>("shmem_broadcast8");
        shmem_gotcha_t::template configure<37, void, void*, const void*, size_t, int, int,
                                           int, int, long*>("shmem_broadcast4");
        shmem_gotcha_t::template configure<38, void, void*, const void*, size_t, int, int,
                                           int, int, long*>("shmem_broadcast128");

        shmem_gotcha_t::template configure<39, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_collect32");
        shmem_gotcha_t::template configure<40, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_collect64");
        shmem_gotcha_t::template configure<41, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_collect8");
        shmem_gotcha_t::template configure<42, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_collect4");
        shmem_gotcha_t::template configure<43, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_collect128");

        shmem_gotcha_t::template configure<44, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_fcollect32");
        shmem_gotcha_t::template configure<45, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_fcollect64");
        shmem_gotcha_t::template configure<46, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_fcollect8");
        shmem_gotcha_t::template configure<47, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_fcollect4");
        shmem_gotcha_t::template configure<48, void, void*, const void*, size_t, int, int,
                                           int, long*>("shmem_fcollect128");

        shmem_gotcha_t::template configure<49, void, void*, const void*, size_t, size_t,
                                           size_t, int, int, int, long*>(
            "shmem_alltoall32");
        shmem_gotcha_t::template configure<50, void, void*, const void*, size_t, size_t,
                                           size_t, int, int, int, long*>(
            "shmem_alltoall64");

        shmem_gotcha_t::template configure<51, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_sum_to_all32");
        shmem_gotcha_t::template configure<52, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_sum_to_all64");
        shmem_gotcha_t::template configure<53, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_min_to_all32");
        shmem_gotcha_t::template configure<54, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_min_to_all64");
        shmem_gotcha_t::template configure<55, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_max_to_all32");
        shmem_gotcha_t::template configure<56, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_max_to_all64");
        shmem_gotcha_t::template configure<57, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_and_to_all32");
        shmem_gotcha_t::template configure<58, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_and_to_all64");
        shmem_gotcha_t::template configure<59, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_or_to_all32");
        shmem_gotcha_t::template configure<60, void, void*, const void*, size_t, int, int,
                                           int, void*, long*>("shmem_or_to_all64");

        shmem_gotcha_t::template configure<61, void, void*, int, int>("shmem_set32");
        shmem_gotcha_t::template configure<62, void, void*, long, int>("shmem_set64");
        shmem_gotcha_t::template configure<63, int, void*, int, int, int>(
            "shmem_cswap32");
        shmem_gotcha_t::template configure<64, long, void*, long, long, int>(
            "shmem_cswap64");
        shmem_gotcha_t::template configure<65, int, void*, int, int, int>("shmem_fadd32");
        shmem_gotcha_t::template configure<66, long, void*, long, int, int>(
            "shmem_fadd64");
        shmem_gotcha_t::template configure<67, int, void*, int>("shmem_finc32");
        shmem_gotcha_t::template configure<68, long, void*, int>("shmem_finc64");
        shmem_gotcha_t::template configure<69, void, void*, int, int>("shmem_add32");
        shmem_gotcha_t::template configure<70, void, void*, long, int>("shmem_add64");
        shmem_gotcha_t::template configure<71, void, void*, int>("shmem_inc32");
        shmem_gotcha_t::template configure<72, void, void*, int>("shmem_inc64");
        shmem_gotcha_t::template configure<73, void, void*, int, int>("shmem_set8");
        shmem_gotcha_t::template configure<74, void, void*, int, int>("shmem_set16");
        shmem_gotcha_t::template configure<75, int, void*, int, int, int>("shmem_swap32");
        shmem_gotcha_t::template configure<76, long, void*, long, long, int>(
            "shmem_swap64");

        shmem_gotcha_t::template configure<77, int, void*, int>("shmem_fetch32");
        shmem_gotcha_t::template configure<78, long, void*, int>("shmem_fetch64");
        shmem_gotcha_t::template configure<79, int, void*, int>("shmem_g32");
        shmem_gotcha_t::template configure<80, long, void*, int>("shmem_g64");

        shmem_gotcha_t::template configure<81, void*, size_t>("shmem_malloc");
        shmem_gotcha_t::template configure<82, void, void*>("shmem_free");
        shmem_gotcha_t::template configure<83, void*, size_t>("shmem_shmalloc");
        shmem_gotcha_t::template configure<84, void, void*>("shmem_shfree");
        shmem_gotcha_t::template configure<85, void*, size_t, size_t>("shmem_align");
        shmem_gotcha_t::template configure<86, void*, void*, size_t>("shmem_realloc");

        shmem_gotcha_t::template configure<87, void, void*, const void*, size_t, int>(
            "shmem_int_put");
        shmem_gotcha_t::template configure<88, void, void*, const void*, size_t, int>(
            "shmem_int_get");
        shmem_gotcha_t::template configure<89, void, void*, const void*, size_t, int>(
            "shmem_long_put");
        shmem_gotcha_t::template configure<90, void, void*, const void*, size_t, int>(
            "shmem_long_get");
        shmem_gotcha_t::template configure<91, void, void*, const void*, size_t, int>(
            "shmem_float_put");
        shmem_gotcha_t::template configure<92, void, void*, const void*, size_t, int>(
            "shmem_float_get");
        shmem_gotcha_t::template configure<93, void, void*, const void*, size_t, int>(
            "shmem_double_put");
        shmem_gotcha_t::template configure<94, void, void*, const void*, size_t, int>(
            "shmem_double_get");
        shmem_gotcha_t::template configure<95, void, void*, const void*, size_t, int>(
            "shmem_short_put");
        shmem_gotcha_t::template configure<96, void, void*, const void*, size_t, int>(
            "shmem_short_get");

        shmem_gotcha_t::template configure<97, int, void*, int, int, int>(
            "shmem_fetch_and_set32");
        shmem_gotcha_t::template configure<98, long, void*, long, int, int>(
            "shmem_fetch_and_set64");
        shmem_gotcha_t::template configure<99, int, void*, int, int, int>(
            "shmem_fetch_and_add32");
        shmem_gotcha_t::template configure<100, long, void*, long, int, int>(
            "shmem_fetch_and_add64");
    };

    shmem_gotcha_t::get_reject_list() = []() {
        std::set<std::string> _reject;
        auto                  reject_list = tim::get_env<std::string>(
            TIMEMORY_SETTINGS_PREFIX "ROCPROFSYS_SHMEM_REJECT_LIST", "");
        for(const auto& itr : tim::delimit(reject_list))
            _reject.insert(itr);
        return _reject;
    };

    shmem_gotcha_t::get_permit_list() = []() {
        std::set<std::string> _permit;
        auto                  permit_list = tim::get_env<std::string>(
            TIMEMORY_SETTINGS_PREFIX "ROCPROFSYS_SHMEM_PERMIT_LIST", "");
        for(const auto& itr : tim::delimit(permit_list))
            _permit.insert(itr);
        return _permit;
    };
}

template <typename SHMEMPolicy>
void
shmem_gotcha<SHMEMPolicy>::shutdown()
{
    using shmem_gotcha_t = typename SHMEMPolicy::shmem_gotcha_t;
    shmem_gotcha_t::disable();
}

template <typename SHMEMPolicy>
void
shmem_gotcha<SHMEMPolicy>::start()
{
    using shmem_gotcha_t = typename SHMEMPolicy::shmem_gotcha_t;

    if(!detail::get_shmem_gotcha<SHMEMPolicy>()
            .template get<shmem_gotcha_t>()
            ->get_is_running())
    {
        configure();
        SHMEMPolicy::comm_data::start();
        detail::get_shmem_gotcha<SHMEMPolicy>().template get<shmem_gotcha_t>()->start();
    }
}

template <typename SHMEMPolicy>
void
shmem_gotcha<SHMEMPolicy>::stop()
{}

template <typename SHMEMPolicy>
void
shmem_gotcha<SHMEMPolicy>::audit(const typename SHMEMPolicy::gotcha_data& _data,
                                 audit::outgoing, void* ret)
{
    SHMEMPolicy::category_region::stop(std::string_view{ _data.tool_id }, "return", ret);
}

template <typename SHMEMPolicy>
void
shmem_gotcha<SHMEMPolicy>::audit(const typename SHMEMPolicy::gotcha_data& _data,
                                 audit::outgoing, int ret)
{
    SHMEMPolicy::category_region::stop(std::string_view{ _data.tool_id }, "return", ret);
}

}  // namespace component

struct DefaultSHMEMPolicy
{
    using comm_data       = component::comm_data;
    using gotcha_data     = tim::component::gotcha_data;
    using category_region = component::category_region<category::shmem>;

    using component_t = component::shmem_gotcha<DefaultSHMEMPolicy>;

    using shmem_bundle_t = tim::component_bundle<category::shmem, component_t, comm_data>;
    using shmem_gotcha_t = tim::component::gotcha<component_t::gotcha_capacity,
                                                  shmem_bundle_t, category::shmem>;
};

}  // namespace rocprofsys
