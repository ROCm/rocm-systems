// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/interfaces/info_registration_writer_interface.hpp"
#include "writers/interfaces/kernel_dispatch_writer_interface.hpp"
#include "writers/interfaces/memory_alloc_writer_interface.hpp"
#include "writers/interfaces/memory_copy_writer_interface.hpp"
#include "writers/interfaces/pmc_event_writer_interface.hpp"
#include "writers/interfaces/region_writer_interface.hpp"

#include <type_traits>

namespace profiler_hub
{

namespace detail
{

template <typename, typename = void>
struct has_required_type_aliases : std::false_type
{};

template <typename Policy>
struct has_required_type_aliases<Policy,
                                 std::void_t<typename Policy::schema_tag_t,
                                             typename Policy::insert_statements_t,
                                             typename Policy::common_ops_t,
                                             typename Policy::info_writer_t,
                                             typename Policy::region_writer_t,
                                             typename Policy::kernel_dispatch_writer_t,
                                             typename Policy::memory_copy_writer_t,
                                             typename Policy::memory_alloc_writer_t,
                                             typename Policy::pmc_event_writer_t>>
: std::true_type
{};

template <typename Writer, template <typename> class Interface>
struct satisfies_interface : std::is_base_of<Interface<Writer>, Writer>
{};

template <typename Policy, typename = void>
struct has_valid_writer_interfaces : std::false_type
{};

template <typename Policy>
struct has_valid_writer_interfaces<
    Policy,
    std::enable_if_t<has_required_type_aliases<Policy>::value>>
: std::bool_constant<satisfies_interface<typename Policy::info_writer_t,
                                         info_registration_writer_interface>::value &&
                     satisfies_interface<typename Policy::region_writer_t,
                                         region_writer_interface>::value &&
                     satisfies_interface<typename Policy::kernel_dispatch_writer_t,
                                         kernel_dispatch_writer_interface>::value &&
                     satisfies_interface<typename Policy::memory_copy_writer_t,
                                         memory_copy_writer_interface>::value &&
                     satisfies_interface<typename Policy::memory_alloc_writer_t,
                                         memory_alloc_writer_interface>::value &&
                     satisfies_interface<typename Policy::pmc_event_writer_t,
                                         pmc_event_writer_interface>::value>
{};

}  // namespace detail

template <typename Policy>
struct is_valid_writer_policy
: std::bool_constant<detail::has_required_type_aliases<Policy>::value &&
                     detail::has_valid_writer_interfaces<Policy>::value>
{};

template <typename Policy>
inline constexpr bool is_valid_writer_policy_v = is_valid_writer_policy<Policy>::value;

}  // namespace profiler_hub
