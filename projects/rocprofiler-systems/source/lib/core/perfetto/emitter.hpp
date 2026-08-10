// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/categories.hpp"
#include "core/concepts.hpp"
#include "core/demangler.hpp"
#include "core/perfetto/category_registry.hpp"
#include "core/perfetto/engine.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace rocprofsys
{
namespace core::perfetto
{
using hash_value_t = std::uint64_t;

constexpr std::uint64_t
hash_combine(std::uint64_t seed, std::uint64_t value) noexcept
{
    constexpr std::uint64_t golden_ratio = 0x9e3779b97f4a7c17ULL;
    return seed ^ (value + golden_ratio + (seed << 6) + (seed >> 2));
}

template <typename... Args>
inline std::uint64_t
hash_combine_all(std::uint64_t seed, Args&&... args) noexcept
{
    ((seed =
          hash_combine(seed, std::hash<std::decay_t<Args>>{}(std::forward<Args>(args)))),
     ...);
    return seed;
}

std::unordered_map<hash_value_t, std::string>&
get_perfetto_track_uuids();

std::mutex&
get_perfetto_track_uuids_mutex();

::perfetto::Track
get_active_process_track();

void
ensure_synthetic_process_track_emitted(int pid);

template <typename T>
    requires std::is_const_v<T>
auto
get_perfetto_string(T& name)
{
    return ::perfetto::StaticString{ name };
}

template <typename T>
    requires(!std::is_const_v<T>)
auto
get_perfetto_string(T& name)
{
    return ::perfetto::DynamicString{ name };
}

template <typename Np, typename Tp>
auto
add_perfetto_annotation(::perfetto::EventContext& ctx, Np&& name, Tp&& value,
                        std::int64_t idx = -1)
{
    using named_type = std::remove_reference_t<std::remove_cv_t<std::decay_t<Np>>>;
    using value_type = std::remove_reference_t<std::remove_cv_t<std::decay_t<Tp>>>;

    static_assert(concepts::is_string_type<named_type>::value,
                  "Error! name is not a string type");

    auto get_debug_annotation = [&]() {
        auto* debug_annotation = ctx.event()->add_debug_annotations();
        if(idx >= 0)
        {
            auto arg_name = fmt::format("arg{}-{}", idx, std::forward<Np>(name));
            debug_annotation->set_name(arg_name);
        }
        else
        {
            debug_annotation->set_name(std::string_view{ std::forward<Np>(name) }.data());
        }
        return debug_annotation;
    };

    if constexpr(std::is_same<value_type, std::string_view>::value)
    {
        get_debug_annotation()->set_string_value(value.data());
    }
    else if constexpr(concepts::is_string_type<value_type>::value)
    {
        get_debug_annotation()->set_string_value(std::forward<Tp>(value));
    }
    else if constexpr(std::is_same<value_type, bool>::value)
    {
        get_debug_annotation()->set_bool_value(value);
    }
    else if constexpr(std::is_enum<value_type>::value)
    {
        get_debug_annotation()->set_int_value(static_cast<std::int64_t>(value));
    }
    else if constexpr(std::is_floating_point<value_type>::value)
    {
        get_debug_annotation()->set_double_value(static_cast<double>(value));
    }
    else if constexpr(std::is_integral<value_type>::value)
    {
        if constexpr(std::is_unsigned<value_type>::value)
            get_debug_annotation()->set_uint_value(value);
        else
            get_debug_annotation()->set_int_value(value);
    }
    else if constexpr(std::is_pointer<value_type>::value)
    {
        get_debug_annotation()->set_pointer_value(reinterpret_cast<std::uint64_t>(value));
    }
    else if constexpr(concepts::string_like<value_type>)
    {
        get_debug_annotation()->set_string_value(
            fmt::format("{}", std::forward<Tp>(value)));
    }
    else
    {
        static_assert(std::is_empty<value_type>::value, "Error! unsupported data type");
    }
}

template <typename CategoryT, typename... Args>
auto
get_perfetto_category_uuid(Args&&... args)
{
    const auto seed = std::hash<std::string>{}(
        fmt::format("rocprofsys_{}", ::tim::trait::name<CategoryT>::value));
    return hash_combine_all(seed, std::forward<Args>(args)...,
                            static_cast<std::int64_t>(core::get_emitting_pid()));
}

template <typename CategoryT, typename TrackT = ::perfetto::Track, typename FuncT,
          typename... Args>
auto
get_perfetto_track(CategoryT, FuncT&& desc_generator, Args&&... args)
{
    auto uuid = get_perfetto_category_uuid<CategoryT>(std::forward<Args>(args)...);

    std::lock_guard<std::mutex> lock{ get_perfetto_track_uuids_mutex() };
    auto&                       track_uuids = get_perfetto_track_uuids();

    if(track_uuids.find(uuid) == track_uuids.end())
    {
        const auto track = TrackT(uuid, get_active_process_track());
        auto       desc  = track.Serialize();

        auto name = std::forward<FuncT>(desc_generator)(std::forward<Args>(args)...);
        desc.set_name(name);
        ::perfetto::TrackEvent::SetTrackDescriptor(track, desc);

        LOG_TRACE("[{}] Created {}({}) with description: \"{}\"",
                  ::tim::trait::name<CategoryT>::value,
                  rocprofsys::utility::demangle<TrackT>(), uuid, name);

        track_uuids.emplace(uuid, name);
    }

#if defined(ROCPROFSYS_CI) && ROCPROFSYS_CI > 0
    auto name = std::forward<FuncT>(desc_generator)(std::forward<Args>(args)...);
    if(track_uuids.at(uuid) != name)
    {
        throw std::runtime_error(
            fmt::format("Error! Multiple invocations of UUID {} produced different "
                        "descriptions: \"{}\" and \"{}\"",
                        uuid, track_uuids.at(uuid), name));
    }
#endif

    return TrackT(uuid, get_active_process_track());
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto_track(CategoryT, const char* name, ::perfetto::Track track,
                    std::uint64_t timestamp, Args&&... args)
{
    if(!::tim::trait::runtime_enabled<CategoryT>::get()) return;

    TRACE_EVENT_BEGIN(::tim::trait::name<CategoryT>::value, get_perfetto_string(name),
                      track, timestamp, std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_track(CategoryT, const char*, ::perfetto::Track track,
                   std::uint64_t timestamp, Args&&... args)
{
    if(!::tim::trait::runtime_enabled<CategoryT>::get()) return;

    TRACE_EVENT_END(::tim::trait::name<CategoryT>::value, track, timestamp,
                    std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto(CategoryT, const char* name, ::perfetto::Track track,
              std::uint64_t timestamp, Args&&... args)
{
    push_perfetto_track(CategoryT{}, name, track, timestamp, std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto(CategoryT, const char* name, ::perfetto::Track track,
             std::uint64_t timestamp, Args&&... args)
{
    pop_perfetto_track(CategoryT{}, name, track, timestamp, std::forward<Args>(args)...);
}
}  // namespace core::perfetto
}  // namespace rocprofsys
