// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys::core
{
// Non-owning, type-erased view over any object exposing
//   void on_source_drained(int, std::vector<char>);
//   void finalize();
class polymorphic_sink_view;

template <typename T>
concept polymorphic_sink_target =
    !std::same_as<std::remove_cvref_t<T>, polymorphic_sink_view> &&
    requires(T& target, int source_id, std::vector<char> bytes) {
        { target.on_source_drained(source_id, std::move(bytes)) } -> std::same_as<void>;
        { target.finalize() } -> std::same_as<void>;
    };

class polymorphic_sink_view
{
public:
    template <polymorphic_sink_target T>
    explicit polymorphic_sink_view(T& target) noexcept
    : m_target{ &target }
    , m_drain_fn{ +[](void* p, int source_id, std::vector<char> bytes) {
        static_cast<T*>(p)->on_source_drained(source_id, std::move(bytes));
    } }
    , m_finalize_fn{ +[](void* p) { static_cast<T*>(p)->finalize(); } }
    {}

    polymorphic_sink_view(const polymorphic_sink_view&)            = default;
    polymorphic_sink_view(polymorphic_sink_view&&)                 = default;
    polymorphic_sink_view& operator=(const polymorphic_sink_view&) = default;
    polymorphic_sink_view& operator=(polymorphic_sink_view&&)      = default;
    ~polymorphic_sink_view()                                       = default;

    void on_source_drained(int source_id, std::vector<char> bytes)
    {
        m_drain_fn(m_target, source_id, std::move(bytes));
    }
    void finalize() { m_finalize_fn(m_target); }

private:
    using drain_fn_t    = void (*)(void*, int, std::vector<char>);
    using finalize_fn_t = void (*)(void*);

    void*         m_target{ nullptr };
    drain_fn_t    m_drain_fn{ nullptr };
    finalize_fn_t m_finalize_fn{ nullptr };
};
}  // namespace rocprofsys::core
