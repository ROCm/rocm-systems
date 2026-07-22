// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <functional>
#include <string_view>

namespace rocprofsys::control
{
enum class action
{
    skip,
    trace,
    pause
};

class trigger
{
public:
    using action_setter = std::function<void(action)>;

    virtual ~trigger() = default;

    trigger(const trigger&)            = delete;
    trigger& operator=(const trigger&) = delete;
    trigger(trigger&&)                 = delete;
    trigger& operator=(trigger&&)      = delete;

    [[nodiscard]] virtual std::string_view name() const noexcept           = 0;
    [[nodiscard]] virtual action           initial_action() const noexcept = 0;

protected:
    trigger() = default;

    void bind_action_setter(action_setter setter) noexcept
    {
        assert(!m_set_action);
        m_set_action = std::move(setter);
    }

    void publish_action(action act) const { m_set_action(act); }

private:
    action_setter m_set_action;
};
}  // namespace rocprofsys::control
