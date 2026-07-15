// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>

namespace rocprofsys::control
{
enum class vote
{
    abstain,
    active,
    paused
};

struct vote_entry
{
    std::string name{};
    vote        current_vote{ vote::active };
};

class trigger
{
public:
    virtual ~trigger() = default;

    trigger(const trigger&)            = delete;
    trigger& operator=(const trigger&) = delete;
    trigger(trigger&&)                 = delete;
    trigger& operator=(trigger&&)      = delete;

    [[nodiscard]] virtual std::string_view name() const noexcept         = 0;
    [[nodiscard]] virtual vote             initial_vote() const noexcept = 0;

protected:
    trigger() = default;
};
}  // namespace rocprofsys::control
