// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

namespace rocpdsna
{

template <typename Derived>
class api_writer_base
{
protected:
    Derived&       self() noexcept { return static_cast<Derived&>(*this); }
    const Derived& self() const noexcept { return static_cast<const Derived&>(*this); }
};

}  // namespace rocpdsna
