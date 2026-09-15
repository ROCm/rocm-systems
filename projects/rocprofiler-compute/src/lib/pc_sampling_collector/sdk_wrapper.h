// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <memory>
#include <string_view>

namespace rocprofiler_compute_tool
{
class sdk_wrapper_t
{
public:
    using ptr = std::shared_ptr<sdk_wrapper_t>;
    static ptr create();

    virtual ~sdk_wrapper_t()                                = default;
    virtual std::string_view source_frame_separator() const = 0;
};

class sdk_wrapper_impl_t : public sdk_wrapper_t
{
public:
    std::string_view source_frame_separator() const override;
};
}  // namespace rocprofiler_compute_tool
