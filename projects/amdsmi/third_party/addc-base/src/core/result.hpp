// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace addc::detail
{

enum class OperationCode
{
    InvalidArgument,
    ParseError,
    DecodeError,
    IoError,
    OutOfResources,
    InternalError,
};

struct OperationError
{
    OperationCode code = OperationCode::InternalError;
    std::string message;
    std::string source;
    std::optional<std::size_t> offset;
};

template <typename T>
struct OperationResult
{
    std::optional<T> value;
    std::optional<OperationError> error;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return value.has_value();
    }

    [[nodiscard]] static OperationResult success(T result)
    {
        return OperationResult{std::move(result), std::nullopt};
    }

    [[nodiscard]] static OperationResult failure(OperationError failure)
    {
        return OperationResult{std::nullopt, std::move(failure)};
    }
};

} // namespace addc::detail
