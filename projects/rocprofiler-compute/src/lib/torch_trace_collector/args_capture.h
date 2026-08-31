// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace torch_trace_collector::detail
{

// Maximum length of an args blob.
inline constexpr std::size_t kMaxArgsLen = 512;

// Maximum number of operator inputs rendered into an args blob.
inline constexpr std::size_t kMaxArgItems = 32;

// Operator-argument capture configuration.
struct ArgsCaptureConfig
{
    std::atomic<bool> capture_args{true};
    std::atomic<bool> capture_values{false};
};

inline ArgsCaptureConfig g_args_capture;

// Truncate an args blob to kMaxArgsLen and append "...".
inline std::string cap_args_blob(std::string blob)
{
    if (blob.size() <= kMaxArgsLen)
    {
        return blob;
    }
    blob.resize(kMaxArgsLen);
    blob += "...";
    return blob;
}

// Whether operator args are captured.
inline bool args_capture_enabled()
{
    return g_args_capture.capture_args.load();
}

// Whether scalar values are recorded. Requires args capture.
inline bool args_values_enabled()
{
    return g_args_capture.capture_args.load() && g_args_capture.capture_values.load();
}

// Percent-encode '%', '|', ';', and newlines in an args blob.
inline std::string encode_args(const std::string& args)
{
    std::string out;
    out.reserve(args.size());
    for (char c : args)
    {
        switch (c)
        {
        case '%':
            out += "%25";
            break;
        case '|':
            out += "%7C";
            break;
        case ';':
            out += "%3B";
            break;
        case '\r':
            out += "%0D";
            break;
        case '\n':
            out += "%0A";
            break;
        default:
            out += c;
        }
    }
    return out;
}

// Tensor dtype name used by the Python tiers (Float -> float32).
// Unlisted types use the lowercased ATen name.
inline std::string scalar_type_name(c10::ScalarType type)
{
    switch (type)
    {
    case c10::ScalarType::Float:
        return "float32";
    case c10::ScalarType::Double:
        return "float64";
    case c10::ScalarType::Half:
        return "float16";
    case c10::ScalarType::BFloat16:
        return "bfloat16";
    case c10::ScalarType::Long:
        return "int64";
    case c10::ScalarType::Int:
        return "int32";
    case c10::ScalarType::Short:
        return "int16";
    case c10::ScalarType::Char:
        return "int8";
    case c10::ScalarType::Byte:
        return "uint8";
    case c10::ScalarType::Bool:
        return "bool";
    case c10::ScalarType::ComplexHalf:
        return "complex32";
    case c10::ScalarType::ComplexFloat:
        return "complex64";
    case c10::ScalarType::ComplexDouble:
        return "complex128";
    default:
        break;
    }
    std::string name = c10::toString(type);
    std::transform(name.begin(),
                   name.end(),
                   name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name;
}

// Render one operator input as "dtype[d0xd1]" for tensors, scalar values when
// value capture is enabled, or the value's type tag otherwise.
inline std::string render_leaf_ivalue(const c10::IValue& iv, bool values)
{
    try
    {
        if (iv.isTensor())
        {
            const auto& tensor = iv.toTensor();
            if (!tensor.defined())
            {
                return "None";
            }
            std::string dims;
            bool        first = true;
            for (const auto dim : tensor.sizes())
            {
                if (!first)
                {
                    dims += "x";
                }
                first = false;
                dims += std::to_string(dim);
            }
            return scalar_type_name(tensor.scalar_type()) + "[" + dims + "]";
        }
        if (iv.isTensorList())
        {
            std::string inner;
            bool        first = true;
            for (const auto& tensor : iv.toTensorVector())
            {
                if (!first)
                {
                    inner += ", ";
                }
                first = false;
                inner += render_leaf_ivalue(c10::IValue(tensor), values);
            }
            return "[" + inner + "]";
        }
        if (values)
        {
            if (iv.isInt())
            {
                return std::to_string(iv.toInt());
            }
            if (iv.isBool())
            {
                return iv.toBool() ? "True" : "False";
            }
            if (iv.isDouble())
            {
                return std::to_string(iv.toDouble());
            }
        }
        return iv.tagKind();
    }
    catch (...)
    {
        return "?";
    }
}

// Unencoded args blob: "(name=dtype[shape], ...)". Empty when capture is off
// or the operator has no inputs.
inline std::string build_leaf_args(const at::RecordFunction& record_fn)
{
    if (!args_capture_enabled())
    {
        return "";
    }
    std::string out;
    try
    {
        std::vector<std::string> names;
        const auto               op_name = record_fn.operator_name();
        if (op_name.has_value())
        {
            const auto handle = c10::Dispatcher::singleton().findSchema(op_name.value());
            if (handle.has_value())
            {
                const auto& schema_args = handle->schema().arguments();
                names.reserve(schema_args.size());
                for (const auto& arg : schema_args)
                {
                    names.push_back(arg.name());
                }
            }
        }

        const bool  values = args_values_enabled();
        const auto& inputs = record_fn.inputs();
        out                = "(";
        std::size_t count  = 0;
        for (const auto& iv : inputs)
        {
            if (count >= kMaxArgItems)
            {
                break;
            }
            if (count > 0)
            {
                out += ", ";
            }
            const std::string rendered = render_leaf_ivalue(iv, values);
            if (count < names.size() && !names[count].empty())
            {
                out += names[count] + "=" + rendered;
            }
            else
            {
                out += rendered;
            }
            ++count;
        }
        if (count == 0)
        {
            return "";
        }
        out += ")";
    }
    catch (...)
    {
        return "";
    }
    return cap_args_blob(std::move(out));
}

// Append "|args=<encoded>" to wire_string when args is non-empty.
inline void append_args_segment(std::string& wire_string, const std::string& args)
{
    if (args.empty())
    {
        return;
    }
    wire_string += "|args=";
    wire_string += encode_args(args);
}

}  // namespace torch_trace_collector::detail
