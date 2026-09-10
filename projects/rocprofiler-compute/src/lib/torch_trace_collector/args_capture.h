// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "schema_arg_names.h"
#include "wire_format.h"

#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace torch_trace_collector::detail
{

inline constexpr std::size_t kMaxArgsLen        = 512;
inline constexpr std::size_t kMaxArgItems       = 32;
inline constexpr std::size_t kMaxNestedArgItems = 8;

inline std::string cap_args_blob(std::string blob)
{
    if(blob.size() <= kMaxArgsLen)
    {
        return blob;
    }
    const bool balanced = !blob.empty() && blob.front() == '(' && blob.back() == ')';
    blob.resize(kMaxArgsLen);
    blob += balanced ? "...)" : "...";
    return blob;
}

inline std::string encode_args(const std::string& args)
{
    std::string out;
    out.reserve(args.size());
    for(char c : args)
    {
        switch(c)
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

inline std::string scalar_type_name(c10::ScalarType type)
{
    switch(type)
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
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return name;
}

inline std::string render_leaf_ivalue(const c10::IValue& iv)
{
    try
    {
        if(iv.isTensor())
        {
            const auto& tensor = iv.toTensor();
            if(!tensor.defined())
            {
                return "None";
            }
            std::string dims;
            bool        first = true;
            for(const auto dim : tensor.sizes())
            {
                if(!first)
                {
                    dims += "x";
                }
                first = false;
                dims += std::to_string(dim);
            }
            return scalar_type_name(tensor.scalar_type()) + "[" + dims + "]";
        }
        if(iv.isTensorList())
        {
            const auto  tensors      = iv.toTensorList();
            const auto  render_count = std::min(tensors.size(), kMaxNestedArgItems);
            std::string inner;
            for(std::size_t i = 0; i < render_count; ++i)
            {
                if(i > 0)
                {
                    inner += ", ";
                }
                inner += render_leaf_ivalue(c10::IValue(tensors.get(i)));
            }
            return "[" + inner + "]";
        }
        return iv.tagKind();
    }
    catch(...)
    {
        return "?";
    }
}

inline std::string capture_record_function_args(const at::RecordFunction& record_fn)
{
    std::string out;
    try
    {
        std::vector<std::string> argument_names;
        const auto               operator_name = record_fn.operator_name();
        if(operator_name.has_value())
        {
            argument_names = schema_arg_names(operator_name.value());
        }

        const auto& inputs    = record_fn.inputs();
        out                   = "(";
        std::size_t arg_index = 0;
        for(const auto& input : inputs)
        {
            if(arg_index >= kMaxArgItems)
            {
                break;
            }
            if(arg_index > 0)
            {
                out += ", ";
            }
            const std::string rendered = render_leaf_ivalue(input);
            if(arg_index < argument_names.size() && !argument_names[arg_index].empty())
            {
                out += argument_names[arg_index] + "=" + rendered;
            }
            else
            {
                out += rendered;
            }
            ++arg_index;
        }
        if(arg_index == 0)
        {
            return std::string{kUnavailable};
        }
        out += ")";
    }
    catch(...)
    {
        return std::string{kUnavailable};
    }
    return encode_args(cap_args_blob(std::move(out)));
}

}  // namespace torch_trace_collector::detail
