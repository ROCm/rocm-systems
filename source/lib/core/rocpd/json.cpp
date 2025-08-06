// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "json.hpp"
#include <sstream>

namespace rocpd
{

std::shared_ptr<json>
json::create()
{
    return std::shared_ptr<json>(new json());
}

void
json::set(const std::string& key, const json_value& value)
{
    data[key] = std::make_shared<json_value>(value);
}

std::string
json::to_string() const
{
    std::ostringstream oss;
    oss << "{";
    bool first = true;

    for(const auto& [key, value] : data)
    {
        if(!first) oss << ", ";
        first = false;

        oss << "\"" << key << "\": " << stringify(value);
    }

    oss << "}";
    return oss.str();
}

std::string
json::stringify(const std::shared_ptr<json_value>& value)
{
    std::ostringstream oss;
    std::visit(
        [&oss](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr(std::is_same_v<T, std::string>)
                oss << "\"" << arg << "\"";
            else if constexpr(std::is_same_v<T, bool>)
                oss << (arg ? "true" : "false");
            else if constexpr(std::is_same_v<T, std::nullptr_t>)
                oss << "null";
            else if constexpr(std::is_same_v<T, std::vector<json>>)
            {
                oss << "[";
                bool first = true;
                for(const auto& item : arg)
                {
                    if(!first) oss << ", ";
                    first = false;
                    oss << item.to_string();
                }
                oss << "]";
            }
            else if constexpr(std::is_same_v<T, std::shared_ptr<json>>)
            {
                oss << arg->to_string();
            }
            else
            {
                // handle int + double
                oss << arg;
            }
        },
        *value);
    return oss.str();
}

}  // namespace rocpd
