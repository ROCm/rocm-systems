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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rocpd
{

class json
{
public:
    static std::shared_ptr<json> create();
    static std::shared_ptr<json> parse_string(const std::string& json_str);
    static std::shared_ptr<json> load_file(const std::string& filepath);

    using json_value =
        std::variant<std::string, int, double, long long, bool, std::vector<json>,
                     std::nullptr_t, std::shared_ptr<json>>;

    void                        set(const std::string& key, const json_value& value);
    std::shared_ptr<json_value> get(const std::string& key) const;

    std::string to_string() const;

private:
    json() = default;

private:
    static std::string           stringify(const std::shared_ptr<json_value>& value);
    static std::shared_ptr<json> parse_json_content(const std::string& content);
    static json_value            parse_value(const std::string& content, size_t& pos);
    static std::string parse_string_literal(const std::string& content, size_t& pos);
    static void        skip_whitespace(const std::string& content, size_t& pos);

private:
    std::unordered_map<std::string, std::shared_ptr<json_value>> data;
};

}  // namespace rocpd
