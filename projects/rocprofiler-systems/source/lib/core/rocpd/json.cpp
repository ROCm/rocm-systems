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
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace rocpd
{

std::shared_ptr<json>
json::create()
{
    return std::shared_ptr<json>(new json());
}

std::shared_ptr<json>
json::parse_string(const std::string& json_str)
{
    return parse_json_content(json_str);
}

std::shared_ptr<json>
json::load_file(const std::string& filepath)
{
    std::ifstream file(filepath);
    if(!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    return parse_json_content(content);
}

std::shared_ptr<json>
json::parse_json_content(const std::string& content)
{
    auto   result = std::shared_ptr<json>(new json());
    size_t pos    = 0;

    skip_whitespace(content, pos);
    if(pos >= content.length() || content[pos] != '{')
    {
        throw std::runtime_error("Expected '{' at start of JSON object");
    }

    pos++;  // skip '{'
    skip_whitespace(content, pos);

    if(pos < content.length() && content[pos] == '}')
    {
        return result;  // empty object
    }

    while(pos < content.length())
    {
        skip_whitespace(content, pos);

        // Parse key
        if(content[pos] != '"')
        {
            throw std::runtime_error("Expected string key");
        }
        std::string key = parse_string_literal(content, pos);

        skip_whitespace(content, pos);
        if(pos >= content.length() || content[pos] != ':')
        {
            throw std::runtime_error("Expected ':' after key");
        }
        pos++;  // skip ':'

        // Parse value
        json_value value = parse_value(content, pos);
        result->set(key, value);

        skip_whitespace(content, pos);
        if(pos >= content.length()) break;

        if(content[pos] == '}')
        {
            break;
        }
        else if(content[pos] == ',')
        {
            pos++;
            continue;
        }
        else
        {
            throw std::runtime_error("Expected ',' or '}' in object");
        }
    }

    return result;
}

json::json_value
json::parse_value(const std::string& content, size_t& pos)
{
    skip_whitespace(content, pos);

    if(pos >= content.length())
    {
        throw std::runtime_error("Unexpected end of input");
    }

    char c = content[pos];

    if(c == '"')
    {
        return parse_string_literal(content, pos);
    }
    else if(c == 't' || c == 'f')
    {
        if(content.substr(pos, 4) == "true")
        {
            pos += 4;
            return true;
        }
        else if(content.substr(pos, 5) == "false")
        {
            pos += 5;
            return false;
        }
        throw std::runtime_error("Invalid boolean value");
    }
    else if(c == 'n')
    {
        if(content.substr(pos, 4) == "null")
        {
            pos += 4;
            return nullptr;
        }
        throw std::runtime_error("Invalid null value");
    }
    else if(std::isdigit(c) || c == '-')
    {
        size_t start = pos;
        if(c == '-') pos++;

        while(pos < content.length() && std::isdigit(content[pos]))
            pos++;

        if(pos < content.length() && content[pos] == '.')
        {
            pos++;
            while(pos < content.length() && std::isdigit(content[pos]))
                pos++;
            return std::stod(content.substr(start, pos - start));
        }
        else
        {
            return std::stoll(content.substr(start, pos - start));
        }
    }

    throw std::runtime_error("Unexpected character in JSON");
}

std::string
json::parse_string_literal(const std::string& content, size_t& pos)
{
    if(content[pos] != '"')
    {
        throw std::runtime_error("Expected '\"' at start of string");
    }

    pos++;  // skip opening quote
    size_t start = pos;

    while(pos < content.length() && content[pos] != '"')
    {
        if(content[pos] == '\\') pos++;  // skip escape sequence
        pos++;
    }

    if(pos >= content.length())
    {
        throw std::runtime_error("Unterminated string");
    }

    std::string result = content.substr(start, pos - start);
    pos++;  // skip closing quote
    return result;
}

void
json::skip_whitespace(const std::string& content, size_t& pos)
{
    while(pos < content.length() && (std::isspace(content[pos]) != 0))
    {
        pos++;
    }
}

void
json::set(const std::string& key, const json_value& value)
{
    data[key] = std::make_shared<json_value>(value);
}

std::shared_ptr<json::json_value>
json::get(const std::string& key) const
{
    return data.at(key);
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
