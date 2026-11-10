// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk-collector/schema.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <utility>

namespace rocprofiler
{
namespace collector
{
bool
load_probe_schema(const std::string& path, probe_schema* schema, std::string* error)
{
    auto input = std::ifstream{path};
    if(!input)
    {
        *error = "cannot open probe schema " + path;
        return false;
    }
    try
    {
        auto doc        = nlohmann::json::parse(input);
        auto result     = probe_schema{};
        result.provider = doc.at("provider").get<std::string>();
        if(doc.contains("version"))
        {
            const auto& version = doc.at("version");
            result.version      = version.is_string() ? version.get<std::string>() : version.dump();
        }
        for(const auto& itr : doc.at("events"))
        {
            auto event = schema_event{};
            event.api  = itr.at("api").get<std::string>();
            for(const auto& field_doc : itr.at("fields"))
            {
                auto field    = schema_field{};
                field.name    = field_doc.at("name").get<std::string>();
                field.role    = field_doc.at("role").get<std::string>();
                field.param   = field_doc.value("param", std::string{});
                field.capture = field_doc.value("capture", std::string{});
                // A member name, or an array index (a number).
                if(field_doc.contains("path"))
                    for(const auto& element : field_doc.at("path"))
                        field.path.push_back(element.is_string() ? element.get<std::string>()
                                                                 : element.dump());
                if(field.role == "argument" &&
                   std::find(event.params.begin(), event.params.end(), field.param) ==
                       event.params.end())
                    event.params.push_back(field.param);
                event.fields.push_back(std::move(field));
            }
            result.events.emplace(itr.at("name").get<std::string>(), std::move(event));
        }
        *schema = std::move(result);
        return true;
    } catch(const std::exception& e)
    {
        *error = "probe schema " + path + ": " + e.what();
        return false;
    }
}

const schema_field*
find_schema_field(const schema_event& event, const std::string& name)
{
    for(const auto& itr : event.fields)
        if(itr.name == name) return &itr;
    return nullptr;
}
}  // namespace collector
}  // namespace rocprofiler
