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

#include "common/defines.h"

#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/type_registry.hpp"

#include "logger/debug.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

namespace rocprofsys
{
namespace trace_cache
{
template <typename TypeIdentifierEnum, typename... SupportedTypes>
class storage_parser
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");

    static_assert(sizeof...(SupportedTypes) != 0, "SupportedTypes must be non-empty");

public:
    storage_parser(std::string _filename)
    : m_filename(std::move(_filename))
    {}

    template <typename TypeProcessing>
    void load(std::shared_ptr<TypeProcessing> _type_processing)
    {
        static_assert(
            type_traits::has_execute_processing<TypeProcessing, TypeIdentifierEnum,
                                                cacheable_t>::value,
            "TypeProcessing must have member function "
            "execute_sample_processing(TypeIdentifierEnum, const cacheable_t&)");

        if(_type_processing == nullptr)
        {
            throw std::runtime_error("TypeProcessing is nullptr");
        }

        LOG_DEBUG("Loading storage from file: {}", m_filename);

        std::ifstream ifs(m_filename, std::ios::binary);
        if(!ifs.good())
        {
            throw std::runtime_error(
                fmt::format("Error opening file for reading: {}", m_filename));
        }

        struct __attribute__((packed)) sample_header
        {
            TypeIdentifierEnum type;
            size_t             sample_size;
        };

        sample_header header;

        std::vector<uint8_t> sample;
        sample.reserve(4096);
        size_t last_capacity = sample.capacity();

        while(!ifs.eof())
        {
            if(!ifs.good())
            {
                throw std::runtime_error(fmt::format(
                    "Stream not in good state, stopping parse. File: {}", m_filename));
            }

            ifs.read(reinterpret_cast<char*>(&header), sizeof(header));

            if(header.sample_size == 0 || ifs.eof())
            {
                continue;
            }

            if(ROCPROFSYS_UNLIKELY(header.sample_size > last_capacity))
            {
                sample.reserve(header.sample_size);
                last_capacity = sample.capacity();
            }

            sample.resize(header.sample_size);
            ifs.read(reinterpret_cast<char*>(sample.data()), header.sample_size);

            if(ifs.fail())
            {
                throw std::runtime_error(
                    fmt::format("Bad read while consuming buffered storage. Filename: {} "
                                "Bytes read: {}",
                                m_filename, static_cast<int>(ifs.tellg())));
            }

            if(header.type == TypeIdentifierEnum::fragmented_space)
            {
                LOG_TRACE("Skipping fragmented space in storage");
                continue;
            }

            auto* data = sample.data();

            auto sample_value = m_registry.get_type(header.type, data);
            if(sample_value.has_value())
            {
                _type_processing->execute_sample_processing(
                    header.type, std::visit(
                                     [](auto& arg) -> cacheable_t& {
                                         return static_cast<cacheable_t&>(arg);
                                     },
                                     sample_value.value()));
            }
            else
            {
                LOG_TRACE("Unknown sample type encountered, skipping");
                continue;
            }
        }

        ifs.close();
        LOG_DEBUG("Storage parsing complete from {}", m_filename);
    }

private:
    std::string                                          m_filename;
    type_registry<TypeIdentifierEnum, SupportedTypes...> m_registry;
};

}  // namespace trace_cache
}  // namespace rocprofsys
