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

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/type_registry.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>

namespace rocprofsys
{
namespace trace_cache
{
/**
 * @brief Parser for reading and processing serialized data from buffered storage files
 *
 * This template class provides functionality to read binary files created by the buffered
 * storage system, deserialize the contained data objects, and process them through a
 * user-provided processing handler. It handles the binary format with type headers and
 * automatically cleans up the file after successful parsing.
 *
 * The parser reads a binary format consisting of:
 * - Type identifier (TypeIdentifierEnum value)
 * - Sample size (size_t)
 * - Serialized data payload
 *
 * @tparam TypeIdentifierEnum Enum class that identifies different serialized types
 * @tparam SupportedTypes Variadic template pack of types that can be deserialized
 *
 * @note All SupportedTypes must be compatible with the type_registry requirements
 */
template <typename TypeIdentifierEnum, typename... SupportedTypes>
class storage_parser
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");

public:
    /**
     * @brief Construct a new storage parser
     *
     * @param _filename Path to the binary file containing serialized data
     */
    storage_parser(std::string _filename)
    : m_filename(std::move(_filename))
    {}

    /**
     * @brief Register a callback to be executed when parsing is finished
     *
     * The callback will be invoked after the file has been successfully parsed
     * and removed from the filesystem, but before the load() method returns.
     *
     * @param callback Unique pointer to a function object that will be called on
     * completion The parser takes ownership of the callback
     */
    void register_on_finished_callback(std::unique_ptr<std::function<void()>> callback)
    {
        m_on_finished_callback = std::move(callback);
    }

    /**
     * @brief Load and process all data from the storage file
     *
     * Reads the binary file sequentially, deserializes each data sample based on its
     * type identifier, and forwards it to the processing handler. The file is
     * automatically removed after successful parsing.
     *
     * Binary format processed:
     * - Header: TypeIdentifierEnum (type) + size_t (sample_size)
     * - Payload: Raw serialized data of specified size
     *
     * @tparam ProcessingType Type of the processing handler
     * @param processing Reference to processing handler that will receive deserialized
     * objects
     *
     * @throws std::runtime_error if the file cannot be opened for reading
     *
     * @note ProcessingType must have a method with signature:
     *       void execute_sample_processing(TypeIdentifierEnum, const cacheable_t&)
     *
     * @note The file is automatically deleted after successful parsing
     * @note Fragmented space entries (used for buffer management) are automatically
     * skipped
     * @note Invalid or corrupted data entries are logged and skipped
     * @note The registered callback (if any) is invoked after file processing completes
     */
    template <typename ProcessingType>
    void load(ProcessingType& processing)
    {
        static_assert(
            type_traits::has_execute_processing<ProcessingType, TypeIdentifierEnum,
                                                cacheable_t>::value,
            "ProcessingType must have proper execute processing member function: "
            "ProcessingType::execute_sample_processing(TypeIdentifierEnum, const "
            "cacheable_t&)");

        std::cout << "Consuming buffered storage with filename: " << m_filename
                  << std::endl;

        std::ifstream ifs(m_filename, std::ios::binary);
        if(!ifs)
        {
            std::stringstream ss;
            ss << "Error opening file for reading: " << m_filename << "\n";
            throw std::runtime_error(ss.str());
        }

        struct __attribute__((packed)) sample_header
        {
            TypeIdentifierEnum type;
            size_t             sample_size;
        };

        sample_header header;

        while(!ifs.eof())
        {
            ifs.read(reinterpret_cast<char*>(&header), sizeof(header));

            if(header.sample_size == 0 || ifs.eof())
            {
                continue;
            }

            std::vector<uint8_t> sample;
            sample.reserve(header.sample_size);
            ifs.read(reinterpret_cast<char*>(sample.data()), header.sample_size);

            if(ifs.fail())
            {
                std::cout << "Bad read while consuming buffered storage. Filename: "
                          << m_filename
                          << " Bytes read: " << static_cast<int>(ifs.tellg())
                          << std::endl;
                continue;
            }

            if(header.type == TypeIdentifierEnum::fragmented_space)
            {
                continue;
            }

            uint8_t* data = sample.data();

            auto sample_value = m_registry.get_type(header.type, data);
            if(sample_value.has_value())
            {
                processing.execute_sample_processing(
                    header.type, std::visit(
                                     [](auto& arg) -> cacheable_t& {
                                         return static_cast<cacheable_t&>(arg);
                                     },
                                     sample_value.value()));
            }
            else
            {
                std::cout << "Unsupported type detected. Skipping current sample."
                          << std::endl;
                continue;
            }
        }

        ifs.close();
        std::cout << "File parsing finished. Removing " << m_filename
                  << " from file system." << std::endl;
        std::remove(m_filename.c_str());

        if(m_on_finished_callback != nullptr)
        {
            (*m_on_finished_callback)();
        }
    }

private:
    std::string m_filename;  ///< Path to the binary file to be parsed
    std::unique_ptr<std::function<void()>> m_on_finished_callback{
        nullptr
    };  ///< Optional callback invoked when parsing completes
    type_registry<TypeIdentifierEnum, SupportedTypes...>
        m_registry;  ///< Type registry for deserializing stored objects
};

}  // namespace trace_cache
}  // namespace rocprofsys
