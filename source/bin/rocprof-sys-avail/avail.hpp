// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "defines.hpp"

#include <timemory/settings/macros.hpp>
#include <timemory/tpls/cereal/archives.hpp>
#include <timemory/tpls/cereal/cereal/external/base64.hpp>
#include <timemory/utility/demangle.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if !defined(TIMEMORY_CEREAL_EPILOGUE_FUNCTION_NAME)
#    define TIMEMORY_CEREAL_EPILOGUE_FUNCTION_NAME epilogue
#endif

#if !defined(TIMEMORY_CEREAL_PROLOGUE_FUNCTION_NAME)
#    define TIMEMORY_CEREAL_PROLOGUE_FUNCTION_NAME prologue
#endif

//======================================================================================//

namespace tim
{
namespace cereal
{
class SettingsTextArchive
: public OutputArchive<SettingsTextArchive>
, public traits::TextArchive
{
public:
    using width_type = std::vector<uint64_t>;
    using value_type = std::string;
    using entry_type = std::map<std::string, value_type>;
    using array_type = std::vector<entry_type>;
    using unique_set = std::set<std::string>;
    using int_stack  = std::stack<uint32_t>;

public:
    //! Construct, outputting to the provided stream
    /// \param stream The array of output data
    SettingsTextArchive(array_type& stream, unique_set exclude)
    : OutputArchive<SettingsTextArchive>(this)
    , output_stream(&stream)
    , exclude_stream(std::move(exclude))
    {
        name_counter.push(0);
    }

    ~SettingsTextArchive() override = default;

    void saveBinaryValue(const void* data, size_t size, const char* name = nullptr)
    {
        setNextName(name);
        writeName();

        auto base64string =
            base64::encode(reinterpret_cast<const unsigned char*>(data), size);
        saveValue(base64string);
    }

    void startNode() { name_counter.push(0); }

    void finishNode() { name_counter.pop(); }

    //! Sets the name for the next node created with startNode
    void setNextName(const char* name)
    {
        if(exclude_stream.count(name) > 0) return;

        if((current_entry != nullptr) && value_keys.count(name) > 0)
        {
            current_entry->insert({ name, "" });
            current_value = &((*current_entry)[name]);
            return;
        }

        if(value_keys.count(name) > 0)
        {
            return;
        }

        current_value = nullptr;
        output_stream->push_back(entry_type{});
        current_entry = &(output_stream->back());

        current_entry->insert({ "identifier", name });
        std::string       func   = name;
        const std::string prefix = TIMEMORY_SETTINGS_PREFIX;
        func                     = func.erase(0, prefix.length());
        std::transform(func.begin(), func.end(), func.begin(),
                       [](char& c) { return tolower(c); });
        {
            std::stringstream ss;
            ss << "settings::" << func << "()";
            current_entry->insert({ "static_accessor", ss.str() });
        }
        {
            std::stringstream ss;
            ss << "settings::instance()->get_" << func << "()";
            current_entry->insert({ "member_accessor", ss.str() });
        }
        {
            std::stringstream ss;
            ss << "settings." << func;
            current_entry->insert({ "python_accessor", ss.str() });
        }
    }

    void setNextType(const char*) {}

public:
    template <typename Tp>
    inline void saveValue(Tp _val)
    {
        std::stringstream ssval;
        ssval << std::boolalpha << _val;
        if(current_value)
        {
            *current_value = ssval.str();
        }
    }

    void writeName() {}

    void makeArray() {}

private:
    value_type* current_value  = nullptr;
    entry_type* current_entry  = nullptr;
    array_type* output_stream  = nullptr;
    unique_set  exclude_stream = {};
    int_stack   name_counter;
    unique_set  value_keys = { "name",    "value",     "description", "count",
                               "environ", "max_count", "cmdline",     "data_type",
                               "initial", "categories" };
};

//======================================================================================//
//
//      prologue and epilogue functions
//
//======================================================================================//

//--------------------------------------------------------------------------------------//
//! Prologue for NVPs for settings archive
/*! NVPs do not start or finish nodes - they just set up the names */
template <typename T>
inline void
TIMEMORY_CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive&, const NameValuePair<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Epilogue for NVPs for settings archive
/*! NVPs do not start or finish nodes - they just set up the names */
template <typename T>
inline void
TIMEMORY_CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive&, const NameValuePair<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Prologue for deferred data for settings archive
/*! Do nothing for the defer wrapper */
template <typename T>
inline void
TIMEMORY_CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive&, const DeferredData<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Epilogue for deferred for settings archive
/*! NVPs do not start or finish nodes - they just set up the names */
template <typename T>
inline void
TIMEMORY_CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive&, const DeferredData<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Prologue for SizeTags for settings archive
/*! SizeTags are ignored */
template <typename T>
inline void
TIMEMORY_CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive& ar, const SizeTag<T>&)
{
    ar.makeArray();
}

//--------------------------------------------------------------------------------------//
//! Epilogue for SizeTags for settings archive
/*! SizeTags are ignored */
template <typename T>
inline void
TIMEMORY_CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive&, const SizeTag<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Prologue for all other types for settings archive
/*! Starts a new node, named either automatically or by some NVP,
    that may be given data by the type about to be archived*/
template <typename T>
inline void
TIMEMORY_CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive& ar, const T&)
{
    ar.startNode();
}

//--------------------------------------------------------------------------------------//
//! Epilogue for all other types other for settings archive
/*! Finishes the node created in the prologue*/
template <typename T>
inline void
TIMEMORY_CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive& ar, const T&)
{
    ar.finishNode();
}

//--------------------------------------------------------------------------------------//
//! Prologue for arithmetic types for settings archive
inline void
TIMEMORY_CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive&, const std::nullptr_t&)
{}

//--------------------------------------------------------------------------------------//
//! Epilogue for arithmetic types for settings archive
inline void
TIMEMORY_CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive&, const std::nullptr_t&)
{}

//======================================================================================//
//
//  Common serialization functions
//
//======================================================================================//

//! Serializing NVP types
template <typename T>
inline void
TIMEMORY_CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive& ar, const NameValuePair<T>& t)
{
    ar.setNextName(t.name);
    if(std::is_same<T, std::string>::value)
    {
        ar.setNextType("string");
    }
    else
    {
        ar.setNextType(tim::demangle<T>().c_str());
    }
    ar(t.value);
}

template <typename CharT, typename Traits, typename Alloc>
inline void
TIMEMORY_CEREAL_SAVE_FUNCTION_NAME(
    SettingsTextArchive&                                          ar,
    const NameValuePair<std::basic_string<CharT, Traits, Alloc>>& t)
{
    ar.setNextName(t.name);
    ar.setNextType("string");
    ar(t.value);
}

//! Saving for nullptr
inline void
TIMEMORY_CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive&, const std::nullptr_t&)
{}

//! Saving for arithmetic
template <typename T, traits::EnableIf<std::is_arithmetic<T>::value> = traits::sfinae>
inline void
TIMEMORY_CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive& ar, const T& t)
{
    if(std::is_same<T, std::string>::value) ar.setNextType("string");
    ar.saveValue(t);
}

//! saving string
template <typename CharT, typename Traits, typename Alloc>
inline void
TIMEMORY_CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive&                           ar,
                                   const std::basic_string<CharT, Traits, Alloc>& str)
{
    ar.setNextType("string");
    ar.saveValue(str);
}

//--------------------------------------------------------------------------------------//
//! Saving SizeTags
template <typename T>
inline void
TIMEMORY_CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive&, const SizeTag<T>&)
{
    // nothing to do here, we don't explicitly save the size
}

}  // namespace cereal
}  // namespace tim

// register archives for polymorphic support
TIMEMORY_CEREAL_REGISTER_ARCHIVE(SettingsTextArchive)
