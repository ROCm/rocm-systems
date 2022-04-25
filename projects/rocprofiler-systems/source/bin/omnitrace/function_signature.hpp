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

#include "fwd.hpp"

struct function_signature
{
    using location_t = std::pair<unsigned long, unsigned long>;

    TIMEMORY_DEFAULT_OBJECT(function_signature)

    function_signature(string_t _ret, const string_t& _name, string_t _file,
                       location_t _row = { 0, 0 }, location_t _col = { 0, 0 },
                       bool _loop = false, bool _info_beg = false,
                       bool _info_end = false);

    function_signature(const string_t& _ret, const string_t& _name, const string_t& _file,
                       const std::vector<string_t>& _params, location_t _row = { 0, 0 },
                       location_t _col = { 0, 0 }, bool _loop = false,
                       bool _info_beg = false, bool _info_end = false);

    static string_t get(function_signature& sig);
    string_t        get(bool _all = false, bool _save = true) const;
    string_t        get_coverage(bool _is_basic_block) const;

    bool             m_loop      = false;
    bool             m_info_beg  = false;
    bool             m_info_end  = false;
    location_t       m_row       = { 0, 0 };
    location_t       m_col       = { 0, 0 };
    string_t         m_return    = {};
    string_t         m_name      = {};
    string_t         m_params    = "()";
    string_t         m_file      = {};
    mutable string_t m_signature = {};

    friend bool operator==(const function_signature& lhs, const function_signature& rhs)
    {
        return lhs.get() == rhs.get();
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& _ar, const unsigned)
    {
        namespace cereal = tim::cereal;
        (void) get();
        _ar(cereal::make_nvp("loop", m_loop), cereal::make_nvp("info_beg", m_info_beg),
            cereal::make_nvp("info_end", m_info_end), cereal::make_nvp("row", m_row),
            cereal::make_nvp("col", m_col), cereal::make_nvp("return", m_return),
            cereal::make_nvp("name", m_name), cereal::make_nvp("params", m_params),
            cereal::make_nvp("file", m_file), cereal::make_nvp("signature", m_signature));
        (void) get();
    }
};

struct basic_block_signature
{
    using address_t = Dyninst::Address;

    address_t          start_address = {};
    address_t          last_address  = {};
    function_signature signature     = {};
};
