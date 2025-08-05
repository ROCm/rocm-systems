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

#include <tuple>

struct function_signature
{
    using location_t = std::pair<unsigned long, unsigned long>;

    TIMEMORY_DEFAULT_OBJECT(function_signature)

    function_signature(std::string_view _ret, std::string_view _name,
                       std::string_view _file, location_t _row = { 0, 0 },
                       location_t _col = { 0, 0 }, bool _loop = false,
                       bool _info_beg = false, bool _info_end = false);

    function_signature(std::string_view _ret, std::string_view _name,
                       std::string_view _file, const std::vector<std::string>& _params,
                       location_t _row = { 0, 0 }, location_t _col = { 0, 0 },
                       bool _loop = false, bool _info_beg = false,
                       bool _info_end = false);

    function_signature& set_loop_number(uint32_t _n)
    {
        m_loop_num = _n;
        return *this;
    }

    static std::string get(function_signature& sig);
    std::string        get(bool _all = false, bool _save = true) const;
    std::string        get_coverage(bool _is_basic_block) const;

    bool                m_loop      = false;
    bool                m_info_beg  = false;
    bool                m_info_end  = false;
    uint32_t            m_loop_num  = std::numeric_limits<uint32_t>::max();
    location_t          m_row       = { 0, 0 };
    location_t          m_col       = { 0, 0 };
    std::string         m_return    = {};
    std::string         m_name      = {};
    std::string         m_params    = "()";
    std::string         m_file      = {};
    mutable std::string m_signature = {};

    friend bool operator==(const function_signature& lhs, const function_signature& rhs)
    {
        return lhs.get() == rhs.get();
    }

    friend bool operator<(const function_signature& lhs, const function_signature& rhs)
    {
        const auto loop_max = std::numeric_limits<uint32_t>::max();
        if(lhs.m_loop && !rhs.m_loop) return false;
        if(!lhs.m_loop && rhs.m_loop) return true;
        if(lhs.m_loop_num < loop_max && rhs.m_loop_num == loop_max) return false;
        if(lhs.m_loop_num == loop_max && rhs.m_loop_num < loop_max) return true;
        return std::tie(lhs.m_file, lhs.m_name, lhs.m_return, lhs.m_params,
                        lhs.m_row.first, lhs.m_col.first, lhs.m_loop_num) <
               std::tie(rhs.m_file, rhs.m_name, rhs.m_return, rhs.m_params,
                        rhs.m_row.first, rhs.m_col.first, rhs.m_loop_num);
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
