// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <timemory/mpl/concepts.hpp>
#include <timemory/units.hpp>

namespace rocprofsys
{
namespace component
{
struct cpu_freq
: comp::empty_base
, tim::component::base_format<cpu_freq>
, tim::component::base_data<std::vector<uint64_t>, 1>
{
    using base_type    = comp::empty_base;
    using this_type    = cpu_freq;
    using value_type   = std::vector<uint64_t>;
    using storage_type = tim::storage<cpu_freq, value_type>;
    using cpu_id_set_t = std::set<uint64_t>;

    ROCPROFSYS_DEFAULT_OBJECT(cpu_freq)

    // string id for component
    static std::string label();
    static std::string description();
    static int64_t     unit();
    static std::string display_unit();

    static void          configure();
    static cpu_id_set_t& get_enabled_cpus();
    static value_type    record();

    // this will get called right before fork
    void        start();
    void        stop();
    cpu_freq&   sample();
    std::string as_string() const;

    float              at(size_t _idx, int64_t _unit = unit()) const;
    std::vector<float> get(int64_t _unit = unit()) const;

public:
    static auto          get_label() { return label(); }
    static auto          get_description() { return description(); }
    static auto          get_unit() { return unit(); }
    static auto          get_display_unit() { return display_unit(); }
    static int64_t       get_laps() { return 0; }
    static storage_type* get_storage() { return nullptr; }

    auto get_display() const { return as_string(); }

    friend std::ostream& operator<<(std::ostream& _os, const cpu_freq& _v)
    {
        return (_os << _v.as_string());
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& _ar, const unsigned _version)
    {
        if constexpr(tim::concepts::is_output_archive<ArchiveT>::value)
            operation::serialization<cpu_freq>{}(*this, _ar, _version);
        else
            _ar(tim::cereal::make_nvp("value", value));
        (void) _version;
    }

    this_type& operator+=(const this_type& _rhs)
    {
        using namespace tim::stl;
        value += _rhs.value;
        return *this;
    }

    this_type& operator-=(const this_type& _rhs)
    {
        using namespace tim::stl;
        value -= _rhs.value;
        return *this;
    }

private:
    using tim::component::base_data<value_type, 1>::value;
};
}  // namespace component
}  // namespace rocprofsys
