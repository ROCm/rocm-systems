// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>

#include <cstddef>
#include <set>
#include <string>

#if !defined(ROCPROFSYS_SERIALIZE)
#    define ROCPROFSYS_SERIALIZE(MEMBER_VARIABLE)                                        \
        ar(::tim::cereal::make_nvp(#MEMBER_VARIABLE, MEMBER_VARIABLE))
#endif

namespace rocprofsys
{
namespace coverage
{
#if !defined(ROCPROFSYS_PYBIND11_SOURCE) || ROCPROFSYS_PYBIND11_SOURCE == 0
void
post_process();
#endif

//--------------------------------------------------------------------------------------//
//
/// \struct code_coverage
/// \brief Summary information about the code coverage
//
//--------------------------------------------------------------------------------------//

struct code_coverage
{
    using int_set_t = std::set<size_t>;
    using str_set_t = std::set<std::string>;

    enum Category
    {
        STANDARD = 0,
        ADDRESS,
        MODULE,
        FUNCTION
    };

    struct data
    {
        int_set_t addresses = {};
        str_set_t modules   = {};
        str_set_t functions = {};

        data& operator+=(const data& rhs);
        data  operator+(const data& rhs) const;

        template <typename ArchiveT>
        void serialize(ArchiveT& ar, const unsigned version);
    };

    double operator()(Category _c = STANDARD) const;
    double get(Category _c = STANDARD) const { return (*this)(_c); }

    int_set_t get_uncovered_addresses() const;
    str_set_t get_uncovered_modules() const;
    str_set_t get_uncovered_functions() const;

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned version);

    size_t count    = 0;
    size_t size     = 0;
    data   covered  = {};
    data   possible = {};
};
//
template <typename ArchiveT>
void
code_coverage::serialize(ArchiveT& ar, const unsigned version)
{
    ROCPROFSYS_SERIALIZE(count);
    ROCPROFSYS_SERIALIZE(size);
    ROCPROFSYS_SERIALIZE(covered);
    ROCPROFSYS_SERIALIZE(possible);
    if constexpr(tim::concepts::is_output_archive<ArchiveT>::value)
    {
        ar.setNextName("coverage");
        ar.startNode();
        ar(tim::cereal::make_nvp("total", get(STANDARD)));
        ar(tim::cereal::make_nvp("addresses", get(ADDRESS)));
        ar(tim::cereal::make_nvp("modules", get(MODULE)));
        ar(tim::cereal::make_nvp("functions", get(FUNCTION)));
        ar.finishNode();
    }
    (void) version;
}
//
template <typename ArchiveT>
void
code_coverage::data::serialize(ArchiveT& ar, const unsigned version)
{
    ROCPROFSYS_SERIALIZE(addresses);
    ROCPROFSYS_SERIALIZE(modules);
    ROCPROFSYS_SERIALIZE(functions);
    (void) version;
}

//--------------------------------------------------------------------------------------//
//
/// \struct coverage_data
/// \brief Detailed information about the code coverage
//
//--------------------------------------------------------------------------------------//

struct coverage_data
{
    using data_tuple_t = std::tuple<std::string_view, std::string_view, size_t>;

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned version);

    coverage_data& operator+=(const coverage_data& rhs);
    coverage_data  operator+(const coverage_data& rhs) const;
    bool           operator==(const coverage_data& rhs) const;
    bool           operator==(const data_tuple_t& rhs) const;
    bool           operator!=(const coverage_data& rhs) const;
    bool           operator<(const coverage_data& rhs) const;
    bool           operator<=(const coverage_data& rhs) const;
    bool           operator>(const coverage_data& rhs) const;
    bool           operator>=(const coverage_data& rhs) const;

    size_t      count    = 0;
    size_t      address  = 0;
    size_t      line     = 0;
    std::string module   = {};
    std::string function = {};
    std::string source   = {};
};
//
template <typename ArchiveT>
void
coverage_data::serialize(ArchiveT& ar, const unsigned version)
{
    ROCPROFSYS_SERIALIZE(count);
    ROCPROFSYS_SERIALIZE(line);
    ROCPROFSYS_SERIALIZE(address);
    ROCPROFSYS_SERIALIZE(module);
    ROCPROFSYS_SERIALIZE(function);
    ROCPROFSYS_SERIALIZE(source);
    (void) version;
}
//
}  // namespace coverage
}  // namespace rocprofsys
