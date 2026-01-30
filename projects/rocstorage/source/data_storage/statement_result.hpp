// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstring>
#include <sqlite3.h>

#include "traits.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace rocstorage::data_storage
{

namespace detail
{
template <typename T>
void
column_value_impl(sqlite3_stmt* stmt, int position, T& value)
{
    using decayed_t = std::decay_t<T>;

    if constexpr(common::traits::is_optional_v<decayed_t>)
    {
        if(sqlite3_column_type(stmt, position) == SQLITE_NULL)
        {
            value = std::nullopt;
        }
        else
        {
            using inner_type_t = typename decayed_t::value_type;
            inner_type_t inner_value;
            column_value_impl(stmt, position, inner_value);
            value = inner_value;
        }
    }
    else if constexpr(common::traits::is_text_bindable_v<decayed_t>)
    {
        constexpr const char* empty_string = "";
        const unsigned char*  text         = sqlite3_column_text(stmt, position);
        if(text != nullptr)
        {
            value = strdup(reinterpret_cast<const char*>(text));
        }
        else
        {
            value = empty_string;
        }
    }
    else if constexpr(common::traits::is_double_bindable_v<decayed_t>)
    {
        value = sqlite3_column_double(stmt, position);
    }
    else if constexpr(common::traits::is_int64_bindable_v<decayed_t>)
    {
        if constexpr(std::is_same_v<decayed_t, size_t>)
        {
            value = static_cast<size_t>(sqlite3_column_int64(stmt, position));
        }
        else
        {
            value = sqlite3_column_int64(stmt, position);
        }
    }
    else if constexpr(common::traits::is_int32_bindable_v<decayed_t>)
    {
        value = sqlite3_column_int(stmt, position);
    }
    else
    {
        static_assert(!std::is_same_v<decayed_t, decayed_t>,
                      "Unsupported type for column value");
    }
}
}  // namespace detail

template <typename T>
class statement_result
{
public:
    template <typename... Members>
    statement_result(std::shared_ptr<sqlite3_stmt> stmt, Members T::*... members)
    : m_stmt(std::move(stmt))
    , m_extractor([members...](sqlite3_stmt* stmt_ptr, T& obj) {
        int position = 0;
        ((detail::column_value_impl(stmt_ptr, position++, obj.*members)), ...);
    })
    {}

    std::vector<T> to_vector()
    {
        std::vector<T> results;
        sqlite3_stmt*  stmt = m_stmt.get();

        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            results.emplace_back();
            m_extractor(stmt, results.back());
        }
        return results;
    }

private:
    std::shared_ptr<sqlite3_stmt>          m_stmt;
    std::function<void(sqlite3_stmt*, T&)> m_extractor;
};

}  // namespace rocstorage::data_storage
