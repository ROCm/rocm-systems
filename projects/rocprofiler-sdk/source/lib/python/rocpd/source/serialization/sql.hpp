// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/output/sql/common.hpp"

#include <fmt/format.h>
#include <sqlite3.h>
#include <cereal/cereal.hpp>
#include <cereal/external/base64.hpp>

#include <limits>
#include <new>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

namespace cereal
{
namespace impl
{
using namespace ::rocprofiler::tool::sql;
}  // namespace impl

class SQLite3InputArchive
: public InputArchive<SQLite3InputArchive>
, public traits::TextArchive
{
public:
    SQLite3InputArchive(sqlite3*         conn,
                        std::string_view query,
                        int64_t          len       = 0,
                        int64_t          chunk_len = 0);

    ~SQLite3InputArchive() CEREAL_NOEXCEPT override
    {
        sqlite3_finalize(m_stmt);  // Finalize statement
    }

public:
    //! Retrieves the current node name
    /*! @return nullptr if no name exists */
    static const char* getNodeName() { return nullptr; }

    static int64_t getRowCount(sqlite3* conn, std::string_view query);

    void startNode() { ++m_iterator; }
    void finishNode() const {}

    //! Sets the name for the next node created with startNode
    void setNextName(const char* name) { m_next_name = name; }

    template <typename Tp>
    void loadValue(std::string_view colname, Tp& val);

    template <typename Tp>
    void loadValue(Tp& val);

    void loadBinaryValue(void* data, size_t size, const char* name = nullptr);
    void loadSize(size_type& size) const;

    int64_t search(std::string_view);
    int64_t search(std::string_view, std::nothrow_t);

    void set_chunk_index(size_t idx);

private:
    int64_t search();

private:
    class Iterator
    {
    public:
        Iterator(SQLite3InputArchive* ar)
        : m_archive{ar}
        {}

        //! Advance to the next node
        Iterator& operator++()
        {
            m_step_ret = sqlite3_step(m_archive->m_stmt);
            return *this;
        }

    private:
        int                  m_step_ret = 0;
        SQLite3InputArchive* m_archive  = nullptr;
    };

    friend class Iterator;

private:
    sqlite3*                                 m_conn         = nullptr;
    sqlite3_stmt*                            m_stmt         = nullptr;
    const char*                              m_next_name    = nullptr;
    int64_t                                  m_row_count    = 0;
    int64_t                                  m_column_count = 0;
    int64_t                                  m_chunk_size   = 0;
    std::string                              m_query        = {};
    std::unordered_map<std::string, int64_t> m_column_names = {};
    Iterator                                 m_iterator;
    std::vector<size_t>                      m_sizes    = {};
    mutable size_t                           m_size_idx = 0;
};

template <typename Tp>
void
SQLite3InputArchive::loadValue(std::string_view colname, Tp& val)
{
    auto col = search(colname);
    auto ret = impl::extract_column<Tp>(m_stmt, col);
    if(ret) val = *ret;
}

template <typename Tp>
void
SQLite3InputArchive::loadValue(Tp& val)
{
    auto col = search();
    auto ret = impl::extract_column<Tp>(m_stmt, col);
    if(ret) val = *ret;
}

// ######################################################################
// SQLite3Archive prologue and epilogue functions
// ######################################################################

// ######################################################################
//! Prologue for NVPs for SQLite3 archives
/*! NVPs do not start or finish nodes - they just set up the names */
template <class T>
inline void
prologue(SQLite3InputArchive&, NameValuePair<T> const&)
{}

// ######################################################################
//! Epilogue for NVPs for SQLite3 archives
/*! NVPs do not start or finish nodes - they just set up the names */
template <class T>
inline void
epilogue(SQLite3InputArchive&, NameValuePair<T> const&)
{}

// ######################################################################
//! Prologue for deferred data for SQLite3 archives
/*! Do nothing for the defer wrapper */
template <class T>
inline void
prologue(SQLite3InputArchive&, DeferredData<T> const&)
{}

// ######################################################################
//! Epilogue for deferred for SQLite3 archives
/*! NVPs do not start or finish nodes - they just set up the names */
/*! Do nothing for the defer wrapper */
template <class T>
inline void
epilogue(SQLite3InputArchive&, DeferredData<T> const&)
{}

// ######################################################################
//! Prologue for SizeTags for SQLite3 archives
/*! SizeTags are strictly ignored for SQLite3, they just indicate
    that the current node should be made into an array */
template <class T>
inline void
prologue(SQLite3InputArchive&, SizeTag<T> const&)
{}

// ######################################################################
//! Epilogue for SizeTags for SQLite3 archives
/*! SizeTags are strictly ignored for SQLite3 */
template <class T>
inline void
epilogue(SQLite3InputArchive&, SizeTag<T> const&)
{}

// ######################################################################
//! Prologue for all other types for SQLite3 archives (except minimal types)
/*! Starts a new node, named either automatically or by some NVP,
    that may be given data by the type about to be archived

    Minimal types do not start or finish nodes */
template <
    class T,
    traits::EnableIf<
        !std::is_arithmetic<T>::value,
        !traits::has_minimal_base_class_serialization<T,
                                                      traits::has_minimal_input_serialization,
                                                      SQLite3InputArchive>::value,
        !traits::has_minimal_input_serialization<T, SQLite3InputArchive>::value> = traits::sfinae>
inline void
prologue(SQLite3InputArchive& ar, T const&)
{
    ar.startNode();
}

// ######################################################################
//! Epilogue for all other types other for SQLite3 archives (except minimal types)
/*! Finishes the node created in the prologue

    Minimal types do not start or finish nodes */
template <
    class T,
    traits::EnableIf<
        !std::is_arithmetic<T>::value,
        !traits::has_minimal_base_class_serialization<T,
                                                      traits::has_minimal_input_serialization,
                                                      SQLite3InputArchive>::value,
        !traits::has_minimal_input_serialization<T, SQLite3InputArchive>::value> = traits::sfinae>
inline void
epilogue(SQLite3InputArchive& ar, T const&)
{
    ar.finishNode();
}

// ######################################################################
//! Prologue for arithmetic types for SQLite3 archives
inline void
prologue(SQLite3InputArchive&, std::nullptr_t const&)
{}

// ######################################################################
//! Epilogue for arithmetic types for SQLite3 archives
inline void
epilogue(SQLite3InputArchive&, std::nullptr_t const&)
{}

// ######################################################################
//! Prologue for arithmetic types for SQLite3 archives
template <class T, traits::EnableIf<std::is_arithmetic<T>::value> = traits::sfinae>
inline void
prologue(SQLite3InputArchive&, T const&)
{}

// ######################################################################
//! Epilogue for arithmetic types for SQLite3 archives
template <class T, traits::EnableIf<std::is_arithmetic<T>::value> = traits::sfinae>
inline void
epilogue(SQLite3InputArchive&, T const&)
{}

// ######################################################################
//! Prologue for strings for SQLite3 archives
template <class CharT, class Traits, class Alloc>
inline void
prologue(SQLite3InputArchive&, std::basic_string<CharT, Traits, Alloc> const&)
{}

// ######################################################################
//! Epilogue for strings for SQLite3 archives
template <class CharT, class Traits, class Alloc>
inline void
epilogue(SQLite3InputArchive&, std::basic_string<CharT, Traits, Alloc> const&)
{}

// ######################################################################
// Common SQLite3Archive serialization functions
// ######################################################################
//! Serializing NVP types to SQLite3
template <class T>
inline void
CEREAL_LOAD_FUNCTION_NAME(SQLite3InputArchive& ar, NameValuePair<T>& t)
{
    ar.setNextName(t.name);
    ar(t.value);
}

//! Loading nullptr from SQLite3
inline void
CEREAL_LOAD_FUNCTION_NAME(SQLite3InputArchive& ar, std::nullptr_t& t)
{
    ar.loadValue(t);
}

//! Loading byte array from SQLite3
template <size_t N>
inline void
CEREAL_LOAD_FUNCTION_NAME(SQLite3InputArchive& ar, std::array<uint8_t, N>& t)
{
    ar.loadValue(t);
}

template <size_t N>
inline void
CEREAL_LOAD_FUNCTION_NAME(SQLite3InputArchive& ar, std::array<int8_t, N>& t)
{
    ar.loadValue(t);
}

//! Loading arithmetic from SQLite3
template <class T, traits::EnableIf<std::is_arithmetic<T>::value> = traits::sfinae>
inline void
CEREAL_LOAD_FUNCTION_NAME(SQLite3InputArchive& ar, T& t)
{
    ar.loadValue(t);
}

//! loading string from SQLite3
template <class CharT, class Traits, class Alloc>
inline void
CEREAL_LOAD_FUNCTION_NAME(SQLite3InputArchive& ar, std::basic_string<CharT, Traits, Alloc>& str)
{
    ar.loadValue(str);
}

// ######################################################################
//! Loading SizeTags from SQLite3
template <class T>
inline void
CEREAL_LOAD_FUNCTION_NAME(SQLite3InputArchive& ar, SizeTag<T>& st)
{
    ar.loadSize(st.size);
}
}  // namespace cereal

// register archives for polymorphic support
CEREAL_REGISTER_ARCHIVE(cereal::SQLite3InputArchive)
