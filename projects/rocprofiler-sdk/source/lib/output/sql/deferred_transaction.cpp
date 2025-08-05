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

#include "lib/output/sql/deferred_transaction.hpp"
#include "lib/output/sql/common.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"

#include <fmt/format.h>
#include <sqlite3.h>

#include <iomanip>
#include <sstream>
#include <thread>

namespace rocprofiler
{
namespace tool
{
namespace sql
{
deferred_transaction::deferred_transaction(sqlite3* conn)
: m_conn{conn}
{
    ROCP_CI_LOG_IF(INFO, m_conn == nullptr) << "rocprofiler::tool::sql::deferred_transaction "
                                               "constructed will nullptr to sqlite3 connection";
    if(m_conn)
    {
        execute_raw_sql_statements(m_conn, "BEGIN DEFERRED TRANSACTION");
    }
}

deferred_transaction::~deferred_transaction()
{
    if(m_conn)
    {
        execute_raw_sql_statements(m_conn, "END TRANSACTION");
    }
}
}  // namespace sql
}  // namespace tool
}  // namespace rocprofiler
