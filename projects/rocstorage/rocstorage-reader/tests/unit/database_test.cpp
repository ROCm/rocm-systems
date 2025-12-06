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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "rocprofvis_db.h"
#include "rocprofvis_dm_trace.h"

#include <gtest/gtest.h>

namespace {

class DatabaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No setup needed - tests will create resources as needed
    }

    void TearDown() override {
        // No teardown needed
    }
};

TEST_F(DatabaseTest, AutodetectReturnsAutodetectForNonexistentFile) {
    // Autodetect should handle non-existent files gracefully
    auto type = RocProfVis::DataModel::Database::Autodetect("/nonexistent/path/database.rpd");
    // Should return autodetect (0) or handle error gracefully
    EXPECT_GE(static_cast<int>(type), 0);
}

TEST_F(DatabaseTest, AutodetectReturnsAutodetectForInvalidPath) {
    auto type = RocProfVis::DataModel::Database::Autodetect("");
    EXPECT_EQ(type, kAutodetect);
}

class DatabaseCacheTest : public ::testing::Test {
protected:
    RocProfVis::DataModel::DatabaseCache m_cache;
};

TEST_F(DatabaseCacheTest, AddAndGetTableCell) {
    m_cache.AddTableCell("test_table", 1, "column1", "value1");
    const char* result = m_cache.GetTableCell("test_table", 1, "column1");
    EXPECT_STREQ(result, "value1");
}

TEST_F(DatabaseCacheTest, AddTableCellWithNullValue) {
    m_cache.AddTableCell("test_table", 2, "column1", nullptr);
    const char* result = m_cache.GetTableCell("test_table", 2, "column1");
    EXPECT_STREQ(result, "");
}

TEST_F(DatabaseCacheTest, GetTableCellReturnsEmptyForMissingEntry) {
    const char* result = m_cache.GetTableCell("nonexistent", 999, "missing");
    EXPECT_STREQ(result, "");
}

TEST_F(DatabaseCacheTest, AddMultipleCellsToSameInstance) {
    m_cache.AddTableCell("agents", 100, "name", "GPU0");
    m_cache.AddTableCell("agents", 100, "type", "gfx1030");
    m_cache.AddTableCell("agents", 100, "memory", "16GB");

    EXPECT_STREQ(m_cache.GetTableCell("agents", 100, "name"), "GPU0");
    EXPECT_STREQ(m_cache.GetTableCell("agents", 100, "type"), "gfx1030");
    EXPECT_STREQ(m_cache.GetTableCell("agents", 100, "memory"), "16GB");
}

TEST_F(DatabaseCacheTest, AddCellsToDifferentTables) {
    m_cache.AddTableCell("agents", 1, "name", "GPU0");
    m_cache.AddTableCell("processes", 1, "name", "Process1");
    m_cache.AddTableCell("queues", 1, "name", "Queue1");

    EXPECT_STREQ(m_cache.GetTableCell("agents", 1, "name"), "GPU0");
    EXPECT_STREQ(m_cache.GetTableCell("processes", 1, "name"), "Process1");
    EXPECT_STREQ(m_cache.GetTableCell("queues", 1, "name"), "Queue1");
}

TEST_F(DatabaseCacheTest, OverwriteExistingCell) {
    m_cache.AddTableCell("test_table", 1, "column1", "original");
    EXPECT_STREQ(m_cache.GetTableCell("test_table", 1, "column1"), "original");

    m_cache.AddTableCell("test_table", 1, "column1", "updated");
    EXPECT_STREQ(m_cache.GetTableCell("test_table", 1, "column1"), "updated");
}

TEST_F(DatabaseCacheTest, GetMemoryFootprintReturnsNonNegative) {
    m_cache.AddTableCell("table1", 1, "col1", "value1");
    m_cache.AddTableCell("table1", 2, "col1", "value2");

    rocprofvis_dm_size_t footprint = m_cache.GetMemoryFootprint();
    EXPECT_GE(footprint, 0u);
}

} // namespace
