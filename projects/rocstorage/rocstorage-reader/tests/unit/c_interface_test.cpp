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

#include "rocprofvis_c_interface.h"

#include <gtest/gtest.h>

namespace {

class CInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_trace = rocprofvis_dm_create_trace();
    }

    void TearDown() override {
        if (m_trace != nullptr) {
            rocprofvis_dm_delete_trace(m_trace);
            m_trace = nullptr;
        }
    }

    rocprofvis_dm_trace_t m_trace;
};

TEST_F(CInterfaceTest, CreateTraceReturnsValidHandle) {
    ASSERT_NE(m_trace, nullptr);
}

TEST_F(CInterfaceTest, GetStartTimeReturnsZeroInitially) {
    uint64_t value = rocprofvis_dm_get_property_as_uint64(
        m_trace, kRPVDMStartTimeUInt64, 0);
    EXPECT_EQ(value, 0u);
}

TEST_F(CInterfaceTest, GetEndTimeReturnsZeroInitially) {
    uint64_t value = rocprofvis_dm_get_property_as_uint64(
        m_trace, kRPVDMEndTimeUInt64, 0);
    EXPECT_EQ(value, 0u);
}

TEST_F(CInterfaceTest, GetNumberOfTracksReturnsZeroInitially) {
    uint64_t value = rocprofvis_dm_get_property_as_uint64(
        m_trace, kRPVDMNumberOfTracksUInt64, 0);
    EXPECT_EQ(value, 0u);
}

TEST_F(CInterfaceTest, GetNumberOfTablesReturnsZeroInitially) {
    uint64_t value = rocprofvis_dm_get_property_as_uint64(
        m_trace, kRPVDMNumberOfTablesUInt64, 0);
    EXPECT_EQ(value, 0u);
}

TEST_F(CInterfaceTest, GetDatabaseHandleReturnsNullInitially) {
    rocprofvis_dm_handle_t db = rocprofvis_dm_get_property_as_handle(
        m_trace, kRPVDMDatabaseHandle, 0);
    EXPECT_EQ(db, nullptr);
}

TEST_F(CInterfaceTest, DeleteTraceSucceeds) {
    rocprofvis_dm_trace_t trace = rocprofvis_dm_create_trace();
    ASSERT_NE(trace, nullptr);

    rocprofvis_dm_result_t result = rocprofvis_dm_delete_trace(trace);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(CInterfaceTest, DeleteAllTimeSlicesSucceedsOnEmptyTrace) {
    rocprofvis_dm_result_t result = rocprofvis_dm_delete_all_time_slices(m_trace);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(CInterfaceTest, DeleteAllTablesSucceedsOnEmptyTrace) {
    rocprofvis_dm_result_t result = rocprofvis_dm_delete_all_tables(m_trace);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(CInterfaceTest, DeleteAllEventPropertiesForFlowTraceSucceeds) {
    rocprofvis_dm_result_t result = rocprofvis_dm_delete_all_event_properties_for(
        m_trace, kRPVDMEventFlowTrace);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(CInterfaceTest, DeleteAllEventPropertiesForStackTraceSucceeds) {
    rocprofvis_dm_result_t result = rocprofvis_dm_delete_all_event_properties_for(
        m_trace, kRPVDMEventStackTrace);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(CInterfaceTest, DeleteAllEventPropertiesForExtDataSucceeds) {
    rocprofvis_dm_result_t result = rocprofvis_dm_delete_all_event_properties_for(
        m_trace, kRPVDMEventExtData);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

class CInterfaceFutureTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_future = rocprofvis_db_future_alloc(nullptr, nullptr);
    }

    void TearDown() override {
        if (m_future != nullptr) {
            rocprofvis_db_future_free(m_future);
            m_future = nullptr;
        }
    }

    rocprofvis_db_future_t m_future;
};

TEST_F(CInterfaceFutureTest, AllocFutureReturnsValidHandle) {
    ASSERT_NE(m_future, nullptr);
}

TEST_F(CInterfaceFutureTest, FreeFutureSucceeds) {
    rocprofvis_db_future_t future = rocprofvis_db_future_alloc(nullptr, nullptr);
    ASSERT_NE(future, nullptr);
    // Should not crash
    rocprofvis_db_future_free(future);
}

TEST_F(CInterfaceFutureTest, CancelFutureDoesNotCrash) {
    // Cancel should not crash even on a newly allocated future
    rocprofvis_db_future_cancel(m_future);
}

class CInterfacePropertyAccessTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_trace = rocprofvis_dm_create_trace();
    }

    void TearDown() override {
        if (m_trace != nullptr) {
            rocprofvis_dm_delete_trace(m_trace);
            m_trace = nullptr;
        }
    }

    rocprofvis_dm_trace_t m_trace;
};

TEST_F(CInterfacePropertyAccessTest, GetPropertyAsUint64WithValidProperty) {
    uint64_t value = 0;
    rocprofvis_dm_result_t result = rocprofvis_dm_get_property_as_uint64(
        m_trace, kRPVDMStartTimeUInt64, 0, &value);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
    EXPECT_EQ(value, 0u);
}

TEST_F(CInterfacePropertyAccessTest, GetPropertyAsHandleWithValidProperty) {
    rocprofvis_dm_handle_t value = nullptr;
    rocprofvis_dm_result_t result = rocprofvis_dm_get_property_as_handle(
        m_trace, kRPVDMDatabaseHandle, 0, &value);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
    EXPECT_EQ(value, nullptr);
}

TEST_F(CInterfacePropertyAccessTest, GetMemoryFootprintReturnsNonNegative) {
    uint64_t footprint = rocprofvis_dm_get_property_as_uint64(
        m_trace, kRPVDMTraceMemoryFootprintUInt64, 0);
    EXPECT_GE(footprint, 0u);
}

class CInterfaceBindingTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_trace = rocprofvis_dm_create_trace();
    }

    void TearDown() override {
        if (m_trace != nullptr) {
            rocprofvis_dm_delete_trace(m_trace);
            m_trace = nullptr;
        }
    }

    rocprofvis_dm_trace_t m_trace;
};

TEST_F(CInterfaceBindingTest, BindToNullDatabaseFails) {
    rocprofvis_dm_result_t result = rocprofvis_dm_bind_trace_to_database(
        m_trace, nullptr);
    EXPECT_EQ(result, kRocProfVisDmResultInvalidParameter);
}

} // namespace
