// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_code_object_writer.h"
#include "nlohmann/json.hpp"

TEST_F(test_code_object_writer_t, ProvidedNoData_ReturnsMinimalJson)
{
    const auto& result = m_writer.get_result();
    EXPECT_FALSE(result.empty());
    EXPECT_TRUE(nlohmann::json::accept(result));
}

TEST_F(test_code_object_writer_t, ProvidedStartCodeObjWithoutEnd_Throws)
{
    m_writer.start_code_obj(0);
    EXPECT_THROW(m_writer.get_result(), std::runtime_error);
}

TEST_F(test_code_object_writer_t, ProvidedEndCodeObjWithoutStart_Throws)
{
    m_writer.end_code_obj();
    EXPECT_THROW(m_writer.get_result(), std::runtime_error);
}

TEST_F(test_code_object_writer_t, ProvidedCodeObjDesc_SerializesIt)
{
    constexpr uint32_t id0 = 10;
    constexpr uint32_t id1 = 20;
    m_writer.start_code_obj(id0);
    m_writer.end_code_obj();
    m_writer.start_code_obj(id1);
    m_writer.end_code_obj();
    const auto& result = m_writer.get_result();
    const auto json = nlohmann::json::parse(result);
    EXPECT_EQ(json["code_objects"][0]["id"], id0);
    EXPECT_EQ(json["code_objects"][1]["id"], id1);
}


void test_code_object_writer_t::SetUp()
{
}