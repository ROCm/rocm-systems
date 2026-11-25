// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_output_registry.h"

#include <stdexcept>

using namespace rocprofiler_compute_tool;

void CountingWriter::write(tool_data_t& /*tool_data*/)
{
    m_write_count.fetch_add(1);
}

void ThrowingWriter::write(tool_data_t& /*tool_data*/)
{
    throw std::runtime_error{"writer failed"};
}

void TestOutputRegistry::SetUp()
{
    m_first  = std::make_shared<CountingWriter>();
    m_second = std::make_shared<CountingWriter>();
}

TEST_F(TestOutputRegistry, GenerateAll_RunsEveryWriterOnce)
{
    m_registry.register_writer(m_first);
    m_registry.register_writer(m_second);

    m_registry.generate_all(m_tool_data);

    EXPECT_EQ(m_first->write_count(), 1);
    EXPECT_EQ(m_second->write_count(), 1);
}

TEST_F(TestOutputRegistry, ThrowingWriter_DoesNotStopTheOthers)
{
    m_registry.register_writer(m_first);
    m_registry.register_writer(std::make_shared<ThrowingWriter>());
    m_registry.register_writer(m_second);

    EXPECT_NO_THROW(m_registry.generate_all(m_tool_data));

    EXPECT_EQ(m_first->write_count(), 1);
    EXPECT_EQ(m_second->write_count(), 1);
}

TEST_F(TestOutputRegistry, ReplaceWriter_SwapsTheOneOfThatName)
{
    m_registry.register_writer(m_first);

    EXPECT_TRUE(m_registry.replace_writer(m_first->name(), m_second));
    m_registry.generate_all(m_tool_data);

    EXPECT_EQ(m_first->write_count(), 0);
    EXPECT_EQ(m_second->write_count(), 1);
}

TEST_F(TestOutputRegistry, ReplaceWriter_UnknownName_ChangesNothing)
{
    m_registry.register_writer(m_first);

    EXPECT_FALSE(m_registry.replace_writer("absent", m_second));
    m_registry.generate_all(m_tool_data);

    EXPECT_EQ(m_first->write_count(), 1);
    EXPECT_EQ(m_second->write_count(), 0);
}

TEST_F(TestOutputRegistry, Reset_DropsTheWriters)
{
    m_registry.register_writer(m_first);
    m_registry.reset();

    m_registry.generate_all(m_tool_data);

    EXPECT_EQ(m_first->write_count(), 0);
}
