// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_agent_writer.h"

#include <unistd.h>

#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>

using namespace rocprofiler_compute_tool;

namespace
{
constexpr const char* kHeader = "node_id,logical_node_id,name,product_name\n";

std::filesystem::path test_directory()
{
    return std::filesystem::temp_directory_path() / ("agent_writer_test_" + std::to_string(::getpid()));
}
}  // namespace

std::string TestAgentWriter::format()
{
    std::string csv;

    EXPECT_TRUE(format_agents_csv(m_tool_data,
                                  [&csv](std::string_view batch)
                                  {
                                      csv.append(batch);
                                      return true;
                                  }));

    return csv;
}

void TestAgentWriter::add_agent(uint64_t handle, uint32_t node_id, const std::string& product_name)
{
    agent_record_t agent{};
    agent.node_id              = node_id;
    agent.logical_node_id      = static_cast<int32_t>(node_id);
    agent.name                 = "gfx942";
    agent.product_name         = product_name;
    m_tool_data.agents[handle] = std::move(agent);
}

TEST_F(TestAgentWriter, NoAgents_WritesOnlyTheHeader)
{
    EXPECT_EQ(format(), kHeader);
}

TEST_F(TestAgentWriter, Agents_AreWrittenInNodeIdOrder)
{
    // Handle order is not node order, and the map preserves neither.
    add_agent(4001, 3, "AMD Instinct MI300X");
    add_agent(4000, 2, "AMD Instinct MI300X");

    EXPECT_EQ(format(),
              std::string{kHeader} + "2,2,\"gfx942\",\"AMD Instinct MI300X\"\n" +
                  "3,3,\"gfx942\",\"AMD Instinct MI300X\"\n");
}

TEST_F(TestAgentWriter, ProductNameWithAComma_IsQuoted)
{
    add_agent(4000, 2, "AMD Instinct MI300X, OAM");

    EXPECT_EQ(format(), std::string{kHeader} + "2,2,\"gfx942\",\"AMD Instinct MI300X, OAM\"\n");
}

TEST_F(TestAgentWriter, SinkFailure_IsReported)
{
    add_agent(4000, 2, "AMD Instinct MI300X");

    EXPECT_FALSE(format_agents_csv(m_tool_data, [](std::string_view) { return false; }));
}

TEST_F(TestAgentWriter, NoAgents_WritesNoFile)
{
    const auto directory = test_directory();
    std::filesystem::remove_all(directory);
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    m_tool_data.agents_filename = (directory / "1234_agents.csv.gz").string();

    AgentWriter writer;
    writer.write(m_tool_data);

    EXPECT_EQ(std::distance(std::filesystem::directory_iterator{directory},
                            std::filesystem::directory_iterator{}),
              0);

    std::filesystem::remove_all(directory);
}

TEST_F(TestAgentWriter, AgentsFilename_IsTheOnlyFileWritten)
{
    const auto directory = test_directory();
    std::filesystem::remove_all(directory);
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    const auto path = directory / "1234_agents.csv.gz";

    m_tool_data.agents_filename = path.string();
    add_agent(4000, 2, "AMD Instinct MI300X");

    AgentWriter writer;
    writer.write(m_tool_data);

    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator{directory},
                            std::filesystem::directory_iterator{}),
              1);

    std::filesystem::remove_all(directory);
}

TEST_F(TestAgentWriter, UnopenableOutput_DoesNotCrash)
{
    const std::filesystem::path path = "/nonexistent-directory/agents.csv.gz";
    m_tool_data.agents_filename      = path.string();
    add_agent(4000, 2, "AMD Instinct MI300X");

    AgentWriter writer;
    EXPECT_NO_THROW(writer.write(m_tool_data));
    EXPECT_FALSE(std::filesystem::exists(path));
}
