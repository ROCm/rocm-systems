// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace aqlprofile
{
namespace packet_dump
{
class aqlprofile_packet_golden_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const char* fixture_root_path = std::getenv(fixture_root_path_env);
        ASSERT_NE(fixture_root_path, nullptr) << fixture_root_path_env << " is not set";
        fixture_root_path_ = fixture_root_path;
        ASSERT_TRUE(std::filesystem::is_directory(fixture_root_path_)) << fixture_root_path_;

        const char* packet_dump_path = std::getenv(packet_dump_path_env);
        ASSERT_NE(packet_dump_path, nullptr) << packet_dump_path_env << " is not set";
        packet_dump_path_ = packet_dump_path;
    }

    void select_fixture(const std::string& gfx)
    {
        const std::filesystem::path fixture_path = fixture_root_path_ / gfx;
        ASSERT_TRUE(std::filesystem::is_directory(fixture_path)) << fixture_path;
        ASSERT_EQ(::setenv(ip_discovery_path_env, fixture_path.c_str(), 1), 0);
    }

    static constexpr const char* ip_discovery_path_env = "AQLPROFILE_CUSTOM_IP_DISCOVERY_PATH";
    static constexpr const char* fixture_root_path_env = "AQLPROFILE_PACKET_FIXTURE_PATH";
    static constexpr const char* packet_dump_path_env  = "AQLPROFILE_PACKET_DUMP_PATH";

    std::filesystem::path fixture_root_path_{};
    std::filesystem::path packet_dump_path_{};
};

}  // namespace packet_dump
}  // namespace aqlprofile
