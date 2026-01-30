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

#include "rocstorage/reader.hpp"
#include "rocstorage/storage.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace
{

class reader_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<rocstorage::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<rocstorage::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                            m_database_path{ ROCPD_DB_PATH };
    std::unique_ptr<rocstorage::storage_t> m_storage;
    std::shared_ptr<rocstorage::reader_t>  m_reader;
};

TEST_F(reader_test, create_reader_instance) { ASSERT_NE(m_reader, nullptr); }

TEST_F(reader_test, get_node_list_returns_correct_value)
{
    auto node_list = m_reader->get_node_list();
    ASSERT_EQ(node_list.size(), 1);

    ASSERT_EQ(node_list[0]->node_id, 9162464413581981795);
    ASSERT_EQ(node_list[0]->hash, 9162464413581981795);
    ASSERT_STREQ(node_list[0]->machine_id, "7cd7e017ddf442f5b7ce8428af366498");
    ASSERT_STREQ(node_list[0]->system_name, "Linux");
    ASSERT_STREQ(node_list[0]->hostname, "smci350-zts-gtu-c14-05");
    ASSERT_STREQ(node_list[0]->release, "5.15.0-70-generic");
    ASSERT_STREQ(node_list[0]->version, "#77-Ubuntu SMP Tue Mar 21 14:02:37 UTC 2023");
    ASSERT_STREQ(node_list[0]->hardware_name, "x86_64");
    ASSERT_STREQ(node_list[0]->domain_name, "(none)");
}

}  // namespace
