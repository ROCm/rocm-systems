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

#include <rocstorage/storage.hpp>
#include <rocstorage/writer.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace {

class writer_test : public ::testing::Test {
protected:
  void SetUp() override {
    m_database_path =
        "test_writer_" +
        std::to_string(
            ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
        ".db";
    m_uuid = "12345";
    m_storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
    m_writer = m_storage->get_writer();
  }

  void TearDown() override {
    m_writer.reset();
    m_storage.reset();
    std::remove(m_database_path.c_str());
  }

  std::string m_database_path;
  std::string m_uuid;
  std::unique_ptr<rocm::storage> m_storage;
  std::shared_ptr<rocstorage::writer> m_writer;
};

// ----------------------------------------------------------------
// rocpd_string table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_string_returns_valid_id) {
  auto id = m_writer->insert_string("test_string");
  EXPECT_GT(id, 0u);
}

TEST_F(writer_test, insert_multiple_strings_returns_unique_ids) {
  auto id1 = m_writer->insert_string("string_one");
  auto id2 = m_writer->insert_string("string_two");
  auto id3 = m_writer->insert_string("string_three");

  EXPECT_NE(id1, id2);
  EXPECT_NE(id2, id3);
  EXPECT_NE(id1, id3);
}

// ----------------------------------------------------------------
// rocpd_info_node table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_node_info) {
  EXPECT_NO_THROW(m_writer->insert_node_info(
      1, 0x12345678, "machine-001", "Linux", "test-host", "5.15.0", "#1 SMP",
      "x86_64", "localdomain"));
}

// ----------------------------------------------------------------
// rocpd_info_process table tests (requires node)
// ----------------------------------------------------------------

TEST_F(writer_test, insert_process_info_with_valid_node) {
  m_writer->insert_node_info(1, 0xAABBCCDD, "machine-proc", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");

  EXPECT_NO_THROW(m_writer->insert_process_info(1, 1, 1234, 0, 0, 1000000,
                                                2000000, "/usr/bin/test"));
}

TEST_F(writer_test, insert_process_info_with_environment_and_extdata) {
  m_writer->insert_node_info(1, 0x11223344, "machine-env", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");

  EXPECT_NO_THROW(m_writer->insert_process_info(1, 1, 5678, 100, 200, 1000,
                                                2000, "/bin/app", "{}", "{}"));
}

// ----------------------------------------------------------------
// rocpd_info_thread table tests (requires node and process)
// ----------------------------------------------------------------

TEST_F(writer_test, insert_thread_info_with_valid_parents) {
  m_writer->insert_node_info(1, 0xDEADBEEF, "machine-thread", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 1000, 0, 0, 100, 200, "/bin/app");

  auto thread_pk = m_writer->insert_thread_info(1, 1000, 1000, 12345,
                                                "worker_thread", 150, 180);

  EXPECT_GT(thread_pk, 0u);
}

TEST_F(writer_test, map_thread_id_to_primary_key) {
  m_writer->insert_node_info(1, 0x11112222, "machine-map", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 1000, 0, 0, 100, 200, "/bin/app");

  auto pk1 = m_writer->insert_thread_info(1, 1000, 1000, 111, "t1", 100, 200);
  auto pk2 = m_writer->insert_thread_info(1, 1000, 1000, 222, "t2", 100, 200);

  auto mapped_pk1 = m_writer->map_thread_id_to_primary_key(111);
  auto mapped_pk2 = m_writer->map_thread_id_to_primary_key(222);
  auto mapped_pk1_again = m_writer->map_thread_id_to_primary_key(111);

  EXPECT_EQ(pk1, mapped_pk1);
  EXPECT_EQ(pk2, mapped_pk2);
  EXPECT_EQ(mapped_pk1, mapped_pk1_again);
  EXPECT_NE(mapped_pk1, mapped_pk2);
}

// ----------------------------------------------------------------
// rocpd_info_agent table tests (requires node and process)
// ----------------------------------------------------------------

TEST_F(writer_test, insert_gpu_agent_with_valid_parents) {
  m_writer->insert_node_info(1, 0xCAFEBABE, "machine-agent", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 2000, 0, 0, 100, 200, "/bin/rocm_app");

  auto agent_id = m_writer->insert_agent(
      1, 2000, "GPU", 0, 0, 0, 0xABCDEF, "gfx90a", "AMD Instinct MI200",
      "Advanced Micro Devices", "MI210", "default");

  EXPECT_GT(agent_id, 0u);
}

TEST_F(writer_test, insert_cpu_agent) {
  m_writer->insert_node_info(1, 0xFEEDFACE, "machine-cpu", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 3000, 0, 0, 100, 200, "/bin/cpu_app");

  auto agent_id =
      m_writer->insert_agent(1, 3000, "CPU", 0, 0, 0, 0, "cpu0", "AMD EPYC",
                             "AMD", "EPYC 7763", "host");

  EXPECT_GT(agent_id, 0u);
}

// ----------------------------------------------------------------
// rocpd_info_queue and rocpd_info_stream table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_queue_info) {
  m_writer->insert_node_info(1, 0x11111111, "machine-queue", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 4000, 0, 0, 100, 200, "/bin/queue_app");

  EXPECT_NO_THROW(m_writer->insert_queue_info(1, 1, 4000, "hsa_queue_0"));
}

TEST_F(writer_test, insert_stream_info) {
  m_writer->insert_node_info(1, 0x22222222, "machine-stream", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 5000, 0, 0, 100, 200, "/bin/stream_app");

  EXPECT_NO_THROW(m_writer->insert_stream_info(1, 1, 5000, "hip_stream_0"));
}

// ----------------------------------------------------------------
// rocpd_track table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_track_without_thread) {
  m_writer->insert_node_info(1, 0x33333333, "machine-track", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 6000, 0, 0, 100, 200, "/bin/track_app");

  EXPECT_NO_THROW(m_writer->insert_track("main_track", 1, 6000, std::nullopt));
}

TEST_F(writer_test, insert_track_with_thread) {
  m_writer->insert_node_info(1, 0x44444444, "machine-track-t", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 7000, 0, 0, 100, 200, "/bin/track_app");
  auto thread_pk =
      m_writer->insert_thread_info(1, 7000, 7000, 7001, "worker", 100, 200);

  EXPECT_NO_THROW(m_writer->insert_track("thread_track", 1, 7000, thread_pk));
}

// ----------------------------------------------------------------
// rocpd_event table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_event_returns_valid_id) {
  auto category_id = m_writer->insert_string("hip_api");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 100);

  EXPECT_GT(event_id, 0u);
}

TEST_F(writer_test, insert_event_with_json_data) {
  auto category_id = m_writer->insert_string("kernel_execution");
  auto event_id = m_writer->insert_event(
      category_id, 1, 0, 200, R"([{"file":"main.cpp","line":42}])",
      R"({"source":"app.cpp","line":100})", R"({"duration_ns":1234567})");

  EXPECT_GT(event_id, 0u);
}

// ----------------------------------------------------------------
// rocpd_arg table tests (requires event)
// ----------------------------------------------------------------

TEST_F(writer_test, insert_args_for_event) {
  auto category_id = m_writer->insert_string("hip_api_call");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 300);

  EXPECT_NO_THROW(
      m_writer->insert_args(event_id, 0, "void*", "dst", "0x7fff0000"));
  EXPECT_NO_THROW(m_writer->insert_args(event_id, 1, "size_t", "size", "1024"));
}

// ----------------------------------------------------------------
// rocpd_region table tests (requires node, process, thread, string, event)
// ----------------------------------------------------------------

TEST_F(writer_test, insert_region) {
  m_writer->insert_node_info(1, 0x55555555, "machine-region", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 8000, 0, 0, 100, 200, "/bin/region_app");
  auto thread_pk =
      m_writer->insert_thread_info(1, 8000, 8000, 8001, "main", 100, 200);

  auto name_id = m_writer->insert_string("hipMemcpy");
  auto category_id = m_writer->insert_string("hip_api");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 400);

  EXPECT_NO_THROW(m_writer->insert_region(1, 8000, thread_pk, 1000000, 1500000,
                                          name_id, event_id));
}

// ----------------------------------------------------------------
// rocpd_sample table tests (requires track, event)
// ----------------------------------------------------------------

TEST_F(writer_test, insert_sample) {
  m_writer->insert_node_info(1, 0x66666666, "machine-sample", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 9000, 0, 0, 100, 200, "/bin/sample_app");
  m_writer->insert_track("sample_track", 1, 9000, std::nullopt);

  auto category_id = m_writer->insert_string("counter");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 500);

  EXPECT_NO_THROW(m_writer->insert_sample("sample_track", 2000000, event_id));
}

// ----------------------------------------------------------------
// rocpd_info_pmc and rocpd_pmc_event table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_pmc_description) {
  m_writer->insert_node_info(1, 0x77777777, "machine-pmc", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 10000, 0, 0, 100, 200, "/bin/pmc_app");
  auto agent_id = m_writer->insert_agent(1, 10000, "GPU", 0, 0, 0, 0xABC,
                                         "gfx90a", "MI200", "AMD", "MI210", "");

  EXPECT_NO_THROW(m_writer->insert_pmc_description(
      1, 10000, agent_id, "GPU", 0x100, 0, "SQ_WAVES", "SQ_WAVES", "Wave count",
      "Number of waves", "SQ", "waves", "ACCUM", "SQ", "", 0, 0));
}

TEST_F(writer_test, insert_pmc_event) {
  m_writer->insert_node_info(1, 0x88888888, "machine-pmc-ev", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 11000, 0, 0, 100, 200, "/bin/pmc_ev");
  auto agent_id = m_writer->insert_agent(1, 11000, "GPU", 0, 0, 0, 0xDEF,
                                         "gfx90a", "MI200", "AMD", "MI210", "");

  m_writer->insert_pmc_description(
      1, 11000, agent_id, "GPU", 0x100, 0, "SQ_WAVES", "SQ_WAVES", "Wave count",
      "Number of waves", "SQ", "waves", "ACCUM", "SQ", "", 0, 0);

  auto category_id = m_writer->insert_string("pmc");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 600);

  EXPECT_NO_THROW(
      m_writer->insert_pmc_event(event_id, agent_id, "SQ_WAVES", 42.5));
}

// ----------------------------------------------------------------
// rocpd_info_code_object and rocpd_info_kernel_symbol table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_code_object) {
  m_writer->insert_node_info(1, 0x99999999, "machine-cobj", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 12000, 0, 0, 100, 200, "/bin/cobj_app");
  auto agent_id = m_writer->insert_agent(1, 12000, "GPU", 0, 0, 0, 0x111,
                                         "gfx90a", "MI200", "AMD", "MI210", "");

  EXPECT_NO_THROW(m_writer->insert_code_object(1, 1, 12000, agent_id,
                                               "file:///path/to/kernel.hsaco",
                                               0x10000, 0x1000, 0, "FILE"));
}

TEST_F(writer_test, insert_kernel_symbol) {
  m_writer->insert_node_info(1, 0xAAAAAAAA, "machine-ksym", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 13000, 0, 0, 100, 200, "/bin/ksym_app");
  auto agent_id = m_writer->insert_agent(1, 13000, "GPU", 0, 0, 0, 0x222,
                                         "gfx90a", "MI200", "AMD", "MI210", "");
  m_writer->insert_code_object(1, 1, 13000, agent_id, "file:///kernel.hsaco",
                               0x10000, 0x1000, 0, "FILE");

  EXPECT_NO_THROW(m_writer->insert_kernel_symbol(
      1, 1, 13000, 1, "vectorAdd", "vectorAdd(float*, float*, float*, int)",
      0x1234, 256, 8, 65536, 0, 32, 64, 0));
}

// ----------------------------------------------------------------
// rocpd_kernel_dispatch table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_kernel_dispatch) {
  m_writer->insert_node_info(1, 0xBBBBBBBB, "machine-kdisp", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 14000, 0, 0, 100, 200, "/bin/kdisp_app");
  auto thread_pk = m_writer->insert_thread_info(1, 14000, 14000, 14001,
                                                "dispatch_thread", 100, 200);
  auto agent_id = m_writer->insert_agent(1, 14000, "GPU", 0, 0, 0, 0x333,
                                         "gfx90a", "MI200", "AMD", "MI210", "");
  m_writer->insert_queue_info(1, 1, 14000, "dispatch_queue");
  m_writer->insert_stream_info(1, 1, 14000, "dispatch_stream");
  m_writer->insert_code_object(1, 1, 14000, agent_id, "file:///k.hsaco",
                               0x10000, 0x1000, 0, "FILE");
  m_writer->insert_kernel_symbol(1, 1, 14000, 1, "kernel", "kernel()", 0x100,
                                 256, 8, 65536, 0, 32, 64, 0);

  auto region_name_id = m_writer->insert_string("kernel_region");
  auto category_id = m_writer->insert_string("kernel");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 700);

  EXPECT_NO_THROW(m_writer->insert_kernel_dispatch(
      1, 14000, thread_pk, agent_id, 1, 1, 1, 1, 5000000, 6000000, 0, 65536,
      256, 1, 1, 1024, 1, 1, region_name_id, event_id));
}

// ----------------------------------------------------------------
// rocpd_memory_copy table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_memory_copy) {
  m_writer->insert_node_info(1, 0xCCCCCCCC, "machine-mcopy", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 15000, 0, 0, 100, 200, "/bin/mcopy_app");
  auto thread_pk = m_writer->insert_thread_info(1, 15000, 15000, 15001,
                                                "copy_thread", 100, 200);
  auto gpu_agent = m_writer->insert_agent(
      1, 15000, "GPU", 0, 0, 0, 0x444, "gfx90a", "MI200", "AMD", "MI210", "");
  auto cpu_agent = m_writer->insert_agent(1, 15000, "CPU", 1, 0, 0, 0, "cpu",
                                          "EPYC", "AMD", "EPYC", "");
  m_writer->insert_queue_info(1, 1, 15000, "copy_queue");
  m_writer->insert_stream_info(1, 1, 15000, "copy_stream");

  auto name_id = m_writer->insert_string("hipMemcpyHtoD");
  auto region_name_id = m_writer->insert_string("memcpy_region");
  auto category_id = m_writer->insert_string("memory");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 800);

  EXPECT_NO_THROW(m_writer->insert_memory_copy(
      1, 15000, thread_pk, 7000000, 7500000, name_id, gpu_agent, 0x7FFF00000000,
      cpu_agent, 0x100000, 1048576, 1, 1, region_name_id, event_id));
}

// ----------------------------------------------------------------
// rocpd_memory_allocate table tests
// ----------------------------------------------------------------

TEST_F(writer_test, insert_memory_alloc) {
  m_writer->insert_node_info(1, 0xDDDDDDDD, "machine-malloc", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 16000, 0, 0, 100, 200, "/bin/malloc_app");
  auto thread_pk = m_writer->insert_thread_info(1, 16000, 16000, 16001,
                                                "alloc_thread", 100, 200);
  auto agent_id = m_writer->insert_agent(1, 16000, "GPU", 0, 0, 0, 0x555,
                                         "gfx90a", "MI200", "AMD", "MI210", "");
  m_writer->insert_queue_info(1, 1, 16000, "alloc_queue");
  m_writer->insert_stream_info(1, 1, 16000, "alloc_stream");

  auto category_id = m_writer->insert_string("allocation");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 900);

  EXPECT_NO_THROW(m_writer->insert_memory_alloc(
      1, 16000, thread_pk, agent_id, "ALLOC", "REAL", 8000000, 8100000,
      0x7FFF80000000, 4194304, 1, 1, event_id));
}

TEST_F(writer_test, insert_memory_alloc_without_agent) {
  m_writer->insert_node_info(1, 0xEEEEEEEE, "machine-malloc2", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 17000, 0, 0, 100, 200, "/bin/malloc2");
  auto thread_pk = m_writer->insert_thread_info(1, 17000, 17000, 17001,
                                                "alloc_thread", 100, 200);
  m_writer->insert_queue_info(1, 1, 17000, "host_queue");
  m_writer->insert_stream_info(1, 1, 17000, "host_stream");

  auto category_id = m_writer->insert_string("host_allocation");
  auto event_id = m_writer->insert_event(category_id, 0, 0, 1000);

  EXPECT_NO_THROW(m_writer->insert_memory_alloc(
      1, 17000, thread_pk, std::nullopt, "ALLOC", "REAL", 9000000, 9050000,
      0x100000, 1024, 1, 1, event_id));
}

// ----------------------------------------------------------------
// Flush tests
// ----------------------------------------------------------------

TEST_F(writer_test, flush_completes_without_error) {
  m_writer->insert_string("data_to_flush");
  EXPECT_NO_THROW(m_writer->flush());
}

TEST_F(writer_test, flush_after_complex_workflow) {
  m_writer->insert_node_info(1, 0xFFFFFFFF, "machine-flush", "Linux", "host",
                             "5.15.0", "v1", "x86_64", "local");
  m_writer->insert_process_info(1, 0, 18000, 0, 0, 100, 200, "/bin/flush_app");
  m_writer->insert_thread_info(1, 18000, 18000, 18001, "main", 100, 200);

  for (int i = 0; i < 10; ++i) {
    m_writer->insert_string(("string_" + std::to_string(i)).c_str());
  }

  auto category_id = m_writer->insert_string("test");
  for (int i = 0; i < 5; ++i) {
    m_writer->insert_event(category_id, i, 0, 1000 + i);
  }

  EXPECT_NO_THROW(m_writer->flush());
}

} // namespace
