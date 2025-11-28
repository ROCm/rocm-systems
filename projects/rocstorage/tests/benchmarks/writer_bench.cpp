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

#include <benchmark/benchmark.h>

#include "utility.hpp"

#include <cstdio>
#include <memory>
#include <string>

namespace {

class writer_fixture : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State &_state) override {
    m_database_path =
        "benchmark_writer_" + std::to_string(_state.thread_index()) + ".db";
    m_uuid = std::to_string(_state.thread_index());
    m_storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
    m_writer = m_storage->get_writer();
    setup_full_schema();
  }

  void TearDown(const benchmark::State &) override {
    m_writer.reset();
    m_storage.reset();
    std::remove(m_database_path.c_str());
  }

  std::string get_db_size_label() const {
    size_t size = utility::get_file_size(m_database_path);
    return "DB: " + utility::format_file_size(size);
  }

protected:
  void setup_full_schema() {
    m_writer->insert_node_info(1, 0xDEADBEEF, "bench-machine", "Linux",
                               "benchmark-host", "6.0.0", "v1", "x86_64",
                               "local");
    m_writer->insert_process_info(1, 0, 1000, 0, 0, 0, 0, "/bin/benchmark");

    m_thread_pk =
        m_writer->insert_thread_info(1, 1000, 1000, 1001, "main", 0, 0);

    m_gpu_agent = m_writer->insert_agent(1, 1000, "GPU", 0, 0, 0, 0xABCD,
                                         "gfx90a", "MI200", "AMD", "MI210", "");
    m_cpu_agent = m_writer->insert_agent(1, 1000, "CPU", 1, 0, 0, 0, "cpu0",
                                         "EPYC", "AMD", "EPYC7763", "");

    m_writer->insert_queue_info(1, 1, 1000, "hsa_queue_0");
    m_writer->insert_stream_info(1, 1, 1000, "hip_stream_0");

    m_writer->insert_code_object(1, 1, 1000, m_gpu_agent,
                                 "file:///kernels.hsaco", 0x10000, 0x1000, 0,
                                 "FILE");
    m_writer->insert_kernel_symbol(1, 1, 1000, 1, "vectorAdd",
                                   "vectorAdd(float*,float*,float*,int)",
                                   0x1234, 256, 8, 65536, 0, 32, 64, 0);

    m_writer->insert_pmc_description(1, 1000, m_gpu_agent, "GPU", 0x100, 0,
                                     "SQ_WAVES", "SQ_WAVES", "Wave count",
                                     "Number of waves", "SQ", "waves", "ACCUM",
                                     "SQ", "", 0, 0);
    m_writer->insert_pmc_description(
        1, 1000, m_gpu_agent, "GPU", 0x101, 0, "SQ_INSTS", "SQ_INSTS",
        "Instruction count", "Number of instructions", "SQ", "insts", "ACCUM",
        "SQ", "", 0, 0);
    m_writer->insert_pmc_description(1, 1000, m_gpu_agent, "GPU", 0x102, 0,
                                     "TA_BUSY", "TA_BUSY", "TA busy cycles",
                                     "Texture addresser busy", "TA", "cycles",
                                     "ACCUM", "TA", "", 0, 0);

    const char *smi_metrics[] = {
        "gfx_busy",     "ucm_busy",     "mm_busy",         "temp",
        "power",        "mem_usage",    "xgmi_link_width", "xgmi_link_speed",
        "pcie_link_w",  "pcie_link_s",  "pcie_bw_acc",     "pcie_bw_inst",
        "xgmi_read_0",  "xgmi_read_1",  "xgmi_read_2",     "xgmi_read_3",
        "xgmi_read_4",  "xgmi_read_5",  "xgmi_read_6",     "xgmi_read_7",
        "xgmi_write_0", "xgmi_write_1", "xgmi_write_2",    "xgmi_write_3",
        "xgmi_write_4", "xgmi_write_5", "xgmi_write_6",    "xgmi_write_7",
        "jpeg_0",       "jpeg_1",       "jpeg_2",          "jpeg_3",
        "jpeg_4",       "jpeg_5",       "jpeg_6",          "jpeg_7"};
    for (size_t i = 0; i < 36; ++i) {
      m_writer->insert_pmc_description(1, 1000, m_gpu_agent, "GPU", 0x200 + i,
                                       0, smi_metrics[i], smi_metrics[i],
                                       "SMI metric", "AMD SMI metric", "SMI",
                                       "value", "ABS", "SMI", "", 0, 0);
    }

    const char *cpu_metrics[] = {"freq", "usage",  "idle", "user",
                                 "sys",  "iowait", "irq"};
    for (size_t i = 0; i < 7; ++i) {
      m_writer->insert_pmc_description(1, 1000, m_cpu_agent, "CPU", 0x300 + i,
                                       0, cpu_metrics[i], cpu_metrics[i],
                                       "CPU metric", "CPU frequency metric",
                                       "CPU", "value", "ABS", "CPU", "", 0, 0);
    }

    m_writer->insert_track("gpu_kernel", 1, 1000, m_thread_pk);
    m_writer->insert_track("gpu_memcpy", 1, 1000, m_thread_pk);
    m_writer->insert_track("cpu_sample", 1, 1000, m_thread_pk);
    m_writer->insert_track("amd_smi", 1, 1000, std::nullopt);
    m_writer->insert_track("cpu_freq", 1, 1000, std::nullopt);
  }

  std::string m_database_path;
  std::string m_uuid;
  std::unique_ptr<rocm::storage> m_storage;
  std::shared_ptr<rocstorage::writer> m_writer;

  size_t m_thread_pk = 0;
  size_t m_gpu_agent = 0;
  size_t m_cpu_agent = 0;
};

BENCHMARK_DEFINE_F(writer_fixture, kernel_dispatch)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  size_t category_id = m_writer->insert_string("kernel_dispatch");

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      size_t name_id = m_writer->insert_string("vectorAdd");
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      m_writer->insert_kernel_dispatch(
          1, 1000, m_thread_pk, m_gpu_agent, 1, i, 1, 1, i * 1000,
          i * 1000 + 500, 0, 65536, 256, 1, 1, 1024, 1, 1, name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, kernel_dispatch)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000);

BENCHMARK_DEFINE_F(writer_fixture, memory_copy)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  size_t category_id = m_writer->insert_string("memory_copy");
  size_t name_id = m_writer->insert_string("hipMemcpy");

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      m_writer->insert_memory_copy(
          1, 1000, m_thread_pk, i * 1000, i * 1000 + 200, name_id, m_gpu_agent,
          0x7FFF00000000 + i * 4096, m_cpu_agent, 0x100000 + i * 4096, 4096, 1,
          1, name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, memory_copy)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000);

BENCHMARK_DEFINE_F(writer_fixture, memory_alloc)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  size_t category_id = m_writer->insert_string("memory_alloc");

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      m_writer->insert_memory_alloc(
          1, 1000, m_thread_pk, m_gpu_agent, "ALLOC", "REAL", i * 1000,
          i * 1000 + 100, 0x7FFF00000000 + i * 4096, 4096, 1, 1, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, memory_alloc)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000);

BENCHMARK_DEFINE_F(writer_fixture, region_variable_args)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  const char *arg_types[] = {"int",    "void*",  "size_t",   "float",
                             "double", "char*",  "uint64_t", "int32_t",
                             "bool",   "uint8_t"};
  const char *arg_names[] = {"arg0", "arg1", "arg2", "arg3", "arg4",
                             "arg5", "arg6", "arg7", "arg8", "arg9"};
  const char *arg_values[] = {"42",    "0x7fff0000", "4096", "3.14", "2.718",
                              "hello", "12345",      "-1",   "true", "255"};

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      size_t name_id = m_writer->insert_string("region_name");
      size_t category_id = m_writer->insert_string("hip_api");
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      size_t num_args = i % 11;
      for (size_t arg = 0; arg < num_args; ++arg) {
        m_writer->insert_args(event_id, arg, arg_types[arg], arg_names[arg],
                              arg_values[arg]);
      }
      m_writer->insert_region(1, 1000, m_thread_pk, i * 500, i * 500 + 100,
                              name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, region_variable_args)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000);

BENCHMARK_DEFINE_F(writer_fixture, backtrace_region)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      size_t name_id = m_writer->insert_string("backtrace_region");
      size_t category_id = m_writer->insert_string("backtrace");
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      m_writer->insert_region(1, 1000, m_thread_pk, i * 500, i * 500 + 100,
                              name_id, event_id);
      m_writer->insert_sample("cpu_sample", i * 500, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, backtrace_region)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000);

BENCHMARK_DEFINE_F(writer_fixture, in_time_sample)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      size_t track_name_id = m_writer->insert_string("in_time_track");
      auto event_id = m_writer->insert_event(track_name_id, i, 0, i);
      m_writer->insert_sample("cpu_sample", i * 100, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, in_time_sample)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000);

BENCHMARK_DEFINE_F(writer_fixture, pmc_event_with_sample)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  size_t category_id = m_writer->insert_string("pmc_event");

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      m_writer->insert_sample("gpu_kernel", i * 1000, event_id);
      m_writer->insert_pmc_event(event_id, m_gpu_agent, "SQ_WAVES",
                                 static_cast<double>(i * 100));
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, pmc_event_with_sample)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000);

BENCHMARK_DEFINE_F(writer_fixture, amd_smi_sample_min)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  size_t category_id = m_writer->insert_string("amd_smi");
  const char *min_metrics[] = {"gfx_busy", "ucm_busy", "mm_busy",
                               "temp",     "power",    "mem_usage"};
  constexpr size_t min_metric_count = 6;

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      m_writer->insert_string("amd_smi_sample");
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      for (size_t m = 0; m < min_metric_count; ++m) {
        m_writer->insert_pmc_event(event_id, m_gpu_agent, min_metrics[m],
                                   static_cast<double>(i + m));
        m_writer->insert_sample("amd_smi", i * 1000, event_id);
      }
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, amd_smi_sample_min)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000);

BENCHMARK_DEFINE_F(writer_fixture, amd_smi_sample_max)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  size_t category_id = m_writer->insert_string("amd_smi");

  const char *base_metrics[] = {
      "gfx_busy",     "ucm_busy",     "mm_busy",         "temp",
      "power",        "mem_usage",    "xgmi_link_width", "xgmi_link_speed",
      "pcie_link_w",  "pcie_link_s",  "pcie_bw_acc",     "pcie_bw_inst",
      "xgmi_read_0",  "xgmi_read_1",  "xgmi_read_2",     "xgmi_read_3",
      "xgmi_read_4",  "xgmi_read_5",  "xgmi_read_6",     "xgmi_read_7",
      "xgmi_write_0", "xgmi_write_1", "xgmi_write_2",    "xgmi_write_3",
      "xgmi_write_4", "xgmi_write_5", "xgmi_write_6",    "xgmi_write_7",
      "jpeg_0",       "jpeg_1",       "jpeg_2",          "jpeg_3",
      "jpeg_4",       "jpeg_5",       "jpeg_6",          "jpeg_7"};
  constexpr size_t base_count = 36;
  constexpr size_t max_n = 386;

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      m_writer->insert_string("amd_smi_sample");
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      for (size_t m = 0; m < max_n; ++m) {
        const char *metric_name = base_metrics[m % base_count];
        m_writer->insert_pmc_event(event_id, m_gpu_agent, metric_name,
                                   static_cast<double>(i + m));
        m_writer->insert_sample("amd_smi", i * 1000, event_id);
      }
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, amd_smi_sample_max)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(300 * 60)       // 300 samples per second for 1 minute
    ->Arg(1000 * 60)      // 1000 samples per second for 1 minute
    ->Arg(300 * 60 * 5)   // 300 samples per second for 5 minute
    ->Arg(1000 * 60 * 5); // 1000 samples per second for 5 minute

BENCHMARK_DEFINE_F(writer_fixture, cpu_freq)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  const auto cpu_core_count = static_cast<size_t>(_state.range(1));
  size_t category_id = m_writer->insert_string("cpu_freq");
  const size_t n = 7 + cpu_core_count * 7;
  const char *cpu_metrics[] = {"freq", "usage",  "idle", "user",
                               "sys",  "iowait", "irq"};

  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      auto event_id = m_writer->insert_event(category_id, i, 0, i);
      for (size_t m = 0; m < n; ++m) {
        m_writer->insert_pmc_event(event_id, m_cpu_agent, cpu_metrics[m % 7],
                                   static_cast<double>(i + m));
        m_writer->insert_sample("cpu_freq", i * 100, event_id);
      }
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_fixture, cpu_freq)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({300 * 60, 1})
    ->Args({300 * 60, 4})
    ->Args({300 * 60, 8})
    ->Args({300 * 60, 16})
    ->Args({300 * 60, 32})
    ->Args({1000 * 60, 1})
    ->Args({1000 * 60, 4})
    ->Args({1000 * 60, 8})
    ->Args({1000 * 60, 16})
    ->Args({1000 * 60, 32});

} // namespace
