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

#include <sys/stat.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr size_t k_1m = 1000000;
constexpr size_t k_5m = 5000000;
constexpr size_t k_10m = 10000000;

std::string format_file_size(size_t _bytes) {
  constexpr double k_kb = 1024.0;
  constexpr double k_mb = k_kb * 1024.0;
  constexpr double k_gb = k_mb * 1024.0;

  char buffer[64];
  if (_bytes >= k_gb) {
    std::snprintf(buffer, sizeof(buffer), "%.2f GB", _bytes / k_gb);
  } else if (_bytes >= k_mb) {
    std::snprintf(buffer, sizeof(buffer), "%.2f MB", _bytes / k_mb);
  } else if (_bytes >= k_kb) {
    std::snprintf(buffer, sizeof(buffer), "%.2f KB", _bytes / k_kb);
  } else {
    std::snprintf(buffer, sizeof(buffer), "%zu B", _bytes);
  }
  return buffer;
}

size_t get_file_size(const std::string &_path) {
  struct stat st;
  if (stat(_path.c_str(), &st) == 0) {
    return static_cast<size_t>(st.st_size);
  }
  return 0;
}

class writer_intensive_fixture : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State &_state) override {
    m_database_path =
        "benchmark_writer_" + std::to_string(_state.thread_index()) + ".db";
    m_uuid = std::to_string(_state.thread_index());
    m_storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
    m_writer = m_storage->get_writer();

    setup_base_schema();
  }

  void TearDown(const benchmark::State &) override {
    m_writer.reset();
    m_storage.reset();
    std::remove(m_database_path.c_str());
  }

  std::string get_db_size_label() const {
    size_t size = get_file_size(m_database_path);
    return "DB: " + format_file_size(size);
  }

protected:
  void setup_base_schema() {
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

    m_category_id = m_writer->insert_string("hip_api");
    m_kernel_name_id = m_writer->insert_string("vectorAdd");
    m_memcpy_name_id = m_writer->insert_string("hipMemcpy");
    m_region_name_id = m_writer->insert_string("user_region");

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

    m_writer->flush();
  }

  std::string m_database_path;
  std::string m_uuid;
  std::unique_ptr<rocm::storage> m_storage;
  std::shared_ptr<rocstorage::writer> m_writer;

  size_t m_thread_pk = 0;
  size_t m_gpu_agent = 0;
  size_t m_cpu_agent = 0;
  size_t m_category_id = 0;
  size_t m_kernel_name_id = 0;
  size_t m_memcpy_name_id = 0;
  size_t m_region_name_id = 0;
};

// ============================================================================
// Events with Args Benchmarks (1M events, varying args per event)
// ============================================================================

BENCHMARK_DEFINE_F(writer_intensive_fixture, events_with_args_1m_2args)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_1m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      m_writer->insert_args(event_id, 0, "void*", "dst", "0x7fff0000");
      m_writer->insert_args(event_id, 1, "size_t", "size", "1024");
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_1m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, events_with_args_1m_2args)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, events_with_args_1m_5args)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_1m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      m_writer->insert_args(event_id, 0, "void*", "dst", "0x7fff0000");
      m_writer->insert_args(event_id, 1, "void*", "src", "0x7fff1000");
      m_writer->insert_args(event_id, 2, "size_t", "size", "1024");
      m_writer->insert_args(event_id, 3, "int", "kind", "1");
      m_writer->insert_args(event_id, 4, "hipStream_t", "stream", "0x1234");
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_1m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, events_with_args_1m_5args)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, events_with_args_1m_10args)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_1m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      for (size_t arg = 0; arg < 10; ++arg) {
        m_writer->insert_args(event_id, arg, "int", "arg", "42");
      }
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_1m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, events_with_args_1m_10args)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

// ============================================================================
// Kernel Dispatch Benchmarks (1M, 5M, 10M)
// ============================================================================

BENCHMARK_DEFINE_F(writer_intensive_fixture, kernel_dispatches_1m)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_1m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 500;
      m_writer->insert_kernel_dispatch(1, 1000, m_thread_pk, m_gpu_agent, 1, i,
                                       1, 1, start, end, 0, 65536, 256, 1, 1,
                                       1024, 1, 1, m_kernel_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_1m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, kernel_dispatches_1m)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, kernel_dispatches_5m)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_5m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 500;
      m_writer->insert_kernel_dispatch(1, 1000, m_thread_pk, m_gpu_agent, 1, i,
                                       1, 1, start, end, 0, 65536, 256, 1, 1,
                                       1024, 1, 1, m_kernel_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_5m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, kernel_dispatches_5m)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, kernel_dispatches_10m)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_10m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 500;
      m_writer->insert_kernel_dispatch(1, 1000, m_thread_pk, m_gpu_agent, 1, i,
                                       1, 1, start, end, 0, 65536, 256, 1, 1,
                                       1024, 1, 1, m_kernel_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_10m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, kernel_dispatches_10m)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

// ============================================================================
// Region Benchmarks (1M, 5M, 10M)
// ============================================================================

BENCHMARK_DEFINE_F(writer_intensive_fixture, regions_1m)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_1m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 500;
      m_writer->insert_region(1, 1000, m_thread_pk, start, end,
                              m_region_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_1m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, regions_1m)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, regions_5m)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_5m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 500;
      m_writer->insert_region(1, 1000, m_thread_pk, start, end,
                              m_region_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_5m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, regions_5m)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, regions_10m)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_10m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 500;
      m_writer->insert_region(1, 1000, m_thread_pk, start, end,
                              m_region_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_10m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, regions_10m)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

// ============================================================================
// Memory Copy Benchmarks (1M, 5M)
// ============================================================================

BENCHMARK_DEFINE_F(writer_intensive_fixture, memory_copies_1m)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_1m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 200;
      m_writer->insert_memory_copy(
          1, 1000, m_thread_pk, start, end, m_memcpy_name_id, m_gpu_agent,
          0x7FFF00000000 + i * 4096, m_cpu_agent, 0x100000 + i * 4096, 4096, 1,
          1, m_region_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_1m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, memory_copies_1m)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, memory_copies_5m)
(benchmark::State &_state) {
  for (auto _ : _state) {
    for (size_t i = 0; i < k_5m; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 200;
      m_writer->insert_memory_copy(
          1, 1000, m_thread_pk, start, end, m_memcpy_name_id, m_gpu_agent,
          0x7FFF00000000 + i * 4096, m_cpu_agent, 0x100000 + i * 4096, 4096, 1,
          1, m_region_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * k_5m);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, memory_copies_5m)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

// ============================================================================
// Target DB Size Benchmark (mixed workload: kernels, memcpy, regions, PMC)
// Args: total record count (~3.2M for 500MB, ~6.5M for 1GB, ~32M for 5GB)
// Distribution: 40% kernel dispatch, 20% memory copy, 30% regions, 10% PMC
// ============================================================================

BENCHMARK_DEFINE_F(writer_intensive_fixture, mixed_target_size)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  const size_t kernel_count = count * 40 / 100;
  const size_t memcpy_count = count * 20 / 100;
  const size_t region_count = count * 30 / 100;
  const size_t pmc_count = count * 10 / 100;
  const char *pmc_names[] = {"SQ_WAVES", "SQ_INSTS", "TA_BUSY"};

  for (auto _ : _state) {
    for (size_t i = 0; i < kernel_count; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 500;
      m_writer->insert_kernel_dispatch(1, 1000, m_thread_pk, m_gpu_agent, 1, i,
                                       1, 1, start, end, 0, 65536, 256, 1, 1,
                                       1024, 1, 1, m_kernel_name_id, event_id);
      m_writer->insert_args(event_id, 0, "void*", "ptr", "0x7fff0000");
      m_writer->insert_args(event_id, 1, "size_t", "n", "1048576");
    }
    for (size_t i = 0; i < memcpy_count; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 1000;
      size_t end = start + 200;
      m_writer->insert_memory_copy(
          1, 1000, m_thread_pk, start, end, m_memcpy_name_id, m_gpu_agent,
          0x2000000 + i * 4096, m_cpu_agent, 0x1000000 + i * 4096, 4096, 1, 1,
          m_memcpy_name_id, event_id);
    }
    for (size_t i = 0; i < region_count; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      size_t start = i * 500;
      size_t end = start + 100;
      m_writer->insert_region(1, 1000, m_thread_pk, start, end,
                              m_region_name_id, event_id);
      m_writer->insert_args(event_id, 0, "int", "iteration", "0");
    }
    for (size_t i = 0; i < pmc_count; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      m_writer->insert_pmc_event(event_id, m_gpu_agent, pmc_names[i % 3],
                                 static_cast<double>(i * 1000 + 42));
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, mixed_target_size)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(3200000)
    ->Arg(6500000)
    ->Arg(32000000);

// ============================================================================
// Throughput benchmarks (items per second)
// ============================================================================

BENCHMARK_DEFINE_F(writer_intensive_fixture, throughput_events_only)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      benchmark::DoNotOptimize(event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, throughput_events_only)
    ->Arg(100000)
    ->Arg(500000)
    ->Arg(1000000)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, throughput_kernel_dispatch)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      m_writer->insert_kernel_dispatch(1, 1000, m_thread_pk, m_gpu_agent, 1, i,
                                       1, 1, i * 1000, i * 1000 + 500, 0, 65536,
                                       256, 1, 1, 1024, 1, 1, m_kernel_name_id,
                                       event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, throughput_kernel_dispatch)
    ->Arg(100000)
    ->Arg(500000)
    ->Arg(1000000)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

BENCHMARK_DEFINE_F(writer_intensive_fixture, throughput_regions)
(benchmark::State &_state) {
  const auto count = static_cast<size_t>(_state.range(0));
  for (auto _ : _state) {
    for (size_t i = 0; i < count; ++i) {
      auto event_id = m_writer->insert_event(m_category_id, i, 0, i);
      m_writer->insert_region(1, 1000, m_thread_pk, i * 1000, i * 1000 + 500,
                              m_region_name_id, event_id);
    }
    m_writer->flush();
  }
  _state.SetItemsProcessed(_state.iterations() * count);
  _state.SetLabel(get_db_size_label());
}
BENCHMARK_REGISTER_F(writer_intensive_fixture, throughput_regions)
    ->Arg(100000)
    ->Arg(500000)
    ->Arg(1000000)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

} // namespace
