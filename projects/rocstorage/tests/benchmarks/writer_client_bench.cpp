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

#include <benchmark/benchmark.h>
#include <rocstorage/storage.hpp>
#include <rocstorage/writer.hpp>

#include "utility.hpp"

#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

// ============================================================================
// Sample types mimicking rocprofiler-systems trace_cache samples
// ============================================================================

struct kernel_dispatch_sample {
  size_t thread_id;
  size_t agent_id_handle;
  size_t kernel_id;
  size_t dispatch_id;
  size_t queue_id_handle;
  size_t stream_handle;
  size_t start_timestamp;
  size_t end_timestamp;
  size_t private_segment_size;
  size_t group_segment_size;
  size_t workgroup_size_x;
  size_t workgroup_size_y;
  size_t workgroup_size_z;
  size_t grid_size_x;
  size_t grid_size_y;
  size_t grid_size_z;
  size_t correlation_id_internal;
  size_t correlation_id_ancestor;
};

struct memory_copy_sample {
  size_t thread_id;
  size_t dst_agent_id_handle;
  size_t src_agent_id_handle;
  size_t start_timestamp;
  size_t end_timestamp;
  size_t dst_address_value;
  size_t src_address_value;
  size_t bytes;
  size_t stream_handle;
  size_t correlation_id_internal;
  size_t correlation_id_ancestor;
};

struct region_sample {
  struct region_sample_args {
    size_t position;
    std::string_view type;
    std::string_view name;
    std::string_view value;
  };

  size_t thread_id;
  std::string_view name;
  std::string_view category;
  size_t start_timestamp;
  size_t end_timestamp;
  std::string_view call_stack;
  std::vector<region_sample_args> args;
  size_t correlation_id_internal;
  size_t correlation_id_ancestor;
};

struct backtrace_sample {
  uint32_t type;
  size_t thread_id;
  std::string_view track_name;
  std::string_view name;
  size_t start_timestamp;
  size_t end_timestamp;
  std::string_view category;
  std::string_view call_stack;
  std::string_view line_info;
  std::string_view extdata;
};

struct amd_smi_sample {
  uint32_t device_id;
  size_t timestamp;
  uint32_t gfx_activity;
  uint32_t umc_activity;
  uint32_t mm_activity;
  uint32_t power;
  int64_t temperature;
  size_t mem_usage;
};

struct cpu_freq_sample {
  struct core_freq_sample {
    size_t id;
    double value;
  };

  size_t timestamp;
  int64_t page_rss;
  int64_t virt_mem_usage;
  int64_t peak_rss;
  int64_t context_switch_count;
  int64_t page_faults;
  int64_t user_mode_time;
  int64_t kernel_mode_time;
  std::vector<core_freq_sample> freq_values;
};

struct memory_alloc_sample {
  size_t thread_id;
  size_t start_timestamp;
  size_t end_timestamp;
  size_t address;
  size_t size;
  size_t correlation_id_internal;
  size_t correlation_id_ancestor;
};

class db_write_processor {
public:
  db_write_processor(const std::string &_db_path, const std::string &_uuid)
      : m_db_path(_db_path),
        m_storage(std::make_unique<rocm::storage>(_db_path, _uuid)),
        m_writer(m_storage->get_writer()) {}

  void post_process_metadata(size_t _num_threads, size_t _num_kernels) {
    m_writer->insert_node_info(1, 0xDEADBEEF, "bench-machine", "Linux",
                               "benchmark-host", "6.0.0", "v1", "x86_64",
                               "local");

    m_writer->insert_process_info(1, 0, 1000, 0, 0, 0, 0, "/bin/benchmark",
                                  "{}");

    m_gpu_agent = m_writer->insert_agent(1, 1000, "GPU", 0, 0, 0, 0xABCD,
                                         "gfx90a", "MI200", "AMD", "MI210", "");
    m_cpu_agent = m_writer->insert_agent(1, 1000, "CPU", 1, 0, 0, 0, "cpu0",
                                         "EPYC", "AMD", "EPYC7763", "");

    for (size_t i = 0; i < _num_threads; ++i) {
      std::string name = "Thread " + std::to_string(i);
      m_thread_pks.push_back(m_writer->insert_thread_info(1, 0, 1000, i + 1000,
                                                          name.data(), 0, 0));
    }

    m_writer->insert_queue_info(1, 1, 1000, "Queue 1");
    m_writer->insert_stream_info(1, 1, 1000, "Stream 1");

    m_writer->insert_code_object(1, 1, 1000, m_gpu_agent,
                                 "file:///kernels.hsaco", 0x10000, 0x1000, 0,
                                 "FILE");

    for (size_t i = 0; i < _num_kernels; ++i) {
      std::string name = "kernel_" + std::to_string(i);
      std::string display = name + "(float*, float*, int)";
      m_writer->insert_kernel_symbol(i + 1, 1, 1000, 1, name.data(),
                                     display.data(), 0x1234 + i, 256, 8, 65536,
                                     0, 32, 64, 0);
      m_kernel_name_ids.push_back(m_writer->insert_string(display.data()));
    }

    m_category_kernel = m_writer->insert_string("rocm_kernel_dispatch");
    m_category_memcpy = m_writer->insert_string("rocm_memory_copy");
    m_category_region = m_writer->insert_string("hip_api");
    m_category_alloc = m_writer->insert_string("memory_alloc");
    m_category_backtrace = m_writer->insert_string("backtrace");
    m_category_smi = m_writer->insert_string("amd_smi");
    m_category_cpu_freq = m_writer->insert_string("cpu_freq");

    m_writer->insert_track("cpu_sample", 1, 1000, m_thread_pks[0]);
    m_writer->insert_track("amd_smi", 1, 1000, std::nullopt);
    m_writer->insert_track("cpu_freq", 1, 1000, std::nullopt);

    const char *smi_metrics[] = {"gfx_busy", "ucm_busy", "mm_busy",
                                 "temp",     "power",    "mem_usage"};
    for (size_t i = 0; i < 6; ++i) {
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
  }

  void handle(const kernel_dispatch_sample &_kds) {
    auto thread_pk = m_thread_pks[_kds.thread_id % m_thread_pks.size()];
    auto kernel_name_id =
        m_kernel_name_ids[_kds.kernel_id % m_kernel_name_ids.size()];

    auto event_id =
        m_writer->insert_event(m_category_kernel, _kds.correlation_id_internal,
                               _kds.correlation_id_ancestor, 0);

    m_writer->insert_kernel_dispatch(
        1, 1000, thread_pk, m_gpu_agent, _kds.kernel_id, _kds.dispatch_id,
        _kds.queue_id_handle, _kds.stream_handle, _kds.start_timestamp,
        _kds.end_timestamp, _kds.private_segment_size, _kds.group_segment_size,
        _kds.workgroup_size_x, _kds.workgroup_size_y, _kds.workgroup_size_z,
        _kds.grid_size_x, _kds.grid_size_y, _kds.grid_size_z, kernel_name_id,
        event_id);
  }

  void handle(const memory_copy_sample &_mcs) {
    auto thread_pk = m_thread_pks[_mcs.thread_id % m_thread_pks.size()];
    auto name_pk = m_writer->insert_string("hipMemcpy");

    auto event_id =
        m_writer->insert_event(m_category_memcpy, _mcs.correlation_id_internal,
                               _mcs.correlation_id_ancestor, 0);

    m_writer->insert_memory_copy(1, 1000, thread_pk, _mcs.start_timestamp,
                                 _mcs.end_timestamp, name_pk, m_gpu_agent,
                                 _mcs.dst_address_value, m_cpu_agent,
                                 _mcs.src_address_value, _mcs.bytes, 1,
                                 _mcs.stream_handle, name_pk, event_id);
  }

  void handle(const region_sample &_rs) {
    auto thread_pk = m_thread_pks[_rs.thread_id % m_thread_pks.size()];
    auto name_pk = m_writer->insert_string(_rs.name.data());
    auto category_pk = m_writer->insert_string(_rs.category.data());

    auto event_id = m_writer->insert_event(
        category_pk, _rs.correlation_id_internal, _rs.correlation_id_ancestor,
        0, _rs.call_stack.data());

    for (const auto &arg : _rs.args) {
      m_writer->insert_args(event_id, arg.position, arg.type.data(),
                            arg.name.data(), arg.value.data());
    }

    m_writer->insert_region(1, 1000, thread_pk, _rs.start_timestamp,
                            _rs.end_timestamp, name_pk, event_id);
  }

  void handle(const memory_alloc_sample &_mas) {
    auto thread_pk = m_thread_pks[_mas.thread_id % m_thread_pks.size()];
    auto event_id =
        m_writer->insert_event(m_category_alloc, _mas.correlation_id_internal,
                               _mas.correlation_id_ancestor, 0);
    m_writer->insert_memory_alloc(
        1, 1000, thread_pk, m_gpu_agent, "ALLOC", "REAL", _mas.start_timestamp,
        _mas.end_timestamp, _mas.address, _mas.size, 1, 1, event_id);
  }

  void handle(const backtrace_sample &_bs) {
    auto thread_pk = m_thread_pks[_bs.thread_id % m_thread_pks.size()];
    auto name_pk = m_writer->insert_string(_bs.name.data());
    auto category_pk = m_writer->insert_string(_bs.category.data());
    auto event_id = m_writer->insert_event(
        category_pk, 0, 0, 0, _bs.call_stack.data(), _bs.line_info.data());
    m_writer->insert_region(1, 1000, thread_pk, _bs.start_timestamp,
                            _bs.end_timestamp, name_pk, event_id);
    m_writer->insert_sample(_bs.track_name.data(), _bs.start_timestamp,
                            event_id);
  }

  void handle(const amd_smi_sample &_smi) {
    auto event_id = m_writer->insert_event(m_category_smi, 0, 0, 0);
    m_writer->insert_pmc_event(event_id, m_gpu_agent, "gfx_busy",
                               static_cast<double>(_smi.gfx_activity));
    m_writer->insert_pmc_event(event_id, m_gpu_agent, "ucm_busy",
                               static_cast<double>(_smi.umc_activity));
    m_writer->insert_pmc_event(event_id, m_gpu_agent, "mm_busy",
                               static_cast<double>(_smi.mm_activity));
    m_writer->insert_pmc_event(event_id, m_gpu_agent, "temp",
                               static_cast<double>(_smi.temperature));
    m_writer->insert_pmc_event(event_id, m_gpu_agent, "power",
                               static_cast<double>(_smi.power));
    m_writer->insert_pmc_event(event_id, m_gpu_agent, "mem_usage",
                               static_cast<double>(_smi.mem_usage));
    m_writer->insert_sample("amd_smi", _smi.timestamp, event_id);
  }

  void handle(const cpu_freq_sample &_cfs) {
    auto event_id = m_writer->insert_event(m_category_cpu_freq, 0, 0, 0);
    for (const auto &freq : _cfs.freq_values) {
      m_writer->insert_pmc_event(event_id, m_cpu_agent, "freq", freq.value);
      m_writer->insert_sample("cpu_freq", _cfs.timestamp, event_id);
    }
  }

  void finalize() { m_writer->flush(); }

  size_t get_db_size() const { return utility::get_file_size(m_db_path); }

  std::string get_db_path() const { return m_db_path; }

private:
  std::string m_db_path;
  std::unique_ptr<rocm::storage> m_storage;
  std::shared_ptr<rocstorage::writer> m_writer;

  size_t m_gpu_agent = 0;
  size_t m_cpu_agent = 0;
  std::vector<size_t> m_thread_pks;
  std::vector<size_t> m_kernel_name_ids;
  size_t m_category_kernel = 0;
  size_t m_category_memcpy = 0;
  size_t m_category_region = 0;
  size_t m_category_alloc = 0;
  size_t m_category_backtrace = 0;
  size_t m_category_smi = 0;
  size_t m_category_cpu_freq = 0;
};

// ============================================================================
// Sample generator (mimics real profiling data stream)
// ============================================================================

class sample_generator {
public:
  sample_generator(size_t _num_threads, size_t _num_kernels)
      : m_num_threads(_num_threads), m_num_kernels(_num_kernels), m_rng(42) {}

  kernel_dispatch_sample gen_kernel_dispatch(size_t _idx) {
    kernel_dispatch_sample s;
    auto rand_value = m_rng();
    s.thread_id = rand_value % m_num_threads;
    s.agent_id_handle = 1;
    s.kernel_id = (_idx % m_num_kernels) + 1;
    s.dispatch_id = _idx;
    s.queue_id_handle = 1;
    s.stream_handle = 1;
    s.start_timestamp = rand_value;
    s.end_timestamp = rand_value;
    s.private_segment_size = 0;
    s.group_segment_size = 65536;
    s.workgroup_size_x = 256;
    s.workgroup_size_y = 1;
    s.workgroup_size_z = 1;
    s.grid_size_x = 1024;
    s.grid_size_y = 1;
    s.grid_size_z = 1;
    s.correlation_id_internal = rand_value;
    s.correlation_id_ancestor = rand_value;
    return s;
  }

  memory_copy_sample gen_memory_copy(size_t _idx) {
    memory_copy_sample s;
    auto rand_value = m_rng();
    s.thread_id = rand_value % m_num_threads;
    s.dst_agent_id_handle = 1;
    s.src_agent_id_handle = 2;
    s.start_timestamp = rand_value;
    s.end_timestamp = rand_value;
    s.dst_address_value = 0x7FFF00000000 + _idx * 4096;
    s.src_address_value = 0x100000 + _idx * 4096;
    s.bytes = rand_value;
    s.stream_handle = 1;
    s.correlation_id_internal = rand_value;
    s.correlation_id_ancestor = rand_value;
    return s;
  }

  region_sample gen_region(size_t _idx, size_t _num_args) {
    region_sample s;
    auto rand_value = m_rng();
    s.thread_id = rand_value % m_num_threads;
    s.name = c_region_name;
    s.category = c_region_category;
    s.start_timestamp = rand_value;
    s.end_timestamp = rand_value;
    s.call_stack = c_empty_json;
    for (size_t i = 0; i < _num_args; ++i) {
      s.args.push_back({i, "int", "arg" + std::to_string(i), "0"});
    }
    s.correlation_id_internal = rand_value;
    s.correlation_id_ancestor = rand_value;
    return s;
  }

  memory_alloc_sample gen_memory_alloc(size_t _idx) {
    memory_alloc_sample s;
    auto rand_value = m_rng();
    s.thread_id = rand_value % m_num_threads;
    s.start_timestamp = rand_value;
    s.end_timestamp = rand_value;
    s.address = 0x7FFF00000000 + _idx * 4096;
    s.size = 4096 * (1 + m_rng() % 16);
    s.correlation_id_internal = rand_value;
    s.correlation_id_ancestor = rand_value;
    return s;
  }

  backtrace_sample gen_backtrace(size_t _idx) {
    backtrace_sample s;
    auto rand_value = m_rng();
    s.type = 1;
    s.thread_id = rand_value % m_num_threads;
    s.track_name = c_track_name;
    s.name = c_backtrace_name;
    s.start_timestamp = rand_value;
    s.end_timestamp = rand_value;
    s.category = c_backtrace_category;
    s.call_stack = c_empty_json;
    s.line_info = c_empty_json;
    s.extdata = c_empty_json;
    return s;
  }

  amd_smi_sample gen_amd_smi(size_t _idx) {
    amd_smi_sample s;
    auto rand_value = m_rng();
    s.device_id = 0;
    s.timestamp = rand_value;
    s.gfx_activity = rand_value;
    s.umc_activity = rand_value;
    s.mm_activity = rand_value;
    s.power = 100 + (rand_value % 200);
    s.temperature = 40 + static_cast<int64_t>(rand_value % 40);
    s.mem_usage = 1024ULL * 1024 * 1024 * (1 + rand_value % 16);
    return s;
  }

  cpu_freq_sample gen_cpu_freq(size_t _idx, size_t _num_cores) {
    cpu_freq_sample s;
    auto rand_value = m_rng();
    s.timestamp = rand_value;
    s.page_rss = rand_value;
    s.virt_mem_usage = rand_value;
    s.peak_rss = rand_value;
    s.context_switch_count = rand_value;
    s.page_faults = rand_value;
    s.user_mode_time = rand_value;
    s.kernel_mode_time = rand_value;
    for (size_t i = 0; i < _num_cores; ++i) {
      s.freq_values.push_back({i, static_cast<double>(rand_value)});
    }
    return s;
  }

private:
  size_t m_num_threads;
  size_t m_num_kernels;
  std::mt19937 m_rng;

  const char *c_track_name = "cpu_sample";
  const char *c_smi_track_name = "amd_smi";
  const char *c_cpu_freq_track_name = "cpu_freq";
  const char *c_backtrace_name = "sample";
  const char *c_backtrace_category = "backtrace";
  const char *c_empty_json = "{}";
  const char *c_region_name = "hipLaunchKernel";
  const char *c_region_category = "hip_api";
};

// ============================================================================
// Benchmark fixture
// ============================================================================

class client_benchmark_fixture : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State &_state) override {
    m_db_path = "bench_client_" + std::to_string(_state.thread_index()) + ".db";
  }

  void TearDown(const benchmark::State &) override {
    std::remove(m_db_path.c_str());
  }

protected:
  std::string m_db_path;
};

// ============================================================================
// Realistic client-style benchmark (mimics rocpd_processor_t usage)
// ============================================================================

BENCHMARK_DEFINE_F(client_benchmark_fixture, realistic_client_workload)
(benchmark::State &_state) {
  const auto total_events = static_cast<size_t>(_state.range(0));

  constexpr size_t num_threads = 12;
  constexpr size_t num_kernels = 100;
  constexpr size_t num_cpu_cores = 16;

  for (auto _ : _state) {
    db_write_processor processor(m_db_path, "benchmark_uuid");
    sample_generator generator(num_threads, num_kernels);

    processor.post_process_metadata(num_threads, num_kernels);

    for (size_t i = 0; i < total_events; ++i) {
      size_t sample_type = i % 1000;

      if (sample_type < 3) {
        processor.handle(generator.gen_kernel_dispatch(i));
      } else if (sample_type < 5) {
        processor.handle(generator.gen_memory_copy(i));
      } else if (sample_type < 7) {
        processor.handle(generator.gen_memory_alloc(i));
      } else if (sample_type < 10) {
        processor.handle(generator.gen_backtrace(i));
      } else if (sample_type < 310) {
        processor.handle(generator.gen_amd_smi(i));
      } else if (sample_type < 320) {
        processor.handle(generator.gen_cpu_freq(i, num_cpu_cores));
      } else {
        processor.handle(generator.gen_region(i, i % 11));
      }
    }

    processor.finalize();

    _state.counters["db_size_mb"] =
        static_cast<double>(processor.get_db_size()) / (1024.0 * 1024.0);
  }

  _state.SetItemsProcessed(_state.iterations() * total_events);
}

BENCHMARK_REGISTER_F(client_benchmark_fixture, realistic_client_workload)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(100000)
    ->Arg(500000)
    ->Arg(1000000);

} // namespace
