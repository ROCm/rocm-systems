#include <benchmark/benchmark.h>
#include <rocstorage/storage.hpp>
#include <rocstorage/writer.hpp>

#include <atomic>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
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
  size_t thread_id;
  std::string name;
  std::string category;
  size_t start_timestamp;
  size_t end_timestamp;
  std::string call_stack;
  std::string args_str;
  size_t correlation_id_internal;
  size_t correlation_id_ancestor;
};

// ============================================================================
// Utility functions
// ============================================================================

std::string format_file_size(size_t _bytes) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2);
  if (_bytes < 1024 * 1024) {
    ss << static_cast<double>(_bytes) / 1024.0 << " KB";
  } else if (_bytes < 1024ULL * 1024 * 1024) {
    ss << static_cast<double>(_bytes) / (1024.0 * 1024.0) << " MB";
  } else {
    ss << static_cast<double>(_bytes) / (1024.0 * 1024.0 * 1024.0) << " GB";
  }
  return ss.str();
}

size_t get_file_size(const std::string &_path) {
  struct stat st;
  if (stat(_path.c_str(), &st) == 0) {
    return static_cast<size_t>(st.st_size);
  }
  return 0;
}

// ============================================================================
// Client-style processor (mimics rocpd_processor_t)
// ============================================================================

class rocpd_processor {
public:
  rocpd_processor(const std::string &_db_path, const std::string &_uuid)
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
                                                          name.c_str(), 0, 0));
    }

    m_writer->insert_queue_info(1, 1, 1000, "Queue 1");
    m_writer->insert_stream_info(1, 1, 1000, "Stream 1");

    m_writer->insert_code_object(1, 1, 1000, m_gpu_agent,
                                 "file:///kernels.hsaco", 0x10000, 0x1000, 0,
                                 "FILE");

    for (size_t i = 0; i < _num_kernels; ++i) {
      std::string name = "kernel_" + std::to_string(i);
      std::string display = name + "(float*, float*, int)";
      m_writer->insert_kernel_symbol(i + 1, 1, 1000, 1, name.c_str(),
                                     display.c_str(), 0x1234 + i, 256, 8, 65536,
                                     0, 32, 64, 0);
      m_kernel_name_ids.push_back(m_writer->insert_string(display.c_str()));
    }

    m_category_kernel = m_writer->insert_string("rocm_kernel_dispatch");
    m_category_memcpy = m_writer->insert_string("rocm_memory_copy");
    m_category_region = m_writer->insert_string("hip_api");
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
    auto name_pk = m_writer->insert_string(_rs.name.c_str());
    auto category_pk = m_writer->insert_string(_rs.category.c_str());

    auto event_id = m_writer->insert_event(
        category_pk, _rs.correlation_id_internal, _rs.correlation_id_ancestor,
        0, _rs.call_stack.c_str());

    if (!_rs.args_str.empty()) {
      m_writer->insert_args(event_id, 0, "int", "arg0", "0");
      m_writer->insert_args(event_id, 1, "void*", "ptr", "0x7fff0000");
      m_writer->insert_args(event_id, 2, "size_t", "size", "4096");
    }

    m_writer->insert_region(1, 1000, thread_pk, _rs.start_timestamp,
                            _rs.end_timestamp, name_pk, event_id);
  }

  void finalize() { m_writer->flush(); }

  size_t get_db_size() const { return get_file_size(m_db_path); }

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
    s.thread_id = m_rng() % m_num_threads;
    s.agent_id_handle = 1;
    s.kernel_id = (_idx % m_num_kernels) + 1;
    s.dispatch_id = _idx;
    s.queue_id_handle = 1;
    s.stream_handle = 1;
    s.start_timestamp = _idx * 10000;
    s.end_timestamp = s.start_timestamp + 5000 + (m_rng() % 1000);
    s.private_segment_size = 0;
    s.group_segment_size = 65536;
    s.workgroup_size_x = 256;
    s.workgroup_size_y = 1;
    s.workgroup_size_z = 1;
    s.grid_size_x = 1024;
    s.grid_size_y = 1;
    s.grid_size_z = 1;
    s.correlation_id_internal = _idx;
    s.correlation_id_ancestor = _idx > 0 ? _idx - 1 : 0;
    return s;
  }

  memory_copy_sample gen_memory_copy(size_t _idx) {
    memory_copy_sample s;
    s.thread_id = m_rng() % m_num_threads;
    s.dst_agent_id_handle = 1;
    s.src_agent_id_handle = 2;
    s.start_timestamp = _idx * 1000;
    s.end_timestamp = s.start_timestamp + 200 + (m_rng() % 100);
    s.dst_address_value = 0x7FFF00000000 + _idx * 4096;
    s.src_address_value = 0x100000 + _idx * 4096;
    s.bytes = 4096 * (1 + m_rng() % 16);
    s.stream_handle = 1;
    s.correlation_id_internal = _idx;
    s.correlation_id_ancestor = _idx > 0 ? _idx - 1 : 0;
    return s;
  }

  region_sample gen_region(size_t _idx, bool _with_args = true) {
    region_sample s;
    s.thread_id = m_rng() % m_num_threads;
    s.name = "hipLaunchKernel";
    s.category = "hip_api";
    s.start_timestamp = _idx * 500;
    s.end_timestamp = s.start_timestamp + 100 + (m_rng() % 50);
    s.call_stack = "{}";
    s.args_str = _with_args ? "has_args" : "";
    s.correlation_id_internal = _idx;
    s.correlation_id_ancestor = _idx > 0 ? _idx - 1 : 0;
    return s;
  }

private:
  size_t m_num_threads;
  size_t m_num_kernels;
  std::mt19937 m_rng;
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
// Distribution: 99.67% regions, 0.33% kernel dispatch (from real 8GB database)
// ============================================================================

BENCHMARK_DEFINE_F(client_benchmark_fixture, realistic_client_workload)
(benchmark::State &_state) {
  const auto total_events = static_cast<size_t>(_state.range(0));
  const size_t region_count = total_events * 9967 / 10000;
  const size_t kernel_count = total_events * 33 / 10000;

  constexpr size_t num_threads = 12;
  constexpr size_t num_kernels = 100;

  for (auto _ : _state) {
    rocpd_processor processor(m_db_path, "benchmark_uuid");
    sample_generator generator(num_threads, num_kernels);

    processor.post_process_metadata(num_threads, num_kernels);

    for (size_t i = 0; i < region_count; ++i) {
      processor.handle(generator.gen_region(i, i % 2 == 0));
    }

    for (size_t i = 0; i < kernel_count; ++i) {
      processor.handle(generator.gen_kernel_dispatch(i));
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
    ->Arg(2300000)   // ~1GB
    ->Arg(4600000)   // ~2GB
    ->Arg(9200000)   // ~4GB
    ->Arg(18700000); // ~8GB

// ============================================================================
// Mixed interleaved workload (mimics real profiling where samples arrive mixed)
// ============================================================================

BENCHMARK_DEFINE_F(client_benchmark_fixture, interleaved_workload)
(benchmark::State &_state) {
  const auto total_events = static_cast<size_t>(_state.range(0));

  constexpr size_t num_threads = 12;
  constexpr size_t num_kernels = 100;

  for (auto _ : _state) {
    rocpd_processor processor(m_db_path, "benchmark_uuid");
    sample_generator generator(num_threads, num_kernels);

    processor.post_process_metadata(num_threads, num_kernels);

    for (size_t i = 0; i < total_events; ++i) {
      size_t sample_type = i % 1000;

      if (sample_type < 3) {
        processor.handle(generator.gen_kernel_dispatch(i));
      } else if (sample_type < 5) {
        processor.handle(generator.gen_memory_copy(i));
      } else {
        processor.handle(generator.gen_region(i, sample_type % 2 == 0));
      }
    }

    processor.finalize();

    _state.counters["db_size_mb"] =
        static_cast<double>(processor.get_db_size()) / (1024.0 * 1024.0);
  }

  _state.SetItemsProcessed(_state.iterations() * total_events);
}

BENCHMARK_REGISTER_F(client_benchmark_fixture, interleaved_workload)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Arg(1000000)
    ->Arg(5000000)
    ->Arg(10000000);

// ============================================================================
// Parallel client benchmark (multiple processes writing to separate DBs)
// ============================================================================

BENCHMARK_DEFINE_F(client_benchmark_fixture, parallel_clients)
(benchmark::State &_state) {
  const auto num_clients = static_cast<size_t>(_state.range(0));
  const auto events_per_client = static_cast<size_t>(_state.range(1));

  constexpr size_t num_threads = 12;
  constexpr size_t num_kernels = 100;

  for (auto _ : _state) {
    std::vector<std::string> db_paths;
    std::vector<std::thread> threads;
    std::atomic<size_t> total_db_size{0};

    for (size_t i = 0; i < num_clients; ++i) {
      db_paths.push_back("bench_client_parallel_" + std::to_string(i) + ".db");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_clients; ++i) {
      threads.emplace_back([&db_paths, i, events_per_client, num_threads,
                            num_kernels, &total_db_size]() {
        rocpd_processor processor(db_paths[i], "uuid_" + std::to_string(i));
        sample_generator generator(num_threads, num_kernels);

        processor.post_process_metadata(num_threads, num_kernels);

        const size_t region_count = events_per_client * 9967 / 10000;
        const size_t kernel_count = events_per_client * 33 / 10000;

        for (size_t j = 0; j < region_count; ++j) {
          processor.handle(generator.gen_region(j, j % 2 == 0));
        }

        for (size_t j = 0; j < kernel_count; ++j) {
          processor.handle(generator.gen_kernel_dispatch(j));
        }

        processor.finalize();
        total_db_size += processor.get_db_size();
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    for (const auto &path : db_paths) {
      std::remove(path.c_str());
    }

    _state.SetIterationTime(static_cast<double>(duration) / 1000.0);
    _state.counters["total_db_size_mb"] =
        static_cast<double>(total_db_size) / (1024.0 * 1024.0);
    _state.counters["clients"] = static_cast<double>(num_clients);
  }

  _state.SetItemsProcessed(_state.iterations() * num_clients *
                           events_per_client);
}

// Args: {num_clients, events_per_client}
BENCHMARK_REGISTER_F(client_benchmark_fixture, parallel_clients)
    ->Unit(benchmark::kSecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Args({2, 1150000})  // 2 clients × ~500MB each
    ->Args({2, 2300000})  // 2 clients × ~1GB each
    ->Args({4, 1150000})  // 4 clients × ~500MB each
    ->Args({4, 2300000})  // 4 clients × ~1GB each
    ->Args({8, 1150000}); // 8 clients × ~500MB each

} // namespace
