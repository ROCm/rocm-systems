#include <benchmark/benchmark.h>
#include <rocstorage/storage.hpp>
#include <rocstorage/writer.hpp>

#include <atomic>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace {

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

struct thread_context {
  std::string database_path;
  std::unique_ptr<rocm::storage> storage;
  std::shared_ptr<rocstorage::writer> writer;

  size_t context_id;

  size_t thread_pk = 0;
  size_t gpu_agent = 0;
  size_t category_id = 0;
  size_t region_name_id = 0;
  size_t kernel_name_id = 0;

  void setup(size_t _thread_id) {
    context_id = _thread_id;

    database_path =
        "benchmark_writer_parallel_" + std::to_string(_thread_id) + ".db";
    std::string uuid = "parallel_" + std::to_string(_thread_id);
    storage = std::make_unique<rocm::storage>(database_path, uuid);
    writer = storage->get_writer();

    writer->insert_node_info(1, 0xDEADBEEF + _thread_id, "bench-machine",
                             "Linux", "benchmark-host", "6.0.0", "v1", "x86_64",
                             "local");
    writer->insert_process_info(1, 0, 1000 + _thread_id, 0, 0, 0, 0,
                                "/bin/benchmark");

    thread_pk =
        writer->insert_thread_info(1, 1000 + _thread_id, 1000 + _thread_id,
                                   2000 + _thread_id, "worker", 0, 0);

    gpu_agent =
        writer->insert_agent(1, 1000 + _thread_id, "GPU", 0, 0, 0, 0xABCD,
                             "gfx90a", "MI200", "AMD", "MI210", "");

    writer->insert_queue_info(1, 1, 1000 + _thread_id, "hsa_queue_0");
    writer->insert_stream_info(1, 1, 1000 + _thread_id, "hip_stream_0");

    writer->insert_code_object(1, 1, 1000 + _thread_id, gpu_agent,
                               "file:///kernels.hsaco", 0x10000, 0x1000, 0,
                               "FILE");
    writer->insert_kernel_symbol(1, 1, 1000 + _thread_id, 1, "vectorAdd",
                                 "vectorAdd(float*,float*,float*,int)", 0x1234,
                                 256, 8, 65536, 0, 32, 64, 0);

    category_id = writer->insert_string("hip_api");
    region_name_id = writer->insert_string("user_region");
    kernel_name_id = writer->insert_string("vectorAdd");
  }

  void teardown() {
    writer.reset();
    storage.reset();
    std::remove(database_path.c_str());
  }

  size_t get_db_size() const { return get_file_size(database_path); }
};

void run_realistic_workload(thread_context &_ctx, size_t _total_events) {
  const size_t region_count = _total_events * 9967 / 10000;
  const size_t kernel_count = _total_events * 33 / 10000;

  for (size_t i = 0; i < region_count; ++i) {
    auto event_id = _ctx.writer->insert_event(_ctx.category_id, i, 0, i);
    size_t start = i * 1000;
    size_t end = start + 500;
    _ctx.writer->insert_region(1, 1000 + _ctx.context_id, _ctx.thread_pk, start,
                               end, _ctx.region_name_id, event_id);
    _ctx.writer->insert_args(event_id, 0, "int", "level", "0");
    _ctx.writer->insert_args(event_id, 1, "const char*", "name", "region");
    if (i % 2 == 0) {
      _ctx.writer->insert_args(event_id, 2, "size_t", "id", "12345");
    }
  }

  for (size_t i = 0; i < kernel_count; ++i) {
    auto event_id = _ctx.writer->insert_event(_ctx.category_id, i, 0, i);
    size_t start = i * 10000;
    size_t end = start + 5000;
    _ctx.writer->insert_kernel_dispatch(
        1, 1000 + _ctx.context_id, _ctx.thread_pk, _ctx.gpu_agent, 1, i, 1, 1,
        start, end, 0, 65536, 256, 1, 1, 1024, 1, 1, _ctx.kernel_name_id,
        event_id);
    _ctx.writer->insert_args(event_id, 0, "void*", "ptr", "0x7fff0000");
    _ctx.writer->insert_args(event_id, 1, "size_t", "n", "1048576");
    _ctx.writer->insert_args(event_id, 2, "int", "stream", "0");
  }

  _ctx.writer->flush();
}

// ============================================================================
// Parallel Writer Benchmark
// Each thread creates its own isolated database
// ============================================================================

class parallel_writer_fixture : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State &) override {}
  void TearDown(const benchmark::State &) override {}
};

BENCHMARK_DEFINE_F(parallel_writer_fixture, parallel_databases)
(benchmark::State &_state) {
  const auto num_threads = static_cast<size_t>(_state.range(0));
  const auto events_per_thread = static_cast<size_t>(_state.range(1));

  for (auto _ : _state) {
    std::vector<thread_context> contexts(num_threads);
    std::vector<std::thread> threads;
    std::atomic<size_t> total_events{0};

    for (size_t i = 0; i < num_threads; ++i) {
      contexts[i].setup(i);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back([&contexts, i, events_per_thread, &total_events]() {
        run_realistic_workload(contexts[i], events_per_thread);
        total_events += events_per_thread;
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    size_t total_db_size = 0;
    for (size_t i = 0; i < num_threads; ++i) {
      total_db_size += contexts[i].get_db_size();
    }

    for (size_t i = 0; i < num_threads; ++i) {
      contexts[i].teardown();
    }

    _state.SetIterationTime(static_cast<double>(duration) / 1000.0);
    _state.counters["total_events"] = static_cast<double>(total_events);
    _state.counters["total_db_size_mb"] =
        static_cast<double>(total_db_size) / (1024.0 * 1024.0);
    _state.counters["threads"] = static_cast<double>(num_threads);
  }

  _state.SetItemsProcessed(_state.iterations() * num_threads *
                           events_per_thread);
}

// Args: {num_threads, events_per_thread}
// ~1.15M events ≈ 500MB, ~2.3M events ≈ 1GB, ~4.6M events ≈ 2GB
BENCHMARK_REGISTER_F(parallel_writer_fixture, parallel_databases)
    ->Unit(benchmark::kSecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Args({2, 1150000})  // 2 threads × 500MB each = 1GB total
    ->Args({2, 2300000})  // 2 threads × 1GB each = 2GB total
    ->Args({2, 4600000})  // 2 threads × 2GB each = 4GB total
    ->Args({4, 1150000})  // 4 threads × 500MB each = 2GB total
    ->Args({4, 2300000})  // 4 threads × 1GB each = 4GB total
    ->Args({8, 1150000})  // 8 threads × 500MB each = 4GB total
    ->Args({8, 2300000}); // 8 threads × 1GB each = 8GB total

// ============================================================================
// Scaling Benchmark - Same total work, varying thread count
// ============================================================================

BENCHMARK_DEFINE_F(parallel_writer_fixture, scaling_test)
(benchmark::State &_state) {
  const auto num_threads = static_cast<size_t>(_state.range(0));
  constexpr size_t total_events = 4600000; // ~2GB total work
  const size_t events_per_thread = total_events / num_threads;

  for (auto _ : _state) {
    std::vector<thread_context> contexts(num_threads);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
      contexts[i].setup(i);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back([&contexts, i, events_per_thread]() {
        run_realistic_workload(contexts[i], events_per_thread);
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    size_t total_db_size = 0;
    for (size_t i = 0; i < num_threads; ++i) {
      total_db_size += contexts[i].get_db_size();
    }

    for (size_t i = 0; i < num_threads; ++i) {
      contexts[i].teardown();
    }

    _state.SetIterationTime(static_cast<double>(duration) / 1000.0);
    _state.counters["total_db_size_mb"] =
        static_cast<double>(total_db_size) / (1024.0 * 1024.0);
    _state.counters["events_per_thread"] =
        static_cast<double>(events_per_thread);
  }

  _state.SetItemsProcessed(_state.iterations() * total_events);
}

BENCHMARK_REGISTER_F(parallel_writer_fixture, scaling_test)
    ->Unit(benchmark::kSecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Arg(1) // baseline single-threaded
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

} // namespace
