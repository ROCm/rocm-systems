// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Performance benchmark for NIC transport backends
 *
 * Compares ioctl vs netlink performance for ethtool operations.
 * Measures latency in microseconds per operation.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "smi_nic_transport.h"

using namespace amd::smi::nic::transport;
using namespace std::chrono;

// Timing Utilities

/**
 * @brief Measure average execution time of an operation
 *
 * @param operation Lambda/function to measure
 * @param iterations Number of times to run the operation
 * @return Average time in microseconds per operation
 */
template <typename Func>
double measure_operation(Func&& operation, int iterations) {
  auto start = high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    operation();
  }

  auto end = high_resolution_clock::now();
  auto duration = duration_cast<nanoseconds>(end - start).count();

  return static_cast<double>(duration) / iterations / 1000.0;  // Convert to µs
}

/**
 * @brief Measure operation with statistical analysis
 *
 * Runs multiple trials to compute mean, stddev, min, max
 */
struct BenchmarkResult {
  double mean;
  double stddev;
  double min;
  double max;
  int iterations;

  void print(const std::string& label) const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::setw(12) << std::left << label;
    std::cout << std::setw(8) << std::right << mean << " µs/op";
    std::cout << "  (σ=" << stddev << ", min=" << min << ", max=" << max << ")\n";
  }
};

template <typename Func>
BenchmarkResult measure_with_stats(Func&& operation, int trials, int iterations_per_trial) {
  std::vector<double> times;
  times.reserve(trials);

  for (int trial = 0; trial < trials; trial++) {
    double time = measure_operation(operation, iterations_per_trial);
    times.push_back(time);
  }

  BenchmarkResult result;
  result.iterations = trials * iterations_per_trial;
  result.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  result.min = *std::min_element(times.begin(), times.end());
  result.max = *std::max_element(times.begin(), times.end());

  double variance = 0.0;
  for (double t : times) {
    variance += (t - result.mean) * (t - result.mean);
  }
  result.stddev = std::sqrt(variance / times.size());

  return result;
}

// Benchmark Tests

constexpr const char kRule[] =
    "+-------------------------------------------------------------------+\n";

void print_header(const std::string& title) {
  std::cout << "\n" << kRule;
  std::cout << "|  " << title << "\n";
  std::cout << kRule;
}

void print_separator() { std::cout << kRule; }

void compare_results(const BenchmarkResult& ioctl_result, const BenchmarkResult& netlink_result,
                     const std::string& /* operation_name */) {
  double ratio = netlink_result.mean / ioctl_result.mean;

  std::cout << std::fixed << std::setprecision(1);
  if (ratio > 1.0) {
    std::cout << "  → Ioctl is " << ((ratio - 1.0) * 100) << "% faster (ratio: " << ratio << "x)\n";
  } else {
    std::cout << "  → Netlink is " << ((1.0 / ratio - 1.0) * 100)
              << "% faster (ratio: " << (1.0 / ratio) << "x)\n";
  }
}

// Main Benchmark

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <interface> [iterations]\n";
    std::cerr << "\n";
    std::cerr << "Examples:\n";
    std::cerr << "  " << argv[0] << " eth0          # Default: 1000 iterations\n";
    std::cerr << "  " << argv[0] << " eth0 10000    # High precision: 10000 iterations\n";
    std::cerr << "  " << argv[0] << " enp1s0 100    # Quick test: 100 iterations\n";
    return 1;
  }

  std::string iface = argv[1];
  int iterations = (argc >= 3) ? std::atoi(argv[2]) : 1000;
  int trials = 10;  // Run each benchmark 10 times for statistics

  std::cout << "\n";
  std::cout << "+-------------------------------------------------------------------+\n";
  std::cout << "|  AMDSMI NIC Transport Backend Performance Benchmark               |\n";
  std::cout << "+-------------------------------------------------------------------+\n";
  std::cout << "\n";
  std::cout << "Interface:         " << iface << "\n";
  std::cout << "Iterations/trial:  " << iterations << "\n";
  std::cout << "Trials:            " << trials << "\n";
  std::cout << "Total ops/test:    " << (iterations * trials) << "\n";

  auto ioctl_transport = create_transport(NicBackend_t::Ioctl);
  auto netlink_transport = create_transport(NicBackend_t::Netlink);
  auto auto_transport = create_transport(NicBackend_t::Auto);

  std::cout << "\nBackend availability:\n";
  std::cout << "  Ioctl:   " << ioctl_transport->backend_name() << "\n";

  bool netlink_available = false;
#ifdef HAVE_LIBNL3
  auto test_nl = netlink_transport->get_pause_params(iface);
  netlink_available = (test_nl.success || test_nl.error_code != ENOTSUP);
  std::cout << "  Netlink: " << netlink_transport->backend_name();
  if (!netlink_available) {
    std::cout << " (NOT AVAILABLE - initialization failed)";
  }
  std::cout << "\n";
#else
  std::cout << "  Netlink: NOT COMPILED (libnl-3 not available at build time)\n";
#endif

  std::cout << "  Auto:    " << auto_transport->backend_name() << "\n";

  if (!netlink_available) {
    std::cout << "\nWARNING: Netlink backend not available.\n";
    std::cout << "   Only ioctl benchmarks will run.\n";
    std::cout << "   To enable netlink:\n";
    std::cout << "   - Install libnl-3-dev\n";
    std::cout << "   - Rebuild with libnl-3 support\n";
    std::cout << "   - Run on kernel 5.6+\n";
  }

  print_header("Warmup (100 operations to warm caches)");
  for (int i = 0; i < 100; i++) {
    (void)ioctl_transport->get_pause_params(iface);
    if (netlink_available) {
      (void)netlink_transport->get_pause_params(iface);
    }
  }
  std::cout << "Warmup complete.\n";

  print_header("Benchmark 2: Pause Parameters (get_pause_params)");

  auto ioctl_pause = measure_with_stats([&]() { (void)ioctl_transport->get_pause_params(iface); },
                                        trials, iterations);
  ioctl_pause.print("Ioctl:");

  BenchmarkResult netlink_pause{};
  if (netlink_available) {
    netlink_pause = measure_with_stats([&]() { (void)netlink_transport->get_pause_params(iface); },
                                       trials, iterations);
    netlink_pause.print("Netlink:");

    print_separator();
    compare_results(ioctl_pause, netlink_pause, "Pause");
  }

  print_header("Benchmark 3: Link Settings (get_link_settings)");

  auto ioctl_link = measure_with_stats([&]() { (void)ioctl_transport->get_link_settings(iface); },
                                       trials, iterations);
  ioctl_link.print("Ioctl:");

  BenchmarkResult netlink_link{};
  if (netlink_available) {
    netlink_link = measure_with_stats([&]() { (void)netlink_transport->get_link_settings(iface); },
                                      trials, iterations);
    netlink_link.print("Netlink:");

    print_separator();
    compare_results(ioctl_link, netlink_link, "Link Settings");
  }

  print_header("Benchmark 4: Driver Info (get_driver_info) - Ioctl only");

  auto ioctl_driver = measure_with_stats(
      [&]() { auto result = ioctl_transport->get_driver_info(iface); }, trials, iterations);
  ioctl_driver.print("Ioctl:");

  std::cout << "\n  Note: Netlink API does not support driver info queries.\n";
  std::cout << "        AutoBackend always uses ioctl for this operation.\n";

  print_header("Benchmark 5: Statistics (get_statistics) - Ioctl only");

  auto ioctl_stats =
      measure_with_stats([&]() { auto result = ioctl_transport->get_statistics(iface); }, trials,
                         iterations / 10);  // Stats are expensive, fewer iterations
  ioctl_stats.print("Ioctl:");

  std::cout << "\n  Note: Netlink statistics parsing is incomplete.\n";
  std::cout << "        AutoBackend always uses ioctl for this operation.\n";

  print_header("Benchmark 6: Batch Query (Pause + Link)");
  std::cout << "Simulates typical monitoring: query multiple parameters at once\n";
  print_separator();

  auto ioctl_batch = measure_with_stats(
      [&]() {
        (void)ioctl_transport->get_pause_params(iface);
        (void)ioctl_transport->get_link_settings(iface);
      },
      trials, iterations);
  ioctl_batch.print("Ioctl batch:");

  if (netlink_available) {
    auto netlink_batch = measure_with_stats(
        [&]() {
          (void)netlink_transport->get_pause_params(iface);
          (void)netlink_transport->get_link_settings(iface);
        },
        trials, iterations);
    netlink_batch.print("Netlink batch:");

    print_separator();
    compare_results(ioctl_batch, netlink_batch, "Batch");
  }

  print_header("Benchmark 7: Auto Backend (production configuration)");
  std::cout << "This is what actually runs in SmiNicPort\n";
  print_separator();

  auto auto_pause = measure_with_stats([&]() { (void)auto_transport->get_pause_params(iface); },
                                       trials, iterations);
  auto_pause.print("Auto Pause:");

  auto auto_driver = measure_with_stats(
      [&]() { auto result = auto_transport->get_driver_info(iface); }, trials, iterations);
  auto_driver.print("Auto Driver:");

  std::cout << "\n  Note: Auto backend tries netlink first (if available), falls back to ioctl.\n";
  std::cout << "        Driver info always uses ioctl (netlink doesn't support it).\n";

  print_header("Summary");

  if (netlink_available) {
    std::cout << "\nAverage latency comparison:\n";
    std::cout << std::fixed << std::setprecision(2);

    double avg_ioctl = (ioctl_pause.mean + ioctl_link.mean) / 2.0;
    double avg_netlink = (netlink_pause.mean + netlink_link.mean) / 2.0;

    std::cout << "  Ioctl average:   " << avg_ioctl << " µs/op\n";
    std::cout << "  Netlink average: " << avg_netlink << " µs/op\n";

    print_separator();

    if (avg_netlink < avg_ioctl) {
      double improvement = ((avg_ioctl - avg_netlink) / avg_ioctl) * 100;
      std::cout << "\nNetlink is " << improvement << "% faster on average\n";
      std::cout << "  Recommendation: Use 'auto' backend (already configured)\n";
    } else {
      double overhead = ((avg_netlink - avg_ioctl) / avg_ioctl) * 100;
      std::cout << "\nNetlink has " << overhead << "% overhead on average\n";

      if (overhead > 100) {
        std::cout << "  Recommendation: Consider using 'ioctl' backend for performance\n";
      } else {
        std::cout
            << "  Recommendation: Keep 'auto' backend (overhead negligible for NIC monitoring)\n";
        std::cout << "                  Benefits: Future-proof, kernel's preferred interface\n";
      }
    }
  } else {
    std::cout << "\nNetlink not available - using ioctl backend.\n";
    std::cout << "This is the fallback behavior and is expected on:\n";
    std::cout << "  - Kernel < 5.6\n";
    std::cout << "  - Systems without libnl-3\n";
    std::cout << "  - Systems where netlink ethtool is not enabled\n";
  }

  std::cout << "\n";
  std::cout << "+-------------------------------------------------------------------+\n";
  std::cout << "|  Benchmark Complete                                               |\n";
  std::cout << "+-------------------------------------------------------------------+\n";
  std::cout << "\n";

  return 0;
}
