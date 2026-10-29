/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_BENCH_METRICS_H_
#define NCCL_M2N_BENCH_METRICS_H_

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

static constexpr const char* BENCH_METRICS_SCHEMA = "nccl_m2n.benchmark.metrics";
static constexpr int BENCH_METRICS_SCHEMA_VERSION = 1;

// Common schema-v1 writer for per-iteration benchmark metric series. A
// benchmark owns collection and reduction semantics; this class owns the
// shared envelope, validation, and serialization contract.
class BenchMetrics {
public:
  BenchMetrics(const char* benchmark, int worldSize, int iterations)
    : benchmark_(benchmark == nullptr ? "" : benchmark), worldSize_(worldSize), iterations_(iterations) {
    if (benchmark_.empty()) {
      throw std::invalid_argument("benchmark name must not be empty");
    }
    if (worldSize_ <= 0) {
      throw std::invalid_argument("world size must be positive");
    }
    if (iterations_ <= 0) {
      throw std::invalid_argument("iteration count must be positive");
    }
  }

  BenchMetrics& addSamples(const char* name, const char* unit, const std::vector<double>& values,
                           nlohmann::json attributes = nlohmann::json::object()) {
    const std::string metricName = name == nullptr ? "" : name;
    const std::string metricUnit = unit == nullptr ? "" : unit;
    if (metricName.empty()) {
      throw std::invalid_argument("metric name must not be empty");
    }
    if (metricUnit.empty()) {
      throw std::invalid_argument("metric unit must not be empty");
    }
    if (values.size() != static_cast<size_t>(iterations_)) {
      throw std::invalid_argument("metric sample count must match iterations");
    }
    if (!attributes.is_object()) {
      throw std::invalid_argument("metric attributes must be an object");
    }
    for (double value : values) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("metric samples must be finite");
      }
    }
    for (const nlohmann::json& metric : metrics_) {
      if (metric.at("name") == metricName) {
        throw std::invalid_argument("metric names must be unique");
      }
    }

    metrics_.push_back({
      {"name", metricName},
      {"type", "samples"},
      {"unit", metricUnit},
      {"attributes", std::move(attributes)},
      {"values", values},
    });
    return *this;
  }

  void write(const char* path) const {
    if (path == nullptr || *path == '\0') {
      throw std::invalid_argument("metrics output path must not be empty");
    }
    if (metrics_.empty()) {
      throw std::invalid_argument("metrics output must contain at least one metric");
    }

    const nlohmann::json document = {
      {"schema", BENCH_METRICS_SCHEMA}, {"schema_version", BENCH_METRICS_SCHEMA_VERSION},
      {"benchmark", benchmark_},        {"world_size", worldSize_},
      {"iterations", iterations_},      {"metrics", metrics_},
    };
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path);
    output << document.dump(2) << '\n';
    output.close();
  }

private:
  std::string benchmark_;
  int worldSize_;
  int iterations_;
  std::vector<nlohmann::json> metrics_;
};

#endif  // NCCL_M2N_BENCH_METRICS_H_
