// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/statistics.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

namespace {

class StatisticsSessionTest : public testing::Test {
protected:
  void SetUp() override {
    directory_ = std::filesystem::temp_directory_path() /
                 ("util-statistics-session-test-" +
                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  }
  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }
  std::filesystem::path directory_;
};

TEST_F(StatisticsSessionTest, ConcurrentCountersAndRetainedHandlesOwnTheirStorage) {
  stats::Config config{.enabled = true, .directory = directory_.string()};
  auto result = stats::Session::create(config);
  ASSERT_TRUE(result.succeeded());
  std::shared_ptr<stats::Session> session = std::move(result).value();
  auto scope = session->scope("worker-0", "worker");
  auto counter = scope->counter("cache.hits", "requests", "Demand hits");
  std::vector<std::thread> threads;
  for (unsigned index = 0; index != 4; ++index)
    threads.emplace_back([counter] {
      for (unsigned increment = 0; increment != 1000; ++increment)
        counter->increment();
    });
  for (std::thread &thread : threads)
    thread.join();
  EXPECT_EQ(counter->value(), 4000u);
  session->finish(true);
  session.reset();
  scope.reset();
  counter->increment();
  EXPECT_EQ(counter->value(), 4001u);
}

TEST_F(StatisticsSessionTest, CounterAndGaugeOverflowIsExplicit) {
  stats::RegisteredCounter counter("count", "items", "", "model");
  counter.add(std::numeric_limits<uint64_t>::max());
  EXPECT_FALSE(counter.overflowed());
  counter.increment();
  EXPECT_EQ(counter.value(), std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(counter.overflowed());
  stats::RegisteredGauge gauge("pending", "items", "", "model");
  gauge.set(std::numeric_limits<int64_t>::min());
  gauge.add(-1);
  EXPECT_EQ(gauge.value(), std::numeric_limits<int64_t>::min());
  std::ostringstream output;
  gauge.write_sample(output, "worker-0");
  EXPECT_NE(output.str().find("\"overflow\":true"), std::string::npos);
}

TEST_F(StatisticsSessionTest, HistogramCountOverflowDoesNotInvalidateExactBuckets) {
  constexpr std::array bounds{0.};
  util::FailureOr<stats::Histogram> lower = stats::Histogram::create(bounds);
  util::FailureOr<stats::Histogram> upper = stats::Histogram::create(bounds);
  ASSERT_TRUE(lower.succeeded());
  ASSERT_TRUE(upper.succeeded());
  lower.value().record(0);
  upper.value().record(1);
  for (unsigned iteration = 0; iteration < 63; ++iteration) {
    ASSERT_TRUE(lower.value().merge(lower.value()).succeeded());
    ASSERT_TRUE(upper.value().merge(upper.value()).succeeded());
  }
  ASSERT_TRUE(lower.value().merge(upper.value()).succeeded());
  stats::RegisteredHistogram histogram("sizes", "bytes", "", "model", std::move(lower).value());
  std::ostringstream output;
  histogram.write_sample(output, "worker-0");
  EXPECT_NE(output.str().find("\"metric_id\":\"worker-0:sizes.count\","
                              "\"value\":\"18446744073709551615\",\"overflow\":true"),
            std::string::npos);
  for (std::string_view bound : {"0", "null"}) {
    EXPECT_NE(output.str().find("\"upper_bound\":" + std::string(bound) +
                                ",\"count\":\"9223372036854775808\",\"overflow\":false"),
              std::string::npos);
  }
  EXPECT_NE(output.str().find("\"metric_id\":\"worker-0:sizes.sum\","
                              "\"value\":null,\"availability\":\"unavailable\""),
            std::string::npos);
}

TEST_F(StatisticsSessionTest, FiltersAreAppliedBeforeMetricAllocation) {
  stats::Config config{
      .enabled = true, .include = {"cache.*", "memory.pending"}, .directory = directory_.string()};
  auto result = stats::Session::create(config);
  ASSERT_TRUE(result.succeeded());
  auto scope = result.value()->scope("worker-0", "worker");
  EXPECT_NE(scope->counter("cache.hits", "requests", ""), nullptr);
  EXPECT_NE(scope->gauge("memory.pending", "operations", ""), nullptr);
  EXPECT_EQ(scope->counter("execution.instructions", "instructions", ""), nullptr);
  EXPECT_EQ(scope->counter("memory.pending.extra", "operations", ""), nullptr);
  EXPECT_EQ(result.value()->scope("worker-0", "worker"), scope);
  EXPECT_NE(result.value()->scope("worker-1", "worker"), scope);
  EXPECT_EQ(result.value()->find_scope("worker-0"), scope);
  EXPECT_EQ(result.value()->find_scope("unregistered"), nullptr);
}

TEST_F(StatisticsSessionTest, ExportPreservesIntegerPrecisionHistogramBoundariesAndPartialStatus) {
  stats::Config config{.enabled = true, .directory = directory_.string()};
  auto result = stats::Session::create(config);
  ASSERT_TRUE(result.succeeded());
  auto session = std::move(result).value();
  auto scope = session->scope("worker-0", "worker");
  scope->counter("cache.hits", "requests", "Demand hits")->add((uint64_t{1} << 53) + 1);
  constexpr std::array<double, 2> bounds{4, 16};
  auto histogram = scope->histogram("memory.request_bytes", "bytes", "Request size", bounds);
  for (double value : {1.0, 4.0, 5.0, 16.0, 17.0})
    histogram->observe(value);
  session->finish(false);
  session->finish(false);
  std::ifstream file(session->output_path());
  const std::string output{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  EXPECT_NE(output.find("\"value\":\"9007199254740993\""), std::string::npos);
  EXPECT_NE(output.find("\"upper_bound\":4,\"count\":\"2\""), std::string::npos);
  EXPECT_NE(output.find("\"upper_bound\":16,\"count\":\"2\""), std::string::npos);
  EXPECT_NE(output.find("\"upper_bound\":null,\"count\":\"1\""), std::string::npos);
  EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":false"), std::string::npos);
  EXPECT_EQ(output.find("\"record\":\"run\""), output.rfind("\"record\":\"run\""));
}

TEST_F(StatisticsSessionTest, SeparateSessionsNeverTruncateOneAnothersOutput) {
  stats::Config config{.enabled = true, .directory = directory_.string()};
  auto first = stats::Session::create(config);
  auto second = stats::Session::create(config);
  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  EXPECT_NE(first.value()->output_path(), second.value()->output_path());
  first.value()->finish(true);
  second.value()->finish(true);
  EXPECT_GT(std::filesystem::file_size(first.value()->output_path()), 0u);
  EXPECT_GT(std::filesystem::file_size(second.value()->output_path()), 0u);
}

TEST_F(StatisticsSessionTest, ConflictingDescriptorsAndBucketLayoutsReturnAnExplicitFailureStatus) {
  stats::Config config{.enabled = true, .directory = directory_.string()};
  auto result = stats::Session::create(config);
  ASSERT_TRUE(result.succeeded());
  auto scope = result.value()->scope("worker-0", "worker");
  auto counter = scope->counter("count", "requests", "Requests");
  EXPECT_EQ(scope->counter("count", "requests", "Requests"), counter);
  EXPECT_EQ(scope->counter("count", "bytes", "Requests"), nullptr);
  EXPECT_TRUE(scope->registration_failed());
  auto other = result.value()->scope("worker-1", "worker");
  constexpr std::array<double, 2> bounds{4, 16};
  constexpr std::array<double, 2> conflicting_bounds{4, 32};
  auto histogram = other->histogram("sizes", "bytes", "Request size", bounds);
  EXPECT_EQ(other->histogram("sizes", "bytes", "Request size", bounds), histogram);
  EXPECT_EQ(other->histogram("sizes", "bytes", "Request size", conflicting_bounds), nullptr);
  EXPECT_TRUE(other->registration_failed());
  result.value()->finish(true);
  std::ifstream file(result.value()->output_path());
  const std::string output{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":false"), std::string::npos);
}

TEST_F(StatisticsSessionTest, GaugesHaveInstantaneousTemporalityAndOwnedScopeAttributes) {
  stats::Config config{.enabled = true, .directory = directory_.string()};
  auto result = stats::Session::create(config);
  ASSERT_TRUE(result.succeeded());
  auto scope = result.value()->scope("worker-0", "worker");
  std::string kernel = "kernel\"name";
  scope->set_attribute("kernel", kernel);
  kernel.clear();
  scope->gauge("memory.pending", "operations", "Pending");
  std::ostringstream output;
  scope->write_descriptors(output);
  EXPECT_NE(output.str().find("\"temporality\":\"instantaneous\""), std::string::npos);
  EXPECT_NE(output.str().find("kernel\\\"name"), std::string::npos);
}

TEST_F(StatisticsSessionTest, ExpandedSummaryNamesCannotAliasExplicitMetricsInEitherOrder) {
  stats::Config config{.enabled = true, .directory = directory_.string()};
  auto result = stats::Session::create(config);
  ASSERT_TRUE(result.succeeded());
  for (std::string_view suffix : {"count", "sum", "minimum", "maximum", "mean", "variance"}) {
    SCOPED_TRACE(suffix);
    const std::string name = "latency." + std::string(suffix);
    auto explicit_first = result.value()->scope("explicit-first-" + name, "worker");
    ASSERT_NE(explicit_first->counter(name, "items", "Explicit value"), nullptr);
    EXPECT_EQ(explicit_first->summary("latency", "seconds", "Latency"), nullptr);
    EXPECT_TRUE(explicit_first->registration_failed());

    auto summary_first = result.value()->scope("summary-first-" + name, "worker");
    auto summary = summary_first->summary("latency", "seconds", "Latency");
    ASSERT_NE(summary, nullptr);
    EXPECT_EQ(summary_first->counter(name, "items", "Explicit value"), nullptr);
    EXPECT_TRUE(summary_first->registration_failed());
    EXPECT_EQ(summary_first->summary("latency", "seconds", "Latency"), summary);
    EXPECT_NE(summary_first->counter("latency.extra", "items", "Separate quantity"), nullptr);
  }
}

TEST_F(StatisticsSessionTest, HistogramSummaryNamesCannotAliasExplicitMetricsInEitherOrder) {
  stats::Config config{.enabled = true, .directory = directory_.string()};
  auto result = stats::Session::create(config);
  ASSERT_TRUE(result.succeeded());
  constexpr std::array<double, 2> bounds{4, 16};
  for (std::string_view suffix : {"count", "sum", "minimum", "maximum", "mean", "variance"}) {
    SCOPED_TRACE(suffix);
    const std::string name = "sizes." + std::string(suffix);
    auto explicit_first = result.value()->scope("explicit-first-" + name, "worker");
    ASSERT_NE(explicit_first->gauge(name, "items", "Explicit value"), nullptr);
    EXPECT_EQ(explicit_first->histogram("sizes", "bytes", "Request size", bounds), nullptr);
    EXPECT_TRUE(explicit_first->registration_failed());

    auto histogram_first = result.value()->scope("histogram-first-" + name, "worker");
    auto histogram = histogram_first->histogram("sizes", "bytes", "Request size", bounds);
    ASSERT_NE(histogram, nullptr);
    EXPECT_EQ(histogram_first->gauge(name, "items", "Explicit value"), nullptr);
    EXPECT_TRUE(histogram_first->registration_failed());
    EXPECT_EQ(histogram_first->histogram("sizes", "bytes", "Request size", bounds), histogram);
  }
}

TEST_F(StatisticsSessionTest, DisabledAndInvalidConfigurationHaveNoFilesystemEffects) {
  stats::Config config{.directory = directory_.string()};
  auto disabled = stats::Session::create(config);
  ASSERT_TRUE(disabled.succeeded());
  EXPECT_EQ(disabled.value(), nullptr);
  EXPECT_FALSE(std::filesystem::exists(directory_));

  config.enabled = true;
  for (std::string_view pattern : {"", "cache.*.misses", "cache.**"}) {
    config.include = {std::string(pattern)};
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(stats::Session::create(config, diagnostic.emitter()).failed());
    EXPECT_NE(diagnostic.message().find("trailing-*"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(directory_));
  }
  config.include.clear();
  config.format = "unknown";
  EXPECT_TRUE(stats::Session::create(config).failed());
  EXPECT_FALSE(std::filesystem::exists(directory_));
  config.format = "jsonl";
  config.schema.clear();
  EXPECT_TRUE(stats::Session::create(config).failed());
  EXPECT_FALSE(std::filesystem::exists(directory_));
  config.schema = "util.statistics.v1";
  config.directory.clear();
  EXPECT_TRUE(stats::Session::create(config).failed());
  EXPECT_FALSE(std::filesystem::exists(directory_));
}

TEST_F(StatisticsSessionTest, CallerMetadataAndScopeIdentitySurviveFinalExport) {
  stats::Config config{.enabled = true,
                       .directory = directory_.string(),
                       .schema = "example.statistics.v1",
                       .default_domain = "host_events"};
  auto result = stats::Session::create(config);
  ASSERT_TRUE(result.succeeded());
  auto session = std::move(result).value();
  std::string label = "thread\n\"primary";
  auto scope = session->scope("worker-0", "worker", [label](std::ostream &output) {
    output << ",\"worker_label\":" << stats::json_string(label);
  });
  label.clear();
  ASSERT_NE(scope, nullptr);
  EXPECT_EQ(session->scope("worker-0", "ignored"), scope);
  EXPECT_EQ(scope->name(), "worker");
  auto counter = scope->counter("processed", "items", "Completed items");
  ASSERT_NE(counter, nullptr);
  EXPECT_EQ(scope->counter("processed", "items", "Completed items", "host_events"), counter);
  ASSERT_NE(scope->counter("modeled", "items", "Modeled items", "model_events"), nullptr);
  counter->add(7);
  session->finish(
      true,
      [](std::ostream &output) {
        output << ",\"application\":" << stats::json_string("tool\tname");
      },
      [](std::ostream &output) { output << ",\"records_dropped\":\"0\""; });
  bool called_again = false;
  session->finish(false, [&called_again](std::ostream &) { called_again = true; });
  EXPECT_FALSE(called_again);
  EXPECT_EQ(session->scope("late", "late"), nullptr);
  EXPECT_EQ(session->find_scope("worker-0"), nullptr);

  std::ifstream file(session->output_path());
  const std::string output{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  EXPECT_NE(output.find("\"schema\":\"example.statistics.v1\""), std::string::npos);
  EXPECT_NE(output.find("\"worker_label\":\"thread\\u000a\\\"primary\""), std::string::npos);
  EXPECT_NE(output.find("\"application\":\"tool\\u0009name\""), std::string::npos);
  EXPECT_NE(output.find("\"domain\":\"host_events\""), std::string::npos);
  EXPECT_NE(output.find("\"domain\":\"model_events\""), std::string::npos);
  EXPECT_NE(output.find("\"value\":\"7\""), std::string::npos);
  EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":true,\"records_dropped\":\"0\""),
            std::string::npos);
}

} // namespace
