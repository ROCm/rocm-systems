// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/statistics.h"

#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <format>
#include <iterator>
#include <limits>
#include <sstream>
#include <utility>

namespace stats {

static constexpr std::string_view kSummaryFields[] = {"count",   "sum",  "minimum",
                                                      "maximum", "mean", "variance"};

std::string json_string(std::string_view value) {
  std::string result = "\"";
  for (unsigned char character : value) {
    if (character == '"' || character == '\\') {
      result += '\\';
      result += static_cast<char>(character);
    } else if (character < 0x20) {
      result += std::format("\\u{:04x}", character);
    } else {
      result += static_cast<char>(character);
    }
  }
  result += '"';
  return result;
}

Metric::Metric(std::string name, std::string unit, std::string description, std::string domain,
               std::string kind, std::string value_type)
    : name_(std::move(name)), unit_(std::move(unit)), description_(std::move(description)),
      domain_(std::move(domain)), kind_(std::move(kind)), value_type_(std::move(value_type)) {}

bool Metric::matches(std::string_view unit, std::string_view description, std::string_view domain,
                     std::string_view kind, std::string_view value_type) const {
  return unit_ == unit && description_ == description && domain_ == domain && kind_ == kind &&
         value_type_ == value_type;
}

bool Metric::reserves_name(std::string_view name) const {
  if (name == name_)
    return true;
  if ((kind_ != "histogram" && kind_ != "summary") || !name.starts_with(name_) ||
      name.size() <= name_.size() || name[name_.size()] != '.')
    return false;
  name.remove_prefix(name_.size() + 1);
  return std::find(std::begin(kSummaryFields), std::end(kSummaryFields), name) !=
         std::end(kSummaryFields);
}

void Metric::write_descriptor(std::ostream &output, std::string_view scope_id) const {
  output << std::format("{{\"record\":\"metric\",\"metric_id\":{},\"name\":{},\"unit\":{},"
                        "\"description\":{},\"domain\":{},\"kind\":{},\"value_type\":{},"
                        "\"temporality\":{}}}\n",
                        json_string(std::string(scope_id) + ":" + name_), json_string(name_),
                        json_string(unit_), json_string(description_), json_string(domain_),
                        json_string(kind_), json_string(value_type_),
                        json_string(kind_ == "gauge" ? "instantaneous" : "cumulative"));
}

std::string Metric::sample_prefix(std::string_view record, std::string_view scope_id,
                                  std::string_view suffix) const {
  return std::format("{{\"record\":{},\"scope_id\":{},\"metric_id\":{}", json_string(record),
                     json_string(scope_id),
                     json_string(std::string(scope_id) + ":" + name_ + std::string(suffix)));
}

void Metric::write_summary_descriptors(std::ostream &output, std::string_view scope_id) const {
  for (std::string_view suffix : kSummaryFields) {
    const std::string name = name_ + "." + std::string(suffix);
    const std::string unit = suffix == "count"      ? "samples"
                             : suffix == "variance" ? unit_ + "^2"
                                                    : unit_;
    output << std::format("{{\"record\":\"metric\",\"metric_id\":{},\"name\":{},\"unit\":{},"
                          "\"description\":{},\"domain\":{},\"kind\":\"summary\",\"value_type\":{},"
                          "\"temporality\":\"cumulative\"}}\n",
                          json_string(std::string(scope_id) + ":" + name), json_string(name),
                          json_string(unit), json_string(description_), json_string(domain_),
                          json_string(suffix == "count" ? "u64" : "f64"));
  }
}

void Metric::write_summary_samples(std::ostream &output, std::string_view scope_id,
                                   const SummarySnapshot &snapshot) const {
  output << sample_prefix("sample", scope_id, ".count")
         << std::format(",\"value\":\"{}\",\"overflow\":{}}}\n", snapshot.count,
                        snapshot.count_overflowed);
  const std::pair<std::string_view, double> values[] = {{".sum", snapshot.sum},
                                                        {".minimum", snapshot.minimum},
                                                        {".maximum", snapshot.maximum},
                                                        {".mean", snapshot.mean},
                                                        {".variance", snapshot.variance}};
  for (const auto &[suffix, value] : values) {
    output << sample_prefix("sample", scope_id, suffix);
    if (snapshot.invalid || snapshot.count == 0)
      output << ",\"value\":null,\"availability\":\"unavailable\"}\n";
    else
      output << std::format(",\"value\":{:.17g}}}\n", value);
  }
}

RegisteredCounter::RegisteredCounter(std::string name, std::string unit, std::string description,
                                     std::string domain)
    : Metric(std::move(name), std::move(unit), std::move(description), std::move(domain), "counter",
             "u64") {}

void RegisteredCounter::add(uint64_t amount) noexcept {
  uint64_t current = value_.load(std::memory_order_relaxed);
  for (;;) {
    const bool overflow = amount > std::numeric_limits<uint64_t>::max() - current;
    const uint64_t next = overflow ? std::numeric_limits<uint64_t>::max() : current + amount;
    if (value_.compare_exchange_weak(current, next, std::memory_order_relaxed)) {
      if (overflow)
        overflowed_.store(true, std::memory_order_relaxed);
      return;
    }
  }
}

void RegisteredCounter::write_sample(std::ostream &output, std::string_view scope_id) const {
  output << sample_prefix("sample", scope_id)
         << std::format(",\"value\":\"{}\",\"overflow\":{}}}\n", value(), overflowed());
}

RegisteredGauge::RegisteredGauge(std::string name, std::string unit, std::string description,
                                 std::string domain)
    : Metric(std::move(name), std::move(unit), std::move(description), std::move(domain), "gauge",
             "i64") {}

void RegisteredGauge::add(int64_t amount) noexcept {
  int64_t current = value_.load(std::memory_order_relaxed);
  for (;;) {
    const bool high = amount > 0 && current > std::numeric_limits<int64_t>::max() - amount;
    const bool low = amount < 0 && current < std::numeric_limits<int64_t>::min() - amount;
    const int64_t next = high  ? std::numeric_limits<int64_t>::max()
                         : low ? std::numeric_limits<int64_t>::min()
                               : current + amount;
    if (value_.compare_exchange_weak(current, next, std::memory_order_relaxed)) {
      if (high || low)
        overflowed_.store(true, std::memory_order_relaxed);
      return;
    }
  }
}

void RegisteredGauge::write_sample(std::ostream &output, std::string_view scope_id) const {
  output << sample_prefix("sample", scope_id)
         << std::format(",\"value\":\"{}\",\"overflow\":{}}}\n", value(),
                        overflowed_.load(std::memory_order_relaxed));
}

RegisteredHistogram::RegisteredHistogram(std::string name, std::string unit,
                                         std::string description, std::string domain,
                                         Histogram histogram)
    : Metric(std::move(name), std::move(unit), std::move(description), std::move(domain),
             "histogram", "f64"),
      histogram_(std::move(histogram)) {}

void RegisteredHistogram::observe(double value) noexcept {
  std::lock_guard lock(mutex_);
  histogram_.record(value);
}

bool RegisteredHistogram::matches_bounds(std::span<const double> bounds) const {
  std::lock_guard lock(mutex_);
  const HistogramSnapshot snapshot = histogram_.snapshot();
  return std::equal(bounds.begin(), bounds.end(), snapshot.upper_bounds.begin(),
                    snapshot.upper_bounds.end());
}

void RegisteredHistogram::write_descriptor(std::ostream &output, std::string_view scope_id) const {
  Metric::write_descriptor(output, scope_id);
  write_summary_descriptors(output, scope_id);
}

void RegisteredHistogram::write_sample(std::ostream &output, std::string_view scope_id) const {
  std::lock_guard lock(mutex_);
  const HistogramSnapshot snapshot = histogram_.snapshot();
  for (size_t index = 0; index < snapshot.counts.size(); ++index) {
    output << sample_prefix("histogram_bucket", scope_id)
           << std::format(",\"bucket_index\":{},\"upper_bound\":{},\"count\":\"{}\","
                          "\"overflow\":{}}}\n",
                          index,
                          index < snapshot.upper_bounds.size()
                              ? std::format("{:.17g}", snapshot.upper_bounds[index])
                              : "null",
                          snapshot.counts[index], snapshot.overflowed);
  }
  write_summary_samples(output, scope_id, snapshot.summary);
}

RegisteredSummary::RegisteredSummary(std::string name, std::string unit, std::string description,
                                     std::string domain)
    : Metric(std::move(name), std::move(unit), std::move(description), std::move(domain), "summary",
             "f64") {}

void RegisteredSummary::observe(double value) noexcept {
  std::lock_guard lock(mutex_);
  summary_.record(value);
}

void RegisteredSummary::write_descriptor(std::ostream &output, std::string_view scope_id) const {
  write_summary_descriptors(output, scope_id);
}

void RegisteredSummary::write_sample(std::ostream &output, std::string_view scope_id) const {
  std::lock_guard lock(mutex_);
  write_summary_samples(output, scope_id, summary_.snapshot());
}

Scope::Scope(std::string id, std::string name, std::shared_ptr<const Config> config,
             MetadataWriter metadata)
    : id_(std::move(id)), name_(std::move(name)), config_(std::move(config)),
      metadata_(std::move(metadata)) {}

static bool metric_selected(const Config &config, std::string_view name) {
  if (!config.enabled)
    return false;
  if (config.include.empty())
    return true;
  for (const std::string &pattern : config.include) {
    if (pattern == name ||
        (pattern.ends_with('*') &&
         name.starts_with(std::string_view(pattern).substr(0, pattern.size() - 1))))
      return true;
  }
  return false;
}

bool Scope::selected(std::string_view name) const { return metric_selected(*config_, name); }

bool Session::selected(std::string_view metric_name) const {
  return metric_selected(*config_, metric_name);
}

std::shared_ptr<Metric> Scope::find(std::string_view name) const {
  for (const std::shared_ptr<Metric> &metric : metrics_)
    if (metric->name() == name)
      return metric;
  return {};
}

bool Scope::can_register(std::string_view name, bool has_summary) {
  for (const std::shared_ptr<Metric> &metric : metrics_) {
    if (metric->reserves_name(name)) {
      reject_registration(name);
      return false;
    }
    if (has_summary) {
      for (std::string_view suffix : kSummaryFields) {
        if (metric->reserves_name(std::string(name) + "." + std::string(suffix))) {
          reject_registration(name);
          return false;
        }
      }
    }
  }
  return true;
}

void Scope::reject_registration(std::string_view name) {
  registration_failed_.store(true, std::memory_order_relaxed);
  util::Logger::warn("conflicting statistics registration for '", name, "' in '", name_, "'");
}

std::shared_ptr<RegisteredCounter> Scope::counter(std::string name, std::string unit,
                                                  std::string description, std::string domain) {
  if (!selected(name))
    return {};
  if (domain.empty())
    domain = config_->default_domain;
  std::lock_guard lock(mutex_);
  if (std::shared_ptr<Metric> existing = find(name)) {
    if (!existing->matches(unit, description, domain, "counter", "u64")) {
      reject_registration(name);
      return {};
    }
    return std::dynamic_pointer_cast<RegisteredCounter>(existing);
  }
  if (!can_register(name, false))
    return {};
  auto result = std::make_shared<RegisteredCounter>(std::move(name), std::move(unit),
                                                    std::move(description), std::move(domain));
  metrics_.push_back(result);
  return result;
}

std::shared_ptr<RegisteredGauge> Scope::gauge(std::string name, std::string unit,
                                              std::string description, std::string domain) {
  if (!selected(name))
    return {};
  if (domain.empty())
    domain = config_->default_domain;
  std::lock_guard lock(mutex_);
  if (std::shared_ptr<Metric> existing = find(name)) {
    if (!existing->matches(unit, description, domain, "gauge", "i64")) {
      reject_registration(name);
      return {};
    }
    return std::dynamic_pointer_cast<RegisteredGauge>(existing);
  }
  if (!can_register(name, false))
    return {};
  auto result = std::make_shared<RegisteredGauge>(std::move(name), std::move(unit),
                                                  std::move(description), std::move(domain));
  metrics_.push_back(result);
  return result;
}

std::shared_ptr<RegisteredHistogram> Scope::histogram(std::string name, std::string unit,
                                                      std::string description,
                                                      std::span<const double> bounds,
                                                      std::string domain) {
  if (!selected(name))
    return {};
  if (domain.empty())
    domain = config_->default_domain;
  std::lock_guard lock(mutex_);
  if (std::shared_ptr<Metric> existing = find(name)) {
    const auto histogram = std::dynamic_pointer_cast<RegisteredHistogram>(existing);
    if (!histogram || !existing->matches(unit, description, domain, "histogram", "f64") ||
        !histogram->matches_bounds(bounds)) {
      reject_registration(name);
      return {};
    }
    return histogram;
  }
  if (!can_register(name, true))
    return {};
  util::StringDiagnostic diagnostic;
  util::FailureOr<Histogram> histogram = Histogram::create(bounds, diagnostic.emitter());
  if (histogram.failed()) {
    registration_failed_.store(true, std::memory_order_relaxed);
    util::Logger::warn("cannot register statistics histogram: ", diagnostic.message());
    return {};
  }
  auto result = std::make_shared<RegisteredHistogram>(std::move(name), std::move(unit),
                                                      std::move(description), std::move(domain),
                                                      std::move(histogram).value());
  metrics_.push_back(result);
  return result;
}

void Scope::write_descriptors(std::ostream &output) const {
  std::lock_guard lock(mutex_);
  output << std::format("{{\"record\":\"scope\",\"scope_id\":{},\"name\":{}", json_string(id_),
                        json_string(name_));
  if (metadata_)
    metadata_(output);
  output << ",\"attributes\":{";
  bool first = true;
  for (const auto &[key, value] : attributes_) {
    if (!first)
      output << ',';
    first = false;
    output << json_string(key) << ':' << json_string(value);
  }
  output << "}}\n";
  for (const std::shared_ptr<Metric> &metric : metrics_)
    metric->write_descriptor(output, id_);
}

void Scope::set_attribute(std::string key, std::string value) {
  std::lock_guard lock(mutex_);
  attributes_.insert_or_assign(std::move(key), std::move(value));
}

std::shared_ptr<RegisteredSummary> Scope::summary(std::string name, std::string unit,
                                                  std::string description, std::string domain) {
  if (!selected(name))
    return {};
  if (domain.empty())
    domain = config_->default_domain;
  std::lock_guard lock(mutex_);
  if (std::shared_ptr<Metric> existing = find(name)) {
    if (!existing->matches(unit, description, domain, "summary", "f64")) {
      reject_registration(name);
      return {};
    }
    return std::dynamic_pointer_cast<RegisteredSummary>(existing);
  }
  if (!can_register(name, true))
    return {};
  auto result = std::make_shared<RegisteredSummary>(std::move(name), std::move(unit),
                                                    std::move(description), std::move(domain));
  metrics_.push_back(result);
  return result;
}

void Scope::write_samples(std::ostream &output) const {
  std::lock_guard lock(mutex_);
  for (const std::shared_ptr<Metric> &metric : metrics_)
    metric->write_sample(output, id_);
}

Session::Session(const Config &config) : config_(std::make_shared<Config>(config)) {}

util::FailureOr<std::shared_ptr<Session>> Session::create(const Config &config,
                                                          util::DiagnosticEmitter emit_error) {
  if (!config.enabled)
    return std::shared_ptr<Session>{};
  if (config.format != "jsonl" || config.directory.empty() || config.schema.empty())
    return emit_error.emit()
           << "statistics requires jsonl output and a nonempty directory and schema";
  for (const std::string &pattern : config.include) {
    const size_t wildcard = pattern.find('*');
    if (pattern.empty() || (wildcard != std::string::npos && wildcard != pattern.size() - 1))
      return emit_error.emit() << "statistics.include accepts metric names or trailing-* prefixes";
  }
  std::error_code error;
  std::filesystem::create_directories(config.directory, error);
  if (error)
    return emit_error.emit() << "cannot create statistics directory: " << error.message();
  auto session = std::shared_ptr<Session>(new Session(config));
  static std::atomic<uint64_t> sequence{0};
  const auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
  for (unsigned attempt = 0; attempt != 16; ++attempt) {
    session->run_id_ = std::format("{}-{}", timestamp, sequence.fetch_add(1));
    session->output_path_ =
        (std::filesystem::path(config.directory) / (session->run_id_ + ".jsonl")).string();
    session->output_ = std::fopen(session->output_path_.c_str(), "wx");
    if (session->output_)
      return session;
    if (errno != EEXIST)
      break;
  }
  return emit_error.emit() << "cannot open statistics output: " << std::strerror(errno);
}

Session::~Session() {
  if (output_)
    std::fclose(output_);
}

std::shared_ptr<Scope> Session::scope(std::string key, std::string name, MetadataWriter metadata) {
  std::lock_guard lock(mutex_);
  if (finished_)
    return {};
  if (const auto existing = scopes_.find(key); existing != scopes_.end())
    return existing->second;
  auto result = std::make_shared<Scope>(std::to_string(scopes_.size()), std::move(name), config_,
                                        std::move(metadata));
  scopes_.emplace(std::move(key), result);
  return result;
}

std::shared_ptr<Scope> Session::find_scope(std::string_view key) {
  std::lock_guard lock(mutex_);
  if (finished_)
    return {};
  const auto scope = scopes_.find(key);
  return scope == scopes_.end() ? nullptr : scope->second;
}

void Session::finish(bool complete, MetadataWriter run_metadata, MetadataWriter footer_metadata) {
  std::lock_guard lock(mutex_);
  if (finished_)
    return;
  finished_ = true;
  for (const auto &[key, scope] : scopes_)
    if (scope->registration_failed())
      complete = false;
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_).count();
  std::ostringstream output;
  output << std::format("{{\"schema\":{},\"record\":\"run\",\"run_id\":{},"
                        "\"host_seconds\":{:.17g}",
                        json_string(config_->schema), json_string(run_id_), elapsed);
  if (run_metadata)
    run_metadata(output);
  output << "}\n";
  for (const auto &[key, scope] : scopes_)
    scope->write_descriptors(output);
  output << std::format(
      "{{\"record\":\"snapshot_begin\",\"snapshot_id\":\"1\",\"epoch\":\"0\",\"complete\":{}}}\n",
      complete);
  for (const auto &[key, scope] : scopes_)
    scope->write_samples(output);
  output << "{\"record\":\"snapshot_end\",\"snapshot_id\":\"1\",\"epoch\":\"0\"}\n";
  output << std::format("{{\"record\":\"session_end\",\"complete\":{}", complete);
  if (footer_metadata)
    footer_metadata(output);
  output << "}\n";
  const std::string contents = std::move(output).str();
  if (std::fwrite(contents.data(), 1, contents.size(), output_) != contents.size() ||
      std::fflush(output_) != 0)
    util::Logger::warn("cannot write statistics output '", output_path_,
                       "': ", std::strerror(errno));
}

} // namespace stats
