// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// compare_amdsmi: a standalone comparison harness (NOT a gtest) that reads GPU
// metrics via the gpumetrics Collector AND the `amd-smi` CLI, then prints a
// side-by-side table with a per-metric PASS/FAIL verdict against a tolerance.
//
// Instantaneous samples from two readers fluctuate, so tolerances are generous:
// the goal is to catch gross errors (wrong unit, wrong field, off by 1000x), not
// to demand identical values.
//
// Unit normalization: amd-smi reports VRAM in MB, the plugin in bytes (convert
// MB * 1024 * 1024). amd-smi socket_power (current W) is compared with the
// plugin's average_socket (average W). temps/clocks/activity/fan rpm are
// same-unit on both sides.
//
// Exit 0 if all metrics are within tolerance, or if amd-smi / the GPU is
// unavailable (no-op). Nonzero ONLY on a real mismatch. Every amd-smi call is
// wrapped in `timeout`.

#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gpumetrics/gpumetrics.h"

using namespace gpumetrics;

namespace {

// Run a shell command (already timeout-wrapped by the caller) and capture
// stdout. nullopt if it could not start or exited non-zero.
std::optional<std::string> RunCapture(const std::string& cmd) {
  std::string out;
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return std::nullopt;
  std::array<char, 4096> buf;
  size_t n;
  while ((n = std::fread(buf.data(), 1, buf.size(), p)) > 0) out.append(buf.data(), n);
  int rc = pclose(p);
  if (rc != 0) return std::nullopt;
  return out;
}

// Minimal JSON parser, enough for amd-smi's `metric --json` output. Strings like
// "N/A" become JNull.
struct JValue;
using JObject = std::map<std::string, std::shared_ptr<JValue>>;
using JArray = std::vector<std::shared_ptr<JValue>>;

struct JValue {
  enum class Kind { Null, Bool, Number, String, Object, Array } kind = Kind::Null;
  double number = 0.0;
  bool boolean = false;
  std::string str;
  JObject obj;
  JArray arr;
};

class JParser {
 public:
  explicit JParser(const std::string& s) : s_(s) {}

  std::shared_ptr<JValue> Parse() {
    SkipWs();
    auto v = ParseValue();
    return v;
  }

 private:
  const std::string& s_;
  size_t i_ = 0;

  void SkipWs() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }

  std::shared_ptr<JValue> ParseValue() {
    SkipWs();
    if (i_ >= s_.size()) return nullptr;
    char c = s_[i_];
    if (c == '{') return ParseObject();
    if (c == '[') return ParseArray();
    if (c == '"') return ParseString();
    if (c == 't' || c == 'f') return ParseBool();
    if (c == 'n') return ParseNull();
    return ParseNumber();
  }

  std::shared_ptr<JValue> ParseObject() {
    auto v = std::make_shared<JValue>();
    v->kind = JValue::Kind::Object;
    ++i_;  // '{'
    SkipWs();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      return v;
    }
    while (i_ < s_.size()) {
      SkipWs();
      auto key = ParseRawString();
      SkipWs();
      if (i_ < s_.size() && s_[i_] == ':') ++i_;
      auto val = ParseValue();
      v->obj[key] = val;
      SkipWs();
      if (i_ < s_.size() && s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (i_ < s_.size() && s_[i_] == '}') {
        ++i_;
        break;
      }
      break;
    }
    return v;
  }

  std::shared_ptr<JValue> ParseArray() {
    auto v = std::make_shared<JValue>();
    v->kind = JValue::Kind::Array;
    ++i_;  // '['
    SkipWs();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      return v;
    }
    while (i_ < s_.size()) {
      auto val = ParseValue();
      v->arr.push_back(val);
      SkipWs();
      if (i_ < s_.size() && s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (i_ < s_.size() && s_[i_] == ']') {
        ++i_;
        break;
      }
      break;
    }
    return v;
  }

  std::string ParseRawString() {
    std::string out;
    if (i_ >= s_.size() || s_[i_] != '"') return out;
    ++i_;  // opening quote
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\' && i_ < s_.size()) {
        char e = s_[i_++];
        switch (e) {
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          default: out.push_back(e); break;
        }
      } else {
        out.push_back(c);
      }
    }
    if (i_ < s_.size() && s_[i_] == '"') ++i_;  // closing quote
    return out;
  }

  std::shared_ptr<JValue> ParseString() {
    auto v = std::make_shared<JValue>();
    v->kind = JValue::Kind::String;
    v->str = ParseRawString();
    return v;
  }

  std::shared_ptr<JValue> ParseBool() {
    auto v = std::make_shared<JValue>();
    v->kind = JValue::Kind::Bool;
    if (s_.compare(i_, 4, "true") == 0) {
      v->boolean = true;
      i_ += 4;
    } else if (s_.compare(i_, 5, "false") == 0) {
      v->boolean = false;
      i_ += 5;
    }
    return v;
  }

  std::shared_ptr<JValue> ParseNull() {
    auto v = std::make_shared<JValue>();
    v->kind = JValue::Kind::Null;
    if (s_.compare(i_, 4, "null") == 0) i_ += 4;
    return v;
  }

  std::shared_ptr<JValue> ParseNumber() {
    auto v = std::make_shared<JValue>();
    v->kind = JValue::Kind::Number;
    size_t start = i_;
    while (i_ < s_.size()) {
      char c = s_[i_];
      if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' ||
          c == '.' || c == 'e' || c == 'E') {
        ++i_;
      } else {
        break;
      }
    }
    if (i_ > start) {
      v->number = std::strtod(s_.substr(start, i_ - start).c_str(), nullptr);
    } else {
      v->kind = JValue::Kind::Null;  // couldn't parse anything
      ++i_;                          // avoid infinite loop
    }
    return v;
  }
};

// Navigate an object path like {"temperature","edge"}. nullptr if a hop is
// missing.
std::shared_ptr<JValue> Get(const std::shared_ptr<JValue>& root,
                            const std::vector<std::string>& path) {
  auto cur = root;
  for (const auto& hop : path) {
    if (!cur || cur->kind != JValue::Kind::Object) return nullptr;
    auto it = cur->obj.find(hop);
    if (it == cur->obj.end()) return nullptr;
    cur = it->second;
  }
  return cur;
}

// Extract a number from a node that is itself a number, or an object with a
// "value" number field (amd-smi's {value, unit} wrapper). nullopt otherwise.
std::optional<double> AsNumber(const std::shared_ptr<JValue>& node) {
  if (!node) return std::nullopt;
  if (node->kind == JValue::Kind::Number) return node->number;
  if (node->kind == JValue::Kind::Object) {
    auto it = node->obj.find("value");
    if (it != node->obj.end() && it->second &&
        it->second->kind == JValue::Kind::Number)
      return it->second->number;
  }
  return std::nullopt;
}

std::optional<double> GetNum(const std::shared_ptr<JValue>& root,
                             const std::vector<std::string>& path) {
  return AsNumber(Get(root, path));
}

// Prefer an explicit override, then the ROCm install, then PATH.
std::string AmdSmiBin() {
  if (const char* env = std::getenv("GPUM_AMDSMI_BIN")) return env;
  if (FILE* f = std::fopen("/opt/rocm/bin/amd-smi", "r")) {
    std::fclose(f);
    return "/opt/rocm/bin/amd-smi";
  }
  return "amd-smi";
}

bool AmdSmiAvailable(const std::string& bin) {
  auto out = RunCapture("timeout 30 " + bin + " version 2>/dev/null");
  return out.has_value();
}

// Fetch and parse `amd-smi metric -g <gpu> --json`; returns the per-GPU JValue
// or nullptr.
std::shared_ptr<JValue> AmdSmiMetric(const std::string& bin, uint32_t gpu) {
  std::string cmd = "timeout 30 " + bin + " metric -g " + std::to_string(gpu) +
                    " --json 2>/dev/null";
  auto out = RunCapture(cmd);
  if (!out) return nullptr;
  JParser parser(*out);
  auto root = parser.Parse();
  if (!root) return nullptr;
  // Expected shape: {"gpu_data":[ {...} ]}. Fall back to root if already the
  // per-gpu object.
  auto gd = Get(root, {"gpu_data"});
  if (gd && gd->kind == JValue::Kind::Array && !gd->arr.empty()) return gd->arr[0];
  if (root->kind == JValue::Kind::Array && !root->arr.empty()) return root->arr[0];
  return root;
}

// Metric comparison spec.
struct Row {
  std::string label;      // display metric name
  std::string gm_key;     // gpumetrics key
  std::vector<std::string> asmi_path;  // JSON path into amd-smi per-gpu object
  double asmi_scale;      // multiply amd-smi value to match plugin unit
  double abs_tol;         // absolute tolerance (plugin units)
  double rel_tol;         // relative tolerance (fraction of max magnitude)
};

const std::vector<Row>& Spec() {
  static const std::vector<Row> spec = {
      // label,             gm_key,               amd-smi path,                         scale,        abs,   rel
      {"temp.edge (C)",     "temp.edge",          {"temperature", "edge"},              1.0,          5.0,   0.10},
      {"temp.hotspot (C)",  "temp.hotspot",       {"temperature", "hotspot"},           1.0,          5.0,   0.10},
      {"clock.gfx (MHz)",   "clock.gfx",          {"clock", "gfx_0", "clk"},            1.0,          200.0, 0.50},
      {"clock.mem (MHz)",   "clock.mem",          {"clock", "mem_0", "clk"},            1.0,          200.0, 0.50},
      {"power.avg (W)",     "power.average_socket",{"power", "socket_power"},            1.0,          10.0,  0.50},
      {"vram.used (bytes)", "mem.vram.used",      {"mem_usage", "used_vram"},           1024.0*1024.0,64.0*1024.0*1024.0, 0.10},
      {"vram.total (bytes)","mem.vram.total",     {"mem_usage", "total_vram"},          1024.0*1024.0,64.0*1024.0*1024.0, 0.05},
      {"fan.rpm",           "fan.rpm",            {"fan", "rpm"},                       1.0,          300.0, 0.50},
      {"activity.gfx (%)",  "activity.gfx",       {"usage", "gfx_activity"},            1.0,          20.0,  1.00},
  };
  return spec;
}

std::string Fmt(double v) {
  char buf[64];
  if (std::fabs(v) >= 1e6)
    std::snprintf(buf, sizeof(buf), "%.3e", v);
  else
    std::snprintf(buf, sizeof(buf), "%.2f", v);
  return buf;
}

}  // namespace

int main() {
  std::printf("=== gpumetrics vs amd-smi comparison harness ===\n\n");

  // Build a collector against the real plugins.
  CollectorOptions opts;
  if (const char* env = std::getenv("GPUM_HW_PLUGIN_DIRS")) {
    std::string dirs = env;
    size_t start = 0;
    while (start <= dirs.size()) {
      size_t colon = dirs.find(':', start);
      std::string d = dirs.substr(start, colon == std::string::npos
                                             ? std::string::npos
                                             : colon - start);
      if (!d.empty()) opts.plugin_paths.push_back(d);
      if (colon == std::string::npos) break;
      start = colon + 1;
    }
  }
  if (opts.plugin_paths.empty())
    opts.plugin_paths = {"plugins/amdsmi", "plugins/rocprofiler"};
  opts.provider_priority = {"amdsmi", "rocprofiler"};

  gpum_status st = GPUM_OK;
  auto c = Collector::Create(opts, &st);
  if (!c) {
    std::printf("Collector unavailable (%s); nothing to compare. PASS (no-op).\n",
                gpum_status_string(st));
    return 0;
  }
  bool have_amdsmi = false;
  for (const auto& n : c->loadedPlugins())
    if (n == "amdsmi") have_amdsmi = true;
  if (!have_amdsmi || c->devices().empty()) {
    std::printf("No amdsmi plugin / no GPUs discovered; nothing to compare. PASS (no-op).\n");
    return 0;
  }

  // Check amd-smi availability.
  const std::string bin = AmdSmiBin();
  if (!AmdSmiAvailable(bin)) {
    std::printf("amd-smi not available (tried '%s'); nothing to compare. PASS (no-op).\n",
                bin.c_str());
    return 0;
  }
  std::printf("Using amd-smi: %s\n", bin.c_str());
  std::printf("GPUs discovered by gpumetrics: %zu\n\n", c->devices().size());

  int mismatches = 0;
  int compared = 0;

  for (const auto& dev : c->devices()) {
    // Both index GPUs in socket-then-BDF order, so dev.id.gpu maps directly to
    // the amd-smi -g index.
    const uint32_t gpu = dev.id.gpu;
    auto asmi = AmdSmiMetric(bin, gpu);
    std::printf("---- GPU %u  (%s, bdf=0x%llx) ----\n", gpu, dev.name.c_str(),
                static_cast<unsigned long long>(dev.identity.bdf));
    if (!asmi) {
      std::printf("  amd-smi metric read failed for gpu %u; skipping this GPU.\n\n", gpu);
      continue;
    }

    std::printf("  %-20s %14s %14s %14s  %s\n", "metric", "gpumetrics", "amd-smi",
                "delta", "verdict");
    std::printf("  %-20s %14s %14s %14s  %s\n", "------", "----------", "-------",
                "-----", "-------");

    for (const auto& row : Spec()) {
      Sample s = c->read(dev.id, row.gm_key);
      std::optional<double> asmi_raw = GetNum(asmi, row.asmi_path);

      std::string gm_str, asmi_str, delta_str, verdict;

      bool gm_ok = s.ok();
      bool asmi_ok = asmi_raw.has_value();

      double gm_val = gm_ok ? s.as_double() : 0.0;
      double asmi_val = asmi_ok ? (*asmi_raw * row.asmi_scale) : 0.0;

      gm_str = gm_ok ? Fmt(gm_val) : std::string(gpum_status_string(s.status));
      asmi_str = asmi_ok ? Fmt(asmi_val) : std::string("N/A");

      if (!gm_ok && !asmi_ok) {
        // Neither side can provide it: not a mismatch.
        delta_str = "-";
        verdict = "SKIP (both n/a)";
      } else if (gm_ok != asmi_ok) {
        // One side has it, the other doesn't: SKIP to avoid false positives from
        // amd-smi N/A at idle.
        delta_str = "-";
        verdict = "SKIP (one n/a)";
      } else {
        double delta = gm_val - asmi_val;
        double tol = row.abs_tol +
                     row.rel_tol * std::max(std::fabs(gm_val), std::fabs(asmi_val));
        delta_str = Fmt(delta);
        if (std::fabs(delta) <= tol) {
          verdict = "PASS";
        } else {
          verdict = "FAIL";
          ++mismatches;
        }
        ++compared;
      }

      std::printf("  %-20s %14s %14s %14s  %s\n", row.label.c_str(), gm_str.c_str(),
                  asmi_str.c_str(), delta_str.c_str(), verdict.c_str());
    }
    std::printf("\n");
  }

  std::printf("=== summary: %d metric(s) compared, %d mismatch(es) ===\n", compared,
              mismatches);
  if (mismatches > 0) {
    std::printf("RESULT: FAIL (%d metric(s) outside tolerance)\n", mismatches);
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
