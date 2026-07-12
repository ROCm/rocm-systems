// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// gpumetrics CLI: a thin front-end over libgpumetrics.
//   gpumetrics discover                 list sockets / GPUs / partitions
//   gpumetrics list-metrics [--scope]   list available metric keys
//   gpumetrics read -e <sel> -m <keys>  read once
//   gpumetrics dmon  [-e ..] [-m ..] [-i ms] [-c count]   watch loop
//
// Output: --format table|csv|json. Entity selectors: gpu:N, g<N>.<P>, socket:N,
// bdf:DDDD:BB:DD.F, uuid:...

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gpumetrics/gpumetrics.h"

using namespace gpumetrics;

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct Args {
  std::string cmd;
  std::vector<std::string> entities;   // -e (repeatable / comma list)
  std::vector<std::string> metrics;    // -m (repeatable / comma list)
  std::string format = "table";        // --format
  std::string scope;                   // --scope gpu|partition|socket
  uint32_t interval_ms = 1000;         // -i
  int count = 0;                       // -c (0 = infinite)
  std::vector<std::string> plugins;    // --plugin
  std::vector<std::string> plugin_paths;  // --plugin-path
  bool help = false;
};

void split_into(const std::string& s, char sep, std::vector<std::string>& out) {
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, sep))
    if (!tok.empty()) out.push_back(tok);
}

const char* kUsage =
    "gpumetrics - read AMD GPU metrics per GPU / partition / socket\n\n"
    "Usage:\n"
    "  gpumetrics discover [--format table|csv|json]\n"
    "  gpumetrics list-metrics [--scope gpu|partition|socket] [--format ...]\n"
    "  gpumetrics read -e <selector>... -m <key>... [--format ...]\n"
    "  gpumetrics dmon [-e <selector>...] [-m <key>...] [-i <ms>] [-c <count>]\n\n"
    "Selectors: gpu:N | g<N>.<P> | socket:N | bdf:DDDD:BB:DD.F | uuid:HEX | N\n"
    "Options:\n"
    "  -e, --entity      target entity (repeatable, or comma-separated)\n"
    "  -m, --metric      metric key (repeatable, or comma-separated)\n"
    "  -i, --interval    dmon interval in ms (default 1000)\n"
    "  -c, --count       dmon iterations (default 0 = until Ctrl-C)\n"
    "      --format      table|csv|json (default table)\n"
    "      --scope       filter list-metrics by scope\n"
    "      --plugin      load only these plugins (repeatable)\n"
    "      --plugin-path extra plugin search dir (repeatable)\n"
    "  -h, --help\n";

bool parse_args(int argc, char** argv, Args* a) {
  if (argc < 2) {
    a->help = true;
    return true;
  }
  a->cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        return "";
      }
      return argv[++i];
    };
    if (s == "-e" || s == "--entity")
      split_into(next("-e"), ',', a->entities);
    else if (s == "-m" || s == "--metric")
      split_into(next("-m"), ',', a->metrics);
    else if (s == "-i" || s == "--interval")
      a->interval_ms = std::strtoul(next("-i").c_str(), nullptr, 10);
    else if (s == "-c" || s == "--count")
      a->count = std::atoi(next("-c").c_str());
    else if (s == "--format")
      a->format = next("--format");
    else if (s == "--scope")
      a->scope = next("--scope");
    else if (s == "--plugin")
      a->plugins.push_back(next("--plugin"));
    else if (s == "--plugin-path")
      a->plugin_paths.push_back(next("--plugin-path"));
    else if (s == "-h" || s == "--help")
      a->help = true;
    else {
      std::cerr << "unknown argument: " << s << "\n";
      return false;
    }
  }
  return true;
}

uint32_t scope_flag(const std::string& s) {
  if (s == "gpu") return GPUM_SCOPE_GPU;
  if (s == "partition") return GPUM_SCOPE_PARTITION;
  if (s == "socket") return GPUM_SCOPE_SOCKET;
  return 0;
}

std::string bdf_str(uint64_t bdf) {
  if (!bdf) return "";
  unsigned dom = static_cast<unsigned>(bdf >> 16);
  unsigned bus = static_cast<unsigned>((bdf >> 8) & 0xff);
  unsigned dev = static_cast<unsigned>((bdf >> 3) & 0x1f);
  unsigned fn = static_cast<unsigned>(bdf & 0x7);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%x", dom, bus, dev, fn);
  return buf;
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o.push_back('\\');
    o.push_back(c);
  }
  return o;
}

// ---- discover ------------------------------------------------------------
int cmd_discover(Collector& c, const Args& a) {
  if (a.format == "json") {
    std::cout << "{\n  \"plugins\": [";
    auto pl = c.loadedPlugins();
    for (size_t i = 0; i < pl.size(); ++i)
      std::cout << (i ? ", " : "") << "\"" << json_escape(pl[i]) << "\"";
    std::cout << "],\n  \"sockets\": [\n";
    const auto& socks = c.sockets();
    for (size_t si = 0; si < socks.size(); ++si) {
      std::cout << "    {\"index\": " << socks[si].index << ", \"gpus\": [";
      for (size_t g = 0; g < socks[si].gpus.size(); ++g)
        std::cout << (g ? ", " : "") << socks[si].gpus[g];
      std::cout << "]}" << (si + 1 < socks.size() ? "," : "") << "\n";
    }
    std::cout << "  ],\n  \"gpus\": [\n";
    const auto& devs = c.devices();
    for (size_t i = 0; i < devs.size(); ++i) {
      const auto& d = devs[i];
      std::cout << "    {\"gpu\": " << d.id.gpu << ", \"socket\": " << d.id.socket
                << ", \"name\": \"" << json_escape(d.name) << "\", \"bdf\": \""
                << bdf_str(d.identity.bdf) << "\", \"kfd_node_id\": " << d.identity.kfd_node_id
                << ", \"partitions\": [";
      for (size_t p = 0; p < d.partitions.size(); ++p)
        std::cout << (p ? ", " : "") << d.partitions[p];
      std::cout << "], \"providers\": [";
      for (size_t p = 0; p < d.providers.size(); ++p)
        std::cout << (p ? ", " : "") << "\"" << json_escape(d.providers[p]) << "\"";
      std::cout << "]}" << (i + 1 < devs.size() ? "," : "") << "\n";
    }
    std::cout << "  ]\n}\n";
    return 0;
  }

  auto plugins = c.loadedPlugins();
  std::cout << "Plugins loaded:";
  for (auto& p : plugins) std::cout << " " << p;
  if (plugins.empty()) std::cout << " (none)";
  std::cout << "\n\n";

  std::printf("%-6s %-8s %-24s %-16s %-8s %-12s %s\n", "GPU", "SOCKET", "NAME", "BDF", "KFD",
              "PARTITIONS", "PROVIDERS");
  for (const auto& d : c.devices()) {
    std::string parts;
    for (size_t i = 0; i < d.partitions.size(); ++i)
      parts += (i ? "," : "") + std::to_string(d.partitions[i]);
    if (parts.empty()) parts = "-";
    std::string prov;
    for (size_t i = 0; i < d.providers.size(); ++i) prov += (i ? "," : "") + d.providers[i];
    std::printf("%-6u %-8u %-24s %-16s %-8u %-12s %s\n", d.id.gpu, d.id.socket, d.name.c_str(),
                bdf_str(d.identity.bdf).c_str(), d.identity.kfd_node_id, parts.c_str(),
                prov.c_str());
  }
  return 0;
}

// ---- list-metrics --------------------------------------------------------
int cmd_list_metrics(Collector& c, const Args& a) {
  uint32_t want = scope_flag(a.scope);
  std::vector<const MetricDescriptor*> ms;
  for (const auto& m : c.metrics())
    if (!want || (m.scope & want)) ms.push_back(&m);

  if (a.format == "json") {
    std::cout << "[\n";
    for (size_t i = 0; i < ms.size(); ++i) {
      const auto* m = ms[i];
      std::cout << "  {\"key\": \"" << json_escape(m->key) << "\", \"unit\": \""
                << json_escape(m->unit) << "\", \"provider\": \"" << json_escape(m->provider)
                << "\", \"description\": \"" << json_escape(m->description) << "\"}"
                << (i + 1 < ms.size() ? "," : "") << "\n";
    }
    std::cout << "]\n";
    return 0;
  }
  if (a.format == "csv") {
    std::cout << "key,unit,provider,description\n";
    for (const auto* m : ms)
      std::cout << m->key << "," << m->unit << "," << m->provider << ",\"" << m->description
                << "\"\n";
    return 0;
  }
  std::printf("%-28s %-8s %-12s %s\n", "KEY", "UNIT", "PROVIDER", "DESCRIPTION");
  for (const auto* m : ms)
    std::printf("%-28s %-8s %-12s %s\n", m->key.c_str(), m->unit.c_str(), m->provider.c_str(),
                m->description.c_str());
  return 0;
}

// Resolve entity selectors; default to all GPUs when none given.
std::vector<Entity> resolve_entities(Collector& c, const Args& a) {
  std::vector<Entity> out;
  if (a.entities.empty()) return c.entities(GPUM_ENTITY_GPU);
  for (const auto& sel : a.entities) {
    if (sel == "all") {
      for (auto& e : c.entities(GPUM_ENTITY_GPU)) out.push_back(e);
      continue;
    }
    auto e = c.resolve(sel);
    if (!e) {
      std::cerr << "warning: could not resolve entity '" << sel << "'\n";
      continue;
    }
    out.push_back(*e);
  }
  return out;
}

// Default metric set when none specified: everything the loaded plugins offer.
std::vector<std::string> resolve_metrics(Collector& c, const Args& a) {
  if (!a.metrics.empty()) return a.metrics;
  std::vector<std::string> keys;
  for (const auto& m : c.metrics()) keys.push_back(m.key);
  return keys;
}

void print_reading_header(Collector& c, const std::vector<std::string>& keys, const Args& a) {
  if (a.format == "csv") {
    std::cout << "timestamp_ns,entity";
    for (auto& k : keys) std::cout << "," << k;
    std::cout << "\n";
  }
}

int do_read_once(Collector& c, const std::vector<Entity>& ents,
                 const std::vector<std::string>& keys, const Args& a) {
  for (const auto& e : ents) {
    auto samples = c.read(e.id, keys);
    if (a.format == "json") {
      std::cout << "{\"entity\": \"" << e.label << "\", \"metrics\": {";
      for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << (i ? ", " : "") << "\"" << keys[i] << "\": ";
        if (samples[i].ok())
          std::cout << (samples[i].type == GPUM_TYPE_STRING
                            ? "\"" + json_escape(samples[i].to_string()) + "\""
                            : samples[i].to_string());
        else
          std::cout << "null";
      }
      std::cout << "}}\n";
    } else if (a.format == "csv") {
      uint64_t ts = samples.empty() ? 0 : samples[0].timestamp_ns;
      std::cout << ts << "," << e.label;
      for (auto& s : samples) std::cout << "," << (s.ok() ? s.to_string() : "");
      std::cout << "\n";
    } else {
      std::cout << "[" << e.label << "]\n";
      for (size_t i = 0; i < samples.size(); ++i) {
        auto d = c.describe(keys[i]);
        std::string unit = d ? d->unit : "";
        std::printf("  %-28s %s%s%s\n", keys[i].c_str(),
                    samples[i].ok() ? samples[i].to_string().c_str()
                                    : gpum_status_string(samples[i].status),
                    (samples[i].ok() && !unit.empty()) ? " " : "",
                    (samples[i].ok() && !unit.empty()) ? unit.c_str() : "");
      }
    }
  }
  return 0;
}

int cmd_read(Collector& c, const Args& a) {
  auto ents = resolve_entities(c, a);
  auto keys = resolve_metrics(c, a);
  if (ents.empty()) {
    std::cerr << "no entities resolved\n";
    return 2;
  }
  print_reading_header(c, keys, a);
  return do_read_once(c, ents, keys, a);
}

int cmd_dmon(Collector& c, const Args& a) {
  auto ents = resolve_entities(c, a);
  auto keys = resolve_metrics(c, a);
  if (ents.empty()) {
    std::cerr << "no entities resolved\n";
    return 2;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  print_reading_header(c, keys, a);
  int iter = 0;
  while (!g_stop && (a.count == 0 || iter < a.count)) {
    do_read_once(c, ents, keys, a);
    if (a.format == "table") std::cout << "\n";
    std::cout.flush();
    ++iter;
    if (a.count != 0 && iter >= a.count) break;
    // Sleep in small slices so Ctrl-C is responsive.
    for (uint32_t slept = 0; slept < a.interval_ms && !g_stop; slept += 50)
      std::this_thread::sleep_for(std::chrono::milliseconds(std::min<uint32_t>(50, a.interval_ms)));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, &a)) return 2;
  if (a.help || a.cmd == "help") {
    std::cout << kUsage;
    return 0;
  }

  CollectorOptions opts;
  opts.plugins = a.plugins;
  opts.plugin_paths = a.plugin_paths;
  gpum_status st;
  auto c = Collector::Create(opts, &st);
  if (!c) {
    std::cerr << "failed to initialize collector: " << gpum_status_string(st) << "\n";
    return 1;
  }

  if (a.cmd == "discover") return cmd_discover(*c, a);
  if (a.cmd == "list-metrics" || a.cmd == "list") return cmd_list_metrics(*c, a);
  if (a.cmd == "read") return cmd_read(*c, a);
  if (a.cmd == "dmon" || a.cmd == "watch") return cmd_dmon(*c, a);

  std::cerr << "unknown command: " << a.cmd << "\n\n" << kUsage;
  return 2;
}
