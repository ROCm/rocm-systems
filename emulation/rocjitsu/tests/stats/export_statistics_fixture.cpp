// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/stats/statistics.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <utility>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: export_statistics_fixture OUTPUT_DIRECTORY\n";
    return 2;
  }
  rocjitsu::StatisticsConfig config{.enabled = true,
                                    .include = {"cache.*", "precision.*", "memory.*", "host.*"},
                                    .scopes = {"dispatch", "component"},
                                    .directory = argv[1],
                                    .max_dispatch_scopes = 1};
  util::StringDiagnostic diagnostic;
  auto created = rocjitsu::StatisticsSession::create(config, "fixture \"node\"\n", "functional",
                                                     diagnostic.emitter());
  if (created.failed()) {
    std::cerr << diagnostic.message() << '\n';
    return 1;
  }
  auto session = std::move(created).value();
  session->register_gpu(0);
  session->register_gpu(1);
  auto first = session->scope("gpu0.l2 \"cache\"\n", 0);
  auto second = session->scope("gpu1.l2", 1);
  if (!first || !second)
    return 1;
  first->set_attribute("target", "gfx1250");
  first->set_attribute("note", "comma, quote\" and newline\n");
  auto requests = first->counter("cache.requests", "requests", "Observed read requests");
  auto peer_requests = second->counter("cache.requests", "requests", "Observed read requests");
  auto above_double = first->counter("precision.above_double", "items", "Exact integer witness");
  auto maximum = first->counter("precision.max_u64", "items", "Largest exact counter value");
  auto overflow = first->counter("precision.overflow", "items", "Saturated counter witness");
  auto pending = first->gauge("memory.pending", "instructions", "Signed current-value witness");
  constexpr std::array bounds{4., 16.};
  auto sizes = first->histogram("memory.request_bytes", "bytes", "Observed request sizes", bounds);
  auto duration = first->summary("host.dispatch.duration", "seconds", "Recorded host intervals",
                                 "host_wall_time");
  auto empty =
      first->summary("host.empty_duration", "seconds", "No recorded intervals", "host_wall_time");
  if (!requests || !peer_requests || !above_double || !maximum || !overflow || !pending || !sizes ||
      !duration || !empty)
    return 1;

  requests->add(6);
  {
    auto reset = session->observe_reset(0, rocjitsu::StatisticsReset::Kind::FunctionLevel);
    reset.complete();
  }
  requests->add(2);
  peer_requests->add(13);
  above_double->add((uint64_t{1} << 53) + 1);
  maximum->add(std::numeric_limits<uint64_t>::max());
  overflow->add(std::numeric_limits<uint64_t>::max());
  overflow->increment();
  pending->set(5);
  pending->add(-8);
  for (double value : {1., 4., 5., 16., 17.})
    sizes->observe(value);
  for (double value : {0.125, 0.25, 0.375})
    duration->observe(value);
  { auto failed = session->observe_reset(1, rocjitsu::StatisticsReset::Kind::Bus); }
  {
    auto reset = session->observe_reset(1, rocjitsu::StatisticsReset::Kind::LostConnection);
    reset.complete();
  }
  if (!session->scope("dispatch0", 0, "dispatch") || session->scope("dispatch1", 0, "dispatch"))
    return 1;
  session->mark_run_complete();
  session->finish();
  std::error_code error;
  const auto bytes = std::filesystem::file_size(session->output_path(), error);
  return error || bytes == 0 ? 1 : 0;
}
