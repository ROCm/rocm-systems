/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// A crafted kernel launch event, shared by the reader tests (hrr-unit-tests) and
// the handler tests (hrr-handler-tests, through hrr_handler_test_util.h). Kept
// apart from that header because it is POSIX only and the reader tests build
// everywhere.

#include "hrr/hrr_api_args.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hrr_test {

// A hipModuleLaunchKernel event for a kernel named `name` (v10 layout, as capture
// writes it): the header, then the fixed fields and `num_args`/`num_snapshots`
// counts, then `records` verbatim (the argument records, then the snapshot
// records), then the launch-attribute tail unless `attrs_tail` is false. The
// counts are what the event declares; `records` need not carry that many.
inline std::vector<uint8_t> make_launch_event(uint64_t seq, const std::string& name,
                                              uint16_t num_args, uint16_t num_snapshots,
                                              const std::vector<uint8_t>& records,
                                              bool attrs_tail = true) {
  std::vector<uint8_t> p(sizeof(hrr_event_header), 0);
  auto put = [&](const void* d, size_t n) {
    p.insert(p.end(), static_cast<const uint8_t*>(d), static_cast<const uint8_t*>(d) + n);
  };
  const uint64_t stream = 0, co_hash[2] = {0, 0};
  const uint16_t name_len = static_cast<uint16_t>(name.size());
  const uint16_t counts[2] = {num_args, num_snapshots};
  const uint32_t dims[7] = {1, 1, 1, 1, 1, 1, 0}, attrs[2] = {0, 0};
  put(&stream, 8);
  put(&name_len, 2);
  put(name.data(), name.size());
  put(co_hash, 16);
  put(dims, sizeof(dims));
  put(counts, sizeof(counts));
  put(records.data(), records.size());
  if (attrs_tail) put(attrs, sizeof(attrs));
  auto* h = reinterpret_cast<hrr_event_header*>(p.data());
  h->event_type = static_cast<uint16_t>(HRR_API_HIPMODULELAUNCHKERNEL);
  h->sequence_id = seq;
  h->payload_length = static_cast<uint32_t>(p.size());
  return p;
}

}  // namespace hrr_test
