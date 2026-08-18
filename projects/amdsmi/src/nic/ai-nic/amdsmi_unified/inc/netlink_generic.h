// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Generic Netlink Client - Layer 2
 *
 * Provides a high-level interface for generic netlink communication,
 * handling family resolution and query execution.
 */

#ifndef AMDSMI_UNIFIED_NETLINK_GENERIC_H_
#define AMDSMI_UNIFIED_NETLINK_GENERIC_H_

#include <functional>
#include <optional>
#include <string>

#include "netlink_wrapper.h"

namespace amd::nic::netlink {

/**
 * High-level generic netlink client (family resolution + query execution).
 * Not thread-safe: each thread should create its own instance.
 */
class GenericNetlinkClient {
 public:
  // Returns NL_OK, NL_SKIP, NL_STOP, or a negative error code.
  using MessageHandler = std::function<int(struct nl_msg*, void*)>;

  GenericNetlinkClient();
  ~GenericNetlinkClient() = default;

  GenericNetlinkClient(const GenericNetlinkClient&) = delete;
  GenericNetlinkClient& operator=(const GenericNetlinkClient&) = delete;

  GenericNetlinkClient(GenericNetlinkClient&&) = default;
  GenericNetlinkClient& operator=(GenericNetlinkClient&&) = default;

  int connect();

  std::optional<int> resolve_family_id(const std::string& family_name);

  /**
   * build_fn adds attributes before send; handler receives each response.
   * `flags` are the netlink message flags; pass NLM_F_REQUEST | NLM_F_DUMP to
   * run a dump (handler is then invoked once per returned object).
   */
  int query(int family_id, uint8_t cmd, uint8_t version, std::function<int(NLMessage&)> build_fn,
            MessageHandler handler, void* arg, uint16_t flags = NLM_F_REQUEST);

  NLSocket& socket() { return socket_; }

  bool is_connected() const { return connected_; }

 private:
  NLSocket socket_;
  bool connected_;
};

}  // namespace amd::nic::netlink

#endif  // AMDSMI_UNIFIED_NETLINK_GENERIC_H_
