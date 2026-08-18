// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * RAII wrappers for the libnl-3 C API.
 */

#ifndef AMDSMI_UNIFIED_NETLINK_WRAPPER_H_
#define AMDSMI_UNIFIED_NETLINK_WRAPPER_H_

#include <netlink/attr.h>
#include <netlink/genl/ctrl.h>
#include <netlink/genl/genl.h>
#include <netlink/msg.h>
#include <netlink/netlink.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace amd::nic::netlink {

// RAII wrapper for nl_sock.
// Not thread-safe: each thread needs its own socket.
class NLSocket {
 public:
  // Throws std::runtime_error on allocation failure.
  NLSocket();

  ~NLSocket();

  // Disable copy (sockets cannot be copied)
  NLSocket(const NLSocket&) = delete;
  NLSocket& operator=(const NLSocket&) = delete;

  // Enable move
  NLSocket(NLSocket&& other) noexcept;
  NLSocket& operator=(NLSocket&& other) noexcept;

  // protocol is ignored; genl_connect always uses NETLINK_GENERIC.
  int connect(int protocol);

  void set_buffer_size(int rx_size, int tx_size);

  // Non-owning; do not free.
  struct nl_sock* get() const { return sock_; }

  bool is_valid() const { return sock_ != nullptr; }

 private:
  struct nl_sock* sock_;
};

// RAII wrapper for nl_msg.
// Not thread-safe: do not share between threads.
class NLMessage {
 public:
  // Throws std::runtime_error on allocation failure.
  NLMessage();

  // Takes ownership of msg; freed by destructor.
  explicit NLMessage(struct nl_msg* msg);

  ~NLMessage();

  // Disable copy (messages cannot be copied)
  NLMessage(const NLMessage&) = delete;
  NLMessage& operator=(const NLMessage&) = delete;

  // Enable move
  NLMessage(NLMessage&& other) noexcept;
  NLMessage& operator=(NLMessage&& other) noexcept;

  void* put_genl_header(uint32_t port, uint32_t seq, int family, int hdrlen, int flags, uint8_t cmd,
                        uint8_t version);

  // Non-owning; do not free.
  struct nl_msg* get() const { return msg_; }

  bool is_valid() const { return msg_ != nullptr; }

  // Caller assumes ownership and must free.
  struct nl_msg* release();

 private:
  struct nl_msg* msg_;
};

// RAII wrapper for nl_cb.
// Not thread-safe: tied to a specific socket/message flow.
class NLCallback {
 public:
  // Returns NL_OK, NL_SKIP, NL_STOP, or a negative error code.
  using CallbackFn = std::function<int(struct nl_msg*, void*)>;

  // Throws std::runtime_error on allocation failure.
  NLCallback();

  ~NLCallback();

  // Disable copy
  NLCallback(const NLCallback&) = delete;
  NLCallback& operator=(const NLCallback&) = delete;

  // Enable move
  NLCallback(NLCallback&& other) noexcept;
  NLCallback& operator=(NLCallback&& other) noexcept;

  int set(int type, int kind, nl_recvmsg_msg_cb_t func, void* arg);

  // Non-owning; do not free.
  struct nl_cb* get() const { return cb_; }

  bool is_valid() const { return cb_ != nullptr; }

 private:
  struct nl_cb* cb_;
};

// Builds and parses typed netlink attributes.
// Not thread-safe: tied to one message.
class NLAttributes {
 public:
  explicit NLAttributes(NLMessage& msg);

  int put_u32(int type, uint32_t value);
  int put_u16(int type, uint16_t value);
  int put_u8(int type, uint8_t value);
  int put_string(int type, const char* value);
  int put_string(int type, const std::string& value);
  int put_flag(int type);

  struct nlattr* nest_start(int type);

  // nested must be the pointer returned by nest_start().
  void nest_end(struct nlattr* nested);

  static int parse(struct nlmsghdr* nlh, int hdrlen, struct nlattr** tb, int maxtype,
                   struct nla_policy* policy = nullptr);

  static uint32_t get_u32(struct nlattr* attr);
  static uint16_t get_u16(struct nlattr* attr);
  static uint8_t get_u8(struct nlattr* attr);

  // Returned string points into attribute data.
  static const char* get_string(struct nlattr* attr);

  static bool exists(struct nlattr* attr);

 private:
  NLMessage& msg_;
};

}  // namespace amd::nic::netlink

#endif  // AMDSMI_UNIFIED_NETLINK_WRAPPER_H_
