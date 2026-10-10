/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_LIBRARY_SRC_NET_IB_CTRL_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_IB_CTRL_HPP_

#include <cstdint>
#include <vector>

#include "net/ibv.hpp"
#include "net/lane_map.hpp"

namespace rocshmem {
namespace net {

/**
 * @brief Per-lane RC connection info, published to peers via all-gather.
 *
 * Trivially copyable so it can flow straight through addr_exchange /
 * Backend::symm_allgather. On RoCE, @p lid is 0 and addressing uses @p gid.
 */
struct DestInfo {
  uint32_t qpn;       //!< remote QP number
  uint32_t psn;       //!< remote packet sequence number
  uint16_t lid;       //!< remote LID (0 on RoCE)
  uint8_t port;       //!< remote physical port
  uint8_t gid_index;  //!< remote GID index (informational)
  uint8_t gid[16];    //!< remote GID (RoCE addressing)
};

/**
 * @brief The opened HCA/port context a conduit posts through.
 */
struct IbDevice {
  struct ibv_context *ctx{nullptr};
  struct ibv_pd *pd{nullptr};
  uint8_t port{1};
  int gid_index{-1};
  union ibv_gid gid {};
  struct ibv_port_attr port_attr {};
  enum ibv_mtu mtu { IBV_MTU_1024 };
  bool is_roce{false};
};

/**
 * @brief Plain-ibv (non-DevX) RC control plane for a host-side conduit.
 *
 * Opens an HCA/port, chooses a RoCEv2 GID, and creates one CQ per context plus
 * one RC QP per lane (lane = ctx*num_pes + pe), then drives each QP through the
 * RESET -> INIT -> RTR -> RTS state machine. This mirrors the reusable,
 * provider-neutral parts of the GDA host bring-up but posts from the host, so
 * it uses ordinary ibv_create_cq/ibv_create_qp rather than GDA's GPU-visible
 * DevX path.
 */
class IbCtrl {
 public:
  IbCtrl() = default;
  ~IbCtrl();
  IbCtrl(const IbCtrl &) = delete;
  IbCtrl &operator=(const IbCtrl &) = delete;

  /**
   * @brief Open an HCA/port and select a GID.
   * @param ibv      Loaded wrapper (must outlive this object).
   * @param dev_name Substring to match a device name, or nullptr for the first
   *                 device with an ACTIVE port.
   * @param port_hint Physical port to use (default 1).
   */
  bool open(Ibv *ibv, const char *dev_name, uint8_t port_hint = 1);

  /// Create one CQ per context and one RC QP per lane; QPs left in INIT.
  bool create_queues(const LaneMap &lm, int cq_depth, int qp_send_depth);

  /// Populate this PE's DestInfo for every lane (out sized to num_lanes()).
  void fill_local_dest(std::vector<DestInfo> &out) const;

  /**
   * @brief Connect every non-self lane to its peer and move it to RTS.
   * @param my_pe  This PE's rank.
   * @param global All-gathered DestInfo laid out global[peer*num_lanes + lane].
   */
  bool connect(int my_pe, const std::vector<DestInfo> &global);

  /// Connect a single QP to @p remote and drive it to RTS (INIT -> RTR -> RTS).
  bool connect_qp(int lane, const DestInfo &remote);

  struct ibv_qp *qp(int lane) const { return qps_[lane]; }
  struct ibv_cq *cq(int ctx) const { return cqs_[ctx]; }
  int lane_count() const { return lm_.num_lanes(); }
  uint32_t local_psn() const { return psn_; }
  const IbDevice &dev() const { return dev_; }

 private:
  bool select_gid();
  bool to_init(struct ibv_qp *qp);
  bool to_rtr(struct ibv_qp *qp, const DestInfo &remote);
  bool to_rts(struct ibv_qp *qp);
  void destroy();

  Ibv *ibv_{nullptr};
  IbDevice dev_{};
  LaneMap lm_{};
  std::vector<struct ibv_cq *> cqs_;  //!< one per context
  std::vector<struct ibv_qp *> qps_;  //!< one per lane
  uint32_t psn_{0x1234};
};

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_IB_CTRL_HPP_
