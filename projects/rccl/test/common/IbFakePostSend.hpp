/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_IB_FAKE_POST_SEND_HPP
#define RCCL_TEST_IB_FAKE_POST_SEND_HPP

// A recording ibv_post_send for host tests of the IB plugins' send paths. It can reject a chain whole or only its
// signaled tail on one QP; no RDMA device is needed. Include it after the plugin headers, which bring the verbs types.

#include <endian.h>
#include <cerrno>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

namespace RcclIbFake
{

enum class PostFail { None, Head, Tail };

// One posted WR, copied at post time because the builders reuse their WR arrays.
struct WrRec {
  uint64_t wrId;
  int opcode;
  unsigned flags;
  uint64_t raddr;
  uint32_t rkey;
  int nsge;
  const ibv_sge* sgList;
  uint64_t laddr;
  uint32_t len;
  uint32_t lkey;
  uint32_t imm;
};

struct PostRec {
  uint32_t qpNum;
  std::vector<WrRec> wrs;
};

struct FakePost {
  uint32_t failQpNum = 0;
  PostFail mode = PostFail::None;
  int calls = 0;
  int failedCallWrs = 0;
  int failedCallSignaled = 0;
  std::vector<PostRec> posts;
};
inline FakePost g_post;

inline int FakePostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) {
  g_post.calls++;
  PostRec rec{qp->qp_num, {}};
  for (struct ibv_send_wr* w = wr; w != nullptr; w = w->next) {
    WrRec r{w->wr_id, static_cast<int>(w->opcode), static_cast<unsigned>(w->send_flags),
            w->wr.rdma.remote_addr, w->wr.rdma.rkey, w->num_sge, w->sg_list, 0, 0, 0,
            static_cast<uint32_t>(w->imm_data)};
    if (w->num_sge > 0 && w->sg_list != nullptr) {
      r.laddr = w->sg_list[0].addr;
      r.len = w->sg_list[0].length;
      r.lkey = w->sg_list[0].lkey;
    }
    rec.wrs.push_back(r);
  }
  g_post.posts.push_back(rec);
  if (g_post.mode == PostFail::None || qp->qp_num != g_post.failQpNum) return 0;
  struct ibv_send_wr* tail = wr;
  for (struct ibv_send_wr* w = wr; w != nullptr; w = w->next) {
    g_post.failedCallWrs++;
    if (w->send_flags & IBV_SEND_SIGNALED) g_post.failedCallSignaled++;
    tail = w;
  }
  // Tail mode: the NIC took every WR but the signaled last one.
  *bad_wr = (g_post.mode == PostFail::Tail) ? tail : wr;
  return ENOMEM;
}

inline const PostRec* PostOn(uint32_t qpNum) {
  for (const PostRec& p : g_post.posts) {
    if (p.qpNum == qpNum) return &p;
  }
  return nullptr;
}

// Every WR posted on that QP, in post order.
inline std::vector<WrRec> WrsOn(uint32_t qpNum) {
  std::vector<WrRec> all;
  for (const PostRec& p : g_post.posts) {
    if (p.qpNum == qpNum) all.insert(all.end(), p.wrs.begin(), p.wrs.end());
  }
  return all;
}

// One data slice: a single sge with this local range and key, written to this remote address and key.
inline void ExpectWrite(const WrRec& w, uint64_t laddr, uint32_t len, uint32_t lkey, uint64_t raddr, uint32_t rkey) {
  EXPECT_EQ(w.nsge, 1);
  EXPECT_EQ(w.laddr, laddr);
  EXPECT_EQ(w.len, len);
  EXPECT_EQ(w.lkey, lkey);
  EXPECT_EQ(w.raddr, raddr);
  EXPECT_EQ(w.rkey, rkey);
}

inline void ExpectPlain(const WrRec& w) {
  EXPECT_EQ(w.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(w.flags & IBV_SEND_SIGNALED, 0u);
}

inline void ExpectImmTail(const WrRec& w, uint32_t imm) {
  EXPECT_EQ(w.opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_NE(w.flags & IBV_SEND_SIGNALED, 0u);
  EXPECT_EQ(w.imm, htobe32(imm));
}

// The last write of a post that leaves data behind: signaled, no imm.
inline void ExpectSliceEnd(const WrRec& w) {
  EXPECT_EQ(w.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_NE(w.flags & IBV_SEND_SIGNALED, 0u);
}

} // namespace RcclIbFake

#endif
