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

#ifndef ROCSHMEM_LIBRARY_SRC_NET_IBV_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_IBV_HPP_

// Self-contained libibverbs access for the reverse-offload verbs conduit.
//
// Design (see ro_multi_conduit_design.md):
//   - ABI from the real <infiniband/verbs.h> (exact structs/enums), so nothing
//     is hand-vendored and post_send/poll_cq's inline op-table layout is correct.
//   - Functions are resolved at runtime via dlopen("libibverbs.so.1") + dlsym,
//     so there is NO link-time dependency on libibverbs (matching the GDA/MPI
//     dlopen philosophy). The library is only needed at build for the header and
//     at runtime when ROCSHMEM_RO_TRANSPORT=verbs is actually selected.
//   - This is intentionally independent of src/gda/ibv_wrapper.* (which is
//     USE_GDA-only, GPU-posting, and lacks post_send/poll_cq). GDA is untouched.

#include <infiniband/verbs.h>

#include <cstddef>

namespace rocshmem {
namespace net {

/**
 * @brief Thin runtime-loaded wrapper over the libibverbs control plane.
 *
 * One instance owns the dlopen handle and a table of dlsym'd control-plane
 * entry points. Data-plane fast paths (post_send/poll_cq) are NOT dlsym'd --
 * in the libibverbs ABI they are inline dispatchers through the object's
 * context ops, so they are called directly here with no symbol lookup.
 *
 * Not copyable. Thread-safety follows libibverbs: distinct QPs/CQs may be
 * driven concurrently; a single QP/CQ must be serialized by the caller (the
 * conduit uses one QP+CQ per lane on its progress thread).
 */
class Ibv {
 public:
  Ibv() = default;
  ~Ibv();

  Ibv(const Ibv &) = delete;
  Ibv &operator=(const Ibv &) = delete;

  /// dlopen libibverbs and resolve the control-plane symbols.
  /// @return true on success; false if the library or a symbol is missing.
  bool load();

  /// True once load() has succeeded.
  bool available() const { return handle_ != nullptr; }

  // --- device / context ---------------------------------------------------
  struct ibv_device **get_device_list(int *num_devices);
  void free_device_list(struct ibv_device **list);
  const char *get_device_name(struct ibv_device *device);
  struct ibv_context *open_device(struct ibv_device *device);
  int close_device(struct ibv_context *context);

  int query_device(struct ibv_context *context,
                   struct ibv_device_attr *device_attr);
  int query_port(struct ibv_context *context, uint8_t port_num,
                 struct ibv_port_attr *port_attr);
  int query_gid(struct ibv_context *context, uint8_t port_num, int index,
                union ibv_gid *gid);
  /// Query the GID table (needed to distinguish RoCE v1/v2 via gid_type).
  /// @return number of entries filled, or -1 if unsupported.
  int query_gid_table(struct ibv_context *context, struct ibv_gid_entry *entries,
                      size_t max_entries);

  // --- protection domain / memory ----------------------------------------
  struct ibv_pd *alloc_pd(struct ibv_context *context);
  int dealloc_pd(struct ibv_pd *pd);

  struct ibv_mr *reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                        int access);
  struct ibv_mr *reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
                               size_t length, uint64_t iova, int fd,
                               int access);
  int dereg_mr(struct ibv_mr *mr);

  // --- completion / queue pairs ------------------------------------------
  struct ibv_cq *create_cq(struct ibv_context *context, int cqe,
                           void *cq_context, struct ibv_comp_channel *channel,
                           int comp_vector);
  int destroy_cq(struct ibv_cq *cq);

  struct ibv_qp *create_qp(struct ibv_pd *pd,
                           struct ibv_qp_init_attr *qp_init_attr);
  int modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
  int destroy_qp(struct ibv_qp *qp);

  // --- data plane (inline op-table dispatch; no dlsym) --------------------
  /// Post a send work-request chain on @p qp.
  int post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                struct ibv_send_wr **bad_wr) {
    return qp->context->ops.post_send(qp, wr, bad_wr);
  }
  /// Reap up to @p num_entries completions from @p cq.
  int poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    return cq->context->ops.poll_cq(cq, num_entries, wc);
  }

 private:
  struct Table {
    struct ibv_device **(*get_device_list)(int *){nullptr};
    void (*free_device_list)(struct ibv_device **){nullptr};
    const char *(*get_device_name)(struct ibv_device *){nullptr};
    struct ibv_context *(*open_device)(struct ibv_device *){nullptr};
    int (*close_device)(struct ibv_context *){nullptr};
    int (*query_device)(struct ibv_context *, struct ibv_device_attr *){nullptr};
    int (*query_port)(struct ibv_context *, uint8_t,
                      struct ibv_port_attr *){nullptr};
    int (*query_gid)(struct ibv_context *, uint8_t, int,
                     union ibv_gid *){nullptr};
    ssize_t (*query_gid_table)(struct ibv_context *, struct ibv_gid_entry *,
                               size_t, uint32_t, size_t){nullptr};
    struct ibv_pd *(*alloc_pd)(struct ibv_context *){nullptr};
    int (*dealloc_pd)(struct ibv_pd *){nullptr};
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int){nullptr};
    struct ibv_mr *(*reg_dmabuf_mr)(struct ibv_pd *, uint64_t, size_t, uint64_t,
                                    int, int){nullptr};
    int (*dereg_mr)(struct ibv_mr *){nullptr};
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *,
                                struct ibv_comp_channel *, int){nullptr};
    int (*destroy_cq)(struct ibv_cq *){nullptr};
    struct ibv_qp *(*create_qp)(struct ibv_pd *,
                                struct ibv_qp_init_attr *){nullptr};
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int){nullptr};
    int (*destroy_qp)(struct ibv_qp *){nullptr};
  } t_;

  void *handle_{nullptr};
};

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_IBV_HPP_
