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

#include "net/ibv.hpp"

#include <dlfcn.h>

#include <cstdio>

namespace rocshmem {
namespace net {

namespace {
// Resolve a required symbol into a table slot; set ok=false and warn if missing.
template <typename Fn>
void resolve(void *handle, const char *name, Fn &slot, bool &ok) {
  slot = reinterpret_cast<Fn>(dlsym(handle, name));
  if (!slot) {
    fprintf(stderr, "[rocSHMEM] verbs conduit: missing symbol '%s'\n", name);
    ok = false;
  }
}
}  // namespace

bool Ibv::load() {
  if (handle_) {
    return true;
  }
  // Prefer the versioned soname; fall back to the dev symlink.
  handle_ = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!handle_) {
    handle_ = dlopen("libibverbs.so", RTLD_NOW | RTLD_GLOBAL);
  }
  if (!handle_) {
    fprintf(stderr,
            "[rocSHMEM] verbs conduit: cannot dlopen libibverbs (%s)\n",
            dlerror());
    return false;
  }

  bool ok = true;
  resolve(handle_, "ibv_get_device_list", t_.get_device_list, ok);
  resolve(handle_, "ibv_free_device_list", t_.free_device_list, ok);
  resolve(handle_, "ibv_get_device_name", t_.get_device_name, ok);
  resolve(handle_, "ibv_open_device", t_.open_device, ok);
  resolve(handle_, "ibv_close_device", t_.close_device, ok);
  resolve(handle_, "ibv_query_device", t_.query_device, ok);
  resolve(handle_, "ibv_query_port", t_.query_port, ok);
  resolve(handle_, "ibv_query_gid", t_.query_gid, ok);
  resolve(handle_, "ibv_alloc_pd", t_.alloc_pd, ok);
  resolve(handle_, "ibv_dealloc_pd", t_.dealloc_pd, ok);
  resolve(handle_, "ibv_reg_mr", t_.reg_mr, ok);
  resolve(handle_, "ibv_dereg_mr", t_.dereg_mr, ok);
  resolve(handle_, "ibv_create_cq", t_.create_cq, ok);
  resolve(handle_, "ibv_destroy_cq", t_.destroy_cq, ok);
  resolve(handle_, "ibv_create_qp", t_.create_qp, ok);
  resolve(handle_, "ibv_modify_qp", t_.modify_qp, ok);
  resolve(handle_, "ibv_destroy_qp", t_.destroy_qp, ok);

  // Optional: dma-buf registration for GPU pointers (rdma-core >= 1.12).
  // Absence is not fatal; the conduit falls back to ibv_reg_mr.
  t_.reg_dmabuf_mr = reinterpret_cast<decltype(t_.reg_dmabuf_mr)>(
      dlsym(handle_, "ibv_reg_dmabuf_mr"));

  // Optional: GID-table query (rdma-core >= 1.11). The public inline
  // ibv_query_gid_table() forwards to the exported _ibv_query_gid_table.
  // Absence means we cannot detect RoCE v2 and fall back to a default GID.
  t_.query_gid_table = reinterpret_cast<decltype(t_.query_gid_table)>(
      dlsym(handle_, "_ibv_query_gid_table"));

  if (!ok) {
    dlclose(handle_);
    handle_ = nullptr;
    return false;
  }
  return true;
}

Ibv::~Ibv() {
  if (handle_) {
    dlclose(handle_);
    handle_ = nullptr;
  }
}

struct ibv_device **Ibv::get_device_list(int *num_devices) {
  return t_.get_device_list(num_devices);
}
void Ibv::free_device_list(struct ibv_device **list) {
  t_.free_device_list(list);
}
const char *Ibv::get_device_name(struct ibv_device *device) {
  return t_.get_device_name(device);
}
struct ibv_context *Ibv::open_device(struct ibv_device *device) {
  return t_.open_device(device);
}
int Ibv::close_device(struct ibv_context *context) {
  return t_.close_device(context);
}
int Ibv::query_device(struct ibv_context *context,
                      struct ibv_device_attr *device_attr) {
  return t_.query_device(context, device_attr);
}
int Ibv::query_port(struct ibv_context *context, uint8_t port_num,
                    struct ibv_port_attr *port_attr) {
  return t_.query_port(context, port_num, port_attr);
}
int Ibv::query_gid(struct ibv_context *context, uint8_t port_num, int index,
                   union ibv_gid *gid) {
  return t_.query_gid(context, port_num, index, gid);
}
int Ibv::query_gid_table(struct ibv_context *context,
                         struct ibv_gid_entry *entries, size_t max_entries) {
  if (!t_.query_gid_table) {
    return -1;
  }
  return static_cast<int>(t_.query_gid_table(context, entries, max_entries,
                                             0 /*flags*/,
                                             sizeof(struct ibv_gid_entry)));
}
struct ibv_pd *Ibv::alloc_pd(struct ibv_context *context) {
  return t_.alloc_pd(context);
}
int Ibv::dealloc_pd(struct ibv_pd *pd) { return t_.dealloc_pd(pd); }
struct ibv_mr *Ibv::reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                           int access) {
  return t_.reg_mr(pd, addr, length, access);
}
struct ibv_mr *Ibv::reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
                                  size_t length, uint64_t iova, int fd,
                                  int access) {
  if (!t_.reg_dmabuf_mr) {
    return nullptr;
  }
  return t_.reg_dmabuf_mr(pd, offset, length, iova, fd, access);
}
int Ibv::dereg_mr(struct ibv_mr *mr) { return t_.dereg_mr(mr); }
struct ibv_cq *Ibv::create_cq(struct ibv_context *context, int cqe,
                              void *cq_context,
                              struct ibv_comp_channel *channel,
                              int comp_vector) {
  return t_.create_cq(context, cqe, cq_context, channel, comp_vector);
}
int Ibv::destroy_cq(struct ibv_cq *cq) { return t_.destroy_cq(cq); }
struct ibv_qp *Ibv::create_qp(struct ibv_pd *pd,
                              struct ibv_qp_init_attr *qp_init_attr) {
  return t_.create_qp(pd, qp_init_attr);
}
int Ibv::modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) {
  return t_.modify_qp(qp, attr, attr_mask);
}
int Ibv::destroy_qp(struct ibv_qp *qp) { return t_.destroy_qp(qp); }

}  // namespace net
}  // namespace rocshmem
