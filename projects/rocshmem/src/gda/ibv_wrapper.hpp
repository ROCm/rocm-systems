/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_GDA_IBV_WRAPPER_HPP_
#define LIBRARY_SRC_GDA_IBV_WRAPPER_HPP_

#include <infiniband/verbs.h>

namespace rocshmem {

class IBVWrapper;

extern IBVWrapper ibv;

class IBVWrapper {
  public:
    explicit IBVWrapper();
    virtual ~IBVWrapper();

    bool is_initialized{false};

    struct ibv_device** get_device_list(int *num_devices);
    void free_device_list(struct ibv_device **list);

    struct ibv_context* open_device(struct ibv_device *device);
    int close_device(struct ibv_context *context);

    const char* get_device_name(struct ibv_device *device);
    int query_device(struct ibv_context *context, struct ibv_device_attr *device_attr);
    int query_port(struct ibv_context *context, uint8_t port_num,
                   struct ibv_port_attr *port_attr);
    ssize_t query_gid_table(struct ibv_context *context, struct ibv_gid_entry *entries,
                            size_t max_entries, uint32_t flags);
    int query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);

    struct ibv_pd* alloc_pd(struct ibv_context *context);
    struct ibv_pd* alloc_parent_domain(struct ibv_context *context,
                                       struct ibv_parent_domain_init_attr *attr);
    int dealloc_pd(struct ibv_pd *pd);

    struct ibv_mr* reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access);
    int dereg_mr(struct ibv_mr *mr);

    struct ibv_cq_ex* create_cq_ex(struct ibv_context *context,
                                   struct ibv_cq_init_attr_ex *cq_attr);
    struct ibv_cq* cq_ex_to_cq(struct ibv_cq_ex *cq);
    int destroy_cq(struct ibv_cq *cq);

    struct ibv_qp* create_qp_ex(struct ibv_context *context,
                                struct ibv_qp_init_attr_ex *qp_init_attr);
    int modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
    int destroy_qp(struct ibv_qp *qp);

  private:
    struct ibv_funcs_t {
      struct ibv_device** (*get_device_list)(int *num_devices);
      void (*free_device_list)(struct ibv_device **list);

      struct ibv_context* (*open_device)(struct ibv_device *device);
      int (*close_device)(struct ibv_context *context);

      const char* (*get_device_name)(struct ibv_device *device);
      int (*query_device)(struct ibv_context *context, struct ibv_device_attr *device_attr);
      int (*query_port)(struct ibv_context *context, uint8_t port_num,
                        struct ibv_port_attr *port_attr);
      ssize_t (*query_gid_table)(struct ibv_context *context,
                                 struct ibv_gid_entry *entries, size_t max_entries,
                                 uint32_t flags, size_t entry_size);

      int (*query_gid)(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);

      struct ibv_pd* (*alloc_pd)(struct ibv_context *context);
      struct ibv_pd* (*alloc_parent_domain)(struct ibv_context *context,
                                             struct ibv_parent_domain_init_attr *attr);
      int (*dealloc_pd)(struct ibv_pd *pd);

      struct ibv_mr* (*reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
      struct ibv_mr* (*reg_mr_iova2)(struct ibv_pd *pd, void *addr, size_t length,
                                     uint64_t iova, unsigned int access);
      int (*dereg_mr)(struct ibv_mr *mr);

      struct ibv_cq_ex* (*create_cq_ex)(struct ibv_context *context,
                                        struct ibv_cq_init_attr_ex *cq_attr);
      int (*destroy_cq)(struct ibv_cq *cq);

      struct ibv_qp* (*create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
      int (*modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
      int (*destroy_qp)(struct ibv_qp *qp);
    };

    /**
     * @brief External handle to the ibv function table
     */
    struct ibv_funcs_t ibv;

    /**
     * @brief handle used for the dlopen of the InfiniBand Verbs library
     */
    void *ibv_handle = nullptr;

    /**
     * @brief initialize function table
     */
    int init_function_table();
};

} // namespace rocshmem

#endif  /* LIBRARY_SRC_GDA_IBV_WRAPPER_HPP_ */
