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

#ifndef LIBRARY_SRC_GDA_IONIC_GDA_PROVIDER_HPP_
#define LIBRARY_SRC_GDA_IONIC_GDA_PROVIDER_HPP_

#include "util.hpp"

extern "C" {
#include "gda/ionic/ionic_dv.h"
#include "gda/ionic/ionic_fw.h"
}

namespace rocshmem {

template <typename T>
struct ionic_device_queue {
  T*        buf;
  uint64_t* dbreg;
  uint64_t  dbval;
  uint64_t  mask;
  uint32_t  pos;
  uint32_t  dbpos;
  uint32_t  lock;

  __host__ inline ionic_device_queue(T* buf, uint64_t* dbreg, uint64_t dbval, uint16_t mask)
    : buf{buf}, dbreg{dbreg}, dbval{dbval}, mask{static_cast<uint64_t>(mask)},
      pos{0}, dbpos{0}, lock{SPIN_LOCK_UNLOCKED} { }
};

struct ionic_device_cq : public ionic_device_queue<ionic_v1_cqe> {
  // inherit constructors
  using ionic_device_queue::ionic_device_queue;
};
static_assert(sizeof(ionic_device_cq) == 48);

struct ionic_device_sq : public ionic_device_queue<ionic_v1_wqe> {
  uint32_t msn;

  __host__ inline ionic_device_sq(ionic_v1_wqe* sq_buf, uint64_t* sq_dbreg, uint64_t sq_dbval,
                                  uint16_t sq_mask)
    : ionic_device_queue{sq_buf, sq_dbreg, sq_dbval, sq_mask}, msn{0} { }
};
// check if ionic_device_sq::msn is packed into tail padding of ionic_device_queue<ionic_v1_wqe>
static_assert(sizeof(ionic_device_sq) == 48);

struct ionicdv_funcs_t {
  int (*get_ctx)(struct ionic_dv_ctx *dvctx, struct ibv_context *ibctx);
  uint8_t (*qp_get_udma_idx)(struct ibv_qp *ibqp);
  int (*get_cq)(struct ionic_dv_cq *dvcq, struct ibv_cq *ibcq, uint8_t udma_idx);
  int (*get_qp)(struct ionic_dv_qp *dvqp, struct ibv_qp *ibqp);
  int (*pd_set_sqcmb)(struct ibv_pd *ibpd, bool enable, bool expdb, bool require);
  int (*pd_set_rqcmb)(struct ibv_pd *ibpd, bool enable, bool expdb, bool require);
  int (*pd_set_udma_mask)(struct ibv_pd *ibpd, uint8_t udma_mask);
  struct ibv_cq_ex *(*create_cq_ex)(struct ibv_context *ibctx,
                                    struct ibv_cq_init_attr_ex *ex,
                                    struct ionic_cq_init_attr_ex *ionic_ex);
};

}  // namespace rocshmem

#endif  //LIBRARY_SRC_GDA_IONIC_GDA_PROVIDER_HPP_
