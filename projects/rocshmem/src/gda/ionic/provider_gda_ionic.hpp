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

extern "C" {
#include "gda/ionic/ionic_dv.h"
#include "gda/ionic/ionic_fw.h"
}

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

#endif  //LIBRARY_SRC_GDA_IONIC_GDA_PROVIDER_HPP_
