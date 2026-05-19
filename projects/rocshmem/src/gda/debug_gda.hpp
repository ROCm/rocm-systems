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

#ifndef LIBRARY_SRC_GDA_DEBUG_GDA_HPP_
#define LIBRARY_SRC_GDA_DEBUG_GDA_HPP_

#include "log.hpp"


static void dump_ibv_context([[maybe_unused]] struct ibv_context* x) {
  LOG_TRACE("== IBV_CONTEXT =============================================\n"
            "  (ibv_device*)        device              = %p\n"
            "  (int)                cmd_fd              = %d\n"
            "  (int)                async_fd            = %d\n"
            "  (int)                num_comp_vectors    = %d\n"
            "  (void*)              abi_compat          = %p\n"
            "========",
            x->device, x->cmd_fd, x->async_fd, x->num_comp_vectors, x->abi_compat);
}

static void dump_ibv_device([[maybe_unused]] struct ibv_device* x) {
  LOG_TRACE("== IBV_DEVICE ==============================================\n"
            "  (enum ibv_node_type)      node_type      = %d\n"
            "  (enum ibv_transport_type) transport_type = %d\n"
            "  (char[])                  name           = %s\n"
            "  (char[])                  dev_name       = %s\n"
            "  (char[])                  dev_path       = %s\n"
            "  (char[])                  ibdev_path     = %s\n"
            "========",
            x->node_type, x->transport_type, x->name, x->dev_name, x->dev_path, x->ibdev_path);
}

static void dump_ibv_pd([[maybe_unused]] struct ibv_pd* x) {
  LOG_TRACE("== IBV_PD ==================================================\n"
            "  (ibv_context*) context = %p\n"
            "  (uint32_t)     handle  = 0x%x\n"
            "========",
            x->context, x->handle);
}

static void dump_ibv_port_attr([[maybe_unused]] struct ibv_port_attr* x) {
  LOG_TRACE("== IBV_PORT_ATTR ===========================================\n"
            "  (enum ibv_port_state) state           = %u\n"
            "  (enum ibv_mtu)        max_mtu         = %u\n"
            "  (enum ibv_mtu)        active_mtu      = %u\n"
            "  (int)                 gid_tbl_len     = %u\n"
            "  (uint32_t)            port_cap_flags  = 0x%x\n"
            "  (uint32_t)            max_msg_sz      = %u\n"
            "  (uint32_t)            bad_pkey_cntr   = %u\n"
            "  (uint32_t)            qkey_viol_cntr  = %u\n"
            "  (uint16_t)            pkey_tbl_len    = %u\n"
            "  (uint16_t)            lid             = 0x%x\n"
            "  (uint16_t)            sm_lid          = 0x%x\n"
            "  (uint8_t)             lmc             = 0x%x\n"
            "  (uint8_t)             max_vl_num      = 0x%x\n"
            "  (uint8_t)             sm_sl           = 0x%x\n"
            "  (uint8_t)             subnet_timeout  = 0x%x\n"
            "  (uint8_t)             init_type_reply = 0x%x\n"
            "  (uint8_t)             active_width    = 0x%x\n"
            "  (uint8_t)             active_speed    = 0x%x\n"
            "  (uint8_t)             phys_state      = 0x%x\n"
            "  (uint8_t)             link_layer      = 0x%x\n"
            "  (uint8_t)             flags           = 0x%x\n"
            "  (uint16_t)            port_cap_flags2 = 0x%x\n"
            "========",
            x->state, x->max_mtu, x->active_mtu, x->gid_tbl_len, x->port_cap_flags, x->max_msg_sz,
            x->bad_pkey_cntr, x->qkey_viol_cntr, x->pkey_tbl_len, x->lid, x->sm_lid, x->lmc, x->max_vl_num,
            x->sm_sl, x->subnet_timeout, x->init_type_reply, x->active_width, x->active_speed, x->phys_state,
            x->link_layer, x->flags, x->port_cap_flags2);
}

static void dump_ibv_qp([[maybe_unused]] struct ibv_qp *qp, [[maybe_unused]] int conn_num) {
  LOG_TRACE("== IBV_QP CONNECTION#%d ======================================\n"
            "  (ibv_context*)      context          = %p\n"
            "  (void*)             qp_context       = %p\n"
            "  (ibv_pd*)           pd               = %p\n"
            "  (ibv_cq*)           send_cq          = %p\n"
            "  (ibv_cq*)           recv_cq          = %p\n"
            "  (ibv_srq*)          srq              = %p\n"
            "  (uint32_t)          handle           = 0x%x\n"
            "  (uint32_t)          qp_num           = 0x%x\n"
            "  (enum ibv_qp_state) state            = %u\n"
            "  (enum_ibv_qp_type)  qp_type          = %u\n"
            "  (uint32_t)          events_completed = %u\n"
            "========",
            conn_num, qp->context, qp->qp_context, qp->pd, qp->send_cq, qp->recv_cq,
            qp->srq, qp->handle, qp->qp_num, qp->state, qp->qp_type, qp->events_completed);
}


#endif /* LIBRARY_SRC_GDA_DEBUG_GDA_HPP_ */
