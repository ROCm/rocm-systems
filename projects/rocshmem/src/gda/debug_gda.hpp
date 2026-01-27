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

static void dump_ibv_context(struct ibv_context *x);
static void dump_ibv_device(struct ibv_device *x);
static void dump_ibv_pd(struct ibv_pd *x);
static void dump_ibv_port_attr(struct ibv_port_attr *x);
static void dump_ibv_qp(struct ibv_qp *qp, int conn_num);
static void dump_mlx5dv_qp(struct mlx5dv_qp *qp_dv, int conn_num);
static void dump_mlx5dv_cq(struct mlx5dv_cq *cq_dv, int conn_num);

static void dump_ibv_context(struct ibv_context* x) {
  /*
   * struct ibv_context {
   *   struct ibv_device      *device;
   *   struct ibv_context_ops  ops;
   *   int                     cmd_fd;
   *   int                     async_fd;
   *   int                     num_comp_vectors;
   *   pthread_mutex_t         mutex;
   *   void                   *abi_compat;
   * };
   */
  DPRINTF("\n"
         "===============================================\n"
         "                IBV_CONTEXT\n"
         "===============================================\n"
         "  (ibv_device*)        device              = %p\n"
         "  (int)                cmd_fd              = %d\n"
         "  (int)                async_fd            = %d\n"
         "  (int)                num_comp_vectors    = %d\n"
         "  (void*)              abi_compat          = %p\n",
         x->device, x->cmd_fd, x->async_fd, x->num_comp_vectors, x->abi_compat);
};

static void dump_ibv_device(struct ibv_device* x) {
  /*
   * struct ibv_device {
   *   struct _ibv_device_ops  _ops;
   *   enum ibv_node_type node_type;
   *   enum ibv_transport_type transport_type;
   *   char name[IBV_SYSFS_NAME_MAX];
   *   char dev_name[IBV_SYSFS_NAME_MAX];
   *   char dev_path[IBV_SYSFS_PATH_MAX];
   *   char ibdev_path[IBV_SYSFS_PATH_MAX];
   * };
   */
  DPRINTF("\n"
         "===============================================\n"
         "               IBV_DEVICE\n"
         "===============================================\n"
         "  (enum ibv_node_type)      node_type      = %d\n"
         "  (enum ibv_transport_type) transport_type = %d\n"
         "  (char[])                  name           = %s\n"
         "  (char[])                  dev_name       = %s\n"
         "  (char[])                  dev_path       = %s\n"
         "  (char[])                  ibdev_path     = %s\n",
         x->node_type, x->transport_type, x->name, x->dev_name, x->dev_path, x->ibdev_path);
}

static void dump_ibv_pd(struct ibv_pd* x) {
  /*
   * struct ibv_pd {
   *   struct ibv_context     *context;
   *   uint32_t                handle;
   * };
   */
  DPRINTF("\n"
         "===============================================\n"
         "               IBV_PD\n"
         "===============================================\n"
         "  (ibv_context*) context = %p\n"
         "  (uint32_t)     handle  = 0x%x\n",
         x->context, x->handle);
}

static void dump_ibv_port_attr(struct ibv_port_attr* x) {
  /*
   * struct ibv_port_attr {
   *   enum ibv_port_state     state;
   *   enum ibv_mtu            max_mtu;
   *   enum ibv_mtu            active_mtu;
   *   int                     gid_tbl_len;
   *   uint32_t                port_cap_flags;
   *   uint32_t                max_msg_sz;
   *   uint32_t                bad_pkey_cntr;
   *   uint32_t                qkey_viol_cntr;
   *   uint16_t                pkey_tbl_len;
   *   uint16_t                lid;
   *   uint16_t                sm_lid;
   *   uint8_t                 lmc;
   *   uint8_t                 max_vl_num;
   *   uint8_t                 sm_sl;
   *   uint8_t                 subnet_timeout;
   *   uint8_t                 init_type_reply;
   *   uint8_t                 active_width;
   *   uint8_t                 active_speed;
   *   uint8_t                 phys_state;
   *   uint8_t                 link_layer;
   *   uint8_t                 flags;
   *   uint16_t                port_cap_flags2;
   * };
   */
  DPRINTF("\n"
         "===============================================\n"
         "               IBV_PORT_ATTR\n"
         "===============================================\n"
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
         "  (uint16_t)            port_cap_flags2 = 0x%x\n",
         x->state, x->max_mtu, x->active_mtu, x->gid_tbl_len, x->port_cap_flags, x->max_msg_sz,
         x->bad_pkey_cntr, x->qkey_viol_cntr, x->pkey_tbl_len, x->lid, x->sm_lid, x->lmc, x->max_vl_num,
         x->sm_sl, x->subnet_timeout, x->init_type_reply, x->active_width, x->active_speed, x->phys_state,
         x->link_layer, x->flags, x->port_cap_flags2);
}

void dump_ibv_qp(struct ibv_qp *qp, int conn_num) {
  /*
   * struct ibv_qp {
   *   struct ibv_context     *context;
   *   void                   *qp_context;
   *   struct ibv_pd          *pd;
   *   struct ibv_cq          *send_cq;
   *   struct ibv_cq          *recv_cq;
   *   struct ibv_srq         *srq;
   *   uint32_t                handle;
   *   uint32_t                qp_num;
   *   enum ibv_qp_state       state;
   *   enum ibv_qp_type        qp_type;
   *   pthread_mutex_t         mutex;
   *   pthread_cond_t          cond;
   *   uint32_t                events_completed;
   * };
   */
  DPRINTF("\n");
  DPRINTF("============== QP_DUMP CONNECTION#%d ==========\n", conn_num);
  DPRINTF("  (ibv_context*)      context          = %p\n",   qp->context);
  DPRINTF("  (void*)             qp_context       = %p\n",   qp->qp_context);
  DPRINTF("  (ibv_pd*)           pd               = %p\n",   qp->pd);
  DPRINTF("  (ibv_cq*)           send_cq          = %p\n",   qp->send_cq);
  DPRINTF("  (ibv_cq*)           recv_cq          = %p\n",   qp->recv_cq);
  DPRINTF("  (ibv_srq*)          srq              = %p\n",   qp->srq);
  DPRINTF("  (uint32_t)          handle           = 0x%x\n", qp->handle);
  DPRINTF("  (uint32_t)          qp_num           = 0x%x\n", qp->qp_num);
  DPRINTF("  (enum ibv_qp_state) state            = %u\n",   qp->state);
  DPRINTF("  (enum_ibv_qp_type)  qp_type          = %u\n",   qp->qp_type);
  DPRINTF("  (uint32_t)          events_completed = %u\n",   qp->events_completed);
  DPRINTF("=========== QP_DUMP_END CONNECTION#%d  ========\n", conn_num);
}

void dump_mlx5dv_qp(struct mlx5dv_qp *qp_dv, int conn_num) {
  DPRINTF("\n");
  DPRINTF("===============================================\n");
  DPRINTF("     INITIALIZED MLXDV_QP FOR CONNECTION#%d\n", conn_num);
  DPRINTF("===============================================\n");
  DPRINTF("=================== QP_DUMP ===================\n");
  DPRINTF("  (__be32*)  dbrec           = %p\n",     qp_dv->dbrec);
  DPRINTF("  (void*)    sq.buf          = %p\n",     qp_dv->sq.buf);
  DPRINTF("  (uint32_t) sq.wqe_cnt      = %u\n",     qp_dv->sq.wqe_cnt);
  DPRINTF("  (uint32_t) sq.stride       = %u\n",     qp_dv->sq.stride);
  DPRINTF("  (void*)    rq.buf          = %p\n",     qp_dv->rq.buf);
  DPRINTF("  (uint32_t) rq.wqe_cnt      = %u\n",     qp_dv->rq.wqe_cnt);
  DPRINTF("  (uint32_t) rq.stride       = %u\n",     qp_dv->rq.stride);
  DPRINTF("  (void*)    bf.reg          = %p\n",     qp_dv->bf.reg);
  DPRINTF("  (uint32_t) bf.size         = 0x%x\n",   qp_dv->bf.size);
  DPRINTF("  (uint64_t) comp_mask       = 0x%lx\n",  qp_dv->comp_mask);
  DPRINTF("  (off_t)    uar_mmap_offset = 0x%lx\n",  qp_dv->uar_mmap_offset);
  DPRINTF("  (uint32_t) tirn            = 0x%x\n",   qp_dv->tirn);
  DPRINTF("  (uint32_t) tisn            = 0x%x\n",   qp_dv->tisn);
  DPRINTF("  (uint32_t) rqn             = 0x%x\n",   qp_dv->rqn);
  DPRINTF("  (uint32_t) sqn             = 0x%x\n",   qp_dv->sqn);
  DPRINTF("  (uint64_t) tir_icm_addr    = 0x%lx\n",  qp_dv->tir_icm_addr);
  DPRINTF("================== QP_DUMP_END ================\n");
}

void dump_mlx5dv_cq(struct mlx5dv_cq *cq_dv, int conn_num) {
  DPRINTF("\n");
  DPRINTF("===============================================\n");
  DPRINTF("     INITIALIZED MLX5DV_CQ FOR CONNECTION#%d\n", conn_num);
  DPRINTF("===============================================\n");
  DPRINTF("=================== CQ_DUMP ===================\n");
  DPRINTF("  (void*)    buf             = %p\n",     cq_dv->buf);
  DPRINTF("  (__be32*)  dbrec           = %p\n",     cq_dv->dbrec);
  DPRINTF("  (uint32_t) cqe_cnt         = %u\n",     cq_dv->cqe_cnt);
  DPRINTF("  (uint32_t) cqe_size        = %u\n",     cq_dv->cqe_size);
  DPRINTF("  (void*)    cq_uar          = %p\n",     cq_dv->cq_uar);
  DPRINTF("  (uint32_t) cqn             = 0x%x\n",   cq_dv->cqn);
  DPRINTF("  (uint64_t) comp_mask       = 0x%lx\n",  cq_dv->comp_mask);
  DPRINTF("================== CQ_DUMP_END ================\n");
}

#endif /* LIBRARY_SRC_GDA_DEBUG_GDA_HPP_ */
