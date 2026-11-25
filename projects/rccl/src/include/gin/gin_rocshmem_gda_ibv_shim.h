/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#ifndef GIN_ROCSHMEM_GDA_IBV_SHIM_H_
#define GIN_ROCSHMEM_GDA_IBV_SHIM_H_

#include <cstdint>
#include <cstddef>
#include <sys/types.h>

// rccl-side IB verbs backend for the GIN QP factory: (1) it supplies the
// extended verbs the factory needs that rccl's wrap_ibv_* layer did not expose,
// and (2) it is the single seam that lets the factory use rccl's ibv layer
// despite rccl's ibvcore.h and rocSHMEM's ibv_core.hpp defining clashing
// struct ibv_* tags.
//
// The declarations below use *opaque* ibv_* tags on purpose: each translation
// unit supplies its own full struct definitions, and a C++ call mangles on the
// tag name alone, so the factory's rocSHMEM-typed pointers bind to this shim's
// rccl-typed pointers. Both headers model the same libibverbs ABI, so the
// pointed-to layouts match. The adapters also restore rocSHMEM's direct-return
// convention over wrap_ibv_*'s out-param + ncclResult_t one.

struct ibv_device;
struct ibv_context;
struct ibv_pd;
struct ibv_mr;
struct ibv_cq;
struct ibv_cq_ex;
struct ibv_qp;
struct ibv_device_attr;
struct ibv_port_attr;
struct ibv_qp_attr;
struct ibv_gid_entry;
struct ibv_cq_init_attr_ex;
struct ibv_qp_init_attr_ex;
struct ibv_parent_domain_init_attr;
struct ibv_ah;
struct ibv_ah_attr;

// GRH flow-label field mask (20 bits). A rocSHMEM-ism (not a standard libibverbs
// macro) needed by provider_gda_mlx5.cpp's host-side mlx5_calc_flow_label(); it
// reaches that file through this shim (which replaces its rocSHMEM ibv_wrapper
// include). Kept here, out of the installed ibv_core.hpp that user device code
// sees. This header is private (never installed), so defining it cannot pollute
// consumer code.
#ifndef IB_GRH_FLOWLABEL_MASK
#define IB_GRH_FLOWLABEL_MASK (0x000FFFFF)
#endif

// True once rccl's libibverbs symbols have been loaded successfully.
int gin_ibv_available(void);

// Common verbs (direct-return adapters over wrap_ibv_*).
struct ibv_device** gin_ibv_get_device_list(int* num_devices);
void gin_ibv_free_device_list(struct ibv_device** list);
const char* gin_ibv_get_device_name(struct ibv_device* device);
struct ibv_context* gin_ibv_open_device(struct ibv_device* device);
int gin_ibv_close_device(struct ibv_context* context);
int gin_ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr);
int gin_ibv_query_port(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr);
struct ibv_pd* gin_ibv_alloc_pd(struct ibv_context* context);
int gin_ibv_dealloc_pd(struct ibv_pd* pd);
struct ibv_mr* gin_ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access);
struct ibv_mr* gin_ibv_reg_mr_iova2(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova, int access);
struct ibv_mr* gin_ibv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd,
                                     int access);
int gin_ibv_dereg_mr(struct ibv_mr* mr);
int gin_ibv_destroy_cq(struct ibv_cq* cq);
int gin_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask);
int gin_ibv_destroy_qp(struct ibv_qp* qp);

// Extended verbs.
struct ibv_cq_ex* gin_ibv_create_cq_ex(struct ibv_context* context, struct ibv_cq_init_attr_ex* cq_attr);
struct ibv_cq* gin_ibv_cq_ex_to_cq(struct ibv_cq_ex* cq);
struct ibv_pd* gin_ibv_alloc_parent_domain(struct ibv_context* context, struct ibv_parent_domain_init_attr* attr);
struct ibv_qp* gin_ibv_create_qp_ex(struct ibv_context* context, struct ibv_qp_init_attr_ex* qp_init_attr_ex);
ssize_t gin_ibv_query_gid_table(struct ibv_context* context, struct ibv_gid_entry* entries, size_t max_entries,
                                uint32_t flags);
int gin_ibv_is_dmabuf_supported(struct ibv_pd* pd);

// Address handle + RoCE helpers (used by the compiled-in mlx5 provider).
struct ibv_ah* gin_ibv_create_ah(struct ibv_pd* pd, struct ibv_ah_attr* attr);
int gin_ibv_destroy_ah(struct ibv_ah* ah);
uint16_t gin_ibv_flow_label_to_udp_sport(uint32_t flow_label);

#endif  // GIN_ROCSHMEM_GDA_IBV_SHIM_H_
