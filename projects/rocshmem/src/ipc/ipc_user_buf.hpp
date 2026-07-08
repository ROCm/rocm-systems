/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef LIBRARY_SRC_IPC_USER_BUF_HPP_
#define LIBRARY_SRC_IPC_USER_BUF_HPP_

#include <stdint.h>

namespace rocshmem {

struct IpcSymmTable;

// Called during IPC backend init to make the symm_table accessible
// to rocshmem_buffer_register_vmm (free function called from RCCL GIN).
void ipc_user_buf_set_symm_table(IpcSymmTable *table, int num_pes);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_USER_BUF_HPP_
