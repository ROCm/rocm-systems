/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_HANDLE_H_
#define NCCL_M2N_HANDLE_H_

#include <memory>

#include "nccl_m2n.h"

/* Shared process-epoch runtime state.  All handles created in the same
 * init/finalize epoch hold a shared_ptr to one runtime; the final release
 * tears down process-global caches and resets config-derived state. */
struct ncclM2nRuntime {
  ncclM2nConfig_t firstConfig = NCCL_M2N_CONFIG_INITIALIZER;
  ~ncclM2nRuntime();
};

/* Internal state owned by the handle table. Public ncclM2nHandle_t values are
 * opaque monotonic tokens and are never dereferenced. */
struct ncclM2nHandleState {
  ncclM2nConfig_t config = NCCL_M2N_CONFIG_INITIALIZER;
  std::shared_ptr<ncclM2nRuntime> runtime;
};

#endif /* NCCL_M2N_HANDLE_H_ */
