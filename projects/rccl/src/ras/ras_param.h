/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RAS_PARAM_H_
#define NCCL_RAS_PARAM_H_

#include <stdint.h>

// Upper bound for RAS timeouts in ns (~73 years). Callers add timeouts to timestamps and to each other, so this
// stays well below INT64_MAX.
#define RAS_MAX_TIMEOUT_NS (INT64_MAX / 4)

#ifdef __cplusplus
extern "C" {
#endif

float ncclParamRasTimeoutFactor(void);
int64_t rasTimeoutFactorNs(int64_t baseSeconds);
double rasTimeoutFactorSec(int baseSeconds);

#ifdef __cplusplus
}
#endif

#endif // NCCL_RAS_PARAM_H_
