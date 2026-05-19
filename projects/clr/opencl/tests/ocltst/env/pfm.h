/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _PFM_H_
#define _PFM_H_

extern unsigned int SavePFM(const char* filename, const float* buffer, unsigned int width,
                            unsigned int height, unsigned int components);

#endif  // _PFM_H_
