/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

__global__ void kernel();

__global__ void kernel2();

__global__ void kernel_42(int* val);

__global__ void coop_kernel();