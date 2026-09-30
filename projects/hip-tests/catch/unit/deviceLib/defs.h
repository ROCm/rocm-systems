/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#define INTERNAL_BUFFER_SIZE 8
// Test Type
#define TEST_MALLOC_FREE 1
#define TEST_NEW_DELETE 2
// Kernel Params
#define BLOCKSIZE 64
#define GRIDSIZE 32
// Code Obj
#define DEV_ALLOC_SINGKER_COBJ "kerDevAllocSingleKer.code"
#define DEV_ALLOC_SINGKER_COBJ_FUNC "ker_TestDynamicAllocInAllThreads_CodeObj"
#define DEV_ALLOC_MULCOBJ "kerDevAllocMultCO.code"
#define DEV_WRITE_MULCOBJ "kerDevWriteMultCO.code"
#define DEV_FREE_MULCOBJ "kerDevFreeMultCO.code"
#define DEV_ALLOC_MULCODEOBJ_ALLOC "ker_Alloc_MultCodeObj"
#define DEV_ALLOC_MULCODEOBJ_WRITE "ker_Write_MultCodeObj"
#define DEV_ALLOC_MULCODEOBJ_FREE "ker_Free_MultCodeObj"
