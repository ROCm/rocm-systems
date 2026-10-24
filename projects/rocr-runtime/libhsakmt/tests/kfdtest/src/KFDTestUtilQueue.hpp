/*
 * Copyright (C) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef __KFD__TEST__UTIL__QUEUE__H__
#define __KFD__TEST__UTIL__QUEUE__H__

#include "hsakmt/hsakmt.h"
#include <sstream>
#include <vector>

typedef struct {
        HSAuint64 timestamp;
        HSAuint64 timeConsumption;
        HSAuint64 timeBegin;
        HSAuint64 timeEnd;
} TimeStamp;

/* We have three pattern to put timestamp packet,
 * NOTS: No timestamp packet insert.
 * ALLTS: Put timestamp packet around every packet. This is the default behavoir.
 *    It will look like |timestamp|packet|timestamp|...|packet|timestamp|
 * HEAD_TAIL: Put timestmap packet at head and tail to measure the overhead of a bunch of packet.
 *    It will look like |timestamp|packet|...|packet|timestamp|
 */
typedef enum {
    NOTS = 0,
    ALLTS = 1,
    HEAD_TAIL = 2,
} TSPattern;

/* Which engine moves the bytes.
 *
 * COPY_SDMA uses the dedicated DMA engines. COPY_BLIT runs a copy kernel on the
 * CUs, which is what ROCr does for large copies. The two have different peak
 * bandwidth and behave differently over XGMI, so a P2P measurement that only
 * covers one of them is only half the picture.
 *
 * COPY_BLIT requires src, dst and size to be 16-byte aligned.
 */
typedef enum {
    COPY_SDMA = 0,
    COPY_BLIT = 1,
} CopyEngine;

typedef struct {
    /* input values*/
    HSAuint32 node;
    void *src;
    void *dst;
    HSAuint64 size;
    /* input value for internal use.*/
    HSAuint64 group;
    /* output value*/
    HSAuint64 timeConsumption;
    HSAuint64 timeBegin;
    HSAuint64 timeEnd;
    /* private: Output values for internal use.*/
    HSAuint64 queue_id;
    HSAuint64 packet_id;
} GpuCopyParams;

void gpu_multicopy(GpuCopyParams *array, int n, CopyEngine engine,
        HSAuint64 *speedSmall = 0, HSAuint64 *speedLarge = 0, std::stringstream *s = 0);
void gpu_multicopy(std::vector<GpuCopyParams> &array, CopyEngine engine,
        int mashup = 1, TSPattern tsp = ALLTS);
#endif //__KFD__TEST__UTIL__QUEUE__H__
