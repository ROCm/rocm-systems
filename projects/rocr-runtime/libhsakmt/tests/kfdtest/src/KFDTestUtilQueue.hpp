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
} SDMACopyParams;

void sdma_multicopy(SDMACopyParams *array, int n,
        HSAuint64 *speedSmall = 0, HSAuint64 *speedLarge = 0, std::stringstream *s = 0);
void sdma_multicopy(std::vector<SDMACopyParams> &array, int mashup = 1, TSPattern tsp = ALLTS);
#endif //__KFD__TEST__UTIL__QUEUE__H__
