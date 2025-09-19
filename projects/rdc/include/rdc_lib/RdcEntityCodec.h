/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef INCLUDE_RDC_LIB_RDCENTITYCODEC_H_
#define INCLUDE_RDC_LIB_RDCENTITYCODEC_H_

/*
 *
 * See rdc.h for description of entity_index
 * Shifts and masks help get only the bits in question to decode/encode
 *
 * Ex, RDC_ENTITY_TYPE_SHIFT = 29 helps shift the 29 irrelevant bits, so we're
 * only left with the top 3 type bits.
 * Then, the corresponding 3 type bits are anded with the RDC_ENTITY_TYPE_MASK = 0x7
 * which = 111 in binary, "copying" the type bits.
 *
 *
 */
#include <cstdint>
static constexpr uint32_t RDC_ENTITY_TYPE_SHIFT = 29;
static constexpr uint32_t RDC_ENTITY_ROLE_SHIFT = 27;
static constexpr uint32_t RDC_ENTITY_INSTANCE_SHIFT = 11;
static constexpr uint32_t RDC_ENTITY_DEVICE_SHIFT = 0;

static constexpr uint32_t RDC_ENTITY_TYPE_MASK = 0x7;        // 3 bits for type.
static constexpr uint32_t RDC_ENTITY_ROLE_MASK = 0x3;        // 2 bits for role.
static constexpr uint32_t RDC_ENTITY_INSTANCE_MASK = 0x3FF;  // 10 bits for instance.
static constexpr uint32_t RDC_ENTITY_DEVICE_MASK = 0x3FF;    // 10 bits for device.

// entity functions are declared in rdc.h
#endif  // INCLUDE_RDC_LIB_RDCENTITYCODEC_H_
