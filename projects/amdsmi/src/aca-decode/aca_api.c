/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "aca_decode.h"
#include "aca_constants.h"

int decode_afid(const uint64_t *register_array, size_t array_len, uint32_t flag, uint16_t hw_revision)
{
    if (!register_array)
    {
        return -1;
    }

    aca_raw_data_t raw_data;

    if (array_len == ACA_REGISTER_ARRAY_SIZE_32_BYTES) // 32 bytes
    {
        raw_data.aca_status = register_array[0];
        raw_data.aca_addr = register_array[1];
        raw_data.aca_ipid = register_array[2];
        raw_data.aca_synd = register_array[3];
    }
    else if (array_len == ACA_REGISTER_ARRAY_SIZE_128_BYTES) // 128 bytes
    {
        raw_data.aca_status = register_array[1];
        raw_data.aca_addr = register_array[2];
        raw_data.aca_ipid = register_array[5];
        raw_data.aca_synd = register_array[6];
    }
    
    else
    {
        return -1; // Unsupported size
    }

    raw_data.flags = flag;
    raw_data.hw_revision = hw_revision;

    aca_error_info_t error_info = aca_decode(&raw_data);
    return error_info.afid;
}

aca_error_info_t decode_error_info(const uint64_t *register_array, size_t array_len, uint32_t flag, uint16_t hw_revision)
{
    aca_raw_data_t raw_data = {0};
    aca_error_info_t error_info = {0};

    if (!register_array)
    {
        return error_info;
    }    if (array_len == ACA_REGISTER_ARRAY_SIZE_32_BYTES) // 32 bytes
    {
        raw_data.aca_status = register_array[0];
        raw_data.aca_addr = register_array[1];
        raw_data.aca_ipid = register_array[2];
        raw_data.aca_synd = register_array[3];
    }
    else if (array_len == ACA_REGISTER_ARRAY_SIZE_128_BYTES) // 128 bytes
    {
        raw_data.aca_status = register_array[1];
        raw_data.aca_addr = register_array[2];
        raw_data.aca_ipid = register_array[5];
        raw_data.aca_synd = register_array[6];
    }
    else
    {
        return error_info; // Return zero-initialized structure for unsupported size
    }

    raw_data.flags = flag;
    raw_data.hw_revision = hw_revision;

    return aca_decode(&raw_data);
}

