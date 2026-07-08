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

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_uuid.h"

#include <string.h>
#include <cstdio>

typedef struct uuid_s {
	union {
		struct {
			uint32_t did    : 16;
			uint32_t fcn    : 8;
			uint32_t asic_7 : 8;
		};
		uint32_t time_low;
	};
	uint32_t time_mid  : 16;
	uint32_t time_high : 12;
	uint32_t version   : 4;
	uint8_t clk_seq_hi : 6;
	uint8_t variant	   : 2;
	union {
		uint8_t clk_seq_low;
		uint8_t asic_6;
	};
	uint16_t asic_4;
	uint32_t asic_0;
} uuid_t;

static void print_uuid(char *str, uuid_t *uuid)
{
	sprintf(str, "%08x", uuid->time_low);
	sprintf(str + 8, "-");
	sprintf(str + 9, "%04x", uuid->time_mid);
	sprintf(str + 13, "-");
	sprintf(str + 14, "%04x", (uuid->version << 12) | uuid->time_high);
	sprintf(str + 18, "-");
	sprintf(str + 14 + 5, "%02x", (uuid->variant << 6) | uuid->clk_seq_hi);
	sprintf(str + 16 + 5, "%02x", uuid->clk_seq_low);
	sprintf(str + 18 + 5, "-");
	sprintf(str + 19 + 5, "%04x", uuid->asic_4);
	sprintf(str + 23 + 5, "%08x", uuid->asic_0);
	str[31 + 5] = 0;
}

static void insert_asic_serial(uuid_t *uuid, uint64_t serial)
{
	uuid->asic_0 = (uint32_t)serial;
	uuid->asic_4 = (uint16_t)(serial >> 4 * 8) & 0xFFFF;
	uuid->asic_6 = (uint8_t)(serial >> 6 * 8) & 0xFF;
	uuid->asic_7 = (uint32_t)(serial >> 7 * 8) & 0xFF;
}

static void insert_did(uuid_t *uuid, uint16_t did)
{
	uuid->did = did;
}

static void insert_fcn(uuid_t *uuid, uint8_t fcn_idx)
{
	uuid->fcn = fcn_idx;
}

static void insert_clk_seq(uuid_t *uuid, uint16_t seq)
{
	uuid->clk_seq_low = (uint8_t)seq;
	uuid->clk_seq_hi = (seq >> 8) & 0x3fU;
}

amdsmi_status_t amdsmi_uuid_gen(char *str, uint64_t serial, uint16_t did, uint8_t idx)
{
	uuid_t uuid;
	memset(&uuid, 0, sizeof(uuid_t));

	insert_clk_seq(&uuid, 0);
	insert_did(&uuid, did);
	insert_fcn(&uuid, idx);
	insert_asic_serial(&uuid, serial);

	uuid.version = 1;
	uuid.variant = 2;

	print_uuid(str, &uuid);

	return AMDSMI_STATUS_SUCCESS;
}
