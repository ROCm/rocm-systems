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

#pragma once

#include <stddef.h>

#include "ifoe_telemetry.h"

/* C11 requires a message argument; C23 made it optional. Use #expr so the
 * assertion text is preserved as the message on all compiler versions. */
#define STATIC_ASSERT(expr) _Static_assert(expr, #expr)

/* Number of telemetry categories currently supported */
#define TELEMETRY_SRC_CATEGORY_COUNT (8)

/* ID Count for each category */
#define IFOE_TELEMETRY_ID_COUNT (167)
#define SWITCH_TELEMETRY_ID_COUNT (6)
#define CRYPTO_TELEMETRY_ID_COUNT (17)
#define PFC_TELEMETRY_ID_COUNT (90)
#define NETPORT_TELEMETRY_ID_COUNT (114)
#define IFOE_DERIVED_TELEMETRY_ID_COUNT (20)
#define NETPORT_DERIVED_TELEMETRY_ID_COUNT (25)

/* Definition of the fundamental Tag-Length-Value format used to organize
 *  the telemetry
 */
typedef struct ifoe_telem_tlv_s {
  uint64_t tag;
  /* Size of the data following this field i.e. not including the Tag or Length */
  uint64_t len;
} ifoe_telem_tlv_t;

STATIC_ASSERT(offsetof(ifoe_telem_tlv_t, tag) == IFOE_TELEM_TLV_INSTANCE_OFST);
STATIC_ASSERT(offsetof(ifoe_telem_tlv_t, len) == IFOE_TELEM_TLV_LEN_OFST);
STATIC_ASSERT(sizeof(ifoe_telem_tlv_t) == IFOE_TELEM_TLV_LENMIN);

/* Definition of the top-level header TLV */
typedef struct ifoe_telem_tlv_hdr_s {
  ifoe_telem_tlv_t tlv;
  /* Magic value, must be 0x1F0E5CA1AB1EDA7A */
  uint64_t magic;
  /* Telemetry Version in format Major.Minor.Patch */
  uint64_t version;
  /* The total size of the top-level telemetry TLVs (including tags/lengths) */
  uint64_t total_length;
} ifoe_telem_tlv_hdr_t;

STATIC_ASSERT(offsetof(ifoe_telem_tlv_hdr_t, magic) == IFOE_TELEM_TLV_HDR_MAGIC_OFST);
STATIC_ASSERT(offsetof(ifoe_telem_tlv_hdr_t, version) == IFOE_TELEM_TLV_HDR_VERSION_OFST);
STATIC_ASSERT(offsetof(ifoe_telem_tlv_hdr_t, total_length) == IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_OFST);
STATIC_ASSERT(sizeof(ifoe_telem_tlv_hdr_t) == IFOE_TELEM_TLV_HDR_LENMIN);

/* Describes the telemetry data for a Telemetry Category */
typedef struct ifoe_telem_tlv_desc_s {
  ifoe_telem_tlv_t tlv;
  /* List of Telemetry IDs in the order that they will appear in the
   * corresponding Telemetry Data TLV
   */
  uint64_t ids[];
} ifoe_telem_tlv_desc_t;

STATIC_ASSERT(offsetof(ifoe_telem_tlv_desc_t, ids) == IFOE_TELEM_TLV_DESC_IDS_OFST);
STATIC_ASSERT(sizeof(ifoe_telem_tlv_desc_t) == IFOE_TELEM_TLV_DESC_LENMIN);

/* TLV containing the Telemetry Data for an instance of Telemetry Category */
typedef struct ifoe_telem_tlv_data_s {
  ifoe_telem_tlv_t tlv;
  /* Telemetry data values in the order defined in the corresponding
   * Telemetry Description.
   */
  uint64_t values[];
} ifoe_telem_tlv_data_t;

STATIC_ASSERT(offsetof(ifoe_telem_tlv_data_t, values) == IFOE_TELEM_TLV_DATA_VALUES_OFST);
STATIC_ASSERT(sizeof(ifoe_telem_tlv_data_t) == IFOE_TELEM_TLV_DATA_LENMIN);

/* TLV containing the location of each Telemetry Description */
typedef struct ifoe_telem_tlv_desc_locator_s {
  ifoe_telem_tlv_t tlv;
  /* Array of offsets (from the start of the telemetry memory) to the
   * Telemetry Description (ifoe_telem_tlv_desc_t) for each category
   * The value 0 is defined as null and means there is no
   * description for the category. Indexed by category number
   * The telemetry memory may span multiple pages in GPU memory, the offset is
   * calculated as if the telemetry memory were a single contiguous buffer
   */
  uint64_t descs[TELEMETRY_SRC_CATEGORY_COUNT];
} ifoe_telem_tlv_desc_locator_t;

STATIC_ASSERT(offsetof(ifoe_telem_tlv_desc_locator_t, descs) ==
              IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_OFST);
STATIC_ASSERT(sizeof(ifoe_telem_tlv_desc_locator_t) ==
              IFOE_TELEM_TLV_DESC_LOCATOR_LEN(TELEMETRY_SRC_CATEGORY_COUNT));

/* TLV containing the location of each Telemetry Dataset */
typedef struct ifoe_telem_tlv_dataset_locator_s {
  ifoe_telem_tlv_t tlv;
  /* Array of offsets (from the start of the telemetry memory) to the
   * Telemetry Dataset (ifoe_telem_tlv_dataset_hdr_t) for each category
   * The value 0 is defined as null and means there is no data for the
   * category. Indexed by category number
   * The telemetry memory may span multiple pages in GPU memory, the offset is
   * calculated as if the telemetry memory were a single contiguous buffer
   */
  uint64_t datasets[TELEMETRY_SRC_CATEGORY_COUNT];
} ifoe_telem_tlv_dataset_locator_t;

STATIC_ASSERT(offsetof(ifoe_telem_tlv_dataset_locator_t, datasets) ==
              IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_OFST);
STATIC_ASSERT(sizeof(ifoe_telem_tlv_dataset_locator_t) ==
              IFOE_TELEM_TLV_DATASET_LOCATOR_LEN(TELEMETRY_SRC_CATEGORY_COUNT));

/* Header TLV for a Telemetry Dataset */
typedef struct ifoe_telem_tlv_dataset_hdr_s {
  ifoe_telem_tlv_t tlv;
  /* Sequence number incremented each time that the telemetry data is
   * written. The header generation count is always written before the
   * telemetry data
   */
  uint64_t gen_count;
  /* Timestamp of the point in time at which the telemetry was captured
   * Note that this is an up-time for the IFoE firmware in milliseconds, not
   * a time-of-day/UTC value
   */
  uint64_t timestamp;
} ifoe_telem_tlv_dataset_hdr_t;

STATIC_ASSERT(offsetof(ifoe_telem_tlv_dataset_hdr_t, gen_count) ==
              IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_OFST);
STATIC_ASSERT(offsetof(ifoe_telem_tlv_dataset_hdr_t, timestamp) ==
              IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_OFST);
STATIC_ASSERT(sizeof(ifoe_telem_tlv_dataset_hdr_t) == IFOE_TELEM_TLV_DATASET_HDR_LENMIN);

/* Footer TLV for a Telemetry Dataset */
typedef struct ifoe_telem_tlv_dataset_ftr_s {
  ifoe_telem_tlv_t tlv;
  /*  Sequence number incremented each time that the telemetry data is
   *  written. The footer generation count is always written after the
   *  telemetry data
   */
  uint64_t gen_count;
} ifoe_telem_tlv_dataset_ftr_t;

STATIC_ASSERT(offsetof(ifoe_telem_tlv_dataset_ftr_t, gen_count) ==
              IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_OFST);
STATIC_ASSERT(sizeof(ifoe_telem_tlv_dataset_ftr_t) == IFOE_TELEM_TLV_DATASET_FTR_LENMIN);
