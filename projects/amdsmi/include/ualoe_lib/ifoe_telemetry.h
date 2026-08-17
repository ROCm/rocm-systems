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

/* IFOE_TELEM_TAG structuredef: Defines the tags for the various IFoE Telemetry
 * TLVs
 */
#define IFOE_TELEM_TAG_LEN 8
#define IFOE_TELEM_TAG_TAG_ID_OFST 7
#define IFOE_TELEM_TAG_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TAG_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TAG_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TAG_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TAG_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TAG_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TAG_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TAG_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TAG_DATASET_FTR 0x7
#define IFOE_TELEM_TAG_TAG_ID_LBN 56
#define IFOE_TELEM_TAG_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TAG_CATEGORY_OFST 6
#define IFOE_TELEM_TAG_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TAG_CATEGORY_LBN 48
#define IFOE_TELEM_TAG_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TAG_INSTANCE_OFST 0
#define IFOE_TELEM_TAG_INSTANCE_LEN 6
#define IFOE_TELEM_TAG_STATION_IDX_OFST 0
#define IFOE_TELEM_TAG_STATION_IDX_LBN 36
#define IFOE_TELEM_TAG_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TAG_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TAG_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TAG_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TAG_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TAG_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TAG_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TAG_INSTANCE_LBN 0
#define IFOE_TELEM_TAG_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TAG_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TAG_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TAG_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TAG_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TAG_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TAG_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TAG_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TAG_NETPORT_HANDLE_WIDTH 32

/* IFOE_TELEM_TLV structuredef: Definition of the fundamental Tag-Length-Value
 * format used to organize the telemetry.
 */
#define IFOE_TELEM_TLV_LENMIN 16
#define IFOE_TELEM_TLV_LENMAX 248
#define IFOE_TELEM_TLV_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_VALUES_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_LEN_OFST 8
#define IFOE_TELEM_TLV_LEN_LEN 8
#define IFOE_TELEM_TLV_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_LEN_LBN 64
#define IFOE_TELEM_TLV_LEN_WIDTH 64
#define IFOE_TELEM_TLV_VALUES_OFST 16
#define IFOE_TELEM_TLV_VALUES_LEN 8
#define IFOE_TELEM_TLV_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_VALUES_LBN 128
#define IFOE_TELEM_TLV_VALUES_WIDTH 64

/* IFOE_TELEM_TLV_NULL structuredef: Definition of a null TLV. The VALUES field
 * in null TLVs does not contain any valid data so should be ignored by
 * clients. In all other respects handling of the TLV is exactly the same as
 * other types i.e. the LEN field provides the size of the (invalid) data in
 * the VALUES fields and thus provides the position of the next TLV.
 */
#define IFOE_TELEM_TLV_NULL_LENMIN 16
#define IFOE_TELEM_TLV_NULL_LENMAX 248
#define IFOE_TELEM_TLV_NULL_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_NULL_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_NULL_VALUES_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_NULL_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_NULL_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_NULL_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_NULL_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_NULL_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_NULL_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_NULL_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_NULL_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_NULL_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_NULL_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_NULL_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_NULL_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_NULL_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_NULL_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_NULL_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_NULL_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_NULL_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_NULL_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_NULL_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_NULL_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_NULL_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_NULL_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_NULL_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_NULL_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_NULL_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_NULL_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_NULL_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_NULL_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_NULL_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_NULL_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_NULL_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_NULL_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_NULL_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_NULL_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_NULL_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_NULL_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_NULL_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_NULL_LEN_OFST 8
#define IFOE_TELEM_TLV_NULL_LEN_LEN 8
#define IFOE_TELEM_TLV_NULL_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_NULL_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_NULL_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_NULL_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_NULL_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_NULL_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_NULL_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_NULL_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_NULL_LEN_LBN 64
#define IFOE_TELEM_TLV_NULL_LEN_WIDTH 64
#define IFOE_TELEM_TLV_NULL_VALUES_OFST 16
#define IFOE_TELEM_TLV_NULL_VALUES_LEN 8
#define IFOE_TELEM_TLV_NULL_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_NULL_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_NULL_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_NULL_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_NULL_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_NULL_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_NULL_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_NULL_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_NULL_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_NULL_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_NULL_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_NULL_VALUES_LBN 128
#define IFOE_TELEM_TLV_NULL_VALUES_WIDTH 64

/* IFOE_TELEM_TLV_HDR structuredef: Definition of the top-level header TLV */
#define IFOE_TELEM_TLV_HDR_LENMIN 40
#define IFOE_TELEM_TLV_HDR_LENMAX 248
#define IFOE_TELEM_TLV_HDR_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_HDR_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_HDR_VALUES_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_HDR_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_HDR_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_HDR_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_HDR_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_HDR_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_HDR_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_HDR_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_HDR_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_HDR_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_HDR_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_HDR_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_HDR_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_HDR_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_HDR_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_HDR_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_HDR_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_HDR_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_HDR_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_HDR_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_HDR_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_HDR_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_HDR_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_HDR_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_HDR_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_HDR_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_HDR_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_HDR_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_HDR_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_HDR_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_HDR_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_HDR_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_HDR_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_HDR_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_HDR_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_HDR_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_HDR_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_HDR_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_HDR_LEN_OFST 8
#define IFOE_TELEM_TLV_HDR_LEN_LEN 8
#define IFOE_TELEM_TLV_HDR_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_HDR_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_HDR_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_HDR_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_HDR_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_HDR_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_HDR_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_HDR_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_HDR_LEN_LBN 64
#define IFOE_TELEM_TLV_HDR_LEN_WIDTH 64
#define IFOE_TELEM_TLV_HDR_VALUES_OFST 16
#define IFOE_TELEM_TLV_HDR_VALUES_LEN 8
#define IFOE_TELEM_TLV_HDR_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_HDR_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_HDR_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_HDR_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_HDR_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_HDR_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_HDR_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_HDR_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_HDR_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_HDR_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_HDR_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_HDR_VALUES_LBN 128
#define IFOE_TELEM_TLV_HDR_VALUES_WIDTH 64
#define IFOE_TELEM_TLV_HDR_MAGIC_OFST 16
#define IFOE_TELEM_TLV_HDR_MAGIC_LEN 8
#define IFOE_TELEM_TLV_HDR_MAGIC_LO_OFST 16
#define IFOE_TELEM_TLV_HDR_MAGIC_LO_LEN 4
#define IFOE_TELEM_TLV_HDR_MAGIC_LO_LBN 128
#define IFOE_TELEM_TLV_HDR_MAGIC_LO_WIDTH 32
#define IFOE_TELEM_TLV_HDR_MAGIC_HI_OFST 20
#define IFOE_TELEM_TLV_HDR_MAGIC_HI_LEN 4
#define IFOE_TELEM_TLV_HDR_MAGIC_HI_LBN 160
#define IFOE_TELEM_TLV_HDR_MAGIC_HI_WIDTH 32
#define IFOE_TELEM_TLV_HDR_MAGIC_VALUE 0x1f0e5ca1ab1eda7a /* enum */
#define IFOE_TELEM_TLV_HDR_MAGIC_LBN 128
#define IFOE_TELEM_TLV_HDR_MAGIC_WIDTH 64
/* Telemetry Version in format Major.Minor.Patch */
#define IFOE_TELEM_TLV_HDR_VERSION_OFST 24
#define IFOE_TELEM_TLV_HDR_VERSION_LEN 8
#define IFOE_TELEM_TLV_HDR_VERSION_LO_OFST 24
#define IFOE_TELEM_TLV_HDR_VERSION_LO_LEN 4
#define IFOE_TELEM_TLV_HDR_VERSION_LO_LBN 192
#define IFOE_TELEM_TLV_HDR_VERSION_LO_WIDTH 32
#define IFOE_TELEM_TLV_HDR_VERSION_HI_OFST 28
#define IFOE_TELEM_TLV_HDR_VERSION_HI_LEN 4
#define IFOE_TELEM_TLV_HDR_VERSION_HI_LBN 224
#define IFOE_TELEM_TLV_HDR_VERSION_HI_WIDTH 32
#define IFOE_TELEM_TLV_HDR_VERSION_MAJOR_OFST 24
#define IFOE_TELEM_TLV_HDR_VERSION_MAJOR_LBN 48
#define IFOE_TELEM_TLV_HDR_VERSION_MAJOR_WIDTH 16
#define IFOE_TELEM_TLV_HDR_VERSION_MINOR_OFST 24
#define IFOE_TELEM_TLV_HDR_VERSION_MINOR_LBN 32
#define IFOE_TELEM_TLV_HDR_VERSION_MINOR_WIDTH 16
#define IFOE_TELEM_TLV_HDR_VERSION_PATCH_OFST 24
#define IFOE_TELEM_TLV_HDR_VERSION_PATCH_LBN 16
#define IFOE_TELEM_TLV_HDR_VERSION_PATCH_WIDTH 16
#define IFOE_TELEM_TLV_HDR_VERSION_RESERVED_OFST 24
#define IFOE_TELEM_TLV_HDR_VERSION_RESERVED_LBN 0
#define IFOE_TELEM_TLV_HDR_VERSION_RESERVED_WIDTH 16
#define IFOE_TELEM_TLV_HDR_VERSION_LBN 192
#define IFOE_TELEM_TLV_HDR_VERSION_WIDTH 64
/* The total size of the top-level telemetry TLVs (including tags/lengths) */
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_OFST 32
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_LEN 8
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_LO_OFST 32
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_LO_LEN 4
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_LO_LBN 256
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_LO_WIDTH 32
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_HI_OFST 36
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_HI_LEN 4
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_HI_LBN 288
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_HI_WIDTH 32
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_LBN 256
#define IFOE_TELEM_TLV_HDR_TOTAL_LENGTH_WIDTH 64

/* IFOE_TELEM_TLV_DESC_LOCATOR structuredef: TLV containing the location of
 * each Telemetry Description
 */
#define IFOE_TELEM_TLV_DESC_LOCATOR_LENMIN 16
#define IFOE_TELEM_TLV_DESC_LOCATOR_LENMAX 248
#define IFOE_TELEM_TLV_DESC_LOCATOR_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_DESC_LOCATOR_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_DESC_LOCATOR_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_DESC_LOCATOR_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_DESC_LOCATOR_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_DESC_LOCATOR_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_DESC_LOCATOR_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_DESC_LOCATOR_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_DESC_LOCATOR_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_DESC_LOCATOR_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_DESC_LOCATOR_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_DESC_LOCATOR_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_DESC_LOCATOR_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_DESC_LOCATOR_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_DESC_LOCATOR_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_DESC_LOCATOR_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_DESC_LOCATOR_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_DESC_LOCATOR_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_DESC_LOCATOR_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DESC_LOCATOR_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DESC_LOCATOR_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DESC_LOCATOR_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DESC_LOCATOR_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_OFST 8
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_LEN 8
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_LBN 64
#define IFOE_TELEM_TLV_DESC_LOCATOR_LEN_WIDTH 64
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_OFST 16
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_LEN 8
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_LBN 128
#define IFOE_TELEM_TLV_DESC_LOCATOR_VALUES_WIDTH 64
/* Array of pointers/offsets to the Telemetry Description for each category.
 * The value 0 is defined as null and means there is no description for the
 * category. Indexed by category number.
 */
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_OFST 16
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_LEN 8
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_LO_OFST 16
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_LO_LEN 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_LO_LBN 128
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_LO_WIDTH 32
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_HI_OFST 20
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_HI_LEN 4
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_HI_LBN 160
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_HI_WIDTH 32
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_MINNUM 0
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_MAXNUM 29
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_LBN 128
#define IFOE_TELEM_TLV_DESC_LOCATOR_DESCS_WIDTH 64

/* IFOE_TELEM_TLV_DATASET_LOCATOR structuredef: TLV containing the location of
 * each Telemetry Dataset
 */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LENMIN 16
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LENMAX 248
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_DATASET_LOCATOR_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_DATASET_LOCATOR_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_DATASET_LOCATOR_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_DATASET_LOCATOR_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_DATASET_LOCATOR_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_DATASET_LOCATOR_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_DATASET_LOCATOR_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_DATASET_LOCATOR_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_DATASET_LOCATOR_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DATASET_LOCATOR_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DATASET_LOCATOR_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_OFST 8
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_LEN 8
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_LBN 64
#define IFOE_TELEM_TLV_DATASET_LOCATOR_LEN_WIDTH 64
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_OFST 16
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_LEN 8
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_LBN 128
#define IFOE_TELEM_TLV_DATASET_LOCATOR_VALUES_WIDTH 64
/* Array of pointers/offsets to the Telemetry Dataset for each category. The
 * value 0 is defined as null and means there is no data for the category.
 * Indexed by category number.
 */
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_OFST 16
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_LEN 8
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_LO_OFST 16
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_LO_LBN 128
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_HI_OFST 20
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_HI_LBN 160
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_MINNUM 0
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_MAXNUM 29
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_LBN 128
#define IFOE_TELEM_TLV_DATASET_LOCATOR_DATASETS_WIDTH 64

/* IFOE_TELEM_TLV_DESC structuredef: Describes the telemetry data for a
 * Telemetry Category
 */
#define IFOE_TELEM_TLV_DESC_LENMIN 16
#define IFOE_TELEM_TLV_DESC_LENMAX 248
#define IFOE_TELEM_TLV_DESC_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_DESC_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_DESC_IDS_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_DESC_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_DESC_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_DESC_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_DESC_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_DESC_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_DESC_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_DESC_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_DESC_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_DESC_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_DESC_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_DESC_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_DESC_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_DESC_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_DESC_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_DESC_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_DESC_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_DESC_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_DESC_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_DESC_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_DESC_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_DESC_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_DESC_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DESC_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DESC_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DESC_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DESC_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DESC_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DESC_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_DESC_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DESC_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DESC_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DESC_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DESC_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DESC_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DESC_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DESC_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DESC_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_DESC_LEN_OFST 8
#define IFOE_TELEM_TLV_DESC_LEN_LEN 8
#define IFOE_TELEM_TLV_DESC_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_DESC_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_DESC_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_DESC_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_DESC_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_DESC_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_DESC_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_DESC_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_DESC_LEN_LBN 64
#define IFOE_TELEM_TLV_DESC_LEN_WIDTH 64
#define IFOE_TELEM_TLV_DESC_VALUES_OFST 16
#define IFOE_TELEM_TLV_DESC_VALUES_LEN 8
#define IFOE_TELEM_TLV_DESC_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_DESC_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_DESC_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_DESC_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_DESC_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_DESC_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_DESC_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_DESC_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_DESC_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_DESC_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_DESC_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DESC_VALUES_LBN 128
#define IFOE_TELEM_TLV_DESC_VALUES_WIDTH 64
/* List of Telemetry IDs in the order that they will appear in the
 * corresponding Telemetry Data TLV.
 */
#define IFOE_TELEM_TLV_DESC_IDS_OFST 16
#define IFOE_TELEM_TLV_DESC_IDS_LEN 8
#define IFOE_TELEM_TLV_DESC_IDS_LO_OFST 16
#define IFOE_TELEM_TLV_DESC_IDS_LO_LEN 4
#define IFOE_TELEM_TLV_DESC_IDS_LO_LBN 128
#define IFOE_TELEM_TLV_DESC_IDS_LO_WIDTH 32
#define IFOE_TELEM_TLV_DESC_IDS_HI_OFST 20
#define IFOE_TELEM_TLV_DESC_IDS_HI_LEN 4
#define IFOE_TELEM_TLV_DESC_IDS_HI_LBN 160
#define IFOE_TELEM_TLV_DESC_IDS_HI_WIDTH 32
#define IFOE_TELEM_TLV_DESC_IDS_MINNUM 0
#define IFOE_TELEM_TLV_DESC_IDS_MAXNUM 29
#define IFOE_TELEM_TLV_DESC_IDS_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DESC_IDS_LBN 128
#define IFOE_TELEM_TLV_DESC_IDS_WIDTH 64

/* IFOE_TELEM_TLV_DATASET_HDR structuredef: Header TLV for a Telemetry Dataset
 */
#define IFOE_TELEM_TLV_DATASET_HDR_LENMIN 32
#define IFOE_TELEM_TLV_DATASET_HDR_LENMAX 248
#define IFOE_TELEM_TLV_DATASET_HDR_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_DATASET_HDR_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_DATASET_HDR_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_DATASET_HDR_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_DATASET_HDR_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_DATASET_HDR_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_DATASET_HDR_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_DATASET_HDR_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_DATASET_HDR_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_DATASET_HDR_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_DATASET_HDR_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_DATASET_HDR_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_DATASET_HDR_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_DATASET_HDR_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_DATASET_HDR_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_DATASET_HDR_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_DATASET_HDR_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_DATASET_HDR_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_DATASET_HDR_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_DATASET_HDR_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_DATASET_HDR_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_HDR_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_DATASET_HDR_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_DATASET_HDR_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_HDR_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DATASET_HDR_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DATASET_HDR_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_HDR_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DATASET_HDR_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DATASET_HDR_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_DATASET_HDR_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DATASET_HDR_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DATASET_HDR_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DATASET_HDR_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DATASET_HDR_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DATASET_HDR_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DATASET_HDR_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_OFST 8
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_LEN 8
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_LBN 64
#define IFOE_TELEM_TLV_DATASET_HDR_LEN_WIDTH 64
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_OFST 16
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_LEN 8
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_LBN 128
#define IFOE_TELEM_TLV_DATASET_HDR_VALUES_WIDTH 64
/* Sequence number incremented each time that the telemetry data is written.
 * The header generation count is always written before the telemetry data.
 */
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_OFST 16
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_LEN 8
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_LO_OFST 16
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_LO_LBN 128
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_HI_OFST 20
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_HI_LBN 160
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_LBN 128
#define IFOE_TELEM_TLV_DATASET_HDR_GEN_COUNT_WIDTH 64
/* Timestamp of the point in time at which the telemetry was captured. Note
 * that this is an up-time for the IFoE firmware in milliseconds, not a time-
 * of-day/UTC value.
 */
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_OFST 24
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_LEN 8
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_LO_OFST 24
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_LO_LBN 192
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_HI_OFST 28
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_HI_LBN 224
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_LBN 192
#define IFOE_TELEM_TLV_DATASET_HDR_TIMESTAMP_WIDTH 64

/* IFOE_TELEM_TLV_DATA structuredef: TLV containing the Telemetry Data for an
 * instance of Telemetry Category
 */
#define IFOE_TELEM_TLV_DATA_LENMIN 16
#define IFOE_TELEM_TLV_DATA_LENMAX 248
#define IFOE_TELEM_TLV_DATA_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_DATA_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_DATA_VALUES_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_DATA_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_DATA_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_DATA_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_DATA_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_DATA_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_DATA_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_DATA_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_DATA_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_DATA_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_DATA_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_DATA_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_DATA_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_DATA_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_DATA_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_DATA_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_DATA_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_DATA_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_DATA_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_DATA_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_DATA_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_DATA_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_DATA_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DATA_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DATA_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DATA_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DATA_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DATA_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DATA_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_DATA_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DATA_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DATA_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DATA_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DATA_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DATA_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DATA_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DATA_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DATA_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_DATA_LEN_OFST 8
#define IFOE_TELEM_TLV_DATA_LEN_LEN 8
#define IFOE_TELEM_TLV_DATA_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_DATA_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_DATA_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_DATA_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATA_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_DATA_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_DATA_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_DATA_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATA_LEN_LBN 64
#define IFOE_TELEM_TLV_DATA_LEN_WIDTH 64
/* Telemetry data values in the order defined in the corresponding Telemetry
 * Description.
 */
#define IFOE_TELEM_TLV_DATA_VALUES_OFST 16
#define IFOE_TELEM_TLV_DATA_VALUES_LEN 8
#define IFOE_TELEM_TLV_DATA_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_DATA_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_DATA_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_DATA_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATA_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_DATA_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_DATA_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_DATA_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATA_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_DATA_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_DATA_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DATA_VALUES_LBN 128
#define IFOE_TELEM_TLV_DATA_VALUES_WIDTH 64

/* IFOE_TELEM_TLV_DATASET_FTR structuredef: Footer TLV for a Telemetry Dataset
 */
#define IFOE_TELEM_TLV_DATASET_FTR_LENMIN 24
#define IFOE_TELEM_TLV_DATASET_FTR_LENMAX 248
#define IFOE_TELEM_TLV_DATASET_FTR_LENMAX_MCDI2 1016
#define IFOE_TELEM_TLV_DATASET_FTR_LEN(num) (16 + 8 * (num))
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_NUM(len) (((len) - 16) / 8)
#define IFOE_TELEM_TLV_DATASET_FTR_TAG_ID_OFST 7
#define IFOE_TELEM_TLV_DATASET_FTR_TAG_ID_LEN 1
/* enum property: index */
/* enum: Indicates that the TLV is null and all other fields in the
 * IFOE_TELEM_TAG should be ignored by the consumer.
 */
#define IFOE_TELEM_TLV_DATASET_FTR_NULL 0x0
/* enum: Indicates that this is top-level telemetry header TLV tag. */
#define IFOE_TELEM_TLV_DATASET_FTR_HEADER 0x1
/* enum: Indicates that this is a telemetry description locator TLV */
#define IFOE_TELEM_TLV_DATASET_FTR_DESC_LOCATOR 0x2
/* enum: Indicates that this is a telemetry dataset locator TLV */
#define IFOE_TELEM_TLV_DATASET_FTR_DATASET_LOCATOR 0x3
/* enum: Indicates that this is a telemetry description TLV */
#define IFOE_TELEM_TLV_DATASET_FTR_DESC 0x4
/* enum: Indicates that this is a dataset header TLV */
#define IFOE_TELEM_TLV_DATASET_FTR_DATASET_HDR 0x5
/* enum: Indicates that this is a data instance TLV */
#define IFOE_TELEM_TLV_DATASET_FTR_DATA 0x6
/* enum: Indicates that this is a dataset footer TLV */
#define IFOE_TELEM_TLV_DATASET_FTR_DATASET_FTR 0x7
#define IFOE_TELEM_TLV_DATASET_FTR_TAG_ID_LBN 56
#define IFOE_TELEM_TLV_DATASET_FTR_TAG_ID_WIDTH 8
/* Telemetry category */
#define IFOE_TELEM_TLV_DATASET_FTR_CATEGORY_OFST 6
#define IFOE_TELEM_TLV_DATASET_FTR_CATEGORY_LEN 1
/* enum property: index */
/*            Enum values, see field(s): */
/*               IFOE_TELEM_CAT */
#define IFOE_TELEM_TLV_DATASET_FTR_CATEGORY_LBN 48
#define IFOE_TELEM_TLV_DATASET_FTR_CATEGORY_WIDTH 8
/* Field used to define which instance of a telemetry TLV structure applies to.
 * This field can be subdivided as needed for each TLV type that has multiple
 * instances.
 */
#define IFOE_TELEM_TLV_DATASET_FTR_INSTANCE_OFST 0
#define IFOE_TELEM_TLV_DATASET_FTR_INSTANCE_LEN 6
#define IFOE_TELEM_TLV_DATASET_FTR_STATION_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_FTR_STATION_IDX_LBN 36
#define IFOE_TELEM_TLV_DATASET_FTR_STATION_IDX_WIDTH 12
#define IFOE_TELEM_TLV_DATASET_FTR_NETPORT_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_FTR_NETPORT_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DATASET_FTR_NETPORT_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DATASET_FTR_PHY_LANE_REL_IDX_OFST 0
#define IFOE_TELEM_TLV_DATASET_FTR_PHY_LANE_REL_IDX_LBN 32
#define IFOE_TELEM_TLV_DATASET_FTR_PHY_LANE_REL_IDX_WIDTH 4
#define IFOE_TELEM_TLV_DATASET_FTR_INSTANCE_LBN 0
#define IFOE_TELEM_TLV_DATASET_FTR_INSTANCE_WIDTH 48
/* For IFoE telemetry, identifies the IFoE Station handle that the telemetry
 * data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DATASET_FTR_STATION_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DATASET_FTR_STATION_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DATASET_FTR_STATION_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DATASET_FTR_STATION_HANDLE_WIDTH 32
/* For Network Port, PFC and PHY telemetry, identifies the Network Port handle
 * that the telemetry data applies to. This handle is unique within a device.
 */
#define IFOE_TELEM_TLV_DATASET_FTR_NETPORT_HANDLE_OFST 0
#define IFOE_TELEM_TLV_DATASET_FTR_NETPORT_HANDLE_LEN 4
#define IFOE_TELEM_TLV_DATASET_FTR_NETPORT_HANDLE_LBN 0
#define IFOE_TELEM_TLV_DATASET_FTR_NETPORT_HANDLE_WIDTH 32
/* Size of the data following this field i.e. not including the Tag or Length
 */
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_OFST 8
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_LEN 8
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_LO_OFST 8
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_LO_LBN 64
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_HI_OFST 12
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_HI_LBN 96
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_LBN 64
#define IFOE_TELEM_TLV_DATASET_FTR_LEN_WIDTH 64
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_OFST 16
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_LEN 8
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_LO_OFST 16
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_LO_LBN 128
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_HI_OFST 20
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_HI_LBN 160
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_MINNUM 0
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_MAXNUM 29
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_MAXNUM_MCDI2 125
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_LBN 128
#define IFOE_TELEM_TLV_DATASET_FTR_VALUES_WIDTH 64
/* Sequence number incremented each time that the telemetry data is written.
 * The footer generation count is always written after the telemetry data.
 */
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_OFST 16
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_LEN 8
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_LO_OFST 16
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_LO_LEN 4
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_LO_LBN 128
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_LO_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_HI_OFST 20
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_HI_LEN 4
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_HI_LBN 160
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_HI_WIDTH 32
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_LBN 128
#define IFOE_TELEM_TLV_DATASET_FTR_GEN_COUNT_WIDTH 64
