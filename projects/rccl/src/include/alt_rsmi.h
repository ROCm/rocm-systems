/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __ALT_RSMI_H__
#define __ALT_RSMI_H__

/*
** This is a light-weight implementation of the RSMI functionality used in RCCL
** The code is based on the actual rocm_smi_library code, but extracted to contain only
** the bits actually required by RCCL.
*/

#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <map>
#include <cassert>
#include <algorithm>
#include <iomanip>

/**
 ** This is an exact copy of the IO Link types from rocm_smi.h
 ** These definitions are required since we do not know whether the
 ** code will also be compiled such that it includes the rocm_smi.h
 ** file or not. The values have to be identical however
 */
typedef enum _ARSMI_IO_LINK_TYPE {
  ARSMI_IOLINK_TYPE_UNDEFINED      = 0,          //!< unknown type.
  ARSMI_IOLINK_TYPE_PCIEXPRESS,                  //!< PCI Express
  ARSMI_IOLINK_TYPE_XGMI,                        //!< XGMI
  ARSMI_IOLINK_TYPE_NUMIOLINKTYPES,              //!< Number of IO Link types
  ARSMI_IOLINK_TYPE_SIZE           = 0xFFFFFFFF  //!< Max of IO Link types
} ARSMI_IO_LINK_TYPE;

struct ARSMI_linkInfo {
    uint32_t src_node;
    uint32_t dst_node;
    uint64_t hops;
    ARSMI_IO_LINK_TYPE type;
    uint64_t weight;
    uint64_t min_bandwidth;
    uint64_t max_bandwidth;
};
typedef struct ARSMI_linkInfo ARSMI_linkInfo;

int ARSMI_init (void);
int ARSMI_get_num_devices (uint32_t *num_devices);
int ARSMI_dev_pci_id_get(uint32_t dv_ind, uint64_t *bdfid);
int ARSMI_topo_get_link_info(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                             ARSMI_linkInfo *info);

// Firmware version - reads MEC firmware from sysfs
int ARSMI_get_fw_version(uint32_t dv_ind, uint64_t *fw_version);

/*
 * Fabric support via /sys/class/drm/<card>/device/ualink/
 *
 * Enum values are chosen to match their amdsmi counterparts so that
 * amdsmi_wrap.cc can cast between the two without a conversion table.
 */

typedef enum {
    ARSMI_FABRIC_TYPE_UALOE   = 0,
    ARSMI_FABRIC_TYPE_UALLINK = 1,
    ARSMI_FABRIC_TYPE_UNKNOWN = 2
} ARSMI_fabric_type_t;

typedef enum {
    ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNCONFIGURED = 0,
    ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED   = 1,
    ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY        = 2,
    ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE       = 3,
    ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR        = 4,
    ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNKNOWN      = 5
} ARSMI_fabric_accelerator_vpod_state_t;

typedef enum {
    ARSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING       = 0,
    ARSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_IDENTIFICATION = 1,
    ARSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN               = 2
} ARSMI_fabric_npa_address_mode_t;

struct ARSMI_fabricInfo {
    int                                  supported;   /* 1 if UALoE or UALLink and state is ready/active */
    ARSMI_fabric_type_t                  fabric_type;
    ARSMI_fabric_accelerator_vpod_state_t accel_state;
    ARSMI_fabric_npa_address_mode_t      addr_mode;
    uint32_t  accel_id;
    uint8_t   ppod_id[16];   /* 128-bit UUID, binary (parsed from sysfs hex string) */
    uint32_t  ppod_size;
    uint32_t  bandwidth;     /* Mb/s */
    uint32_t  latency;       /* ns */
    uint32_t  vpod_id;
    uint32_t  vpod_size;
};

/* Read fabric info for device dv_ind from /sys/class/drm/<card>/device/ualink/.
 * Returns 0 on success, ENODEV if no ualink directory exists for that device,
 * EINVAL on bad arguments. */
int ARSMI_get_fabric_info(uint32_t dv_ind, struct ARSMI_fabricInfo *info);

const char* ARSMI_fabric_telem_id_to_string(uint64_t telem_id);

#endif
