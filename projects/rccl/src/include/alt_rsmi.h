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

#endif
