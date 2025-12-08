/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_IO_LINK_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_IO_LINK_H_

#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <map>
#include <utility>

#include "rocm_smi/rocm_smi.h"

namespace amd {
namespace smi {

typedef enum _IO_LINK_TYPE {
  IOLINK_TYPE_UNDEFINED      = 0,
  IOLINK_TYPE_HYPERTRANSPORT = 1,
  IOLINK_TYPE_PCIEXPRESS     = 2,
  IOLINK_TYPE_AMBA           = 3,
  IOLINK_TYPE_MIPI           = 4,
  IOLINK_TYPE_QPI_1_1        = 5,
  IOLINK_TYPE_RESERVED1      = 6,
  IOLINK_TYPE_RESERVED2      = 7,
  IOLINK_TYPE_RAPID_IO       = 8,
  IOLINK_TYPE_INFINIBAND     = 9,
  IOLINK_TYPE_RESERVED3      = 10,
  IOLINK_TYPE_XGMI           = 11,
  IOLINK_TYPE_XGOP           = 12,
  IOLINK_TYPE_GZ             = 13,
  IOLINK_TYPE_ETHERNET_RDMA  = 14,
  IOLINK_TYPE_RDMA_OTHER     = 15,
  IOLINK_TYPE_OTHER          = 16,
  IOLINK_TYPE_NUMIOLINKTYPES,
  IOLINK_TYPE_SIZE           = 0xFFFFFFFF
} IO_LINK_TYPE;

typedef enum _LINK_DIRECTORY_TYPE {
  IO_LINK_DIRECTORY          = 0,
  P2P_LINK_DIRECTORY         = 1
} LINK_DIRECTORY_TYPE;

class IOLink {
 public:
    explicit IOLink(uint32_t node_indx, uint32_t link_indx, LINK_DIRECTORY_TYPE link_dir_type) :
                    node_indx_(node_indx), link_indx_(link_indx), link_dir_type_(link_dir_type) {}
    ~IOLink();

    int Initialize();
    int ReadProperties(void);
    int get_property_value(std::string property, uint64_t *value);
    uint32_t get_node_indx(void) const {return node_indx_;}
    uint32_t get_link_indx(void) const {return link_indx_;}
    IO_LINK_TYPE type(void) const {return type_;}
    uint32_t node_from(void) const {return node_from_;}
    uint32_t node_to(void) const {return node_to_;}
    uint64_t weight(void) const {return weight_;}
    LINK_DIRECTORY_TYPE get_directory_type(void) const {return link_dir_type_;}
    uint64_t min_bandwidth(void) const {return min_bandwidth_;}
    uint64_t max_bandwidth(void) const {return max_bandwidth_;}


 private:
    uint32_t node_indx_;
    uint32_t link_indx_;
    IO_LINK_TYPE type_;
    uint32_t node_from_;
    uint32_t node_to_;
    uint64_t weight_;
    uint64_t min_bandwidth_;
    uint64_t max_bandwidth_;
    std::map<std::string, uint64_t> properties_;
    LINK_DIRECTORY_TYPE link_dir_type_;
};

int
DiscoverIOLinksPerNode(uint32_t node_indx, std::map<uint32_t,
                       std::shared_ptr<IOLink>> *links);

int
DiscoverP2PLinksPerNode(uint32_t node_indx, std::map<uint32_t,
                        std::shared_ptr<IOLink>> *links);

int
DiscoverIOLinks(std::map<std::pair<uint32_t, uint32_t>,
                std::shared_ptr<IOLink>> *links);

int
DiscoverP2PLinks(std::map<std::pair<uint32_t, uint32_t>,
                 std::shared_ptr<IOLink>> *links);

}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_IO_LINK_H_
