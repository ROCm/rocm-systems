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

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_IO_LINK_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_IO_LINK_H_

#include <string>
#include <memory>
#include <map>
#include <utility>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi {

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

enum class IOLinkDirectionType_t
{
    kNonDirectional = 0,
    kUniDirectional = 1,
    kBiDirectional = 2,
};


class IOLink {
 public:
    explicit IOLink(uint32_t node_indx, uint32_t link_indx, LINK_DIRECTORY_TYPE link_dir_type) :
                    node_indx_(node_indx), link_indx_(link_indx), link_dir_type_(link_dir_type),
                    link_cap_{UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX} {}
    ~IOLink();

    int Initialize();
    int ReadProperties(void);
    int get_property_value(std::string property, uint64_t *value);
    uint32_t get_node_indx(void) const {return node_indx_;}
    uint32_t get_link_indx(void) const {return link_indx_;}
    IO_LINK_TYPE type(void) const {return type_;}
    uint32_t node_from(void) const {return node_from_;}
    uint32_t node_to(void) const {return node_to_;}
    uint32_t flag(void) const {return flags_;}
    uint64_t weight(void) const {return weight_;}
    LINK_DIRECTORY_TYPE get_directory_type(void) const {return link_dir_type_;}
    uint64_t min_bandwidth(void) const {return min_bandwidth_;}
    uint64_t max_bandwidth(void) const {return max_bandwidth_;}
    const rsmi_p2p_capability_t& get_link_capability(void) const {return link_cap_;}

 protected:
    virtual int UpdateP2pCapability(void);
 private:
    uint32_t node_indx_;
    uint32_t link_indx_;
    IO_LINK_TYPE type_;
    uint32_t node_from_;
    uint32_t node_to_;
    uint32_t flags_;
    uint64_t weight_;
    uint64_t min_bandwidth_;
    uint64_t max_bandwidth_;
    std::map<std::string, uint64_t> properties_;
    LINK_DIRECTORY_TYPE link_dir_type_;
    rsmi_p2p_capability_t link_cap_;
};

using IOLinksPerNodeList_t = std::map<uint32_t, std::shared_ptr<IOLink>>;
using IOLinksPerNodeListPtr_t = IOLinksPerNodeList_t*;

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

IOLinkDirectionType_t DiscoverIOLinkPerNodeDirection(uint32_t src_node_idx,
                                                     uint32_t dst_node_idx);


} // namespace amd::smi

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_IO_LINK_H_
