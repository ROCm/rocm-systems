/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <string>

#include "graph/topo.h"
#include "graph/xml.h"

namespace RcclUnitTesting {

// ncclTopoGetSystemFromXml() requires rail/plane on every net/gin/rma node, which the library only
// adds while enumerating NICs through the net plugin (ncclTopoProcessNet). Topology XML loaded
// straight from a file or built in code skips that, so assign the same defaults here: one rail per
// distinct NIC on each host, in order of appearance, and the plane from the port.
inline ncclResult_t fillMissingRailPlane(struct ncclXml* xml) {
  std::map<std::string, std::map<std::string, int>> railKeys;  // (tag, host) -> NIC ASIC -> rail index
  for (int i = 0; i < xml->maxIndex; i++) {
    struct ncclXmlNode* node = xml->nodes + i;
    if (strcmp(node->name, "net") != 0 && strcmp(node->name, "gin") != 0 && strcmp(node->name, "rma") != 0) continue;
    int rail, plane, port;
    NCCLCHECK(xmlGetAttrIntDefault(node, "rail", &rail, NCCL_TOPO_UNDEF));
    NCCLCHECK(xmlGetAttrIntDefault(node, "plane", &plane, NCCL_TOPO_UNDEF));
    NCCLCHECK(xmlGetAttrIntDefault(node, "port", &port, 0));
    if (rail == NCCL_TOPO_UNDEF) {
      const char* hostHash = nullptr;
      for (struct ncclXmlNode* p = node->parent; p != nullptr && hostHash == nullptr; p = p->parent) {
        if (strcmp(p->name, "cpu") == 0) NCCLCHECK(xmlGetAttr(p, "host_hash", &hostHash));
      }
      // Without a GUID each net is its own ASIC, as ncclTopoAddNetAsic() assumes.
      const char* guidStr;
      NCCLCHECK(xmlGetAttr(node, "guid", &guidStr));
      std::string asic;
      if (guidStr) {
        uint64_t guid;
        NCCLCHECK(xmlGetAttrUint64Default(node, "guid", &guid, 0));
        asic = "guid:" + std::to_string(guid);
      } else {
        int dev;
        NCCLCHECK(xmlGetAttrIntDefault(node, "dev", &dev, i));
        asic = "dev:" + std::to_string(dev);
      }
      auto& keys = railKeys[std::string(node->name) + "/" + (hostHash ? hostHash : "")];
      int index = keys.emplace(asic, static_cast<int>(keys.size())).first->second;
      NCCLCHECK(xmlSetAttrInt(node, "rail", NCCL_TOPO_UNDEF_BIT | index));
    }
    if (plane == NCCL_TOPO_UNDEF) NCCLCHECK(xmlSetAttrInt(node, "plane", NCCL_TOPO_UNDEF_BIT | port));
  }
  return ncclSuccess;
}

} // namespace RcclUnitTesting
