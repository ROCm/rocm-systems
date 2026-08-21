/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "multiplane.h"
#include "core.h"

#include <arpa/inet.h>
#include <map>
#include <string>
#include <vector>
#include <mutex>

// Forward declarations for XML parser internals (defined in xml.cc, external linkage)
typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclXml*, struct ncclXmlNode*);
struct xmlHandler {
  const char* name;
  xmlHandlerFunc_t func;
};
ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head,
                        struct xmlHandler handlers[], int nHandlers);

// Module-scoped VIP GID string -> list of PIP entries
static std::map<std::string, std::vector<ncclIbPipInfo>> gidToPipMap;
static bool multiplaneLoaded = false;
static std::once_flag loadOnceFlag;
static ncclResult_t loadResult = ncclSuccess;

// Convert a 16-byte GID to colon-separated hex string (e.g. "fe80:0000:...:0002")
static void ibCastGidToString(const union ibv_gid* gid, char* buf, size_t bufLen) {
  snprintf(buf, bufLen,
    "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",
    gid->raw[0],  gid->raw[1],  gid->raw[2],  gid->raw[3],
    gid->raw[4],  gid->raw[5],  gid->raw[6],  gid->raw[7],
    gid->raw[8],  gid->raw[9],  gid->raw[10], gid->raw[11],
    gid->raw[12], gid->raw[13], gid->raw[14], gid->raw[15]);
}

// Convert an IPv4 or IPv6 address string to an ibv_gid.
// IPv4 addresses are stored as IPv4-mapped IPv6 (::ffff:x.x.x.x).
static ncclResult_t ibCastIpToGid(const char* ipStr, union ibv_gid* gid) {
  memset(gid, 0, sizeof(union ibv_gid));

  // Try IPv6 first
  struct in6_addr addr6;
  if (inet_pton(AF_INET6, ipStr, &addr6) == 1) {
    memcpy(gid->raw, &addr6, 16);
    return ncclSuccess;
  }

  // Try IPv4 — store as ::ffff:x.x.x.x
  struct in_addr addr4;
  if (inet_pton(AF_INET, ipStr, &addr4) == 1) {
    gid->raw[10] = 0xff;
    gid->raw[11] = 0xff;
    memcpy(&gid->raw[12], &addr4, 4);
    return ncclSuccess;
  }

  WARN("Multiplane: invalid IP address '%s'", ipStr);
  return ncclInvalidArgument;
}

// XML handler for <pip> elements (leaf, no children)
static ncclResult_t xmlLoadPip(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

// XML handler for <interface> elements, containing <pip> children
static ncclResult_t xmlLoadInterface(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"pip", xmlLoadPip}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

// XML handler for <host> elements, containing <interface> children
static ncclResult_t xmlLoadHost(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"interface", xmlLoadInterface}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

// XML handler for <multiplane> root element, containing <host> children
static ncclResult_t xmlLoadMultiplane(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"host", xmlLoadHost}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

// Walk the parsed XML tree and populate gidToPipMap
static ncclResult_t ibCastMultiplanePopulateMap(struct ncclXml* xml) {
  // Find the <multiplane> root node
  struct ncclXmlNode* root = NULL;
  for (int i = 0; i < xml->maxIndex; i++) {
    if (strcmp(xml->nodes[i].name, "multiplane") == 0) {
      root = &xml->nodes[i];
      break;
    }
  }
  if (root == NULL) {
    WARN("Multiplane: no <multiplane> root element found");
    return ncclInvalidArgument;
  }

  // Iterate <host> -> <interface> -> <pip>
  for (int h = 0; h < root->nSubs; h++) {
    struct ncclXmlNode* hostNode = root->subs[h];
    if (strcmp(hostNode->name, "host") != 0) continue;

    for (int i = 0; i < hostNode->nSubs; i++) {
      struct ncclXmlNode* ifNode = hostNode->subs[i];
      if (strcmp(ifNode->name, "interface") != 0) continue;

      const char* gidStr = NULL;
      NCCLCHECK(xmlGetAttr(ifNode, "gid", &gidStr));
      if (gidStr == NULL) {
        WARN("Multiplane: <interface> missing 'gid' attribute");
        return ncclInvalidArgument;
      }

      std::vector<ncclIbPipInfo> pips;
      for (int p = 0; p < ifNode->nSubs; p++) {
        struct ncclXmlNode* pipNode = ifNode->subs[p];
        if (strcmp(pipNode->name, "pip") != 0) continue;

        const char* pipIp = NULL;
        const char* pipIface = NULL;
        NCCLCHECK(xmlGetAttr(pipNode, "ip", &pipIp));
        NCCLCHECK(xmlGetAttr(pipNode, "interface", &pipIface));
        if (pipIp == NULL) {
          WARN("Multiplane: <pip> missing 'ip' attribute");
          return ncclInvalidArgument;
        }

        ncclIbPipInfo info = {};
        strncpy(info.ip, pipIp, MAX_STR_LEN - 1);
        info.ip[MAX_STR_LEN - 1] = '\0';
        if (pipIface) {
          strncpy(info.interface, pipIface, MAX_STR_LEN - 1);
          info.interface[MAX_STR_LEN - 1] = '\0';
        }
        pips.push_back(info);
      }

      std::string key(gidStr);
      gidToPipMap[key] = pips;
      INFO(NCCL_NET, "Multiplane: GID %s -> %zu PIPs", gidStr, pips.size());
      for (size_t p = 0; p < pips.size(); p++) {
        INFO(NCCL_NET, "  PIP[%zu]: ip=%s interface=%s", p, pips[p].ip, pips[p].interface);
      }
    }
  }
  return ncclSuccess;
}

static void ibCastMultiplaneLoadOnce() {
  const char* mapFile = getenv("RCCL_MULTIPLANE_MAP_FILE");
  if (mapFile == NULL || mapFile[0] == '\0') {
    loadResult = ncclSuccess;
    return;
  }

  INFO(NCCL_NET, "Multiplane: loading map file %s", mapFile);

  FILE* file = fopen(mapFile, "r");
  if (file == NULL) {
    WARN("Multiplane: could not open map file %s: %s", mapFile, strerror(errno));
    loadResult = ncclSystemError;
    return;
  }

  struct ncclXml* xml = NULL;
  // Allocate enough nodes for a reasonable map file
  loadResult = xmlAlloc(&xml, 1024);
  if (loadResult != ncclSuccess) {
    fclose(file);
    return;
  }
  xml->maxIndex = 0;

  struct xmlHandler handlers[] = {{"multiplane", xmlLoadMultiplane}};
  loadResult = xmlLoadSub(file, xml, NULL, handlers, 1);
  fclose(file);

  if (loadResult != ncclSuccess) {
    WARN("Multiplane: failed to parse map file %s", mapFile);
    free(xml);
    return;
  }

  loadResult = ibCastMultiplanePopulateMap(xml);
  free(xml);

  if (loadResult == ncclSuccess) {
    multiplaneLoaded = true;
    INFO(NCCL_NET, "Multiplane: loaded %zu GID mappings", gidToPipMap.size());
  }
}

ncclResult_t ibCastMultiplaneLoad(void) {
  std::call_once(loadOnceFlag, ibCastMultiplaneLoadOnce);
  return loadResult;
}

ncclResult_t ibCastMultiplaneEnabled(bool* enabled) {
  const char* mapFile = getenv("RCCL_MULTIPLANE_MAP_FILE");
  *enabled = (mapFile != NULL && mapFile[0] != '\0');
  return ncclSuccess;
}

ncclResult_t ibCastMultiplaneGetPipGids(const union ibv_gid* vipGid, union ibv_gid* pipGids, int* nPips) {
  *nPips = 0;
  if (!multiplaneLoaded) return ncclSuccess;

  char gidStr[64];
  ibCastGidToString(vipGid, gidStr, sizeof(gidStr));

  auto it = gidToPipMap.find(std::string(gidStr));
  if (it == gidToPipMap.end()) {
    INFO(NCCL_NET, "Multiplane: no PIP mapping found for GID %s", gidStr);
    return ncclSuccess;
  }

  const std::vector<ncclIbPipInfo>& pips = it->second;
  int count = (int)pips.size();
  if (count > MULTIPLANE_MAX_PIPS) count = MULTIPLANE_MAX_PIPS;

  for (int i = 0; i < count; i++) {
    NCCLCHECK(ibCastIpToGid(pips[i].ip, &pipGids[i]));
  }
  *nPips = count;
  INFO(NCCL_NET, "Multiplane: GID %s resolved to %d PIPs", gidStr, count);
  return ncclSuccess;
}
