/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/multiplane.cc
// (NET_IB_CAST_MULTIPLANE_CC_PATH): the VIP-to-PIP map file, parsed by the real xml.cc.

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <stdlib.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "../common/LogCapture.hpp"
#include "fakes/env_fakes.h"
#include "net-ib-cast-test-common.h"

#include NET_IB_CAST_MULTIPLANE_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;
using RcclUnitTesting::ScopedDebugLogging;

class NetIbCastMultiplaneMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    char tmpl[] = "/tmp/net-ib-cast-multiplane-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    dir_ = tmpl;
  }
  void TearDown() override {
    ncclIbCastTestMultiplaneReset();
    ResetEnvFakes();
    std::filesystem::remove_all(dir_);
  }

  // Writes the map file and points RCCL_MULTIPLANE_MAP_FILE at it.
  void UseMap(const std::string& xml) {
    const std::string path = dir_ + "/map.xml";
    std::ofstream(path) << xml;
    SetMicroEnv("RCCL_MULTIPLANE_MAP_FILE", path.c_str());
  }

  static union ibv_gid Gid(const char* ip) {
    union ibv_gid gid;
    EXPECT_EQ(ibCastIpToGid(ip, &gid), ncclSuccess) << ip;
    return gid;
  }

  static bool SameGid(const union ibv_gid& gid, const char* ip) {
    union ibv_gid expected = Gid(ip);
    return memcmp(&gid, &expected, sizeof(gid)) == 0;
  }

  int Resolve(const char* vip, union ibv_gid* pips) {
    union ibv_gid vipGid = Gid(vip);
    int nPips = -1;
    EXPECT_EQ(IbCastMultiplaneGetPipGids(&vipGid, pips, &nPips), ncclSuccess);
    return nPips;
  }

  std::string dir_;
};

constexpr const char* kTwoInterfaces =
    "<multiplane>\n"
    "  <host name=\"node0\">\n"
    "    <interface gid=\"fd00::1\">\n"
    "      <pip ip=\"10.0.0.1\" interface=\"eth1\"/>\n"
    "      <pip ip=\"10.0.1.1\" interface=\"eth2\"/>\n"
    "    </interface>\n"
    "    <interface gid=\"192.168.7.9\">\n"
    "      <pip ip=\"fd01::5\"/>\n"
    "    </interface>\n"
    "  </host>\n"
    "</multiplane>\n";

TEST_F(NetIbCastMultiplaneMicrotest, IpToGid_Ipv6_CopiesTheAddress) {
  struct in6_addr expected;
  ASSERT_EQ(inet_pton(AF_INET6, "fe80::1:2", &expected), 1);
  uint8_t gid[16];
  ASSERT_EQ(ncclIbCastTestIpToGid("fe80::1:2", gid), ncclSuccess);
  EXPECT_EQ(memcmp(gid, &expected, sizeof(gid)), 0);
}

TEST_F(NetIbCastMultiplaneMicrotest, IpToGid_Ipv4_StoresTheV4MappedForm) {
  uint8_t gid[16];
  ASSERT_EQ(ncclIbCastTestIpToGid("10.1.2.3", gid), ncclSuccess);
  const uint8_t expected[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 1, 2, 3};
  EXPECT_EQ(memcmp(gid, expected, sizeof(gid)), 0);
}

TEST_F(NetIbCastMultiplaneMicrotest, IpToGid_NotAnAddress_IsRejected) {
  uint8_t gid[16];
  memset(gid, 0xab, sizeof(gid));
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclIbCastTestIpToGid("10.1.2", gid), ncclInvalidArgument); });
  EXPECT_TRUE(LogHas(log, "invalid IP address '10.1.2'")) << log;
  EXPECT_EQ(gid[0], 0xab);  // left alone on failure
}

TEST_F(NetIbCastMultiplaneMicrotest, GidToString_PrintsEightLowercaseGroups) {
  uint8_t gid[16];
  for (int i = 0; i < 16; i++) gid[i] = (uint8_t)(0xA0 + i);
  char buf[64];
  ncclIbCastTestGidToString(gid, buf, sizeof(buf));
  EXPECT_STREQ(buf, "a0a1:a2a3:a4a5:a6a7:a8a9:aaab:acad:aeaf");
}

TEST_F(NetIbCastMultiplaneMicrotest, Load_NoMapFile_SucceedsAndResolvesNothing) {
  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);
  union ibv_gid pips[MULTIPLANE_MAX_PIPS];
  EXPECT_EQ(Resolve("fd00::1", pips), 0);
}

TEST_F(NetIbCastMultiplaneMicrotest, Load_EmptyMapFileName_IsTreatedAsUnset) {
  SetMicroEnv("RCCL_MULTIPLANE_MAP_FILE", "");
  EXPECT_EQ(IbCastMultiplaneLoad(), ncclSuccess);
  union ibv_gid pips[MULTIPLANE_MAX_PIPS];
  EXPECT_EQ(Resolve("fd00::1", pips), 0);
}

TEST_F(NetIbCastMultiplaneMicrotest, Load_MissingFile_FailsAndKeepsFailingWithoutRereading) {
  const std::string path = dir_ + "/absent.xml";
  SetMicroEnv("RCCL_MULTIPLANE_MAP_FILE", path.c_str());
  std::string log = CaptureLog([&] { EXPECT_EQ(IbCastMultiplaneLoad(), ncclSystemError); });
  EXPECT_TRUE(LogHas(log, "could not open map file")) << log;
  // The load runs once: creating the file afterwards changes nothing.
  std::ofstream(path) << kTwoInterfaces;
  EXPECT_EQ(IbCastMultiplaneLoad(), ncclSystemError);
}

TEST_F(NetIbCastMultiplaneMicrotest, Load_ValidMap_ResolvesEachVipToItsPipsInOrder) {
  UseMap(kTwoInterfaces);
  ScopedDebugLogging info(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = CaptureLog([&] { ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "loaded 2 GID mappings")) << log;
  EXPECT_FALSE(LogHas(log, "duplicate")) << log;
  EXPECT_TRUE(LogHas(log, "ip=10.0.0.1 interface=eth1")) << log;

  union ibv_gid pips[MULTIPLANE_MAX_PIPS];
  ASSERT_EQ(Resolve("fd00::1", pips), 2);
  EXPECT_TRUE(SameGid(pips[0], "10.0.0.1"));
  EXPECT_TRUE(SameGid(pips[1], "10.0.1.1"));
  ASSERT_EQ(Resolve("192.168.7.9", pips), 1);
  EXPECT_TRUE(SameGid(pips[0], "fd01::5"));
}

TEST_F(NetIbCastMultiplaneMicrotest, Load_VipKeyIsCanonicalized_LongFormAndUppercaseMatch) {
  UseMap("<multiplane><host><interface gid=\"FD00:0:0:0:0:0:0:1\"><pip ip=\"10.0.0.1\"/>"
         "</interface></host></multiplane>");
  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);
  union ibv_gid pips[MULTIPLANE_MAX_PIPS];
  EXPECT_EQ(Resolve("fd00::1", pips), 1);
}

TEST_F(NetIbCastMultiplaneMicrotest, GetPipGids_UnknownVip_ResolvesNothing) {
  UseMap(kTwoInterfaces);
  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);
  union ibv_gid pips[MULTIPLANE_MAX_PIPS];
  EXPECT_EQ(Resolve("fd00::2", pips), 0);
}

TEST_F(NetIbCastMultiplaneMicrotest, GetPipGids_MorePipsThanTheMax_ReturnsOnlyTheFirstMax) {
  std::string xml = "<multiplane><host><interface gid=\"fd00::1\">";
  for (int i = 0; i <= MULTIPLANE_MAX_PIPS; i++) xml += "<pip ip=\"10.0.0." + std::to_string(i + 1) + "\"/>";
  xml += "</interface></host></multiplane>";
  UseMap(xml);
  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);
  union ibv_gid pips[MULTIPLANE_MAX_PIPS + 1];
  memset(pips, 0, sizeof(pips));
  ASSERT_EQ(Resolve("fd00::1", pips), MULTIPLANE_MAX_PIPS);
  const std::string last = "10.0.0." + std::to_string(MULTIPLANE_MAX_PIPS);
  EXPECT_TRUE(SameGid(pips[MULTIPLANE_MAX_PIPS - 1], last.c_str()));
  union ibv_gid zero;
  memset(&zero, 0, sizeof(zero));
  EXPECT_EQ(memcmp(&pips[MULTIPLANE_MAX_PIPS], &zero, sizeof(zero)), 0);
}

TEST_F(NetIbCastMultiplaneMicrotest, Load_PipWithoutInterface_LeavesTheInterfaceEmpty) {
  UseMap("<multiplane><host><interface gid=\"fd00::1\"><pip ip=\"10.0.0.1\"/></interface></host></multiplane>");
  ScopedDebugLogging info(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = CaptureLog([&] { ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "ip=10.0.0.1 interface=\n")) << log;
}

TEST_F(NetIbCastMultiplaneMicrotest, Load_DuplicateVip_LastEntryWins) {
  UseMap("<multiplane><host>"
         "<interface gid=\"fd00::1\"><pip ip=\"10.0.0.1\"/></interface>"
         "<interface gid=\"fd00:0::1\"><pip ip=\"10.0.0.2\"/><pip ip=\"10.0.0.3\"/></interface>"
         "</host></multiplane>");
  std::string log = CaptureLog([&] { ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "duplicate GID")) << log;
  union ibv_gid pips[MULTIPLANE_MAX_PIPS];
  ASSERT_EQ(Resolve("fd00::1", pips), 2);
  EXPECT_TRUE(SameGid(pips[0], "10.0.0.2"));
}

struct BadMap {
  const char* name;
  const char* xml;
  const char* warning;
};

class NetIbCastMultiplaneBadMapMicrotest : public NetIbCastMultiplaneMicrotest,
                                           public ::testing::WithParamInterface<BadMap> {};

TEST_P(NetIbCastMultiplaneBadMapMicrotest, Load_IsRejectedAndNothingResolves) {
  UseMap(GetParam().xml);
  std::string log = CaptureLog([&] { EXPECT_NE(IbCastMultiplaneLoad(), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, GetParam().warning)) << log;
  union ibv_gid pips[MULTIPLANE_MAX_PIPS];
  EXPECT_EQ(Resolve("fd00::1", pips), 0);
}

INSTANTIATE_TEST_SUITE_P(
    NetIbCastMultiplane, NetIbCastMultiplaneBadMapMicrotest,
    ::testing::Values(
        BadMap{"NoRoot", "<other><host/></other>", "no <multiplane> root element"},
        BadMap{"InterfaceWithoutGid",
               "<multiplane><host><interface><pip ip=\"10.0.0.1\"/></interface></host></multiplane>",
               "<interface> missing 'gid'"},
        BadMap{"PipWithoutIp",
               "<multiplane><host><interface gid=\"fd00::1\"><pip interface=\"eth1\"/></interface></host>"
               "</multiplane>",
               "<pip> missing 'ip'"},
        BadMap{"BadPipIp",
               "<multiplane><host><interface gid=\"fd00::1\"><pip ip=\"10.0.0.300\"/></interface></host>"
               "</multiplane>",
               "invalid IP address '10.0.0.300'"},
        BadMap{"BadVip",
               "<multiplane><host><interface gid=\"fd00::g\"><pip ip=\"10.0.0.1\"/></interface></host>"
               "</multiplane>",
               "invalid IP address 'fd00::g'"},
        BadMap{"Unterminated", "<multiplane><host>", "failed to parse map file"}),
    [](const ::testing::TestParamInfo<BadMap>& info) { return std::string(info.param.name); });

// The parser only hands back the elements it has handlers for, so the "skip anything else"
// branches of the tree walk are driven with a hand-built tree.
class NetIbCastMultiplaneTreeMicrotest : public NetIbCastMultiplaneMicrotest {
 protected:
  struct ncclXmlNode* Add(const char* name, struct ncclXmlNode* parent) {
    struct ncclXmlNode* node = &xml_->nodes[xml_->maxIndex++];
    strcpy(node->name, name);
    if (parent) parent->subs[parent->nSubs++] = node;
    return node;
  }
  void SetUp() override {
    NetIbCastMultiplaneMicrotest::SetUp();
    ASSERT_EQ(xmlAlloc(&xml_, 16), ncclSuccess);
  }
  void TearDown() override {
    free(xml_);
    NetIbCastMultiplaneMicrotest::TearDown();
  }
  struct ncclXml* xml_ = nullptr;
};

TEST_F(NetIbCastMultiplaneTreeMicrotest, PopulateMap_SkipsElementsThatAreNotHostInterfaceOrPip) {
  // Each stray element holds what would fail the load if it were walked into.
  struct ncclXmlNode* root = Add("multiplane", nullptr);
  struct ncclXmlNode* comment = Add("comment", root);
  Add("interface", comment);  // no gid
  struct ncclXmlNode* host = Add("host", root);
  struct ncclXmlNode* note = Add("note", host);
  Add("pip", note);  // no ip
  struct ncclXmlNode* iface = Add("interface", host);
  ASSERT_EQ(xmlSetAttr(iface, "gid", "fd00::1"), ncclSuccess);
  struct ncclXmlNode* label = Add("label", iface);
  ASSERT_EQ(xmlSetAttr(label, "ip", "not-an-ip"), ncclSuccess);
  struct ncclXmlNode* pip = Add("pip", iface);
  ASSERT_EQ(xmlSetAttr(pip, "ip", "10.0.0.1"), ncclSuccess);

  ASSERT_EQ(ibCastMultiplanePopulateMap(xml_), ncclSuccess);
  ASSERT_EQ(gidToPipMap.size(), 1u);
  EXPECT_EQ(gidToPipMap.begin()->second.size(), 1u);
}

TEST_F(NetIbCastMultiplaneMicrotest, Enabled_ReportsTheInitTimeFlagAndResetClearsIt) {
  bool enabled = false;
  IbCastMultiplaneEnable = true;
  ASSERT_EQ(IbCastMultiplaneEnabled(&enabled), ncclSuccess);
  EXPECT_TRUE(enabled);
  ncclIbCastTestMultiplaneReset();
  ASSERT_EQ(IbCastMultiplaneEnabled(&enabled), ncclSuccess);
  EXPECT_FALSE(enabled);
}

}  // namespace
