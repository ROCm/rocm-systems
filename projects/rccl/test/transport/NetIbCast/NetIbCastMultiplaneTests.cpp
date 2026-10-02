/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Host-only unit tests for the multiplane VIP-to-PIP mapping module
 * (src/transport/net_ib_cast/multiplane.cc).  No MPI, no RDMA verbs,
 * no NIC hardware required.
 *
 * Because multiplane.cc uses std::call_once for one-shot loading, tests
 * that need different module states (enabled vs disabled, valid file vs
 * missing file) run in forked child processes via ProcessIsolatedTestRunner.
 * The "post-load lookup" tests share a single fixture that loads once and
 * verifies multiple lookup scenarios.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <gtest/gtest.h>

#include "net_ib_cast_inspect.h"

// multiplane public API — include the actual header for ibv_gid and function declarations
#include "multiplane.h"
#include "common_cast.h"  // for IbCastAinicRoce

namespace {

// ─── Helper: build an IPv4-mapped GID (::ffff:a.b.c.d) ─────────────────
static void MakeGidV4Mapped(uint8_t g[16], uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  memset(g, 0, 16);
  g[10] = 0xff;
  g[11] = 0xff;
  g[12] = a; g[13] = b; g[14] = c; g[15] = d;
}

// ─── Helper: write a minimal multiplane XML to a temp file ──────────────
static std::string WriteTestXml(const char* xml) {
  char path[] = "/tmp/rccl_mp_test_XXXXXX";
  int fd = mkstemp(path);
  EXPECT_NE(fd, -1) << "mkstemp failed";
  FILE* f = fdopen(fd, "w");
  EXPECT_NE(f, nullptr);
  fputs(xml, f);
  fclose(f);
  return std::string(path);
}

// =====================================================================
// 1. ibCastIpToGid / ibCastGidToString  (via inspect wrappers)
// =====================================================================

TEST(NetIbCastMultiplane, IpToGidIPv4) {
  uint8_t gid[16] = {};
  ASSERT_EQ(ncclIbCastTestIpToGid("50.1.1.2", gid), ncclSuccess);

  // Expect ::ffff:32.01.01.02
  uint8_t expected[16] = {};
  MakeGidV4Mapped(expected, 50, 1, 1, 2);
  EXPECT_EQ(memcmp(gid, expected, 16), 0);
}

TEST(NetIbCastMultiplane, IpToGidIPv6) {
  uint8_t gid[16] = {};
  ASSERT_EQ(ncclIbCastTestIpToGid("fe80::1", gid), ncclSuccess);

  // fe80::1 = fe80:0000:0000:0000:0000:0000:0000:0001
  EXPECT_EQ(gid[0], 0xfe);
  EXPECT_EQ(gid[1], 0x80);
  for (int i = 2; i < 15; i++) EXPECT_EQ(gid[i], 0);
  EXPECT_EQ(gid[15], 0x01);
}

TEST(NetIbCastMultiplane, IpToGidInvalid) {
  uint8_t gid[16] = {};
  EXPECT_NE(ncclIbCastTestIpToGid("not-an-ip", gid), ncclSuccess);
}

TEST(NetIbCastMultiplane, GidToStringRoundTrip) {
  uint8_t gid[16] = {};
  MakeGidV4Mapped(gid, 50, 1, 0, 2);

  char buf[64];
  ncclIbCastTestGidToString(gid, buf, sizeof(buf));
  // Expected: "0000:0000:0000:0000:0000:ffff:3201:0002"
  EXPECT_STREQ(buf, "0000:0000:0000:0000:0000:ffff:3201:0002");
}

TEST(NetIbCastMultiplane, GidToStringAllZeros) {
  uint8_t gid[16] = {};
  char buf[64];
  ncclIbCastTestGidToString(gid, buf, sizeof(buf));
  EXPECT_STREQ(buf, "0000:0000:0000:0000:0000:0000:0000:0000");
}

// =====================================================================
// 2. IbCastMultiplaneEnabled  (init-time global)
//
// IbCastMultiplaneEnable is set once at RCCL init in init.cc.  In tests
// we simulate this by setting the global directly and verifying that
// IbCastMultiplaneEnabled() reflects it.
// =====================================================================

TEST_F(MultiplaneLoadTest, EnabledWhenGlobalSet) {
  IbCastMultiplaneEnable = true;
  bool enabled = false;
  ASSERT_EQ(IbCastMultiplaneEnabled(&enabled), ncclSuccess);
  EXPECT_TRUE(enabled);
}

TEST_F(MultiplaneLoadTest, DisabledByDefault) {
  // After reset, IbCastMultiplaneEnable is false
  bool enabled = true;
  ASSERT_EQ(IbCastMultiplaneEnabled(&enabled), ncclSuccess);
  EXPECT_FALSE(enabled);
}

// =====================================================================
// 3. Full XML load + PIP GID resolution
//
// We reset the module state, write a small XML, load it, then verify
// GID lookups.  The reset + reload cycle is safe in a single-threaded
// test process.
// =====================================================================

class MultiplaneLoadTest : public ::testing::Test {
protected:
  void SetUp() override {
    ncclIbCastTestMultiplaneReset();
  }
  void TearDown() override {
    unsetenv("RCCL_MULTIPLANE_MAP_FILE");
    ncclIbCastTestMultiplaneReset();
  }
};

// Minimal 2-host, 1-interface-each XML (IPv4 PIPs).
static const char* kMinimalXml =
  "<multiplane>\n"
  "  <host name=\"nodeA\">\n"
  "    <interface gid=\"0000:0000:0000:0000:0000:ffff:3201:0002\" dev=\"ionic_1\">\n"
  "      <pip ip=\"50.1.1.2\" interface=\"eth1\"/>\n"
  "      <pip ip=\"50.1.2.2\" interface=\"eth2\"/>\n"
  "    </interface>\n"
  "  </host>\n"
  "  <host name=\"nodeB\">\n"
  "    <interface gid=\"0000:0000:0000:0000:0000:ffff:3c01:0002\" dev=\"ionic_1\">\n"
  "      <pip ip=\"60.1.1.2\" interface=\"eth1\"/>\n"
  "      <pip ip=\"60.1.2.2\" interface=\"eth2\"/>\n"
  "      <pip ip=\"60.1.3.2\" interface=\"eth3\"/>\n"
  "    </interface>\n"
  "  </host>\n"
  "</multiplane>\n";

TEST_F(MultiplaneLoadTest, LoadValidXmlAndResolvePips) {
  std::string path = WriteTestXml(kMinimalXml);
  setenv("RCCL_MULTIPLANE_MAP_FILE", path.c_str(), 1);

  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);

  // Look up nodeA's interface (VIP GID = ::ffff:50.1.0.2)
  union ibv_gid vipGid;
  memset(&vipGid, 0, sizeof(vipGid));
  // Build the GID for 0000:0000:0000:0000:0000:ffff:3201:0002
  // which is ::ffff:50.1.0.2
  MakeGidV4Mapped(vipGid.raw, 50, 1, 0, 2);

  union ibv_gid pipGids[16];
  int nPips = 0;
  ASSERT_EQ(IbCastMultiplaneGetPipGids(&vipGid, pipGids, &nPips), ncclSuccess);
  EXPECT_EQ(nPips, 2);

  // Verify PIP[0] = ::ffff:50.1.1.2
  uint8_t expected0[16] = {};
  MakeGidV4Mapped(expected0, 50, 1, 1, 2);
  EXPECT_EQ(memcmp(pipGids[0].raw, expected0, 16), 0);

  // Verify PIP[1] = ::ffff:50.1.2.2
  uint8_t expected1[16] = {};
  MakeGidV4Mapped(expected1, 50, 1, 2, 2);
  EXPECT_EQ(memcmp(pipGids[1].raw, expected1, 16), 0);

  remove(path.c_str());
}

TEST_F(MultiplaneLoadTest, LoadValidXmlSecondHost) {
  std::string path = WriteTestXml(kMinimalXml);
  setenv("RCCL_MULTIPLANE_MAP_FILE", path.c_str(), 1);

  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);

  // Look up nodeB's interface (VIP GID = ::ffff:60.1.0.2)
  union ibv_gid vipGid;
  memset(&vipGid, 0, sizeof(vipGid));
  MakeGidV4Mapped(vipGid.raw, 60, 1, 0, 2);

  union ibv_gid pipGids[16];
  int nPips = 0;
  ASSERT_EQ(IbCastMultiplaneGetPipGids(&vipGid, pipGids, &nPips), ncclSuccess);
  EXPECT_EQ(nPips, 3);

  // Verify PIP[2] = ::ffff:60.1.3.2
  uint8_t expected2[16] = {};
  MakeGidV4Mapped(expected2, 60, 1, 3, 2);
  EXPECT_EQ(memcmp(pipGids[2].raw, expected2, 16), 0);

  remove(path.c_str());
}

TEST_F(MultiplaneLoadTest, UnknownVipReturnsZeroPips) {
  std::string path = WriteTestXml(kMinimalXml);
  setenv("RCCL_MULTIPLANE_MAP_FILE", path.c_str(), 1);

  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);

  // GID that is not in the map
  union ibv_gid unknownGid;
  memset(&unknownGid, 0, sizeof(unknownGid));
  MakeGidV4Mapped(unknownGid.raw, 99, 99, 99, 99);

  union ibv_gid pipGids[16];
  int nPips = -1;
  ASSERT_EQ(IbCastMultiplaneGetPipGids(&unknownGid, pipGids, &nPips), ncclSuccess);
  EXPECT_EQ(nPips, 0);

  remove(path.c_str());
}

TEST_F(MultiplaneLoadTest, DisabledReturnsZeroPips) {
  // Explicitly unset to guard against inheriting from the environment
  unsetenv("RCCL_MULTIPLANE_MAP_FILE");
  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);

  union ibv_gid vipGid;
  memset(&vipGid, 0, sizeof(vipGid));
  MakeGidV4Mapped(vipGid.raw, 50, 1, 0, 2);

  union ibv_gid pipGids[16];
  int nPips = -1;
  ASSERT_EQ(IbCastMultiplaneGetPipGids(&vipGid, pipGids, &nPips), ncclSuccess);
  EXPECT_EQ(nPips, 0);
}

TEST_F(MultiplaneLoadTest, MissingFileReturnsError) {
  setenv("RCCL_MULTIPLANE_MAP_FILE", "/tmp/rccl_no_such_file_12345.xml", 1);
  EXPECT_NE(IbCastMultiplaneLoad(), ncclSuccess);
}

// =====================================================================
// 4. IPv4-to-GID byte-level verification using the sample XML's
//    known IP addresses (from vip_pip_mapping.xml)
// =====================================================================

TEST_F(MultiplaneLoadTest, Ipv4MappedGidBytesMatch) {
  // 50.1.1.2 in network byte order = 0x32, 0x01, 0x01, 0x02
  uint8_t gid[16] = {};
  ASSERT_EQ(ncclIbCastTestIpToGid("50.1.1.2", gid), ncclSuccess);

  EXPECT_EQ(gid[10], 0xff);
  EXPECT_EQ(gid[11], 0xff);
  EXPECT_EQ(gid[12], 0x32);  // 50
  EXPECT_EQ(gid[13], 0x01);  // 1
  EXPECT_EQ(gid[14], 0x01);  // 1
  EXPECT_EQ(gid[15], 0x02);  // 2
}

// =====================================================================
// 5. GID string round-trip matches the XML key format
// =====================================================================

TEST_F(MultiplaneLoadTest, GidStringMatchesXmlKey) {
  // The XML uses GID "0000:0000:0000:0000:0000:ffff:3201:0002"
  // which represents ::ffff:50.1.0.2.  Verify our GID-to-string
  // produces the same key the XML parser stores.
  uint8_t gid[16] = {};
  MakeGidV4Mapped(gid, 50, 1, 0, 2);

  char buf[64];
  ncclIbCastTestGidToString(gid, buf, sizeof(buf));
  EXPECT_STREQ(buf, "0000:0000:0000:0000:0000:ffff:3201:0002");
}

// =====================================================================
// 6. Multi-interface host: verify 4-PIP resolution from real XML
// =====================================================================

static const char* kFourPipXml =
  "<multiplane>\n"
  "  <host name=\"node01\">\n"
  "    <interface gid=\"0000:0000:0000:0000:0000:ffff:3201:0002\" dev=\"ionic_1\">\n"
  "      <pip ip=\"50.1.1.2\" interface=\"enP1p68s0\"/>\n"
  "      <pip ip=\"50.1.2.2\" interface=\"enP1p69s0\"/>\n"
  "      <pip ip=\"50.1.3.2\" interface=\"enP1p70s0\"/>\n"
  "      <pip ip=\"50.1.4.2\" interface=\"enP1p71s0\"/>\n"
  "    </interface>\n"
  "  </host>\n"
  "</multiplane>\n";

TEST_F(MultiplaneLoadTest, FourPipsResolved) {
  std::string path = WriteTestXml(kFourPipXml);
  setenv("RCCL_MULTIPLANE_MAP_FILE", path.c_str(), 1);

  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);

  union ibv_gid vipGid;
  memset(&vipGid, 0, sizeof(vipGid));
  MakeGidV4Mapped(vipGid.raw, 50, 1, 0, 2);

  union ibv_gid pipGids[16];
  int nPips = 0;
  ASSERT_EQ(IbCastMultiplaneGetPipGids(&vipGid, pipGids, &nPips), ncclSuccess);
  EXPECT_EQ(nPips, 4);

  // Verify each PIP's last two bytes (the varying octets: 50.1.{1,2,3,4}.2)
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(pipGids[i].raw[12], 0x32) << "PIP[" << i << "] octet 0";  // 50
    EXPECT_EQ(pipGids[i].raw[13], 0x01) << "PIP[" << i << "] octet 1";  // 1
    EXPECT_EQ(pipGids[i].raw[14], (uint8_t)(i + 1)) << "PIP[" << i << "] octet 2";  // 1,2,3,4
    EXPECT_EQ(pipGids[i].raw[15], 0x02) << "PIP[" << i << "] octet 3";  // 2
  }

  remove(path.c_str());
}

// =====================================================================
// 7. Multiple interfaces on same host
// =====================================================================

static const char* kMultiIfXml =
  "<multiplane>\n"
  "  <host name=\"node01\">\n"
  "    <interface gid=\"0000:0000:0000:0000:0000:ffff:3201:0002\" dev=\"ionic_1\">\n"
  "      <pip ip=\"50.1.1.2\" interface=\"eth1\"/>\n"
  "    </interface>\n"
  "    <interface gid=\"0000:0000:0000:0000:0000:ffff:3202:0002\" dev=\"ionic_2\">\n"
  "      <pip ip=\"50.2.1.2\" interface=\"eth2\"/>\n"
  "      <pip ip=\"50.2.2.2\" interface=\"eth3\"/>\n"
  "    </interface>\n"
  "  </host>\n"
  "</multiplane>\n";

TEST_F(MultiplaneLoadTest, MultipleInterfacesSameHost) {
  std::string path = WriteTestXml(kMultiIfXml);
  setenv("RCCL_MULTIPLANE_MAP_FILE", path.c_str(), 1);

  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);

  // Interface 1: 1 PIP
  {
    union ibv_gid vipGid;
    memset(&vipGid, 0, sizeof(vipGid));
    MakeGidV4Mapped(vipGid.raw, 50, 1, 0, 2);
    union ibv_gid pipGids[16];
    int nPips = 0;
    ASSERT_EQ(IbCastMultiplaneGetPipGids(&vipGid, pipGids, &nPips), ncclSuccess);
    EXPECT_EQ(nPips, 1);
  }

  // Interface 2: 2 PIPs
  {
    union ibv_gid vipGid;
    memset(&vipGid, 0, sizeof(vipGid));
    MakeGidV4Mapped(vipGid.raw, 50, 2, 0, 2);
    union ibv_gid pipGids[16];
    int nPips = 0;
    ASSERT_EQ(IbCastMultiplaneGetPipGids(&vipGid, pipGids, &nPips), ncclSuccess);
    EXPECT_EQ(nPips, 2);
  }

  remove(path.c_str());
}

// =====================================================================
// 8. IPv6 PIP addresses
// =====================================================================

static const char* kIpv6PipXml =
  "<multiplane>\n"
  "  <host name=\"node01\">\n"
  "    <interface gid=\"2001:0db8:0000:0000:0000:0000:0000:0001\" dev=\"ionic_1\">\n"
  "      <pip ip=\"2001:db8::10\" interface=\"eth1\"/>\n"
  "      <pip ip=\"2001:db8::20\" interface=\"eth2\"/>\n"
  "    </interface>\n"
  "  </host>\n"
  "</multiplane>\n";

TEST_F(MultiplaneLoadTest, IPv6PipResolution) {
  std::string path = WriteTestXml(kIpv6PipXml);
  setenv("RCCL_MULTIPLANE_MAP_FILE", path.c_str(), 1);

  ASSERT_EQ(IbCastMultiplaneLoad(), ncclSuccess);

  union ibv_gid vipGid;
  memset(&vipGid, 0, sizeof(vipGid));
  // Build 2001:0db8::0001
  vipGid.raw[0] = 0x20; vipGid.raw[1] = 0x01;
  vipGid.raw[2] = 0x0d; vipGid.raw[3] = 0xb8;
  vipGid.raw[15] = 0x01;

  union ibv_gid pipGids[16];
  int nPips = 0;
  ASSERT_EQ(IbCastMultiplaneGetPipGids(&vipGid, pipGids, &nPips), ncclSuccess);
  EXPECT_EQ(nPips, 2);

  // PIP[0] = 2001:db8::10
  EXPECT_EQ(pipGids[0].raw[0], 0x20);
  EXPECT_EQ(pipGids[0].raw[1], 0x01);
  EXPECT_EQ(pipGids[0].raw[15], 0x10);

  // PIP[1] = 2001:db8::20
  EXPECT_EQ(pipGids[1].raw[15], 0x20);

  remove(path.c_str());
}

}  // namespace
