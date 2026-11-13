/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/diagnostics_xid.cc: Xid/SXid kernel-log parsing, scan windows, guidance and the
// cross-host summary.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <unistd.h>

#include "comm.h"
#include "fakes/ras_diagnostics_test_support.h"
#include "ras/diagnostics.h"

#include RAS_DIAGNOSTICS_XID_CC_PATH

namespace {

using ras_test::MakeReporter;
using ras_test::ReporterState;

int64_t Bdf(unsigned domain, unsigned bus, unsigned device, unsigned function) {
  return ((int64_t)domain << 20) | ((int64_t)bus << 12) | ((int64_t)device << 4) | function;
}

rasDiagnosticsXidEvent Parse(const char* line, bool* ok) {
  rasDiagnosticsXidEvent event;
  *ok = rasDiagnosticsXidParseLine(line, &event);
  return event;
}

rasDiagnosticsXidWindow MonotonicWindow(uint64_t startUsec, uint64_t endUsec) {
  rasDiagnosticsXidWindow window{};
  window.monotonicStartUsec = startUsec;
  window.monotonicEndUsec = endUsec;
  return window;
}

// ---------------------------------------------------------------------------------------------------------------
// Line parsing
// ---------------------------------------------------------------------------------------------------------------

TEST(RasDiagnosticsXidMicrotest, BdfAcceptsBothPciFormsAndTrimsWhitespace) {
  const char* texts[] = {"0000:3b:00.1", " PCI:0000:3b:00.1\t", "pci:0000:3b:00.1"};
  for (const char* text : texts) {
    int64_t bdf = -1;
    EXPECT_TRUE(rasDiagnosticsXidParseBdf(text, text + strlen(text), &bdf)) << text;
    EXPECT_EQ(Bdf(0, 0x3b, 0, 1), bdf) << text;
  }
  const char* noFunction = "0001:3b:1f";
  int64_t bdf = -1;
  EXPECT_TRUE(rasDiagnosticsXidParseBdf(noFunction, noFunction + strlen(noFunction), &bdf));
  EXPECT_EQ(Bdf(1, 0x3b, 0x1f, 0), bdf);
}

TEST(RasDiagnosticsXidMicrotest, BdfRejectsOutOfRangeFieldsAndJunk) {
  for (const char* text : {"0000:3b:20.0", "0000:3b:00.8", "10000:3b:00.0", "0000:100:00.0", "0000:3b", "gpu0",
                           "0000:3b:00.0z", "", "PCI:"}) {
    int64_t bdf = -1;
    EXPECT_FALSE(rasDiagnosticsXidParseBdf(text, text + strlen(text), &bdf)) << text;
  }
  int64_t bdf;
  EXPECT_FALSE(rasDiagnosticsXidParseBdf(nullptr, nullptr, &bdf));
}

TEST(RasDiagnosticsXidMicrotest, XidLineWithBdfYieldsCodeAndBdf) {
  bool ok;
  rasDiagnosticsXidEvent e = Parse("NVRM: Xid (PCI:0000:3b:00): 79, pid=1234, GPU has fallen off the bus.", &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(RAS_DIAG_EVENT_XID, e.type);
  EXPECT_EQ(79u, e.code);
  EXPECT_TRUE(e.bdfValid);
  EXPECT_EQ(Bdf(0, 0x3b, 0, 0), e.bdf);
  EXPECT_EQ(-1, e.link);
  EXPECT_EQ(RAS_DIAG_SXID_CLASS_UNKNOWN, e.sxidClass);
}

TEST(RasDiagnosticsXidMicrotest, XidLineWithoutUsableBdfStillYieldsTheCode) {
  bool ok;
  rasDiagnosticsXidEvent e = Parse("NVRM: Xid 13, Graphics Exception", &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(13u, e.code);
  EXPECT_FALSE(e.bdfValid);
  e = Parse("NVRM: Xid (GPU-1): 31, MMU fault", &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(31u, e.code);
  EXPECT_FALSE(e.bdfValid);
}

TEST(RasDiagnosticsXidMicrotest, XidLineWithoutCodeIsRejected) {
  bool ok;
  for (const char* line : {"NVRM: Xid (PCI:0000:3b:00): pid=1", "NVRM: Xid (PCI:0000:3b:00", "NVRM: Xid none",
                           "kernel: eth0 link up"}) {
    Parse(line, &ok);
    EXPECT_FALSE(ok) << line;
  }
}

TEST(RasDiagnosticsXidMicrotest, SxidSummaryLineCarriesClassAndLink) {
  bool ok;
  rasDiagnosticsXidEvent e =
    Parse("nvidia-nvswitch3: SXid (PCI:0000:05:00.0): 12028, Non-fatal, Link 32 egress non-posted PRIV error", &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(RAS_DIAG_EVENT_SXID, e.type);
  EXPECT_EQ(12028u, e.code);
  EXPECT_EQ(RAS_DIAG_SXID_CLASS_NON_FATAL, e.sxidClass);
  EXPECT_EQ(32, e.link);
  EXPECT_TRUE(e.bdfValid);
  EXPECT_EQ(Bdf(0, 5, 0, 0), e.bdf);

  e = Parse("nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 20034, fatal , Link 2", &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(RAS_DIAG_SXID_CLASS_FATAL, e.sxidClass);
  EXPECT_EQ(2, e.link);
}

TEST(RasDiagnosticsXidMicrotest, SxidDetailAndMalformedLinesAreRejected) {
  bool ok;
  for (const char* line : {"nvidia-nvswitch3: SXid (PCI:0000:05:00.0): 12028, Severity 1 Engine instance 32",
                           "nvidia-nvswitch3: SXid (PCI:0000:05:00.0): 12028, Non-fatal",
                           "nvidia-nvswitch3: SXid (PCI:0000:05:00.0): 12028, Non-fatal, Linked 3",
                           "nvidia-nvswitch3: SXid (PCI:0000:05:00.0): 12028, Non-fatal, Link -1",
                           "nvidia-nvswitch3: SXid (PCI:0000:05:00.0): 12028, Non-fatal, Link 40000",
                           "nvidia-nvswitch3: SXid (PCI:0000:05:00.0): 12028 Non-fatal, Link 3",
                           "nvidia-nvswitch3: SXid 12028, Non-fatal, Link 3"}) {
    Parse(line, &ok);
    EXPECT_FALSE(ok) << line;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Event accumulation and timestamps
// ---------------------------------------------------------------------------------------------------------------

TEST(RasDiagnosticsXidMicrotest, AddEventCountsRepeatsOfTheSameEvent) {
  rasDiagnosticsXidScan scan{};
  bool ok;
  rasDiagnosticsXidEvent a = Parse("Xid (PCI:0000:3b:00): 79", &ok);
  rasDiagnosticsXidEvent otherGpu = Parse("Xid (PCI:0000:3c:00): 79", &ok);
  rasDiagnosticsXidAddEvent(&scan, &a);
  rasDiagnosticsXidAddEvent(&scan, &a);
  rasDiagnosticsXidAddEvent(&scan, &otherGpu);
  ASSERT_EQ(2, scan.nStored);
  EXPECT_EQ(2, scan.events[0].count);
  EXPECT_EQ(1, scan.events[1].count);
  EXPECT_FALSE(scan.truncated);
}

TEST(RasDiagnosticsXidMicrotest, AddEventSaturatesTheCountAndTruncatesPastTheTable) {
  rasDiagnosticsXidScan scan{};
  rasDiagnosticsXidEvent e{};
  e.type = RAS_DIAG_EVENT_XID;
  e.link = -1;
  rasDiagnosticsXidAddEvent(&scan, &e);
  scan.events[0].count = UINT16_MAX;
  rasDiagnosticsXidAddEvent(&scan, &e);
  EXPECT_EQ(UINT16_MAX, scan.events[0].count);
  EXPECT_TRUE(scan.events[0].countSaturated);

  for (uint32_t code = 1; code < RAS_DIAG_XID_MAX_EVENTS; code++) {
    e.code = code;
    rasDiagnosticsXidAddEvent(&scan, &e);
  }
  EXPECT_EQ(RAS_DIAG_XID_MAX_EVENTS, scan.nStored);
  EXPECT_FALSE(scan.truncated);
  e.code = RAS_DIAG_XID_MAX_EVENTS;
  rasDiagnosticsXidAddEvent(&scan, &e);
  EXPECT_EQ(RAS_DIAG_XID_MAX_EVENTS, scan.nStored);
  EXPECT_TRUE(scan.truncated);
}

TEST(RasDiagnosticsXidMicrotest, KmsgWindowIsHalfOpen) {
  rasDiagnosticsXidScan scan{};
  rasDiagnosticsXidWindow window = MonotonicWindow(1000, 2000);
  rasDiagnosticsXidScanKmsgLine("3,10,999,-;NVRM: Xid (PCI:0000:3b:00): 13", &window, &scan);
  rasDiagnosticsXidScanKmsgLine("3,11,1000,-;NVRM: Xid (PCI:0000:3b:00): 31", &window, &scan);
  rasDiagnosticsXidScanKmsgLine("3,12,1999,-,caller=T1;NVRM: Xid (PCI:0000:3b:00): 43", &window, &scan);
  rasDiagnosticsXidScanKmsgLine("3,13,2000,-;NVRM: Xid (PCI:0000:3b:00): 48", &window, &scan);
  rasDiagnosticsXidScanKmsgLine("3,14,abc,-;NVRM: Xid (PCI:0000:3b:00): 62", &window, &scan);
  rasDiagnosticsXidScanKmsgLine("NVRM: Xid (PCI:0000:3b:00): 63", &window, &scan);
  ASSERT_EQ(2, scan.nStored);
  EXPECT_EQ(31u, scan.events[0].code);
  EXPECT_EQ(43u, scan.events[1].code);
}

TEST(RasDiagnosticsXidMicrotest, DmesgTimestampScalesToMicroseconds) {
  uint64_t usec = 0;
  const char* message = nullptr;
  ASSERT_TRUE(rasDiagnosticsXidDmesgTimestamp("[  12.345678] NVRM: hi", &usec, &message));
  EXPECT_EQ(12345678u, usec);
  EXPECT_STREQ(" NVRM: hi", message);
  ASSERT_TRUE(rasDiagnosticsXidDmesgTimestamp("[5.5] x", &usec, &message));
  EXPECT_EQ(5500000u, usec);
  ASSERT_TRUE(rasDiagnosticsXidDmesgTimestamp("[7] x", &usec, &message));
  EXPECT_EQ(7000000u, usec);
  ASSERT_TRUE(rasDiagnosticsXidDmesgTimestamp("[1.1234567] x", &usec, &message));
  EXPECT_EQ(1123456u, usec);
  for (const char* line : {"no bracket", "[abc] x", "[1.5 x", "[18446744073709551615] x"})
    EXPECT_FALSE(rasDiagnosticsXidDmesgTimestamp(line, &usec, &message)) << line;
}

TEST(RasDiagnosticsXidMicrotest, DmesgTextKeepsInWindowEventsAndFlagsOverlongLines) {
  rasDiagnosticsXidScan scan{};
  rasDiagnosticsXidWindow window = MonotonicWindow(10000000, 20000000);
  std::string text = "[    9.999999] NVRM: Xid (PCI:0000:3b:00): 13\n"
                     "[   10.000000] NVRM: Xid (PCI:0000:3b:00): 31\n"
                     "[   15.000000] eth0: link up\n"
                     "[   19.999999] NVRM: Xid (PCI:0000:3b:00): 31\n"
                     "[   20.000000] NVRM: Xid (PCI:0000:3b:00): 48";
  rasDiagnosticsXidScanDmesgText(text.c_str(), &window, &scan);
  ASSERT_EQ(1, scan.nStored);
  EXPECT_EQ(31u, scan.events[0].code);
  EXPECT_EQ(2, scan.events[0].count);
  EXPECT_FALSE(scan.truncated);

  std::string longLine = "[   11.000000] " + std::string(RAS_DIAG_XID_LINE_BYTES, 'x') + "\n";
  rasDiagnosticsXidScanDmesgText(longLine.c_str(), &window, &scan);
  EXPECT_TRUE(scan.truncated);
}

// ---------------------------------------------------------------------------------------------------------------
// Regular-log wall-clock timestamps
// ---------------------------------------------------------------------------------------------------------------

struct tm Tm(int year, int mon, int day, int hour, int min, int sec) {
  struct tm t {};
  t.tm_year = year - 1900;
  t.tm_mon = mon - 1;
  t.tm_mday = day;
  t.tm_hour = hour;
  t.tm_min = min;
  t.tm_sec = sec;
  return t;
}

time_t Utc(int year, int mon, int day, int hour, int min, int sec) {
  struct tm t = Tm(year, mon, day, hour, min, sec);
  return timegm(&t);
}

constexpr time_t kZoneOffsetSec = 5 * 3600 + 30 * 60;

// Local time in the FixedZone below.
time_t Local(int year, int mon, int day, int hour, int min, int sec) {
  return Utc(year, mon, day, hour, min, sec) - kZoneOffsetSec;
}

// Pins local time to UTC+05:30 without DST, so local parsing differs from UTC on any CI host.
class FixedZone {
 public:
  FixedZone() {
    const char* tz = getenv("TZ");
    hadTz_ = tz != nullptr;
    if (hadTz_) savedTz_ = tz;
    setenv("TZ", "XST-05:30", 1);
    tzset();
  }
  ~FixedZone() {
    if (hadTz_) setenv("TZ", savedTz_.c_str(), 1);
    else unsetenv("TZ");
    tzset();
  }

 private:
  bool hadTz_;
  std::string savedTz_;
};

TEST(RasDiagnosticsXidMicrotest, IsoWallTimeHonorsUtcAndOffsets) {
  FixedZone zone;
  const time_t utc = Utc(2026, 11, 12, 10, 0, 0);
  time_t t = 0;
  ASSERT_TRUE(rasDiagnosticsXidParseIsoWallTime("2026-11-12T10:00:00Z host kernel: x", &t));
  EXPECT_EQ(utc, t);
  ASSERT_TRUE(rasDiagnosticsXidParseIsoWallTime("2026-11-12T10:00:00.123456Z x", &t));
  EXPECT_EQ(utc, t);
  ASSERT_TRUE(rasDiagnosticsXidParseIsoWallTime("2026-11-12T15:30:00+05:30 x", &t));
  EXPECT_EQ(utc, t);
  ASSERT_TRUE(rasDiagnosticsXidParseIsoWallTime("2026-11-12T08:00:00-0200 x", &t));
  EXPECT_EQ(utc, t);
  ASSERT_TRUE(rasDiagnosticsXidParseIsoWallTime("2026-11-12 10:00:00 x", &t));
  EXPECT_EQ(Local(2026, 11, 12, 10, 0, 0), t);
}

TEST(RasDiagnosticsXidMicrotest, IsoWallTimeRejectsBadTimesAndOffsets) {
  time_t t;
  for (const char* line : {"Nov 12 10:00:00 host", "2026-11-12T10:00:00+5", "2026-11-12T10:00:00+24:00",
                           "2026-11-12T10:00:00+05:60", "2026-11-12T10:00:00+05x30"})
    EXPECT_FALSE(rasDiagnosticsXidParseIsoWallTime(line, &t)) << line;
}

TEST(RasDiagnosticsXidMicrotest, SyslogWallTimeTakesTheYearFromNow) {
  FixedZone zone;
  const time_t now = Local(2026, 11, 12, 12, 0, 0);
  time_t t = 0;
  ASSERT_TRUE(rasDiagnosticsXidParseSyslogWallTime("Nov 12 10:00:00 host kernel: x", now, &t));
  EXPECT_EQ(Local(2026, 11, 12, 10, 0, 0), t);
  ASSERT_TRUE(rasDiagnosticsXidParseSyslogWallTime("nov  2 03:04:05 host kernel: x", now, &t));
  EXPECT_EQ(Local(2026, 11, 2, 3, 4, 5), t);
}

TEST(RasDiagnosticsXidMicrotest, SyslogWallTimeRollsBackAYearAcrossNewYear) {
  FixedZone zone;
  const time_t now = Local(2027, 1, 1, 0, 30, 0);
  time_t t = 0;
  ASSERT_TRUE(rasDiagnosticsXidParseSyslogWallTime("Dec 31 23:59:00 host kernel: x", now, &t));
  EXPECT_EQ(Local(2026, 12, 31, 23, 59, 0), t);
}

TEST(RasDiagnosticsXidMicrotest, SyslogWallTimeRejectsMalformedStamps) {
  FixedZone zone;
  const time_t now = Local(2026, 11, 12, 12, 0, 0);
  time_t t;
  for (const char* line : {"Foo 12 10:00:00 host kernel", "Nov 32 10:00:00 host kernel", "Nov 12 24:00:00 host kern",
                           "Nov 12 10:60:00 host kernel", "Nov 12 10:00", "Nov 12 10:00:61 host kernel"})
    EXPECT_FALSE(rasDiagnosticsXidParseSyslogWallTime(line, now, &t)) << line;
}

// ---------------------------------------------------------------------------------------------------------------
// Log-file scan
// ---------------------------------------------------------------------------------------------------------------

class RasDiagnosticsXidLogFileMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = ::testing::TempDir() + "xid-log-XXXXXX";
    int fd = mkstemp(&path_[0]);
    ASSERT_NE(-1, fd);
    close(fd);
  }
  void TearDown() override { unlink(path_.c_str()); }

  void Write(const std::string& text) {
    FILE* f = fopen(path_.c_str(), "w");
    ASSERT_NE(nullptr, f);
    fputs(text.c_str(), f);
    fclose(f);
  }

  std::string path_;
};

TEST_F(RasDiagnosticsXidLogFileMicrotest, KeepsEventsInsideTheInclusiveWallClockWindow) {
  Write("2026-11-12T09:59:59Z host kernel: NVRM: Xid (PCI:0000:3b:00): 13\n"
        "2026-11-12T10:00:00Z host kernel: NVRM: Xid (PCI:0000:3b:00): 31\n"
        "kernel: NVRM: Xid (PCI:0000:3b:00): 43\n"
        "2026-11-12T10:05:00Z host kernel: NVRM: Xid (PCI:0000:3b:00): 31\n"
        "2026-11-12T10:05:01Z host kernel: NVRM: Xid (PCI:0000:3b:00): 48\n");
  rasDiagnosticsXidWindow window{};
  window.realtimeStartSec = Utc(2026, 11, 12, 10, 0, 0);
  window.realtimeEndSec = Utc(2026, 11, 12, 10, 5, 0);
  rasDiagnosticsXidScan scan{};
  ASSERT_TRUE(rasDiagnosticsXidScanLogFile(path_.c_str(), &window, &scan));
  EXPECT_TRUE(scan.scanned);
  EXPECT_FALSE(scan.truncated);
  ASSERT_EQ(1, scan.nStored);
  EXPECT_EQ(31u, scan.events[0].code);
  EXPECT_EQ(2, scan.events[0].count);
}

TEST_F(RasDiagnosticsXidLogFileMicrotest, OverlongLineIsSkippedAndMarksTruncation) {
  Write("2026-11-12T10:00:00Z " + std::string(RAS_DIAG_XID_LINE_BYTES * 2, 'x') +
        "\n2026-11-12T10:00:00Z host kernel: NVRM: Xid (PCI:0000:3b:00): 31\n");
  rasDiagnosticsXidWindow window{};
  window.realtimeStartSec = Utc(2026, 11, 12, 10, 0, 0);
  window.realtimeEndSec = window.realtimeStartSec;
  rasDiagnosticsXidScan scan{};
  ASSERT_TRUE(rasDiagnosticsXidScanLogFile(path_.c_str(), &window, &scan));
  EXPECT_TRUE(scan.truncated);
  ASSERT_EQ(1, scan.nStored);
  EXPECT_EQ(31u, scan.events[0].code);
}

TEST_F(RasDiagnosticsXidLogFileMicrotest, MissingFileIsNotScanned) {
  rasDiagnosticsXidWindow window{};
  rasDiagnosticsXidScan scan{};
  EXPECT_FALSE(rasDiagnosticsXidScanLogFile((path_ + ".missing").c_str(), &window, &scan));
}

// ---------------------------------------------------------------------------------------------------------------
// Guidance
// ---------------------------------------------------------------------------------------------------------------

TEST(RasDiagnosticsXidMicrotest, XidGuidanceFollowsTheCatalog) {
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_NO_USER_ACTION, rasDiagnosticsXidClassify(63));
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_RESTART_APPLICATION, rasDiagnosticsXidClassify(13));
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_RESTART_APPLICATION, rasDiagnosticsXidClassify(139));
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_RESET_GPU, rasDiagnosticsXidClassify(46));
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_RESET_GPU, rasDiagnosticsXidClassify(158));
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_CHECK_DOCUMENTATION, rasDiagnosticsXidClassify(79));
}

TEST(RasDiagnosticsXidMicrotest, SxidGuidanceNeedsANonFatalCatalogCode) {
  rasDiagnosticsXidEvent e{};
  e.type = RAS_DIAG_EVENT_SXID;
  e.code = 11012;
  e.sxidClass = RAS_DIAG_SXID_CLASS_NON_FATAL;
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_NO_USER_ACTION, rasDiagnosticsSxidClassify(&e));
  e.sxidClass = RAS_DIAG_SXID_CLASS_FATAL;
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_CHECK_DOCUMENTATION, rasDiagnosticsSxidClassify(&e));
  e.sxidClass = RAS_DIAG_SXID_CLASS_NON_FATAL;
  e.code = 12028;
  EXPECT_EQ(RAS_DIAG_XID_GUIDANCE_CHECK_DOCUMENTATION, rasDiagnosticsSxidClassify(&e));
}

// ---------------------------------------------------------------------------------------------------------------
// Summary across hosts
// ---------------------------------------------------------------------------------------------------------------

class RasDiagnosticsXidSummarizeMicrotest : public ::testing::Test {
 protected:
  void Host(uint64_t hostHash, const char* name, bool scanned = true, bool truncated = false, bool preAmpere = false) {
    rasDiagnosticsXidData d{};
    d.kind = RAS_DIAG_XID_RECORD_SCAN_STATUS;
    d.scan.hostHash = hostHash;
    snprintf(d.scan.hostName, sizeof(d.scan.hostName), "%s", name);
    d.scan.scanned = scanned;
    d.scan.truncated = truncated;
    d.scan.preAmpere = preAmpere;
    records_.push_back(d);
  }
  void Xid(uint64_t hostHash, uint32_t code, uint16_t count, int64_t bdf = 0) {
    rasDiagnosticsXidData d{};
    d.kind = RAS_DIAG_XID_RECORD_EVENT;
    d.event.hostHash = hostHash;
    d.event.value.type = RAS_DIAG_EVENT_XID;
    d.event.value.code = code;
    d.event.value.count = count;
    d.event.value.link = -1;
    d.event.value.bdf = bdf;
    d.event.value.bdfValid = true;
    records_.push_back(d);
  }
  void Sxid(uint64_t hostHash, uint32_t code, uint8_t sxidClass, int16_t link) {
    rasDiagnosticsXidData d{};
    d.kind = RAS_DIAG_XID_RECORD_EVENT;
    d.event.hostHash = hostHash;
    d.event.value.type = RAS_DIAG_EVENT_SXID;
    d.event.value.code = code;
    d.event.value.count = 1;
    d.event.value.sxidClass = sxidClass;
    d.event.value.link = link;
    records_.push_back(d);
  }

  ncclResult_t Summarize() {
    rasDiagnosticsReporter reporter = MakeReporter(&state_);
    return rasDiagnosticsXidSummarize(nullptr, &reporter, reinterpret_cast<const char*>(records_.data()),
                                      (int)(records_.size() * sizeof(rasDiagnosticsXidData)));
  }
  ::testing::AssertionResult AnyLineContains(const std::string& text) {
    std::string all;
    for (const std::string& line : state_.lines) {
      if (line.find(text) != std::string::npos) return ::testing::AssertionSuccess();
      all += "\n  " + line;
    }
    return ::testing::AssertionFailure() << "no line has \"" << text << "\" in:" << all;
  }

  std::vector<rasDiagnosticsXidData> records_;
  ReporterState state_;
};

TEST_F(RasDiagnosticsXidSummarizeMicrotest, RejectsAMissingReporterOrMalformedSizeAndAcceptsEmptyData) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsXidSummarize(nullptr, nullptr, nullptr, 0));
  rasDiagnosticsReporter reporter = MakeReporter(&state_);
  EXPECT_EQ(ncclSuccess, rasDiagnosticsXidSummarize(nullptr, &reporter, nullptr, 0));
  char byte = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsXidSummarize(nullptr, &reporter, &byte, 1));
  EXPECT_TRUE(state_.lines.empty());
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, CleanScanReportsOk) {
  Host(1, "h1");
  Host(2, "h2");
  ASSERT_EQ(ncclSuccess, Summarize());
  ASSERT_EQ(1u, state_.lines.size());
  EXPECT_TRUE(AnyLineContains("[OK]"));
  EXPECT_TRUE(AnyLineContains("across 2 hosts"));
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, UnscannedOrTruncatedHostsAreIncomplete) {
  Host(1, "h1", /*scanned=*/false);
  Host(2, "h2", /*scanned=*/true, /*truncated=*/true);
  ASSERT_EQ(ncclSuccess, Summarize());
  EXPECT_TRUE(AnyLineContains("no kernel-log source was available on host h1"));
  EXPECT_TRUE(AnyLineContains("recent-event scan was truncated on host h2"));
  EXPECT_FALSE(AnyLineContains("[OK]"));
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, XidFindingCarriesCountAndGuidance) {
  Host(1, "h1");
  Xid(1, 13, 2, 0x100);
  Xid(1, 13, 1, 0x200);  // another GPU: same finding
  ASSERT_EQ(ncclSuccess, Summarize());
  ASSERT_EQ(1u, state_.lines.size());
  EXPECT_TRUE(AnyLineContains("Xid 13 detected 3 times on host h1"));
  EXPECT_TRUE(AnyLineContains("Restart the affected application"));
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, DuplicateReportsOfOneEventCountOnce) {
  // Two ranks on the same host report the same scan.
  Host(1, "h1");
  Host(1, "h1");
  Xid(1, 63, 2, 0x100);
  Xid(1, 63, 2, 0x100);
  ASSERT_EQ(ncclSuccess, Summarize());
  ASSERT_EQ(1u, state_.lines.size());
  EXPECT_TRUE(AnyLineContains("Xid 63 detected 2 times"));
  EXPECT_TRUE(AnyLineContains("No user action"));
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, PreAmpereHostsPointAtTheArchive) {
  Host(1, "h1", /*scanned=*/true, /*truncated=*/false, /*preAmpere=*/true);
  Xid(1, 13, 1);
  ASSERT_EQ(ncclSuccess, Summarize());
  EXPECT_TRUE(AnyLineContains("detected 1 time on host h1"));
  EXPECT_TRUE(AnyLineContains("pre-Ampere"));
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, SxidLinksAreGroupedPerFinding) {
  Host(1, "h1");
  Sxid(1, 12028, RAS_DIAG_SXID_CLASS_NON_FATAL, 5);
  Sxid(1, 12028, RAS_DIAG_SXID_CLASS_NON_FATAL, 3);
  Sxid(1, 12028, RAS_DIAG_SXID_CLASS_NON_FATAL, 5);
  ASSERT_EQ(ncclSuccess, Summarize());
  ASSERT_EQ(1u, state_.lines.size());
  EXPECT_TRUE(AnyLineContains("SXid 12028 detected 2 times"));
  EXPECT_TRUE(AnyLineContains("Non-fatal, link(s) {3,5}"));
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, EventWithoutItsHostRecordIsAnError) {
  Host(1, "h1");
  Xid(2, 13, 1);
  EXPECT_EQ(ncclInternalError, Summarize());
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, MalformedEventIsAnError) {
  Host(1, "h1");
  Xid(1, 13, 0);  // a zero count never comes from a scan
  EXPECT_EQ(ncclInternalError, Summarize());
}

TEST_F(RasDiagnosticsXidSummarizeMicrotest, ReporterFailureIsReturned) {
  Host(1, "h1");
  Xid(1, 13, 1);
  state_.emitResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, Summarize());
}

}  // namespace
