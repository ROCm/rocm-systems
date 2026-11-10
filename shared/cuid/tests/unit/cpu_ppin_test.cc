// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The CPU package serial is its PPIN. Only when the CPU has none does the
// SMBIOS hash stand in; a PPIN this caller cannot read is an error, so one
// socket does not get a different permanent CUID depending on whether the msr
// module is loaded.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <string>

#include "src/cuid_cpu.h"
#include "test_common.h"

namespace {

constexpr uint16_t kAmd = 0x1022;
constexpr uint16_t kIntel = 0x8086;

class FakeCpu {
 public:
  FakeCpu() : dir_("cuid_ppin_") {
    base_ = dir_.path() + "/cpu";
    msr_ = dir_.path() + "/msr";
    cpuinfo_ = dir_.path() + "/cpuinfo";
    mkdir(base_.c_str(), 0700);
    mkdir((base_ + "/cpu3").c_str(), 0700);
    mkdir((base_ + "/cpu3/topology").c_str(), 0700);
    mkdir(msr_.c_str(), 0700);
  }

  std::string TopologyPpinPath() const { return base_ + "/cpu3/topology/ppin"; }

  void SetTopologyPpin(const std::string& text) { std::ofstream(TopologyPpinPath()) << text; }

  void SetFlags(const std::string& flags) {
    std::ofstream(cpuinfo_) << "processor\t: 0\nflags\t\t: fpu amd_ppin\n\nprocessor\t: 3\n"
                            << "vendor_id\t: x\nflags\t\t: " << flags << "\n\n";
  }

  // False when this filesystem cannot hold the sparse file a high MSR offset needs.
  bool SetMsr(uint32_t msr, uint64_t value) {
    mkdir((msr_ + "/3").c_str(), 0700);
    const int fd = open((msr_ + "/3/msr").c_str(), O_CREAT | O_WRONLY, 0600);
    if (fd < 0) return false;
    const bool wrote = pwrite(fd, &value, sizeof(value), msr) == sizeof(value);
    close(fd);
    return wrote;
  }

  void SetKernel(const std::string& release, bool amd_ppin_cpuid) {
    release_ = release;
    amd_ppin_cpuid_ = amd_ppin_cpuid;
  }

  amdcuid_status_t Read(uint16_t vendor, uint64_t& ppin) const {
    return read_cpu_ppin(base_, cpuinfo_, msr_, release_, amd_ppin_cpuid_, vendor, 3, ppin);
  }

  const std::string& msr() const { return msr_; }

 private:
  ScopedTempDir dir_;
  std::string base_, msr_, cpuinfo_;
  std::string release_ = "6.8.0-124-generic";
  bool amd_ppin_cpuid_ = true;
};

}  // namespace

#define SKIP_UNLESS_X86() \
  if (!kIsX86) GTEST_SKIP() << "PPIN is read on x86 only"

#ifdef __x86_64__
constexpr bool kIsX86 = true;
#else
constexpr bool kIsX86 = false;
#endif

TEST(cuidtstUnprivileged, PpinFromTopologyIsTheSerial) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  cpu.SetTopologyPpin("0x1234567890abcdef\n");
  uint64_t ppin = 0;
  EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, 0x1234567890abcdefull);
  ppin = 0;
  EXPECT_EQ(cpu.Read(kIntel, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, 0x1234567890abcdefull);
}

TEST(cuidtstUnprivileged, PpinZeroIsAbsent) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  cpu.SetFlags("fpu");
  cpu.SetTopologyPpin("0x0\n");
  uint64_t ppin = 0xdeadbeef;
  EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, 0u);
}

TEST(cuidtstUnprivileged, PpinMissingWithoutKernelFlagIsAbsent) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  // amd_ppin on another logical CPU says nothing about this one.
  cpu.SetFlags("fpu sse");
  uint64_t ppin = 0xdeadbeef;
  EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, 0u);
  EXPECT_EQ(cpu.Read(kIntel, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, 0u);
}

TEST(cuidtstUnprivileged, PpinFlagWithoutReadableMsrIsAnErrorNotAbsent) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  cpu.SetFlags("fpu amd_ppin intel_ppin");
  for (const uint16_t vendor : {kAmd, kIntel}) {
    uint64_t ppin = 0xdeadbeef;
    const amdcuid_status_t status = cpu.Read(vendor, ppin);
    EXPECT_NE(status, AMDCUID_STATUS_SUCCESS)
        << "a PPIN that cannot be read must not fall back to the SMBIOS hash";
    EXPECT_EQ(ppin, 0u);
  }
}

TEST(cuidtstUnprivileged, PpinFromTheVendorMsr) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  const uint64_t amd = 0x1111222233334444ull, intel = 0x5555666677778888ull;
  if (!cpu.SetMsr(0xC00102F1, amd) || !cpu.SetMsr(0x4F, intel))
    GTEST_SKIP() << "no sparse files here";

  uint64_t ppin = 0;
  cpu.SetFlags("amd_ppin");
  EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, amd);
  cpu.SetFlags("intel_ppin");
  EXPECT_EQ(cpu.Read(kIntel, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, intel);
}

TEST(cuidtstUnprivileged, PpinMalformedTopologyIsInvalidFormat) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  for (const char* text : {"", "\n", "not-a-ppin\n", "0x\n", "0x12zz\n", "-1\n"}) {
    cpu.SetTopologyPpin(text);
    uint64_t ppin = 0xdeadbeef;
    EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_INVALID_FORMAT) << text;
    EXPECT_EQ(ppin, 0u);
  }
}

TEST(cuidtstUnprivileged, PpinUnreadableTopologyIsPermissionDenied) {
  SKIP_UNLESS_X86();
  if (geteuid() == 0) GTEST_SKIP() << "root reads any file";
  FakeCpu cpu;
  cpu.SetFlags("fpu");
  cpu.SetTopologyPpin("0x1234\n");
  ASSERT_EQ(chmod(cpu.TopologyPpinPath().c_str(), 0), 0);
  uint64_t ppin = 0xdeadbeef;
  EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_PERMISSION_DENIED);
  EXPECT_EQ(ppin, 0u);
}

// amd_ppin exists only since Linux 5.7. Before that a missing flag says nothing,
// so an AMD CPU is judged the way the kernel's own PPIN setup did: the CPUID
// bit, then PPIN_CTL, then the PPIN MSR.
TEST(cuidtstUnprivileged, PpinOnAnOldKernelWithoutTheFlagReadsTheMsrs) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  cpu.SetFlags("fpu");
  cpu.SetKernel("5.4.0-100-generic", true);
  if (!cpu.SetMsr(0xC00102F0, 0x2) || !cpu.SetMsr(0xC00102F1, 0x1111222233334444ull))
    GTEST_SKIP() << "no sparse files here";
  uint64_t ppin = 0;
  EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, 0x1111222233334444ull);
}

TEST(cuidtstUnprivileged, PpinOnAnOldKernelWithPpinLockedOutIsAbsent) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  cpu.SetFlags("fpu");
  cpu.SetKernel("5.4.0-100-generic", true);
  if (!cpu.SetMsr(0xC00102F0, 0x1) || !cpu.SetMsr(0xC00102F1, 0x1111222233334444ull))
    GTEST_SKIP() << "no sparse files here";
  uint64_t ppin = 0xdeadbeef;
  EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, 0u);
}

TEST(cuidtstUnprivileged, PpinOnAnOldKernelWithoutTheCpuidBitIsAbsent) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  cpu.SetFlags("fpu");
  cpu.SetKernel("5.4.0-100-generic", false);
  uint64_t ppin = 0xdeadbeef;
  EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(ppin, 0u);
}

TEST(cuidtstUnprivileged, PpinOnAnOldKernelWithoutReadableMsrsIsAnError) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  cpu.SetFlags("fpu");
  for (const char* release : {"5.4.0-100-generic", "5.6.19", "not-a-release", ""}) {
    cpu.SetKernel(release, true);
    uint64_t ppin = 0xdeadbeef;
    EXPECT_NE(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS) << release;
    EXPECT_EQ(ppin, 0u);
  }
}

TEST(cuidtstUnprivileged, PpinOnANewKernelWithoutTheFlagIsAbsentWithoutTouchingTheMsrs) {
  SKIP_UNLESS_X86();
  FakeCpu cpu;
  cpu.SetFlags("fpu");
  for (const char* release : {"6.8.0-124-generic", "5.7.0", "5.18.1", "7.0"}) {
    cpu.SetKernel(release, true);
    uint64_t ppin = 0xdeadbeef;
    EXPECT_EQ(cpu.Read(kAmd, ppin), AMDCUID_STATUS_SUCCESS) << release;
    EXPECT_EQ(ppin, 0u);
  }
}
