// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amdsmi_get_gpu_pm_metrics_info() and amdsmi_get_gpu_reg_table_info() return the
// amdsmi_name_value_t records that amdsmi.h declares, holding what rocm_smi's parsers read.
// They used to return rocm_smi's own array, whose records have 64-byte names, so a caller
// indexing the documented 264-byte records read every value and every later name from the
// wrong offset, and read past the end of the array.
//
// A fake sysfs tree holds the pm_metrics and reg_state images, so this runs on any host.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "rocm_smi/rocm_smi.h"
#include "test_fixture_utils.h"

namespace amd::smi {
int present_pmmetrics(const char* fname, rsmi_name_value_t** kv, uint32_t* kvnum);
int present_reg_state(const char* fname, rsmi_reg_type_t reg_type, rsmi_name_value_t** kv,
                      uint32_t* kvnum);
}  // namespace amd::smi

namespace {

struct FreeDeleter {
  void operator()(void* p) const { free(p); }
};
template <typename T>
using MallocPtr = std::unique_ptr<T, FreeDeleter>;

// The size of the buffer rocm_smi's PM metrics parser reads the file into.
constexpr size_t kPmMetricsReadSize = 65536;

// A GPU whose device directory is a fake sysfs tree, registered with RocmSMI and
// AMDSmiSystem for the life of the object so the public API can reach it.
class FakeGpu {
 public:
  FakeGpu()
      : device_(tree_.root() + "/card0"),
        gpu_device_(device_.index(), "card0", amdsmi_bdf_t{}, drm_),
        registration_(&gpu_device_) {}

  std::string Write(const std::string& name, const std::vector<uint8_t>& image) {
    tree_.WriteFile("card0/device/" + name, std::string(image.begin(), image.end()));
    return tree_.root() + "/card0/device/" + name;
  }

  amdsmi_processor_handle handle() const { return registration_.handle(); }

 private:
  ScopedAmdsmiInit init_;
  FakeSysfsTree tree_;
  ScopedRocmSmiDevice device_;
  amd::smi::AMDSmiDrm drm_;
  amd::smi::AMDSmiGPUDevice gpu_device_;
  amd::smi::testing::ScopedProcessorRegistration registration_;
};

// A version 4 PM metrics image, the only version the parser reads (the drivers checked publish
// versions 13 to 21). The image fills the parser's read buffer, so the whole table fits, and its
// bytes vary so that neighbouring records hold different values.
std::vector<uint8_t> PmMetricsImage() {
  std::vector<uint8_t> image(kPmMetricsReadSize);
  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = static_cast<uint8_t>(i * 7 + 1);
  }
  const uint32_t version = 4;
  std::memcpy(&image[12], &version, sizeof(version));
  return image;
}

// An XGMI register state image: 8-byte header, 8-byte instance header, then 16 bytes per SMN
// entry (see binary_parser_test.cc).
std::vector<uint8_t> XgmiRegStateImage(uint16_t num_smn_regs) {
  const size_t size = 16 + static_cast<size_t>(num_smn_regs) * 16;
  std::vector<uint8_t> image(size, 0);
  const uint16_t structure_size = static_cast<uint16_t>(size);
  std::memcpy(&image[0], &structure_size, sizeof(structure_size));
  image[5] = 1;  // num_instances
  std::memcpy(&image[12], &num_smn_regs, sizeof(num_smn_regs));
  for (uint16_t i = 0; i < num_smn_regs; ++i) {
    const size_t offset = 16 + static_cast<size_t>(i) * 16;
    const uint64_t addr = 0x1000u + i;
    const uint32_t value = 0xA0000000u + i;
    std::memcpy(&image[offset], &addr, sizeof(addr));
    std::memcpy(&image[offset + 8], &value, sizeof(value));
  }
  return image;
}

// The public records must hold exactly what rocm_smi's parser read from the same file, with the
// rest of each name field zero rather than bytes copied from rocm_smi's records. Those bytes are
// uninitialized, so whether a copy of the whole 64-byte name shows up depends on the allocator.
void ExpectSameRecords(const amdsmi_name_value_t* actual, uint32_t actual_count,
                       const rsmi_name_value_t* expected, uint32_t expected_count) {
  ASSERT_EQ(actual_count, expected_count);
  for (uint32_t i = 0; i < expected_count; ++i) {
    EXPECT_STREQ(actual[i].name, expected[i].name) << "record " << i;
    EXPECT_EQ(actual[i].value, expected[i].value) << "record " << i;
    for (size_t j = strnlen(actual[i].name, sizeof(actual[i].name)); j < sizeof(actual[i].name);
         ++j) {
      ASSERT_EQ(actual[i].name[j], '\0') << "record " << i << ", name byte " << j;
    }
  }
}

}  // namespace

TEST(GpuUnit, PmMetricsInfoReturnsDocumentedRecords) {
  FakeGpu gpu;
  const std::string path = gpu.Write("pm_metrics", PmMetricsImage());

  rsmi_name_value_t* raw_parsed = nullptr;
  uint32_t parsed_count = 0;
  const int parse_status = amd::smi::present_pmmetrics(path.c_str(), &raw_parsed, &parsed_count);
  MallocPtr<rsmi_name_value_t> parsed(raw_parsed);
  ASSERT_EQ(parse_status, 0);
  // More records than the parser's first block of 64, so some come from its realloc() growth,
  // and neighbouring values differ, so a record read at the wrong offset shows.
  ASSERT_GT(parsed_count, 64u);
  EXPECT_NE(parsed.get()[0].value, parsed.get()[1].value);

  amdsmi_name_value_t* raw_metrics = nullptr;
  uint32_t count = 0;
  ASSERT_EQ(amdsmi_get_gpu_pm_metrics_info(gpu.handle(), &raw_metrics, &count),
            AMDSMI_STATUS_SUCCESS);
  MallocPtr<amdsmi_name_value_t> metrics(raw_metrics);
  ASSERT_NE(metrics, nullptr);
  ASSERT_NO_FATAL_FAILURE(ExpectSameRecords(metrics.get(), count, parsed.get(), parsed_count));
}

TEST(GpuUnit, RegTableInfoReturnsDocumentedRecords) {
  FakeGpu gpu;
  const std::string path = gpu.Write("reg_state", XgmiRegStateImage(3));

  rsmi_name_value_t* raw_parsed = nullptr;
  uint32_t parsed_count = 0;
  const int parse_status =
      amd::smi::present_reg_state(path.c_str(), RSMI_REG_XGMI, &raw_parsed, &parsed_count);
  MallocPtr<rsmi_name_value_t> parsed(raw_parsed);
  ASSERT_EQ(parse_status, 0);
  ASSERT_GT(parsed_count, 4u);

  amdsmi_name_value_t* raw_regs = nullptr;
  uint32_t count = 0;
  ASSERT_EQ(amdsmi_get_gpu_reg_table_info(gpu.handle(), AMDSMI_REG_XGMI, &raw_regs, &count),
            AMDSMI_STATUS_SUCCESS);
  MallocPtr<amdsmi_name_value_t> regs(raw_regs);
  ASSERT_NE(regs, nullptr);
  // The header fields as the image lays them out: 16 + 3 * 16 bytes, one instance.
  EXPECT_STREQ(regs.get()[0].name, "structure_size");
  EXPECT_EQ(regs.get()[0].value, 64u);
  EXPECT_STREQ(regs.get()[4].name, "num_instances");
  EXPECT_EQ(regs.get()[4].value, 1u);
  ASSERT_NO_FATAL_FAILURE(ExpectSameRecords(regs.get(), count, parsed.get(), parsed_count));
}

// The parser fails on an image shorter than its table after it has allocated the records and
// counted part of them. The call reports the failure and returns no array and a count of 0.
TEST(GpuUnit, PmMetricsInfoTruncatedImageReturnsNothing) {
  FakeGpu gpu;
  std::vector<uint8_t> image = PmMetricsImage();
  image.resize(64);
  gpu.Write("pm_metrics", image);

  amdsmi_name_value_t* metrics = nullptr;
  uint32_t count = 12345;
  EXPECT_EQ(amdsmi_get_gpu_pm_metrics_info(gpu.handle(), &metrics, &count),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(metrics, nullptr);
  EXPECT_EQ(count, 0u);
}

TEST(GpuUnit, RegTableInfoTruncatedImageReturnsNothing) {
  FakeGpu gpu;
  std::vector<uint8_t> image = XgmiRegStateImage(3);
  image.resize(24);  // Ends inside the first SMN entry.
  gpu.Write("reg_state", image);

  amdsmi_name_value_t* regs = nullptr;
  uint32_t count = 12345;
  EXPECT_EQ(amdsmi_get_gpu_reg_table_info(gpu.handle(), AMDSMI_REG_XGMI, &regs, &count),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(regs, nullptr);
  EXPECT_EQ(count, 0u);
}

// Before amdsmi_init() the call fails, and the count still reads 0.
TEST(GpuUnit, NameValueInfoBeforeInitZeroesTheCount) {
  amdsmi_name_value_t* records = nullptr;
  uint32_t count = 12345;
  EXPECT_EQ(amdsmi_get_gpu_pm_metrics_info(nullptr, &records, &count), AMDSMI_STATUS_NOT_INIT);
  EXPECT_EQ(count, 0u);
  count = 12345;
  EXPECT_EQ(amdsmi_get_gpu_reg_table_info(nullptr, AMDSMI_REG_XGMI, &records, &count),
            AMDSMI_STATUS_NOT_INIT);
  EXPECT_EQ(count, 0u);
}

// With a count to fill in but no array pointer, the call is invalid rather than a crash.
TEST(GpuUnit, NameValueInfoWithoutArrayPointerIsInvalid) {
  FakeGpu gpu;
  gpu.Write("pm_metrics", PmMetricsImage());
  gpu.Write("reg_state", XgmiRegStateImage(1));

  uint32_t count = 12345;
  EXPECT_EQ(amdsmi_get_gpu_pm_metrics_info(gpu.handle(), nullptr, &count), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(count, 0u);
  count = 12345;
  EXPECT_EQ(amdsmi_get_gpu_reg_table_info(gpu.handle(), AMDSMI_REG_XGMI, nullptr, &count),
            AMDSMI_STATUS_INVAL);
  EXPECT_EQ(count, 0u);
}

// A NULL count asks whether the device supports the call, with or without an array pointer.
TEST(GpuUnit, NameValueInfoNullCountWithoutSysfsFilesIsNotSupported) {
  FakeGpu gpu;
  amdsmi_name_value_t* records = nullptr;
  EXPECT_EQ(amdsmi_get_gpu_pm_metrics_info(gpu.handle(), &records, nullptr),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_get_gpu_pm_metrics_info(gpu.handle(), nullptr, nullptr),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_get_gpu_reg_table_info(gpu.handle(), AMDSMI_REG_XGMI, &records, nullptr),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_get_gpu_reg_table_info(gpu.handle(), AMDSMI_REG_XGMI, nullptr, nullptr),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(records, nullptr);
}

// On a device that supports the call, the missing count is the invalid argument.
TEST(GpuUnit, NameValueInfoNullCountWithSysfsFilesIsInvalid) {
  FakeGpu gpu;
  gpu.Write("pm_metrics", PmMetricsImage());
  gpu.Write("reg_state", XgmiRegStateImage(1));
  amdsmi_name_value_t* records = nullptr;
  EXPECT_EQ(amdsmi_get_gpu_pm_metrics_info(gpu.handle(), &records, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_gpu_pm_metrics_info(gpu.handle(), nullptr, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_gpu_reg_table_info(gpu.handle(), AMDSMI_REG_XGMI, &records, nullptr),
            AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_gpu_reg_table_info(gpu.handle(), AMDSMI_REG_XGMI, nullptr, nullptr),
            AMDSMI_STATUS_INVAL);
  EXPECT_EQ(records, nullptr);
}
