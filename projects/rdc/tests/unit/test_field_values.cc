/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// The values RdcMetricFetcherImpl reads from the fake AMD SMI (tests/fake_amdsmi).
// The fake reads its RDC_FAKE_* knobs once per process, at the first amdsmi_init(),
// so each case runs in a child process with its own knobs.

#include <gtest/gtest.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

extern char** environ;

namespace {

using amd::rdc::RdcMetricFetcherImpl;
using Knobs = std::vector<std::pair<std::string, std::string>>;

// GPU 0's readings in the fake.
struct Gpu0 {
#define FAKE_VALUE(name, base, step, modulo) static constexpr int64_t name = base;
#include "fake_values.inc"
#undef FAKE_VALUE
};

constexpr int64_t kHzPerMHz = 1000000;

// Runs body with an AMD SMI session and a fetcher, in a child process whose fake
// has only these knobs. The test fails unless the child exits 0; a failure of its
// own, not one the test had before, makes it exit 1.
void with_fake(const Knobs& knobs, const std::function<void(RdcMetricFetcherImpl&)>& body) {
  const ::testing::TestResult& result =
      *::testing::UnitTest::GetInstance()->current_test_info()->result();
  const int parts_before = result.total_part_count();
  fflush(nullptr);
  const pid_t pid = fork();
  ASSERT_NE(pid, -1) << strerror(errno);
  if (pid == 0) {
    std::vector<std::string> inherited;
    for (char** e = environ; *e != nullptr; ++e) {
      if (strncmp(*e, "RDC_FAKE_", 9) == 0) inherited.emplace_back(*e, strcspn(*e, "="));
    }
    for (const auto& name : inherited) unsetenv(name.c_str());
    for (const auto& knob : knobs) setenv(knob.first.c_str(), knob.second.c_str(), 1);
    try {
      amd::rdc::SmiSession session;
      RdcMetricFetcherImpl fetcher;
      body(fetcher);
    } catch (const std::exception& e) {
      fprintf(stderr, "child threw: %s\n", e.what());
      fflush(nullptr);
      _exit(1);
    } catch (...) {
      fprintf(stderr, "child threw\n");
      fflush(nullptr);
      _exit(1);
    }
    bool failed = false;
    for (int i = parts_before; i < result.total_part_count(); ++i) {
      failed = failed || result.GetTestPartResult(i).failed();
    }
    fflush(nullptr);
    _exit(failed ? 1 : 0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid) << strerror(errno);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "child wait status " << status;
}

// What a caller's stack can hold before the fetch.
rdc_field_value dirty_value() {
  rdc_field_value value;
  memset(&value, 0xA5, sizeof(value));
  return value;
}

uint32_t partition(uint32_t gpu, uint32_t instance) {
  rdc_entity_info_t info = {};
  info.device_index = gpu;
  info.instance_index = instance;
  info.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
  info.device_type = RDC_DEVICE_TYPE_GPU;
  return rdc_get_entity_index_from_info(info);
}

// Larger than the frames of a fetch, the largest holding an amdsmi_gpu_metrics_t;
// well within the default 8 MB stack.
constexpr size_t kDirt = 1 << 20;
static_assert(sizeof(amdsmi_gpu_metrics_t) * 4 < kDirt, "dirty_stack() covers a fetch's frames");

// Fills the stack below the caller's frame with byte, where the frames of the
// caller's next call go. Not instrumented by AddressSanitizer, which would put a
// redzone between the array and the top of the frame, and so leave the top of the
// next frame clean.
__attribute__((noinline, no_sanitize_address)) void dirty_stack(unsigned char byte) {
  unsigned char dirt[kDirt];
  memset(dirt, byte, sizeof(dirt));
  asm volatile("" : : "r"(dirt) : "memory");
}

// Whether an uninitialized local, in a frame instrumented like the fetcher's,
// shows the byte of dirty_stack(): if not, this build places frames differently,
// and a check after dirty_stack() proves nothing. It reads uninitialized memory on
// purpose, which MemorySanitizer would report.
__attribute__((noinline)) bool shows_dirt(unsigned char byte) {
  unsigned char local[4096];
  asm volatile("" : : "r"(local) : "memory");
  return local[0] == byte && local[sizeof(local) - 1] == byte;
}

amdsmi_processor_handle handle_of(uint32_t index) {
  amdsmi_processor_handle handle = nullptr;
  EXPECT_EQ(amd::rdc::get_processor_handle_from_id(index, &handle), AMDSMI_STATUS_SUCCESS);
  return handle;
}

// A fetch that fails leaves a defined value: the field, an integer 0 and a status
// that is not success. One that returns before reading has no timestamp.
TEST(FieldValues, AFailedFetchLeavesADefinedValue) {
  with_fake({}, [](RdcMetricFetcherImpl& fetcher) {
    const struct {
      uint32_t index;
      rdc_field_t field;
      bool read;
    } cases[] = {
        {100, RDC_FI_GPU_CLOCK, false},              // no such GPU
        {0, static_cast<rdc_field_t>(9999), false},  // no such field
        {0, RDC_FI_GPU_MM_ENC_UTIL, true},           // RDC reports it not supported
    };
    for (const auto& c : cases) {
      SCOPED_TRACE("field " + std::to_string(c.field) + " of " + std::to_string(c.index));
      rdc_field_value value = dirty_value();
      EXPECT_NE(fetcher.fetch_smi_field(c.index, c.field, &value), RDC_ST_OK);
      EXPECT_EQ(value.field_id, c.field);
      EXPECT_NE(value.status, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(value.type, INTEGER);
      EXPECT_EQ(value.value.l_int, 0);
      if (c.read) {
        EXPECT_NE(value.ts, 0u);
      } else {
        EXPECT_EQ(value.ts, 0u);
      }
    }
  });
}

// RDC_HEALTH_EEPROM_CONFIG_VALID: 1 when AMD SMI validates the EEPROM, else 0. (amd-smi
// does not implement the check yet: it reports AMDSMI_STATUS_NOT_SUPPORTED.)
TEST(FieldValues, EepromValidityIsOneOrZero) {
  with_fake({}, [](RdcMetricFetcherImpl& fetcher) {
    rdc_field_value value = dirty_value();
    EXPECT_EQ(fetcher.fetch_smi_field(0, RDC_HEALTH_EEPROM_CONFIG_VALID, &value), RDC_ST_OK);
    EXPECT_EQ(value.type, INTEGER);
    EXPECT_EQ(value.value.l_int, 1);
  });
  const std::string corrupted = std::to_string(AMDSMI_STATUS_CORRUPTED_EEPROM);
  with_fake({{"RDC_FAKE_FAIL", "amdsmi_gpu_validate_ras_eeprom=" + corrupted}},
            [](RdcMetricFetcherImpl& fetcher) {
              rdc_field_value value = dirty_value();
              EXPECT_NE(fetcher.fetch_smi_field(0, RDC_HEALTH_EEPROM_CONFIG_VALID, &value),
                        RDC_ST_OK);
              EXPECT_EQ(value.type, INTEGER);
              EXPECT_EQ(value.value.l_int, 0);
            });
}

// A name that fills AMD SMI's buffer, with no NUL, is cut to fit and terminated.
TEST(FieldValues, ANameFillingItsBufferIsCutToFit) {
  with_fake({{"RDC_FAKE_LONG_STRINGS", "1"}}, [](RdcMetricFetcherImpl& fetcher) {
    amdsmi_asic_info_t info = {};
    ASSERT_EQ(amdsmi_get_gpu_asic_info(handle_of(0), &info), AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(memchr(info.market_name, '\0', sizeof(info.market_name)), nullptr);
    ASSERT_EQ(memchr(info.asic_serial, '\0', sizeof(info.asic_serial)), nullptr);
    for (const auto& name : {std::make_pair(RDC_FI_DEV_NAME, info.market_name[0]),
                             std::make_pair(RDC_FI_UUID, info.asic_serial[0])}) {
      SCOPED_TRACE("field " + std::to_string(name.first));
      rdc_field_value value = dirty_value();
      ASSERT_EQ(fetcher.fetch_smi_field(0, name.first, &value), RDC_ST_OK);
      EXPECT_EQ(value.type, STRING);
      EXPECT_EQ(std::string(value.value.str, strnlen(value.value.str, sizeof(value.value.str))),
                std::string(RDC_MAX_STR_LENGTH - 1, name.second));
    }
  });
}

// When its profile cannot be read, a GPU counts as one partition: a partition
// past that has no clock, rather than one read past the valid clocks.
TEST(FieldValues, APartitionPastThePartitionCountHasNoClock) {
  with_fake(
      {{"RDC_FAKE_GPUS", "1"},
       {"RDC_FAKE_PARTITIONS", "4"},
       {"RDC_FAKE_FAIL",
        "amdsmi_get_gpu_accelerator_partition_profile=AMDSMI_STATUS_NOT_SUPPORTED"}},
      [](RdcMetricFetcherImpl& fetcher) {
        uint16_t count = 0;
        ASSERT_EQ(amd::rdc::get_num_partition(partition(0, 3), &count), AMDSMI_STATUS_SUCCESS);
        ASSERT_EQ(count, 1);
        rdc_field_value value = dirty_value();
        EXPECT_NE(fetcher.fetch_smi_field(partition(0, 3), RDC_FI_GPU_CLOCK, &value), RDC_ST_OK);
        EXPECT_NE(value.status, AMDSMI_STATUS_SUCCESS);
        value = dirty_value();
        ASSERT_EQ(fetcher.fetch_smi_field(partition(0, 0), RDC_FI_GPU_CLOCK, &value), RDC_ST_OK);
        EXPECT_EQ(value.value.l_int, Gpu0::GFX_CLK_MHZ * kHzPerMHz);
      });
}

// With as many valid clocks as partitions, each partition reads a valid clock, not
// the reading in its own slot.
TEST(FieldValues, PartitionClocksSkipUnavailableReadings) {
  with_fake({{"RDC_FAKE_GPUS", "1"}, {"RDC_FAKE_PARTITIONS", "4"}, {"RDC_FAKE_GFXCLK_HOLES", "4"}},
            [](RdcMetricFetcherImpl& fetcher) {
              amdsmi_gpu_metrics_t metrics = {};
              ASSERT_EQ(amdsmi_get_gpu_metrics_info(handle_of(partition(0, 0)), &metrics),
                        AMDSMI_STATUS_SUCCESS);
              for (uint32_t i = 0; i < AMDSMI_MAX_NUM_GFX_CLKS; ++i) {
                ASSERT_EQ(metrics.current_gfxclks[i], i < 4 ? UINT16_MAX : Gpu0::GFX_CLK_MHZ) << i;
              }
              for (uint32_t p = 0; p < 4; ++p) {
                SCOPED_TRACE("partition " + std::to_string(p));
                rdc_field_value value = dirty_value();
                ASSERT_EQ(fetcher.fetch_smi_field(partition(0, p), RDC_FI_GPU_CLOCK, &value),
                          RDC_ST_OK);
                EXPECT_EQ(value.value.l_int, Gpu0::GFX_CLK_MHZ * kHzPerMHz);
              }
            });
}

// More CPER entries than the headers buffer holds, which amd-smi never reports:
// only the buffer's entries are read. Under AddressSanitizer, a read past it is
// reported.
TEST(FieldValues, CperEntriesStopAtTheBuffer) {
  with_fake({{"RDC_FAKE_CPER_COUNT", "300"}}, [](RdcMetricFetcherImpl& fetcher) {
    std::vector<char> data(64 * 1024);
    uint64_t data_size = data.size();
    std::vector<amdsmi_cper_hdr_t*> headers(256, nullptr);
    uint64_t count = headers.size();
    uint64_t cursor = 0;
    const uint32_t severities = (1u << AMDSMI_CPER_SEV_FATAL) |
                                (1u << AMDSMI_CPER_SEV_NON_FATAL_CORRECTED) |
                                (1u << AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED);
    ASSERT_EQ(amdsmi_get_gpu_cper_entries(handle_of(0), severities, data.data(), &data_size,
                                          headers.data(), &count, &cursor),
              AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(count, 300u);
    rdc_field_value value = dirty_value();
    ASSERT_EQ(fetcher.fetch_smi_field(0, RDC_FI_AFID, &value), RDC_ST_OK);
    EXPECT_EQ(value.type, STRING);
    EXPECT_STREQ(value.value.str, "N/A");  // the fake's record has no AFID
  });
}

// A library that reports success without writing its output (RDC_FAKE_UNFILLED)
// leaves RDC's zeroed structure, not what the stack held: the value read is the
// same after the stack held 0xA5 bytes as after it held 0x5A bytes.
struct Unfilled {
  rdc_field_t field;
  const char* api;
};

class UnfilledOutput : public ::testing::TestWithParam<Unfilled> {};

TEST_P(UnfilledOutput, GivesAValueThatDoesNotDependOnTheStack) {
  const Unfilled c = GetParam();
  with_fake({{"RDC_FAKE_UNFILLED", c.api}}, [&c](RdcMetricFetcherImpl& fetcher) {
    rdc_field_value read[2] = {};
    const unsigned char dirt[2] = {0xA5, 0x5A};
    for (int i = 0; i < 2; ++i) {
      dirty_stack(dirt[i]);
      ASSERT_TRUE(shows_dirt(dirt[i])) << "this build leaves no dirt in a new frame";
      dirty_stack(dirt[i]);
      fetcher.fetch_smi_field(0, c.field, &read[i]);
    }
    EXPECT_EQ(read[0].status, read[1].status);
    EXPECT_EQ(read[0].type, read[1].type);
    EXPECT_EQ(memcmp(&read[0].value, &read[1].value, sizeof(read[0].value)), 0)
        << read[0].value.l_int << " then " << read[1].value.l_int;
  });
}

INSTANTIATE_TEST_SUITE_P(
    FieldValues, UnfilledOutput,
    ::testing::Values(Unfilled{RDC_FI_ECC_SDMA_CE, "amdsmi_get_gpu_ecc_count"},
                      Unfilled{RDC_FI_ECC_CORRECT_TOTAL, "amdsmi_get_gpu_ecc_count"},
                      Unfilled{RDC_FI_ECC_SDMA_DE, "amdsmi_get_gpu_ecc_count"},
                      Unfilled{RDC_FI_ECC_DEFERRED_TOTAL, "amdsmi_get_gpu_ecc_count"},
                      Unfilled{RDC_FI_PCIE_BANDWIDTH, "amdsmi_get_gpu_metrics_info"},
                      Unfilled{RDC_FI_GPU_MEMORY_ACTIVITY, "amdsmi_get_gpu_activity"},
                      Unfilled{RDC_FI_GPU_MEMORY_MAX_BANDWIDTH, "amdsmi_get_gpu_vram_info"},
                      Unfilled{RDC_FI_GPU_MEMORY_CUR_BANDWIDTH, "amdsmi_get_gpu_metrics_info"},
                      Unfilled{RDC_FI_GPU_MEMORY_CUR_BANDWIDTH, "amdsmi_get_gpu_vram_info"},
                      Unfilled{RDC_FI_KFD_ID, "amdsmi_get_gpu_kfd_info"},
                      Unfilled{RDC_FI_GPU_UTIL, "amdsmi_get_gpu_activity"},
                      Unfilled{RDC_FI_DEV_NAME, "amdsmi_get_gpu_asic_info"},
                      Unfilled{RDC_FI_UUID, "amdsmi_get_gpu_asic_info"},
                      Unfilled{RDC_FI_GPU_MM_DEC_UTIL, "amdsmi_get_utilization_count"},
                      Unfilled{RDC_HEALTH_PCIE_REPLAY_COUNT, "amdsmi_get_pcie_info"},
                      Unfilled{RDC_HEALTH_LOW_UTIL_ACC, "amdsmi_get_violation_status"}),
    [](const ::testing::TestParamInfo<Unfilled>& info) {
      return "Field" + std::to_string(info.param.field) + "_" + info.param.api;
    });

// The same for the GPU metrics that a bulk fetch reads.
TEST(FieldValues, ABulkFetchOfUnfilledMetricsDoesNotDependOnTheStack) {
  with_fake({{"RDC_FAKE_UNFILLED", "amdsmi_get_gpu_metrics_info"}},
            [](RdcMetricFetcherImpl& fetcher) {
              rdc_gpu_field_t fields[] = {{0, RDC_FI_GPU_CLOCK}, {0, RDC_FI_GPU_TEMP}};
              std::vector<rdc_gpu_field_value_t> read[2];
              const unsigned char dirt[2] = {0xA5, 0x5A};
              for (int i = 0; i < 2; ++i) {
                dirty_stack(dirt[i]);
                ASSERT_TRUE(shows_dirt(dirt[i])) << "this build leaves no dirt in a new frame";
                dirty_stack(dirt[i]);
                ASSERT_EQ(fetcher.bulk_fetch_smi_fields(fields, 2, read[i]), RDC_ST_OK);
              }
              ASSERT_EQ(read[0].size(), 2u);
              ASSERT_EQ(read[1].size(), 2u);
              for (size_t j = 0; j < 2; ++j) {
                const rdc_field_value& a = read[0][j].field_value;
                const rdc_field_value& b = read[1][j].field_value;
                EXPECT_EQ(a.status, b.status) << "field " << a.field_id;
                EXPECT_EQ(memcmp(&a.value, &b.value, sizeof(a.value)), 0)
                    << "field " << a.field_id << ": " << a.value.l_int << " then " << b.value.l_int;
              }
            });
}

}  // namespace
