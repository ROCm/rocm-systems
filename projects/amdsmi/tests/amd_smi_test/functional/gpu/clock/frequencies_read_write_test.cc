// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "frequencies_read_write.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "test_common.h"

TestFrequenciesReadWrite::TestFrequenciesReadWrite() : TestBase() {
  set_title("AMDSMI Frequencies Read/Write Test");
  set_description(
      "The Frequencies tests verify that the frequency "
      "settings can be read and controlled properly.");
}

TestFrequenciesReadWrite::~TestFrequenciesReadWrite(void) {}

void TestFrequenciesReadWrite::SetUp(void) {
  TestBase::SetUp();
  // Capture the per-device performance level so Close() can restore it and
  // avoid leaking a forced 'manual' perf state to later tests.
  SavePerfLevels();
  return;
}

void TestFrequenciesReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestFrequenciesReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFrequenciesReadWrite::Close() {
  // Restore the performance level captured in SetUp() before amdsmi is shut
  // down. Runs even if Run() aborted on a failed assertion, so this test cannot
  // leak a forced 'manual' state to subsequent tests.
  RestorePerfLevels();
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

// Number of writable DPM labels. The deep-sleep pseudo-level ('S') is listed in
// the table but has no settable numeric label, so it is excluded.
static uint32_t settable_level_count(const amdsmi_frequencies_t& f) {
  return (f.has_deep_sleep && f.num_supported > 0) ? f.num_supported - 1 : f.num_supported;
}

// The subset of a requested set-mask that can actually take effect: the bits
// within the settable label range [0, settable_level_count). amd-smi (and the
// driver) drop bits above that, so they never apply.
static uint64_t effective_level_mask(uint64_t requested_mask, const amdsmi_frequencies_t& f) {
  uint32_t settable = settable_level_count(f);
  uint64_t in_range = (settable >= 64) ? ~0ULL : ((1ULL << settable) - 1);
  return requested_mask & in_range;
}

// Binary string for a bitmask, trimmed of leading zeros (down to at least one
// digit) for display.
static std::string to_bit_string(uint64_t mask) {
  std::string s = std::bitset<AMDSMI_MAX_NUM_FREQUENCIES>(mask).to_string();
  s.erase(0, std::min(s.find_first_not_of('0'), s.size() - 1));
  return s;
}

void TestFrequenciesReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_frequencies_t f;
  uint64_t freq_bitmask;
  amdsmi_clk_type_t amdsmi_clk;
  const std::map<amdsmi_clk_type_t, std::string> clk_type_map = {
      {AMDSMI_CLK_TYPE_SYS, "SYS"},     {AMDSMI_CLK_TYPE_GFX, "GFX"},
      {AMDSMI_CLK_TYPE_DF, "DF"},       {AMDSMI_CLK_TYPE_DCEF, "DCEF"},
      {AMDSMI_CLK_TYPE_SOC, "SOC"},     {AMDSMI_CLK_TYPE_MEM, "MEM"},
      {AMDSMI_CLK_TYPE_PCIE, "PCIE"},   {AMDSMI_CLK_TYPE_VCLK0, "VCLK0"},
      {AMDSMI_CLK_TYPE_VCLK1, "VCLK1"}, {AMDSMI_CLK_TYPE_DCLK0, "DCLK0"},
      {AMDSMI_CLK_TYPE_DCLK1, "DCLK1"},
  };

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    // Optional debug filter: set AMDSMI_TEST_CLK to a clock name (e.g. "SYS",
    // "GFX", "DF", "SOC", "MEM", "VCLK0", ...) to restrict this test to a
    // single clock type. Useful for isolating which clock a set error comes
    // from by running the test once per clock. Unset => exercise all clocks.
    const char* only_clk_env = std::getenv("AMDSMI_TEST_CLK");

    for (uint32_t clk = AMDSMI_CLK_TYPE_FIRST; clk <= AMDSMI_CLK_TYPE__MAX; ++clk) {
      amdsmi_clk = (amdsmi_clk_type_t)clk;

      if (only_clk_env != nullptr) {
        auto name_it = clk_type_map.find(amdsmi_clk);
        std::string clk_name = (name_it != clk_type_map.end()) ? name_it->second : "";
        if (clk_name != only_clk_env) {
          continue;  // skip clocks that don't match AMDSMI_TEST_CLK
        }
      }

      auto freq_read = [&]() -> bool {
        // Skip AMDSMI_CLK_TYPE_PCIE, which does not supported in rocm-smi.
        if (auto it = clk_type_map.find(amdsmi_clk); it != clk_type_map.end()) {
          if (amdsmi_clk == AMDSMI_CLK_TYPE_PCIE) {
            return false;  // Quietly skip PCIE clock
                           // Cannot read/write to PCIE clock in driver
          }
          std::cout << "amdsmi_get_clk_freq(" << it->second << ", f)" << std::endl;
        }

        DISPLAY_AMDSMI_API("amdsmi_get_clk_freq(" + std::string(FreqEnumToStr(amdsmi_clk)) + ")",
                           "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
        ret = amdsmi_get_clk_freq(processor_handles_[dv_ind], amdsmi_clk, &f);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
        if (auto it = clk_type_map.find(amdsmi_clk); it != clk_type_map.end()) {
          std::cout << ": " << smi_amdgpu_get_status_string(ret, false) << std::endl;
        }

        IF_VERB(STANDARD) {
          if (ret == AMDSMI_STATUS_SUCCESS) {
            std::cout << "\t**" << FreqEnumToStr(amdsmi_clk) << ":\n"
                      << "\t\t**Is in deep sleep: " << (f.has_deep_sleep ? "Yes" : "No") << "\n"
                      << "\t\t**Number of supported frequencies: " << f.num_supported << "\n"
                      << "\t\t**Current frequency index: " << f.current << "\n"
                      << "\t\t**Supported frequencies: ";
            for (uint32_t i = 0; i < f.num_supported; ++i) {
              std::cout << f.frequency[i] << " Hz";
              if (i < f.num_supported - 1) {
                std::cout << ", ";
              } else {
                std::cout << std::endl;
              }
            }
          }
        }

        if (ret == AMDSMI_STATUS_NOT_SUPPORTED || ret == AMDSMI_STATUS_NOT_YET_IMPLEMENTED) {
          std::cout << "\t**Set " << FreqEnumToStr(amdsmi_clk) << ": Not supported on this machine"
                    << std::endl;
          return false;
        }

        IF_VERB(STANDARD) {
          std::cout << "Initial frequency for clock " << FreqEnumToStr(amdsmi_clk) << " is "
                    << f.current << std::endl;
        }
        return true;
      };

      auto freq_write = [&]() {
        // Exercises the lenient-set contract in four phases:
        // (1) a mask entirely above the settable range must return INVAL,
        // (2) every settable level, singly and paired, must be settable,
        // (3) resetting to "all frequencies" must succeed (or legitimately INVAL if
        // nothing is settable),
        // (4) perf level is restored to AUTO.

        // Set clocks to something other than the usual default of the lowest
        // frequency.
        // Skip AMDSMI_CLK_TYPE_PCIE, which does not supported in rocm-smi.
        if (amdsmi_clk == AMDSMI_CLK_TYPE_PCIE) return;

        uint32_t settable = settable_level_count(f);

        // False when a set failure just means "unsupported on this ASIC", not
        // a real test failure; also prints why the rest of the check is
        // skipped. Shared by the negative case and the per-level loop below.
        auto is_set_supported = [&](amdsmi_status_t set_ret) {
          if ((set_ret == AMDSMI_STATUS_NO_PERM && geteuid() == 0) ||
              (set_ret == AMDSMI_STATUS_NOT_SUPPORTED)) {
            std::cout << "\t**Set " << FreqEnumToStr(amdsmi_clk)
                      << ": Not supported on this machine. Skipping..." << std::endl;
            return false;
          }
          return true;
        };

        // "amdsmi_set_clk_freq(<clock>, <value>)", shared by every set call
        // below so the clock name is only spelled out once.
        auto set_api_str = [&](const std::string& value) {
          return "amdsmi_set_clk_freq(" + std::string(FreqEnumToStr(amdsmi_clk)) + ", " + value +
                 ")";
        };

        // "gpu=<dv_ind>, VALID"/"gpu=<dv_ind>, INVALID" tag for the API call
        // lines below, so it's clear at a glance which outcome is expected.
        auto gpu_tag_str = [&](bool valid) {
          return "gpu=" + std::to_string(dv_ind) + (valid ? ", VALID" : ", INVALID");
        };

        // Re-reads f (the settable range can shrink live); callers assert
        // whatever f-derived condition justifies the INVAL they just saw.
        auto refresh_f = [&]() {
          DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "gpu=" + std::to_string(dv_ind),
                             VERB(STANDARD));
          amdsmi_status_t refresh_ret =
              amdsmi_get_clk_freq(processor_handles_[dv_ind], amdsmi_clk, &f);
          DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, refresh_ret,
                                AMDSMI_STATUS_SUCCESS);
          return refresh_ret;
        };

        // Negative case: a mask entirely above the settable range must be
        // rejected with INVAL, regardless of how many levels are settable.
        if (settable > 0) {
          uint64_t invalid_bitmask = 1ULL << settable;
          std::string invalid_bm_str = to_bit_string(invalid_bitmask);

          IF_VERB(STANDARD) {
            std::cout << "Setting frequency mask for " << FreqEnumToStr(amdsmi_clk) << " to 0b"
                      << invalid_bm_str << " (level " << settable
                      << ", out-of-range, expect AMDSMI_STATUS_INVAL) ..." << std::endl;
          }
          DISPLAY_AMDSMI_API(set_api_str("0b" + invalid_bm_str), gpu_tag_str(false),
                             VERB(STANDARD));
          amdsmi_status_t invalid_ret =
              amdsmi_set_clk_freq(processor_handles_[dv_ind], amdsmi_clk, invalid_bitmask);
          DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, invalid_ret,
                                AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED,
                                AMDSMI_STATUS_NO_PERM);
          ASSERT_TRUE(invalid_ret == AMDSMI_STATUS_INVAL ||
                      invalid_ret == AMDSMI_STATUS_NOT_SUPPORTED ||
                      invalid_ret == AMDSMI_STATUS_NO_PERM);
          if (!is_set_supported(invalid_ret)) {
            return;
          }
          ASSERT_EQ(invalid_ret, AMDSMI_STATUS_INVAL);
        }

        // Positive case: exercise every settable DPM level - bit i alone, or
        // paired with i+1 when both are settable, so both single- and
        // multi-bit masks get a real (non-INVAL) write on every device
        // regardless of table size.
        // f.frequency[] is positional: index 0 is the deep-sleep entry when
        // present, so the first settable level is offset by 1 in that case.
        uint32_t freq_idx_offset = f.has_deep_sleep ? 1 : 0;
        std::string set_api;
        for (uint32_t i = 0; i < settable; ++i) {
          freq_bitmask = (1ULL << i);
          if (i + 1 < settable) {
            freq_bitmask |= (1ULL << (i + 1));
          }

          std::string freq_bm_str = to_bit_string(freq_bitmask);

          IF_VERB(STANDARD) {
            std::cout << "Setting frequency mask for " << FreqEnumToStr(amdsmi_clk) << " to 0b"
                      << freq_bm_str << " (" << f.frequency[i + freq_idx_offset] << " Hz";
            if (i + 1 < settable) {
              std::cout << ", " << f.frequency[i + 1 + freq_idx_offset] << " Hz";
            }
            std::cout << ", expect AMDSMI_STATUS_SUCCESS) ..." << std::endl;
          }
          set_api = set_api_str("0b" + freq_bm_str);
          DISPLAY_AMDSMI_API(set_api, gpu_tag_str(true), VERB(STANDARD));
          ret = amdsmi_set_clk_freq(processor_handles_[dv_ind], amdsmi_clk, freq_bitmask);
          DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                                AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED,
                                AMDSMI_STATUS_NO_PERM);
          ASSERT_TRUE(ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_INVAL ||
                      ret == AMDSMI_STATUS_NOT_SUPPORTED || ret == AMDSMI_STATUS_NO_PERM);
          if (!is_set_supported(ret)) {
            return;
          }
          if (ret == AMDSMI_STATUS_INVAL) {
            // The settable range can shrink between planning this mask and the
            // driver processing it (live power-state fluctuation, same as
            // deep-sleep detection). Re-read after the fact - not before - so
            // there's no window left for the state to change again before we
            // judge whether this INVAL was actually justified.
            ASSERT_EQ(refresh_f(), AMDSMI_STATUS_SUCCESS);
            ASSERT_GE(i, settable_level_count(f));
            continue;
          }
          CHK_ERR_ASRT(ret)
        }

        // Refresh f for the printout below; the reset step's own INVAL
        // check re-reads again regardless, so this is mainly a post-loop
        // sanity check plus display info, not load-bearing for the reset.
        ASSERT_EQ(refresh_f(), AMDSMI_STATUS_SUCCESS);
        IF_VERB(STANDARD) {
          std::cout << "Frequency is now index " << f.current << std::endl;
          std::cout << "Resetting mask to all frequencies." << std::endl;
        }

        freq_bitmask = 0xFFFFFFFF;
        set_api = set_api_str("0xFFFFFFFF");
        DISPLAY_AMDSMI_API(set_api, gpu_tag_str(true), VERB(STANDARD));
        ret = amdsmi_set_clk_freq(processor_handles_[dv_ind], amdsmi_clk, freq_bitmask);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                              AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED,
                              AMDSMI_STATUS_NO_PERM);
        ASSERT_TRUE(ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_INVAL ||
                    ret == AMDSMI_STATUS_NOT_SUPPORTED || ret == AMDSMI_STATUS_NO_PERM);
        if (!is_set_supported(ret)) {
          return;
        }
        if (ret == AMDSMI_STATUS_INVAL) {
          // has_deep_sleep can change between reads, so re-read f here instead
          // of reusing the snapshot taken before this reset attempt.
          ASSERT_EQ(refresh_f(), AMDSMI_STATUS_SUCCESS);
          // Only tolerated for a clock with no settable level (e.g. deep-sleep
          // only); otherwise the "enable all" reset must succeed.
          ASSERT_EQ(effective_level_mask(freq_bitmask, f), 0ULL);
          return;
        }
        CHK_ERR_ASRT(ret)

        DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                           VERB(STANDARD));
        ret = amdsmi_set_gpu_perf_level(processor_handles_[dv_ind], AMDSMI_DEV_PERF_LEVEL_AUTO);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                              AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM);
        ASSERT_TRUE(ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_NOT_SUPPORTED ||
                    ret == AMDSMI_STATUS_NO_PERM);
        if (is_set_supported(ret)) {
          CHK_ERR_ASRT(ret)
        }
      };

      if (freq_read()) {
        CHK_ERR_ASRT(ret)
      } else {
        continue;
      }
      freq_write();
    }
  }
}
