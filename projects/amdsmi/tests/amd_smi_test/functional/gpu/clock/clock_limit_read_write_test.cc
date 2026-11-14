// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TestClockLimitReadWrite: checks that amdsmi_set_gpu_clk_limit() really changes the clock limits.
//
// For each GPU:
//   1. Negative tests: null handles and pointers, invalid clock and limit types (RunNegativeTests).
//   2. For each of SYS, MEM, DF (RunClockTest):
//      a. Read the frequency table and clock info. A clock that is not supported ends here.
//      b. Probe: set max, then min, to their starting values (SweepLimits). A limit type that is
//         rejected cannot be set on this clock and is left out of the rest of the sweep.
//      c. Run the plan from BuildPlan, then put back any limit that moved.
//
// After every set, accepted or rejected, ApplyStep checks:
//   - the status is the one the step expects (SUCCESS, or NOT_SUPPORTED for a rejected step)
//   - min_clk and max_clk read back as expected (a rejected set must change nothing)
//   - the perf level is MANUAL (once, after the first accepted set)
//   - the table's levels are unchanged and the current clock is inside the limits
//     (CheckCurrentLevel)
//
// Example plan for a clock with levels 500 and 2200 MHz where both limits can be set
// ("equal" means min = max; "inverted" is a request the driver must reject):
//   window 1 max=900, window 1 min=500, equal min=900, inverted max=500 (rejected),
//   re-apply max=900, inverted min=1800 (rejected), re-apply min=900, window 2 max=2200,
//   window 2 min=1800, then restore min=500.
//
// A failure message starts with "<clock> <step label>", and the label is the one in BuildPlan. Run
// with verbosity 1 to see every API call and what it returned before the failing check.

#include "clock_limit_read_write.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

namespace {

// amdsmi_set_gpu_clk_limit() only operates on these clocks.
const amdsmi_clk_type_t kLimitClkTypes[] = {AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_TYPE_MEM,
                                            AMDSMI_CLK_TYPE_DF};

// The current level is live telemetry and can trail a limit change; re-read it this long.
const int64_t kSettleMs = 2000;  // 2 seconds
const int kSettlePollMs = 100;   // 0.1 seconds

// The reported current clock can sit a few MHz outside a soft limit (seen: 1789 against a 1800 min,
// 896 against 900). 25 MHz is the amdgpu driver's own "equal" window (smu_cmn_freqs_match).
const uint64_t kClockToleranceMhz = 25;

// amdsmi_get_clk_freq() reports Hz; amdsmi_set_gpu_clk_limit() takes MHz.
uint64_t hz_to_mhz(uint64_t hz) { return hz / 1000000ULL; }

// Index 0 is deep sleep, not a selectable DPM level, when the table has it.
uint32_t lowest_selectable_idx(const amdsmi_frequencies_t& f) {
  return (f.has_deep_sleep && f.num_supported > 0) ? 1 : 0;
}

uint64_t lowest_level_mhz(const amdsmi_frequencies_t& f) {
  return hz_to_mhz(f.frequency[lowest_selectable_idx(f)]);
}

uint64_t highest_level_mhz(const amdsmi_frequencies_t& f) {
  return hz_to_mhz(f.frequency[f.num_supported - 1]);
}

// What the first table read, taken before any limit is set, says about the clock.
struct BaselineTable {
  std::vector<uint64_t> levels_hz;  // the selectable levels, excluding live_clock_hz
  // The driver adds a line for the live clock when it sits between levels, so a current entry that
  // is strictly between the lowest and highest may not be a level. It is kept out of levels_hz
  // (e.g. 500, 1815 current, 2200 gives levels 500 and 2200) and ignored in later reads.
  uint64_t live_clock_hz = 0;
  bool had_current = false;  // the clock reported a current level before any limit was set
};

BaselineTable make_baseline(const amdsmi_frequencies_t& f) {
  BaselineTable base;
  base.had_current = f.current < f.num_supported;
  const uint32_t lowest = lowest_selectable_idx(f);
  for (uint32_t i = lowest; i < f.num_supported; ++i) {
    if (i == f.current && i > lowest && i + 1 < f.num_supported) {
      base.live_clock_hz = f.frequency[i];
    } else {
      base.levels_hz.push_back(f.frequency[i]);
    }
  }
  return base;
}

// A value 25% (min) or 75% (max) of the way through the range, rounded down (min) or up (max) to a
// multiple of 100, or of 10 if that leaves the range. The unrounded value is used if neither stays
// inside the range.
uint64_t in_between_mhz(bool is_max, uint64_t range_min_mhz, uint64_t range_max_mhz) {
  const uint64_t range = range_max_mhz - range_min_mhz;
  const uint64_t raw = range_min_mhz + (is_max ? range * 3 : range) / 4;
  for (const uint64_t step : {100, 10}) {
    const uint64_t rounded = is_max ? (raw + step - 1) / step * step : raw / step * step;
    if (rounded > range_min_mhz && rounded < range_max_mhz) {
      return rounded;
    }
  }
  return raw;
}

// How far `mhz` is outside the limits: negative below min, positive above max, 0 if inside.
int64_t outside_limits_mhz(uint64_t mhz, uint64_t min_mhz, uint64_t max_mhz) {
  if (mhz < min_mhz) return -static_cast<int64_t>(min_mhz - mhz);
  if (mhz > max_mhz) return static_cast<int64_t>(mhz - max_mhz);
  return 0;
}

// One table read, split into its selectable levels and what the driver adds around them: the deep
// sleep line, and (per the amdgpu driver) one extra line for the live clock when it sits between
// levels. That extra line is expected, so it is allowed once and used as the current clock.
struct TableView {
  std::vector<uint64_t> levels_hz;
  uint32_t extra_lines = 0;
  bool current_reported = false;
  bool current_in_sleep = false;
  uint64_t current_mhz = 0;
};

TableView view_table(const amdsmi_frequencies_t& f, const BaselineTable& base) {
  TableView view;
  view.current_reported = f.current < f.num_supported;
  for (uint32_t i = 0; i < f.num_supported; ++i) {
    if (f.has_deep_sleep && i == 0) {
      view.current_in_sleep = (i == f.current);
      continue;
    }
    const bool is_level = std::find(base.levels_hz.begin(), base.levels_hz.end(), f.frequency[i]) !=
                          base.levels_hz.end();
    if (f.frequency[i] == base.live_clock_hz) {
      // The baseline's live clock entry: neither a checked level nor an extra line.
    } else if (is_level) {
      view.levels_hz.push_back(f.frequency[i]);
    } else {
      ++view.extra_lines;
    }
    if (i == f.current) {
      view.current_mhz = hz_to_mhz(f.frequency[i]);
    }
  }
  return view;
}

// True when reading again cannot change the outcome, so the wait can stop: the table is wrong
// (reported by the caller), the clock is in deep sleep (below min by design), the clock reports no
// current level when it never did, or the current clock is inside the limits plus the tolerance.
bool reading_is_final(const TableView& view, const BaselineTable& base, uint64_t min_mhz,
                      uint64_t max_mhz) {
  if (view.levels_hz != base.levels_hz || view.extra_lines > 1 || view.current_in_sleep) {
    return true;
  }
  if (!view.current_reported) {
    return !base.had_current;
  }
  return std::abs(outside_limits_mhz(view.current_mhz, min_mhz, max_mhz)) <=
         static_cast<int64_t>(kClockToleranceMhz);
}

}  // namespace

TestClockLimitReadWrite::TestClockLimitReadWrite() : TestBase() {
  set_title("AMDSMI Clock Limit Read/Write Test");
  set_description(
      "The Clock Limit Read/Write test verifies that the sclk/mclk/fclk minimum "
      "and maximum frequency limits can be set through "
      "amdsmi_set_gpu_clk_limit() and restored to their original values.");
}

TestClockLimitReadWrite::~TestClockLimitReadWrite(void) {}

void TestClockLimitReadWrite::SetUp(void) {
  TestBase::SetUp();
  // amdsmi_set_gpu_clk_limit() forces MANUAL even when it returns NOT_SUPPORTED.
  SavePerfLevels();
  return;
}

void TestClockLimitReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestClockLimitReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestClockLimitReadWrite::Close() {
  RestorePerfLevels();
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

// ---- API call helpers ----

// Prints the API and its status, then expects the status to be `want` or `or_want`.
amdsmi_status_t TestClockLimitReadWrite::CheckCall(const char* api, const std::string& args,
                                                   const std::function<amdsmi_status_t()>& call,
                                                   amdsmi_status_t want, amdsmi_status_t or_want) {
  DISPLAY_AMDSMI_API(api, args, VERB(STANDARD));
  const amdsmi_status_t status = call();
  DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, status, want, or_want);
  EXPECT_TRUE(status == want || status == or_want) << api << "(" << args << ") returned " << status;
  return status;
}

// The getters call the API, expect SUCCESS (or `or_want`) and print what it returned.
amdsmi_status_t TestClockLimitReadWrite::GetFreq(const ClockCtx& clock, amdsmi_frequencies_t* f,
                                                 amdsmi_status_t or_want) {
  *f = {};
  const amdsmi_status_t ret = CheckCall(
      "amdsmi_get_clk_freq", clock.desc,
      [&] { return amdsmi_get_clk_freq(clock.handle, clock.clk, f); }, AMDSMI_STATUS_SUCCESS,
      or_want);
  IF_VERB(STANDARD) {
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    const std::string current = (f->current == UINT32_MAX) ? "N/A" : std::to_string(f->current);
    std::cout << "\t**" << clock.name
              << ": has_deep_sleep=" << (f->has_deep_sleep ? "true" : "false")
              << ", num_supported=" << f->num_supported << ", current_index=" << current
              << std::endl;
    for (uint32_t i = 0; i < f->num_supported; ++i) {
      std::cout << "\t**" << clock.name << ": frequency[" << i << "] = " << f->frequency[i]
                << " Hz (" << hz_to_mhz(f->frequency[i]) << " MHz)"
                << (i < lowest_selectable_idx(*f) ? " (deep sleep)" : "")
                << (i == f->current ? " (current)" : "") << std::endl;
    }
  }
  return ret;
}

amdsmi_status_t TestClockLimitReadWrite::GetInfo(const ClockCtx& clock, amdsmi_clk_info_t* info,
                                                 amdsmi_status_t or_want) {
  *info = {};
  const amdsmi_status_t ret = CheckCall(
      "amdsmi_get_clock_info", clock.desc,
      [&] { return amdsmi_get_clock_info(clock.handle, clock.clk, info); }, AMDSMI_STATUS_SUCCESS,
      or_want);
  IF_VERB(STANDARD) {
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    // UINT32_MAX (and 255 for deep sleep) mark a value the library cannot provide.
    auto mhz = [](uint32_t v) {
      return v == UINT32_MAX ? std::string("N/A") : std::to_string(v) + " MHz";
    };
    const std::string deep_sleep =
        (info->clk_deep_sleep == UINT8_MAX)
            ? "N/A"
            : std::to_string(static_cast<unsigned>(info->clk_deep_sleep)) + " MHz";
    std::cout << "\t**" << clock.name << ": clk (current)=" << mhz(info->clk)
              << ", min_clk=" << mhz(info->min_clk) << ", max_clk=" << mhz(info->max_clk)
              << std::endl;
    std::cout << "\t**" << clock.name << ": clk_locked=" << (info->clk_locked ? "true" : "false")
              << ", clk_deep_sleep=" << deep_sleep << std::endl;
  }
  return ret;
}

amdsmi_status_t TestClockLimitReadWrite::GetPerf(const ClockCtx& clock,
                                                 amdsmi_dev_perf_level_t* pfl) {
  *pfl = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;
  const amdsmi_status_t ret = CheckCall("amdsmi_get_gpu_perf_level", clock.desc, [&] {
    return amdsmi_get_gpu_perf_level(clock.handle, pfl);
  });
  IF_VERB(STANDARD) {
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    std::cout << "\t**" << clock.name << ": perf level=" << GetPerfLevelStr(*pfl) << std::endl;
  }
  return ret;
}

// ---- Negative tests ----

// Bad arguments are rejected before the device is touched, so these run first.
void TestClockLimitReadWrite::RunNegativeTests(uint32_t dv_ind) {
  const amdsmi_processor_handle handle = processor_handles_[dv_ind];
  const std::string gpu = "gpu=" + std::to_string(dv_ind);
  const auto bad_clk = static_cast<amdsmi_clk_type_t>(AMDSMI_CLK_TYPE__MAX + 1);
  const auto bad_limit = static_cast<amdsmi_clk_limit_type_t>(AMDSMI_CLK_LIMIT_MAX + 1);
  amdsmi_frequencies_t f{};
  amdsmi_clk_info_t info{};
  amdsmi_dev_perf_level_t pfl = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;

  auto expect_status = [&](const char* api, const std::function<amdsmi_status_t()>& call,
                           amdsmi_status_t want = AMDSMI_STATUS_INVAL) {
    CheckCall(api, gpu, call, want, want);
  };

  IF_VERB(STANDARD) { std::cout << "\t**Negative tests (" << gpu << ")" << std::endl; }

  expect_status("amdsmi_get_clk_freq(nullptr handle)",
                [&] { return amdsmi_get_clk_freq(nullptr, AMDSMI_CLK_TYPE_SYS, &f); });
  for (auto clk : kLimitClkTypes) {
    // Support is checked before the pointer, so an unsupported clock reports NOT_SUPPORTED.
    const bool supported = amdsmi_get_clk_freq(handle, clk, &f) != AMDSMI_STATUS_NOT_SUPPORTED;
    CheckCall(
        "amdsmi_get_clk_freq(nullptr freq)", gpu + ", clk_type=" + FreqEnumToStr(clk),
        [&] { return amdsmi_get_clk_freq(handle, clk, nullptr); },
        supported ? AMDSMI_STATUS_INVAL : AMDSMI_STATUS_NOT_SUPPORTED,
        supported ? AMDSMI_STATUS_INVAL : AMDSMI_STATUS_NOT_SUPPORTED);
  }

  expect_status("amdsmi_get_clock_info(nullptr handle)",
                [&] { return amdsmi_get_clock_info(nullptr, AMDSMI_CLK_TYPE_SYS, &info); });
  expect_status("amdsmi_get_clock_info(nullptr info)",
                [&] { return amdsmi_get_clock_info(handle, AMDSMI_CLK_TYPE_SYS, nullptr); });
  expect_status("amdsmi_get_clock_info(invalid clk_type)",
                [&] { return amdsmi_get_clock_info(handle, bad_clk, &info); });

  expect_status("amdsmi_get_gpu_perf_level(nullptr handle)",
                [&] { return amdsmi_get_gpu_perf_level(nullptr, &pfl); });
  expect_status("amdsmi_get_gpu_perf_level(nullptr perf)",
                [&] { return amdsmi_get_gpu_perf_level(handle, nullptr); });

  // Any value works: the arguments are rejected before it is used.
  auto set_limit = [&](amdsmi_processor_handle h, amdsmi_clk_type_t clk,
                       amdsmi_clk_limit_type_t type) {
    return amdsmi_set_gpu_clk_limit(h, clk, type, 1000);
  };
  expect_status("amdsmi_set_gpu_clk_limit(nullptr handle)",
                [&] { return set_limit(nullptr, AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_LIMIT_MAX); });
  expect_status("amdsmi_set_gpu_clk_limit(clk_type without limits)",
                [&] { return set_limit(handle, AMDSMI_CLK_TYPE_DCEF, AMDSMI_CLK_LIMIT_MAX); });
  expect_status("amdsmi_set_gpu_clk_limit(invalid clk_type)",
                [&] { return set_limit(handle, bad_clk, AMDSMI_CLK_LIMIT_MAX); });
  expect_status("amdsmi_set_gpu_clk_limit(invalid limit_type)",
                [&] { return set_limit(handle, AMDSMI_CLK_TYPE_SYS, bad_limit); });
}

// ---- Positive tests ----

// Always reads the table and limits, so a clock without limit support still gets those checks.
void TestClockLimitReadWrite::RunClockTest(uint32_t dv_ind, amdsmi_clk_type_t clk) {
  const char* name = FreqEnumToStr(clk);
  const ClockCtx clock{processor_handles_[dv_ind], clk, name, dv_ind,
                       "gpu=" + std::to_string(dv_ind) + ", clk_type=" + name};

  amdsmi_frequencies_t f{};
  const amdsmi_status_t freq_ret = GetFreq(clock, &f, AMDSMI_STATUS_NOT_SUPPORTED);
  if (freq_ret == AMDSMI_STATUS_NOT_SUPPORTED) {
    IF_VERB(STANDARD) {
      std::cout << "\t**" << name << ": Not supported on this machine" << std::endl;
    }
  } else {
    if (freq_ret == AMDSMI_STATUS_NOT_YET_IMPLEMENTED ||
        freq_ret == AMDSMI_STATUS_UNEXPECTED_DATA) {
      std::cerr << "WARN: Clock [" << name << "] on device [" << dv_ind << "] returned "
                << (freq_ret == AMDSMI_STATUS_NOT_YET_IMPLEMENTED ? "an empty frequency table"
                                                                  : "unexpected data")
                << ". Likely a driver issue!" << std::endl;
    }

    CHK_ERR_ASRT(freq_ret)  // DO NOT remove or move this check ^ test should fail if clock
                            // files are not in a valid state (eg. driver issues -
                            // missing DPM tables or unexpected frequency data)
  }

  amdsmi_clk_info_t info{};
  const amdsmi_status_t info_ret = GetInfo(clock, &info, AMDSMI_STATUS_NOT_SUPPORTED);
  if (info_ret == AMDSMI_STATUS_SUCCESS && info.min_clk != UINT32_MAX &&
      info.max_clk != UINT32_MAX) {
    EXPECT_LE(info.min_clk, info.max_clk) << name << ": min limit above max limit";
  }

  if (freq_ret != AMDSMI_STATUS_SUCCESS || info_ret != AMDSMI_STATUS_SUCCESS) {
    IF_VERB(STANDARD) {
      std::cout << "\t**" << name << ": table or limits not readable, so no limit sets were tried"
                << std::endl;
    }
    return;
  }
  SweepLimits(clock, f, info);
}

// The values the sweep uses: the lowest level, a value 25% and one 75% through the starting range,
// and the highest level. Only values inside the starting range are kept.
std::vector<uint64_t> TestClockLimitReadWrite::BuildValues(const amdsmi_frequencies_t& f,
                                                           const Limits& start) {
  std::vector<uint64_t> values = {
      lowest_level_mhz(f), in_between_mhz(false, start.min_mhz, start.max_mhz),
      in_between_mhz(true, start.min_mhz, start.max_mhz), highest_level_mhz(f)};
  values.erase(std::remove_if(values.begin(), values.end(),
                              [&](uint64_t v) { return v < start.min_mhz || v > start.max_mhz; }),
               values.end());
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

// Turns the values (sorted, e.g. 500, 900, 1800, 2200) into steps. A window is two consecutive
// values (low, high), set as max=high then min=low. The plan is:
//   1. the first window                      max=900,  min=500
//   2. an equal window (min = max)           min=900
//   3. requests the driver must reject (min above max), each followed by setting that same limit
//      back, because a rejected request leaves its value stored in the driver
//                                            max=500 (rejected), max=900, min=1800 (rejected),
//                                            min=900
//   4. the remaining windows                 max=2200, min=1800
// A limit type that cannot be set keeps its starting value, which is what the other limit is
// measured against, so the equal step moves only the settable one. Rejected requests need both
// types to be settable.
std::vector<TestClockLimitReadWrite::Step> TestClockLimitReadWrite::BuildPlan(
    const std::vector<uint64_t>& values, const Limits& start, bool max_ok, bool min_ok) {
  std::vector<std::pair<uint64_t, uint64_t>> windows;  // (low, high)
  for (size_t i = 0; i + 1 < values.size(); i += 2) {
    windows.emplace_back(values[i], values[i + 1]);
  }
  if (values.size() > 2 && values.size() % 2 == 1) {
    windows.emplace_back(values[values.size() - 2], values.back());
  }

  std::vector<Step> plan;
  if (windows.empty()) {
    return plan;
  }
  auto set_window = [&](const std::string& name, uint64_t low, uint64_t high) {
    if (max_ok) plan.push_back(Step::Valid(AMDSMI_CLK_LIMIT_MAX, name + " max", high));
    if (min_ok) plan.push_back(Step::Valid(AMDSMI_CLK_LIMIT_MIN, name + " min", low));
  };

  const uint64_t first_low = windows[0].first;
  const uint64_t first_high = windows[0].second;
  set_window("window 1", first_low, first_high);

  // Both settable: the max is already first_high, so raising the min to it makes the window equal.
  // Otherwise move the one settable limit to the other's fixed starting value.
  if (max_ok && min_ok) {
    plan.push_back(Step::Valid(AMDSMI_CLK_LIMIT_MIN, "equal min", first_high));
    plan.push_back(Step::Rejected(AMDSMI_CLK_LIMIT_MAX, "inverted max", first_low));
    plan.push_back(Step::Valid(AMDSMI_CLK_LIMIT_MAX, "re-apply max", first_high));
    const auto above = std::upper_bound(values.begin(), values.end(), first_high);
    if (above != values.end()) {
      plan.push_back(Step::Rejected(AMDSMI_CLK_LIMIT_MIN, "inverted min", *above));
      plan.push_back(Step::Valid(AMDSMI_CLK_LIMIT_MIN, "re-apply min", first_high));
    }
  } else if (min_ok) {
    plan.push_back(Step::Valid(AMDSMI_CLK_LIMIT_MIN, "equal min", start.max_mhz));
  } else {
    plan.push_back(Step::Valid(AMDSMI_CLK_LIMIT_MAX, "equal max", start.min_mhz));
  }

  for (size_t i = 1; i < windows.size(); ++i) {
    set_window("window " + std::to_string(i + 1), windows[i].first, windows[i].second);
  }
  return plan;
}

// Probes whether each limit type can be set, then runs the plan, then puts back any limit that
// moved. The limits must end where they started.
void TestClockLimitReadWrite::SweepLimits(const ClockCtx& clock, const amdsmi_frequencies_t& f,
                                          const amdsmi_clk_info_t& info) {
  if (f.num_supported <= lowest_selectable_idx(f)) {
    IF_VERB(STANDARD) {
      std::cout << "\t**" << clock.name << ": No selectable DPM levels. Read checks only."
                << std::endl;
    }
    return;
  }

  ASSERT_GT(lowest_level_mhz(f), 0u) << clock.name << ": lowest selectable level is 0 MHz";
  ASSERT_LE(lowest_level_mhz(f), highest_level_mhz(f))
      << clock.name << ": lowest level above the highest";
  ASSERT_NE(info.min_clk, UINT32_MAX) << clock.name << ": driver reports no min limit";
  ASSERT_NE(info.max_clk, UINT32_MAX) << clock.name << ": driver reports no max limit";

  const Limits start{info.min_clk, info.max_clk};
  SweepState sw{clock, f, start, false};

  // Probe with the starting values: a limit that is rejected here cannot be set on this clock.
  const bool max_ok = ApplyStep(&sw, Step::Probe(AMDSMI_CLK_LIMIT_MAX, "probe max",
                                                 start.max_mhz)) == AMDSMI_STATUS_SUCCESS;
  const bool min_ok = ApplyStep(&sw, Step::Probe(AMDSMI_CLK_LIMIT_MIN, "probe min",
                                                 start.min_mhz)) == AMDSMI_STATUS_SUCCESS;
  if (!max_ok && !min_ok) {
    IF_VERB(STANDARD) {
      std::cout
          << "\t**" << clock.name
          << ": limits not settable on this clock (both probe sets rejected, limits unchanged)"
          << std::endl;
    }
    return;
  }

  // The first unexpected status ends the plan (it is already reported); the restore still runs.
  for (const Step& step : BuildPlan(BuildValues(f, start), start, max_ok, min_ok)) {
    const amdsmi_status_t ret = ApplyStep(&sw, step);
    if (ret != step.want && ret != step.or_want) {
      break;
    }
  }

  if (min_ok && sw.expect.min_mhz != start.min_mhz) {
    ApplyStep(&sw, Step::Valid(AMDSMI_CLK_LIMIT_MIN, "restore min", start.min_mhz));
  }
  if (max_ok && sw.expect.max_mhz != start.max_mhz) {
    ApplyStep(&sw, Step::Valid(AMDSMI_CLK_LIMIT_MAX, "restore max", start.max_mhz));
  }
  EXPECT_EQ(sw.expect.min_mhz, start.min_mhz) << clock.name << ": min limit not restored";
  EXPECT_EQ(sw.expect.max_mhz, start.max_mhz) << clock.name << ": max limit not restored";
}

// Sets one limit, then checks the device. A rejected set must leave both limits untouched, so the
// checks run either way. Returns the status of the set.
amdsmi_status_t TestClockLimitReadWrite::ApplyStep(SweepState* sw, const Step& step) {
  const ClockCtx& clock = sw->clock;
  const std::string where = std::string(clock.name) + " " + step.label;
  const std::string args =
      clock.desc + ", clk_limit=" + step.label + ", value=" + std::to_string(step.target_mhz);
  const amdsmi_status_t ret = CheckCall(
      "amdsmi_set_gpu_clk_limit", args,
      [&] { return amdsmi_set_gpu_clk_limit(clock.handle, clock.clk, step.type, step.target_mhz); },
      step.want, step.or_want);
  if (ret == AMDSMI_STATUS_SUCCESS) {
    (step.type == AMDSMI_CLK_LIMIT_MAX ? sw->expect.max_mhz : sw->expect.min_mhz) = step.target_mhz;
  }

  // Right after the first accepted set, so nothing else has time to change the mode.
  amdsmi_dev_perf_level_t pfl;
  if (ret == AMDSMI_STATUS_SUCCESS && !sw->manual_checked) {
    sw->manual_checked = true;
    if (GetPerf(clock, &pfl) == AMDSMI_STATUS_SUCCESS) {
      EXPECT_EQ(pfl, AMDSMI_DEV_PERF_LEVEL_MANUAL) << where;
    }
  }

  amdsmi_clk_info_t info;
  if (GetInfo(clock, &info) == AMDSMI_STATUS_SUCCESS) {
    EXPECT_EQ(info.max_clk, sw->expect.max_mhz) << where << ": max";
    EXPECT_EQ(info.min_clk, sw->expect.min_mhz) << where << ": min";
  }
  CheckCurrentLevel(*sw, step.label);
  return ret;
}

bool TestClockLimitReadWrite::ReadFreqUntil(
    const ClockCtx& clock, const std::function<bool(const amdsmi_frequencies_t&)>& settled,
    amdsmi_frequencies_t* out) {
  const auto start = std::chrono::steady_clock::now();
  int reads = 0;
  for (;;) {
    if (GetFreq(clock, out) != AMDSMI_STATUS_SUCCESS) {
      return false;
    }
    ++reads;

    const int64_t waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
    if (settled(*out) || waited_ms >= kSettleMs) {
      if (reads > 1) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" << clock.name << ": current level checked after " << waited_ms
                    << " ms" << std::endl;
        }
      }
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kSettlePollMs));
  }
}

// The table's selectable levels must not change, and the current clock must sit inside the limits
// (give or take kClockToleranceMhz). The current clock is live telemetry, so keep re-reading until
// reading_is_final() says another read would not change the outcome.
void TestClockLimitReadWrite::CheckCurrentLevel(const SweepState& sw, const std::string& label) {
  const ClockCtx& clock = sw.clock;
  const Limits& expect = sw.expect;
  const std::string where = std::string(clock.name) + " " + label;
  const BaselineTable base = make_baseline(sw.baseline);

  amdsmi_frequencies_t cur{};
  const auto settled = [&](const amdsmi_frequencies_t& f) {
    return reading_is_final(view_table(f, base), base, expect.min_mhz, expect.max_mhz);
  };
  if (!ReadFreqUntil(clock, settled, &cur)) {
    return;
  }

  const TableView view = view_table(cur, base);
  EXPECT_EQ(view.levels_hz, base.levels_hz) << where << ": table levels changed";
  // The driver adds at most one extra line, for the live clock when it sits between levels.
  EXPECT_LE(view.extra_lines, 1u) << where << ": unexpected table lines";
  if (view.levels_hz != base.levels_hz || view.current_in_sleep) {
    return;
  }
  if (!view.current_reported) {
    if (base.had_current) {
      std::cerr << "WARN: " << where << ": no current level reported" << std::endl;
    }
    return;
  }

  const int64_t outside = outside_limits_mhz(view.current_mhz, expect.min_mhz, expect.max_mhz);
  if (outside != 0) {
    IF_VERB(STANDARD) {
      std::cout << "\t**" << where << ": current clock " << view.current_mhz << " MHz is "
                << std::abs(outside) << " MHz " << (outside < 0 ? "below min " : "above max ")
                << (outside < 0 ? expect.min_mhz : expect.max_mhz) << " MHz (tolerance "
                << kClockToleranceMhz << " MHz)" << std::endl;
    }
  }
  EXPECT_GE(outside, -static_cast<int64_t>(kClockToleranceMhz))
      << where << ": current clock " << view.current_mhz << " MHz is below the min limit";
  EXPECT_LE(outside, static_cast<int64_t>(kClockToleranceMhz))
      << where << ": current clock " << view.current_mhz << " MHz is above the max limit";
}

void TestClockLimitReadWrite::Run(void) {
  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);
    RunNegativeTests(dv_ind);
    IF_VERB(STANDARD) { std::cout << "\t\n**Positive tests (gpu=" << dv_ind << ")" << std::endl; }
    for (auto clk : kLimitClkTypes) {
      RunClockTest(dv_ind, clk);
    }
  }
}
