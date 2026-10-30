// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/gpu/partition/memorypartition_read_write.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace {

const char* ModeName(amdsmi_memory_partition_type_t mode) {
  switch (mode) {
    case AMDSMI_MEMORY_PARTITION_NPS1:
      return "NPS1";
    case AMDSMI_MEMORY_PARTITION_NPS2:
      return "NPS2";
    case AMDSMI_MEMORY_PARTITION_NPS4:
      return "NPS4";
    case AMDSMI_MEMORY_PARTITION_NPS8:
      return "NPS8";
    default:
      return "UNKNOWN";
  }
}

amdsmi_memory_partition_type_t ModeOf(const std::string& name) {
  for (auto mode : {AMDSMI_MEMORY_PARTITION_NPS1, AMDSMI_MEMORY_PARTITION_NPS2,
                    AMDSMI_MEMORY_PARTITION_NPS4, AMDSMI_MEMORY_PARTITION_NPS8}) {
    if (name == ModeName(mode)) {
      return mode;
    }
  }
  return AMDSMI_MEMORY_PARTITION_UNKNOWN;
}

// Profile index i selects kProfileTypes[i], as the profile config of a real device does.
const amdsmi_accelerator_partition_type_t kProfileTypes[] = {
    AMDSMI_ACCELERATOR_PARTITION_SPX, AMDSMI_ACCELERATOR_PARTITION_DPX,
    AMDSMI_ACCELERATOR_PARTITION_QPX, AMDSMI_ACCELERATOR_PARTITION_CPX};

struct PartitionDevice {
  std::string memory = "NPS2";
  amdsmi_status_t memory_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_accelerator_partition_profile_t profile = {};
  amdsmi_status_t profile_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_memory_partition_type_t staged = AMDSMI_MEMORY_PARTITION_UNKNOWN;
  std::vector<amdsmi_memory_partition_type_t> requests;
  amdsmi_nps_caps_t caps = {{1, 1, 1, 1, 0}};
  // Returned by caps reads once the cycle has keyed the device; the default path reads
  // the caps too and must not see it.
  amdsmi_status_t caps_status = AMDSMI_STATUS_SUCCESS;
  bool keyed = false;
  // Handles of one physical GPU share a bus; only partition 0 speaks for it.
  uint8_t bus = 0;
  uint32_t partition_id = 0;
  // Valid modes the device answers INVAL for, as for a mode the hive cannot run.
  std::vector<amdsmi_memory_partition_type_t> rejects;
  // Statuses returned by successive profile sets before one succeeds.
  std::vector<amdsmi_status_t> profile_set_statuses;
  // Runs after every set request, to change the device partway through a run.
  std::function<void(PartitionDevice&, amdsmi_memory_partition_type_t)> on_set;

  bool RequestedValidMode() const {
    return std::any_of(requests.begin(), requests.end(), [](amdsmi_memory_partition_type_t mode) {
      return mode != AMDSMI_MEMORY_PARTITION_UNKNOWN;
    });
  }
};

// The host the reload cycle sees: sysfs, /proc/self/status and modprobe, following amdgpu.
struct FakeHost {
  bool modprobe_installed = true;
  bool can_load_modules = true;
  bool built_in = false;
  bool loaded = true;
  int other_users = 0;      // amdgpu references held by other processes
  std::string refcnt_text;  // replaces the refcnt file when set
  int unload_failures = 0;
  int load_failures = 0;
  bool load_stuck = false;
  bool applies_staged_modes = true;
  int enumerate_failures = 0;  // device enumerations that fail after each load
  std::vector<std::string> modprobe_calls;
  std::vector<std::string> applied;  // first GPU's mode after each load
};

// Follows the driver: a mode other than the active one stays staged until the next reload,
// and a request for the active mode is ignored. A reload applies each staged mode and resets
// that GPU to the first accelerator profile, the default after an NPS change.
class MemoryPartitionHarness : public TestMemoryPartitionReadWrite {
 public:
  explicit MemoryPartitionHarness(uint32_t count) : devices(count) {
    set_verbosity(0);
    set_dont_fail(false);
    set_num_monitor_devs(count);
    allow_destructive_ = false;
    for (uint32_t i = 0; i < count; ++i) {
      devices[i].profile.profile_type = AMDSMI_ACCELERATOR_PARTITION_SPX;
      devices[i].bus = static_cast<uint8_t>(i);
      processor_handles_[i] = &devices[i];
    }
    api_.sleep_ms = [](uint32_t) {};
    api_.set_profile = [](amdsmi_processor_handle handle, uint32_t index) {
      auto& device = Device(handle);
      if (!device.profile_set_statuses.empty()) {
        const auto status = device.profile_set_statuses.front();
        device.profile_set_statuses.erase(device.profile_set_statuses.begin());
        if (status != AMDSMI_STATUS_SUCCESS) {
          return status;
        }
      }
      device.profile.profile_index = index;
      device.profile.profile_type = kProfileTypes[index];
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_profile = [](amdsmi_processor_handle handle,
                          amdsmi_accelerator_partition_profile_t* profile, uint32_t* ids) {
      const auto& device = Device(handle);
      *profile = device.profile;
      std::memset(ids, 0, 8 * sizeof(*ids));
      return device.profile_status;
    };
    api_.get_profile_config = [](amdsmi_processor_handle,
                                 amdsmi_accelerator_partition_profile_config_t* config) {
      *config = {};
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_xcd_counter = [](amdsmi_processor_handle, uint16_t* count) {
      *count = 8;
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_memory = [](amdsmi_processor_handle handle, char* value, uint32_t size) {
      const auto& device = Device(handle);
      if (!value || !size) {
        return AMDSMI_STATUS_INVAL;
      }
      if (device.memory_status != AMDSMI_STATUS_SUCCESS) {
        return device.memory_status;
      }
      if (size <= device.memory.size()) {
        return AMDSMI_STATUS_INSUFFICIENT_SIZE;
      }
      std::memcpy(value, device.memory.c_str(), device.memory.size() + 1);
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_memory_config = [](amdsmi_processor_handle handle,
                                amdsmi_memory_partition_config_t* config) {
      if (!config) {
        return AMDSMI_STATUS_INVAL;
      }
      *config = {};
      config->mp_mode = ModeOf(Device(handle).memory);
      config->partition_caps = Device(handle).caps;
      return Device(handle).keyed ? Device(handle).caps_status : AMDSMI_STATUS_SUCCESS;
    };
    api_.set_memory = [](amdsmi_processor_handle handle, amdsmi_memory_partition_type_t mode) {
      auto& device = Device(handle);
      device.requests.push_back(mode);
      const bool valid =
          ModeOf(ModeName(mode)) != AMDSMI_MEMORY_PARTITION_UNKNOWN &&
          std::find(device.rejects.begin(), device.rejects.end(), mode) == device.rejects.end();
      if (valid && device.memory != ModeName(mode)) {
        device.staged = mode;
      }
      if (device.on_set) {
        device.on_set(device, mode);
      }
      return valid ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_INVAL;
    };
    api_.get_kfd_info = [](amdsmi_processor_handle handle, amdsmi_kfd_info_t* info) {
      *info = {};
      Device(handle).keyed = true;
      info->current_partition_id = Device(handle).partition_id;
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_bdf = [](amdsmi_processor_handle handle, amdsmi_bdf_t* bdf) {
      *bdf = {};
      bdf->bus_number = Device(handle).bus;
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.shut_down = [this]() {
      library_open = false;
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.read_file = [this](const std::string& path) -> std::string {
      if (path == "/proc/self/status") {
        return std::string("Name:\tamdsmitst\nCapEff:\t") +
               (host.can_load_modules ? "000001ffffffffff" : "0000000000000000") + "\n";
      }
      if (host.built_in || !host.loaded) {
        return "";
      }
      if (path == "/sys/module/amdgpu/refcnt") {
        return host.refcnt_text.empty()
                   ? std::to_string(host.other_users + (library_open ? 1 : 0)) + "\n"
                   : host.refcnt_text;
      }
      return path == "/sys/module/amdgpu/initstate" ? "live\n" : "";
    };
    api_.is_executable = [this](const std::string& path) {
      return host.modprobe_installed && path == "/usr/sbin/modprobe";
    };
    api_.run = [this](const std::string&, const std::vector<std::string>& args,
                      std::string* error) { return Modprobe(args, error); };
  }

  void PrintDeviceHeader(amdsmi_processor_handle) override {}

  bool RefreshHandles(std::string* error) override {
    library_open = false;
    set_num_monitor_devs(0);
    if (!host.loaded || failing_enumerations > 0) {
      failing_enumerations -= host.loaded ? 1 : 0;
      *error = "no devices";
      return false;
    }
    library_open = true;
    set_num_monitor_devs(static_cast<uint32_t>(devices.size()));
    return true;
  }

  RunResult Modprobe(const std::vector<std::string>& args, std::string* error) {
    const bool unload = args.size() == 2 && args[0] == "-r";
    host.modprobe_calls.push_back(unload ? "-r amdgpu" : "amdgpu");
    if (unload) {
      if (host.unload_failures > 0 || host.other_users > 0 || library_open) {
        host.unload_failures -= host.unload_failures > 0 ? 1 : 0;
        *error = "modprobe -r amdgpu failed";
        return RunResult::kFailed;
      }
      host.loaded = false;
      return RunResult::kOk;
    }
    if (host.load_stuck) {
      *error = "modprobe amdgpu ignored SIGKILL";
      return RunResult::kStuck;
    }
    if (host.load_failures > 0) {
      --host.load_failures;
      *error = "modprobe amdgpu failed";
      return RunResult::kFailed;
    }
    host.loaded = true;
    failing_enumerations = host.enumerate_failures;
    for (auto& device : devices) {
      if (device.staged != AMDSMI_MEMORY_PARTITION_UNKNOWN && host.applies_staged_modes) {
        device.memory = ModeName(device.staged);
        device.profile.profile_index = 0;
        device.profile.profile_type = kProfileTypes[0];
      }
      device.staged = AMDSMI_MEMORY_PARTITION_UNKNOWN;
    }
    host.applied.push_back(devices[0].memory);
    return RunResult::kOk;
  }

  void EnableCycle() { allow_destructive_ = true; }

  std::vector<PartitionDevice> devices;
  FakeHost host;
  bool library_open = true;
  int failing_enumerations = 0;

 private:
  static PartitionDevice& Device(amdsmi_processor_handle handle) {
    return *static_cast<PartitionDevice*>(handle);
  }
};

struct Outcome {
  std::vector<std::string> failures;
  bool skipped = false;

  bool Mentions(const std::string& text) const {
    return std::any_of(failures.begin(), failures.end(), [&](const std::string& failure) {
      return failure.find(text) != std::string::npos;
    });
  }
};

Outcome RunTest(TestMemoryPartitionReadWrite* test) {
  testing::TestPartResultArray results;
  {
    testing::ScopedFakeTestPartResultReporter reporter(
        testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
    test->Run();
  }
  Outcome outcome;
  for (int i = 0; i < results.size(); ++i) {
    const auto& result = results.GetTestPartResult(i);
    if (result.skipped()) {
      outcome.skipped = true;
    } else if (result.failed()) {
      outcome.failures.push_back(result.message());
    }
  }
  return outcome;
}

size_t Failures(TestMemoryPartitionReadWrite* test) { return RunTest(test).failures.size(); }

std::string Describe(const Outcome& outcome) {
  std::string text;
  for (const auto& failure : outcome.failures) {
    text += "\n  " + failure;
  }
  return text.empty() ? " none" : text;
}

TEST(GpuUnit, MemoryPartitionRequestsOnlyTheActiveMode) {
  MemoryPartitionHarness test(2);
  test.devices[0].memory = "NPS1";
  test.devices[1].memory = "NPS4";
  EXPECT_EQ(Failures(&test), 0u);
  for (const auto& device : test.devices) {
    EXPECT_EQ(device.staged, AMDSMI_MEMORY_PARTITION_UNKNOWN) << "left a mode staged";
    EXPECT_NE(std::find(device.requests.begin(), device.requests.end(), ModeOf(device.memory)),
              device.requests.end())
        << "never requested the active mode";
  }
}

TEST(GpuUnit, MemoryPartitionFailsOnChangedOrUnreadableFinalState) {
  const std::vector<std::function<void(PartitionDevice&)>> changes = {
      [](PartitionDevice& device) { device.memory = "NPS4"; },
      [](PartitionDevice& device) {
        device.profile.profile_type = AMDSMI_ACCELERATOR_PARTITION_DPX;
      },
      [](PartitionDevice& device) { device.profile.profile_index = 3; },
      [](PartitionDevice& device) { device.memory_status = AMDSMI_STATUS_FILE_ERROR; },
      [](PartitionDevice& device) { device.profile_status = AMDSMI_STATUS_NO_PERM; },
  };
  for (size_t i = 0; i < changes.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "change " << i);
    MemoryPartitionHarness test(1);
    test.devices[0].on_set = [&change = changes[i]](PartitionDevice& device,
                                                    amdsmi_memory_partition_type_t mode) {
      if (mode != AMDSMI_MEMORY_PARTITION_UNKNOWN) {
        change(device);
      }
    };
    EXPECT_GT(Failures(&test), 0u);
  }
}

TEST(GpuUnit, MemoryPartitionRejectsUnusableSnapshotsWithoutSetting) {
  for (const char* memory : {"", "NPS3"}) {
    SCOPED_TRACE(::testing::Message() << "memory=\"" << memory << "\"");
    MemoryPartitionHarness test(1);
    test.devices[0].memory = memory;
    EXPECT_GT(Failures(&test), 0u);
    EXPECT_FALSE(test.devices[0].RequestedValidMode());
  }

  MemoryPartitionHarness test(2);
  test.devices[0].profile_status = AMDSMI_STATUS_FILE_ERROR;
  EXPECT_GT(Failures(&test), 0u);
  EXPECT_FALSE(test.devices[0].RequestedValidMode());
  EXPECT_TRUE(test.devices[1].RequestedValidMode()) << "one unreadable device stopped the rest";
}

TEST(GpuUnit, MemoryPartitionDoesNotRequestAfterExternalChange) {
  MemoryPartitionHarness test(1);
  // Another process applies NPS4 after the test read NPS2.
  test.devices[0].on_set = [](PartitionDevice& device, amdsmi_memory_partition_type_t mode) {
    if (mode == AMDSMI_MEMORY_PARTITION_UNKNOWN) {
      device.memory = "NPS4";
    }
  };
  EXPECT_GT(Failures(&test), 0u);
  EXPECT_EQ(test.devices[0].staged, AMDSMI_MEMORY_PARTITION_UNKNOWN);
  EXPECT_FALSE(test.devices[0].RequestedValidMode());
}

TEST(GpuUnit, MemoryPartitionPassesWhenUnsupported) {
  MemoryPartitionHarness no_memory(1);
  no_memory.devices[0].memory_status = AMDSMI_STATUS_NOT_SUPPORTED;
  EXPECT_EQ(Failures(&no_memory), 0u);
  EXPECT_FALSE(no_memory.devices[0].RequestedValidMode());

  MemoryPartitionHarness no_compute(1);
  no_compute.devices[0].profile_status = AMDSMI_STATUS_NOT_SUPPORTED;
  EXPECT_EQ(Failures(&no_compute), 0u);
  EXPECT_TRUE(no_compute.devices[0].RequestedValidMode());
}

// GPUs in NPS2 with QPX (profile index 2) that support NPS1 and NPS2.
MemoryPartitionHarness& CycleHost(MemoryPartitionHarness& test) {
  for (auto& device : test.devices) {
    device.memory = "NPS2";
    device.profile.profile_index = 2;
    device.profile.profile_type = kProfileTypes[2];
    device.caps = {{1, 1, 0, 0, 0}};
  }
  test.EnableCycle();
  return test;
}

void ExpectStartingState(const MemoryPartitionHarness& test) {
  for (const auto& device : test.devices) {
    EXPECT_EQ(device.memory, "NPS2");
    EXPECT_EQ(device.profile.profile_index, 2u);
    EXPECT_EQ(device.staged, AMDSMI_MEMORY_PARTITION_UNKNOWN) << "left a mode staged";
  }
}

// Requests other than a probe or the active mode; a precondition failure must make none.
bool StagedAnything(const MemoryPartitionHarness& test) {
  for (const auto& device : test.devices) {
    for (auto mode : device.requests) {
      if (mode != AMDSMI_MEMORY_PARTITION_UNKNOWN && mode != ModeOf(device.memory)) {
        return true;
      }
    }
  }
  return false;
}

const std::vector<std::string> kOneCycle = {"-r amdgpu", "amdgpu", "-r amdgpu", "amdgpu"};

TEST(GpuUnit, MemoryPartitionCycleNeedsTheDestructiveFlag) {
  MemoryPartitionHarness test(2);
  EXPECT_EQ(Failures(&test), 0u);
  EXPECT_TRUE(test.host.modprobe_calls.empty());
  EXPECT_FALSE(StagedAnything(test));
}

TEST(GpuUnit, MemoryPartitionCycleAppliesEachModeAndRestores) {
  MemoryPartitionHarness test(2);
  CycleHost(test);
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.failures.empty()) << Describe(outcome);
  EXPECT_FALSE(outcome.skipped);
  EXPECT_EQ(test.host.applied, (std::vector<std::string>{"NPS1", "NPS2"}));
  EXPECT_EQ(test.host.modprobe_calls, kOneCycle);
  ExpectStartingState(test);
}

TEST(GpuUnit, MemoryPartitionCycleChecksEveryPreconditionBeforeStaging) {
  const std::vector<std::pair<std::string, std::function<void(FakeHost&)>>> blockers = {
      {"modprobe not found", [](FakeHost& host) { host.modprobe_installed = false; }},
      {"CAP_SYS_MODULE", [](FakeHost& host) { host.can_load_modules = false; }},
      {"not a loaded module", [](FakeHost& host) { host.built_in = true; }},
      {"in use by another process (refcnt 1)", [](FakeHost& host) { host.other_users = 1; }},
      {"refcnt is unreadable", [](FakeHost& host) { host.refcnt_text = "busy\n"; }},
  };
  for (const auto& blocker : blockers) {
    SCOPED_TRACE(blocker.first);
    MemoryPartitionHarness test(2);
    CycleHost(test);
    blocker.second(test.host);
    const Outcome outcome = RunTest(&test);
    EXPECT_TRUE(outcome.Mentions(blocker.first)) << Describe(outcome);
    EXPECT_TRUE(outcome.Mentions("nothing was staged")) << Describe(outcome);
    EXPECT_FALSE(StagedAnything(test));
    EXPECT_TRUE(test.host.modprobe_calls.empty());
    ExpectStartingState(test);
  }
}

TEST(GpuUnit, MemoryPartitionCycleKeysEachPhysicalGpuOnPartitionZero) {
  // Two GPUs with two partition handles each.
  MemoryPartitionHarness test(4);
  CycleHost(test);
  for (uint32_t i = 0; i < 4; ++i) {
    test.devices[i].bus = static_cast<uint8_t>(i / 2);
    test.devices[i].partition_id = i % 2;
  }
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.failures.empty()) << Describe(outcome);
  for (uint32_t i : {1u, 3u}) {
    EXPECT_EQ(std::count(test.devices[i].requests.begin(), test.devices[i].requests.end(),
                         AMDSMI_MEMORY_PARTITION_NPS1),
              0)
        << "staged a mode through a sub-partition handle";
  }
  ExpectStartingState(test);
}

TEST(GpuUnit, MemoryPartitionCycleDoesNotReloadForModesEveryGpuRejects) {
  MemoryPartitionHarness test(2);
  CycleHost(test);
  for (auto& device : test.devices) {
    device.caps = {{1, 1, 1, 0, 0}};
    device.rejects = {AMDSMI_MEMORY_PARTITION_NPS4};
  }
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.failures.empty()) << Describe(outcome);
  EXPECT_EQ(test.host.applied, (std::vector<std::string>{"NPS1", "NPS2"}));
  ExpectStartingState(test);
}

TEST(GpuUnit, MemoryPartitionCycleSkipsWhenThereIsNothingToCycle) {
  const std::vector<std::pair<std::string, std::function<void(MemoryPartitionHarness&)>>> cases = {
      {"different starting modes",
       [](MemoryPartitionHarness& test) { test.devices[1].memory = "NPS1"; }},
      {"only the starting mode supported",
       [](MemoryPartitionHarness& test) {
         for (auto& device : test.devices) {
           device.caps = {{0, 1, 0, 0, 0}};
         }
       }},
      {"every other mode rejected",
       [](MemoryPartitionHarness& test) {
         for (auto& device : test.devices) {
           device.rejects = {AMDSMI_MEMORY_PARTITION_NPS1};
         }
       }},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(entry.first);
    MemoryPartitionHarness test(2);
    CycleHost(test);
    entry.second(test);
    const Outcome outcome = RunTest(&test);
    EXPECT_TRUE(outcome.skipped) << "passed without cycling anything";
    EXPECT_TRUE(outcome.failures.empty()) << Describe(outcome);
    EXPECT_TRUE(test.host.modprobe_calls.empty());
  }
}

TEST(GpuUnit, MemoryPartitionCycleFailsWhenCapabilitiesAreUnreadable) {
  MemoryPartitionHarness test(2);
  CycleHost(test);
  test.devices[1].caps_status = AMDSMI_STATUS_FILE_ERROR;
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.Mentions("capabilities unreadable on 0000:01:00")) << Describe(outcome);
  EXPECT_FALSE(StagedAnything(test));
  EXPECT_TRUE(test.host.modprobe_calls.empty());
}

TEST(GpuUnit, MemoryPartitionCycleFailsWhenOnlySomeGpusAcceptAMode) {
  MemoryPartitionHarness test(2);
  CycleHost(test);
  test.devices[1].rejects = {AMDSMI_MEMORY_PARTITION_NPS1};
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.Mentions("NPS1 accepted by 1 of 2 GPUs")) << Describe(outcome);
  ExpectStartingState(test);
}

TEST(GpuUnit, MemoryPartitionCycleRestoresWhenTheFirstGpuStillReportsTheStart) {
  // The lowest-BDF GPU rejects NPS1 and the first unload fails, so after the retried reload
  // it still reads NPS2 while the other GPU is on NPS1.
  MemoryPartitionHarness test(2);
  CycleHost(test);
  test.devices[0].rejects = {AMDSMI_MEMORY_PARTITION_NPS1};
  test.host.unload_failures = 1;
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.Mentions("NPS1 accepted by 1 of 2 GPUs")) << Describe(outcome);
  ExpectStartingState(test);
}

TEST(GpuUnit, MemoryPartitionCycleRetriesAReloadThatDidNotUnload) {
  MemoryPartitionHarness test(2);
  CycleHost(test);
  test.host.unload_failures = 1;
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.failures.empty()) << Describe(outcome);
  EXPECT_EQ(test.host.applied, (std::vector<std::string>{"NPS1", "NPS2"}));
  ExpectStartingState(test);
}

TEST(GpuUnit, MemoryPartitionCycleReportsAModeLeftStaged) {
  MemoryPartitionHarness test(1);
  CycleHost(test);
  test.host.unload_failures = 2;
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.Mentions("NPS1 is still staged")) << Describe(outcome);
  EXPECT_EQ(test.host.modprobe_calls, (std::vector<std::string>{"-r amdgpu", "-r amdgpu"}));
  EXPECT_EQ(test.devices[0].staged, AMDSMI_MEMORY_PARTITION_NPS1);
  EXPECT_EQ(test.devices[0].requests.back(), AMDSMI_MEMORY_PARTITION_NPS1)
      << "kept setting modes after a failed reload";
}

TEST(GpuUnit, MemoryPartitionCycleLoadsTwiceThenStops) {
  MemoryPartitionHarness test(1);
  CycleHost(test);
  test.host.load_failures = 2;
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.Mentions("amdgpu is not loaded")) << Describe(outcome);
  EXPECT_TRUE(outcome.Mentions("up in NPS1")) << Describe(outcome);
  EXPECT_EQ(test.host.modprobe_calls, (std::vector<std::string>{"-r amdgpu", "amdgpu", "amdgpu"}));
  EXPECT_EQ(test.devices[0].requests.back(), AMDSMI_MEMORY_PARTITION_NPS1)
      << "kept setting modes without a driver";
}

TEST(GpuUnit, MemoryPartitionCycleDoesNotRetryAStuckLoad) {
  MemoryPartitionHarness test(1);
  CycleHost(test);
  test.host.load_stuck = true;
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.Mentions("ignored SIGKILL")) << Describe(outcome);
  EXPECT_EQ(test.host.modprobe_calls, (std::vector<std::string>{"-r amdgpu", "amdgpu"}));
}

TEST(GpuUnit, MemoryPartitionCycleWaitsForDevicesAfterAReload) {
  MemoryPartitionHarness slow(2);
  CycleHost(slow);
  slow.host.enumerate_failures = 3;
  const Outcome recovered = RunTest(&slow);
  EXPECT_TRUE(recovered.failures.empty()) << Describe(recovered);
  ExpectStartingState(slow);

  MemoryPartitionHarness lost(2);
  CycleHost(lost);
  lost.host.enumerate_failures = 1000;
  const Outcome outcome = RunTest(&lost);
  EXPECT_TRUE(outcome.Mentions("library found no devices")) << Describe(outcome);
  EXPECT_FALSE(outcome.Mentions("amdgpu is not loaded")) << "named the wrong cause";
  EXPECT_EQ(lost.host.modprobe_calls, (std::vector<std::string>{"-r amdgpu", "amdgpu"}));
}

TEST(GpuUnit, MemoryPartitionCycleStopsAfterTheFirstFailedMode) {
  // NPS1 and NPS4 are both candidates; a failure on NPS1 must not go on to NPS4.
  MemoryPartitionHarness not_applied(1);
  CycleHost(not_applied);
  not_applied.devices[0].caps = {{1, 1, 1, 0, 0}};
  not_applied.host.applies_staged_modes = false;
  const Outcome outcome = RunTest(&not_applied);
  EXPECT_TRUE(outcome.Mentions("reload applied the wrong mode")) << Describe(outcome);
  EXPECT_EQ(not_applied.host.modprobe_calls, (std::vector<std::string>{"-r amdgpu", "amdgpu"}));
  EXPECT_EQ(not_applied.devices[0].memory, "NPS2");

  MemoryPartitionHarness not_unloaded(1);
  CycleHost(not_unloaded);
  not_unloaded.devices[0].caps = {{1, 1, 1, 0, 0}};
  not_unloaded.host.unload_failures = 2;
  RunTest(&not_unloaded);
  EXPECT_EQ(not_unloaded.host.modprobe_calls, (std::vector<std::string>{"-r amdgpu", "-r amdgpu"}));
}

TEST(GpuUnit, MemoryPartitionCycleRetriesABusyProfileRestore) {
  MemoryPartitionHarness test(2);
  CycleHost(test);
  for (auto& device : test.devices) {
    device.profile_set_statuses = {AMDSMI_STATUS_BUSY, AMDSMI_STATUS_BUSY};
  }
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(outcome.failures.empty()) << Describe(outcome);
  ExpectStartingState(test);
}

TEST(GpuUnit, MemoryPartitionCycleFailsWhenTheProfileStaysBusy) {
  MemoryPartitionHarness test(1);
  CycleHost(test);
  test.devices[0].profile_set_statuses =
      std::vector<amdsmi_status_t>(MemoryPartitionHarness::kProfileAttempts, AMDSMI_STATUS_BUSY);
  const Outcome outcome = RunTest(&test);
  EXPECT_TRUE(
      outcome.Mentions("accelerator profile not restored to QPX (index 2), last status "
                       "AMDSMI_STATUS_BUSY"))
      << Describe(outcome);
  EXPECT_EQ(test.devices[0].profile.profile_index, 0u);
}

using RunResult = TestMemoryPartitionReadWrite::RunResult;

TEST(GpuUnit, MemoryPartitionRunProgramReportsTheOutcome) {
  std::string error;
  EXPECT_EQ(TestMemoryPartitionReadWrite::RunProgram("/bin/sh", {"-c", "exit 0"}, 5, 1, &error),
            RunResult::kOk);
  EXPECT_EQ(TestMemoryPartitionReadWrite::RunProgram("/bin/sh", {"-c", "exit 3"}, 5, 1, &error),
            RunResult::kFailed);
  EXPECT_NE(error.find("exit status 3"), std::string::npos) << error;
  EXPECT_EQ(TestMemoryPartitionReadWrite::RunProgram("/bin/sh", {"-c", "sleep 30"}, 1, 5, &error),
            RunResult::kTimedOut);
  EXPECT_NE(error.find("timed out after 1s"), std::string::npos) << error;
}

TEST(GpuUnit, MemoryPartitionRunProgramDoesNotPassTheCallersEnvironment) {
  setenv("MODPROBE_OPTIONS", "-C /tmp/elsewhere", 1);
  std::string error;
  const auto result = TestMemoryPartitionReadWrite::RunProgram(
      "/bin/sh", {"-c", "test -z \"$MODPROBE_OPTIONS\""}, 5, 1, &error);
  unsetenv("MODPROBE_OPTIONS");
  EXPECT_EQ(result, RunResult::kOk) << error;
}

}  // namespace
