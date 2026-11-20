// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef ENABLE_ESMI_LIB
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_system.h"
#include "e_smi/e_smi.h"
#include "test_fixture_utils.h"

namespace {
// Only these tests intercept ESMI. The unit test executable normally calls the real library.
constexpr uint32_t kCoresPerSocket = 192;
constexpr uint32_t kPhysical = 2 * kCoresPerSocket;
struct FakeEsmi {
  uint32_t range_socket = UINT32_MAX;
  uint32_t floor_target = UINT32_MAX;
  bool hsmp_supported = true;
  esmi_status_t hsmp_core_status(uint32_t index) const {
    if (!hsmp_supported) return ESMI_NO_HSMP_MSG_SUP;
    return index < kPhysical ? ESMI_SUCCESS : ESMI_INVALID_INPUT;
  }
  set_get_floorlimit floor_mode = GET_FLOOR_FREQUENCY_SOCKET;
};
FakeEsmi* active = nullptr;
}  // namespace

extern "C" {
esmi_status_t __real_esmi_init();
void __real_esmi_exit();
esmi_status_t __real_esmi_number_of_cpus_get(uint32_t*);
esmi_status_t __real_esmi_number_of_sockets_get(uint32_t*);
esmi_status_t __real_esmi_threads_per_core_get(uint32_t*);
esmi_status_t __real_esmi_core_energy_get(uint32_t, uint64_t*);
esmi_status_t __real_esmi_socket_freq_range_get(uint8_t, uint16_t*, uint16_t*);
esmi_status_t __real_esmi_msr_floorlimit_set(uint32_t, uint32_t, set_get_floorlimit, uint32_t);

esmi_status_t __wrap_esmi_init() { return active ? ESMI_SUCCESS : __real_esmi_init(); }
void __wrap_esmi_exit() {
  if (!active) __real_esmi_exit();
}
esmi_status_t __wrap_esmi_number_of_sockets_get(uint32_t* n) {
  if (!active) return __real_esmi_number_of_sockets_get(n);
  *n = 2;
  return ESMI_SUCCESS;
}
esmi_status_t __wrap_esmi_number_of_cpus_get(uint32_t* n) {
  if (!active) return __real_esmi_number_of_cpus_get(n);
  *n = 2 * kPhysical;
  return ESMI_SUCCESS;
}
esmi_status_t __wrap_esmi_threads_per_core_get(uint32_t* n) {
  if (!active) return __real_esmi_threads_per_core_get(n);
  *n = 2;
  return ESMI_SUCCESS;
}
esmi_status_t __wrap_esmi_core_energy_get(uint32_t cpu, uint64_t* energy) {
  if (!active) return __real_esmi_core_energy_get(cpu, energy);
  if (cpu >= kPhysical) return ESMI_INVALID_INPUT;
  *energy = cpu;
  return ESMI_SUCCESS;
}
esmi_status_t __wrap_esmi_socket_freq_range_get(uint8_t socket, uint16_t* fmax, uint16_t* fmin) {
  if (!active) return __real_esmi_socket_freq_range_get(socket, fmax, fmin);
  if (!active->hsmp_supported) return ESMI_NO_HSMP_MSG_SUP;
  active->range_socket = socket;
  *fmax = 3200;
  *fmin = 1200;
  return ESMI_SUCCESS;
}
esmi_status_t __wrap_esmi_msr_floorlimit_set(uint32_t target, uint32_t freq,
                                             set_get_floorlimit mode, uint32_t fmax) {
  if (!active) return __real_esmi_msr_floorlimit_set(target, freq, mode, fmax);
  if (target >= (mode == SET_FLOOR_FREQUENCY_SOCKET ? 2u : kPhysical)) return ESMI_INVALID_INPUT;
  active->floor_target = target;
  active->floor_mode = mode;
  return ESMI_SUCCESS;
}
decltype(esmi_current_freq_limit_core_get) __real_esmi_current_freq_limit_core_get;
esmi_status_t __wrap_esmi_current_freq_limit_core_get(uint32_t cpu, uint32_t* out) {
  if (!active) return __real_esmi_current_freq_limit_core_get(cpu, out);
  *out = cpu;
  return active->hsmp_core_status(cpu);
}
decltype(esmi_core_boostlimit_get) __real_esmi_core_boostlimit_get;
esmi_status_t __wrap_esmi_core_boostlimit_get(uint32_t cpu, uint32_t* out) {
  if (!active) return __real_esmi_core_boostlimit_get(cpu, out);
  *out = cpu;
  return active->hsmp_core_status(cpu);
}
decltype(esmi_core_boostlimit_set) __real_esmi_core_boostlimit_set;
esmi_status_t __wrap_esmi_core_boostlimit_set(uint32_t cpu, uint32_t freq) {
  if (!active) return __real_esmi_core_boostlimit_set(cpu, freq);
  return active->hsmp_core_status(cpu);
}
decltype(esmi_read_ccd_power) __real_esmi_read_ccd_power;
esmi_status_t __wrap_esmi_read_ccd_power(uint32_t cpu, uint32_t* out) {
  if (!active) return __real_esmi_read_ccd_power(cpu, out);
  *out = cpu;
  return active->hsmp_core_status(cpu);
}
decltype(esmi_floorlimit_set_get) __real_esmi_floorlimit_set_get;
esmi_status_t __wrap_esmi_floorlimit_set_get(uint32_t cpu, uint32_t* out, set_get_floorlimit mode) {
  if (!active) return __real_esmi_floorlimit_set_get(cpu, out, mode);
  *out = cpu;
  return active->hsmp_core_status(cpu);
}
}  // extern "C"

namespace {
class CpuRoutingFixture {
 public:
  explicit CpuRoutingFixture(bool sparse = false)
      : esmi_([] { return active; }, [](FakeEsmi* fake) { active = fake; }, &fake) {
    amd::smi::SetSysDevicesSystemRootForTesting(tree_.root().c_str());
    // Package numbering is deliberately interleaved. The second thread for a
    // physical core is one kPhysical stride away, as ESMI's MSR backend expects.
    tree_.WriteFile("cpu/online", sparse ? "0-3,5-767\n" : "0-767\n");
    for (uint32_t cpu = 0; cpu < 2 * kPhysical; ++cpu) {
      const std::string path = "cpu/cpu" + std::to_string(cpu) + "/topology/";
      tree_.WriteFile(path + "physical_package_id", std::to_string(cpu % 2));
      tree_.WriteFile(path + "core_id", std::to_string((cpu % kPhysical) / 2));
    }
    status = amdsmi_init(AMDSMI_INIT_AMD_CPUS);
  }
  ~CpuRoutingFixture() {
    if (status == AMDSMI_STATUS_SUCCESS) amdsmi_shut_down();
    amd::smi::SetSysDevicesSystemRootForTesting(nullptr);
  }
  FakeEsmi fake;
  amdsmi_status_t status;

 private:
  FakeSysfsTree tree_;
  amd::smi::testing::ScopedOverride<FakeEsmi*> esmi_;
};
}  // namespace

TEST(SystemUnit, CpuCoreRoutingUsesPhysicalCpuNotSocketLocalId) {
  CpuRoutingFixture fixture;
  ASSERT_EQ(fixture.status, AMDSMI_STATUS_SUCCESS);
  uint32_t count = 0;
  ASSERT_EQ(amdsmi_get_cpucore_handles(&count, nullptr), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(count, kPhysical);
  std::vector<amdsmi_processor_handle> cores(count);
  ASSERT_EQ(amdsmi_get_cpucore_handles(&count, cores.data()), AMDSMI_STATUS_SUCCESS);
  for (uint32_t socket : {0u, 1u}) {
    for (uint32_t local : {0u, 128u, kCoresPerSocket - 1}) {
      auto handle = cores[socket * kCoresPerSocket + local];
      char name[16]{};
      ASSERT_EQ(amdsmi_get_processor_info(handle, sizeof(name), name), AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(std::string(name), std::to_string(local));
      uint64_t energy = UINT64_MAX;
      ASSERT_EQ(amdsmi_get_cpu_core_energy(handle, &energy), AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(energy, 2 * local + socket);
    }
  }

  // Both setters call the in-memory ESMI recorder, never MSR or HSMP.
  ASSERT_EQ(amdsmi_set_cpu_core_msr_floor_freq_limit(cores.back(), 1500), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(fixture.fake.range_socket, 1u);
  EXPECT_EQ(fixture.fake.floor_target, kPhysical - 1);
  EXPECT_EQ(fixture.fake.floor_mode, SET_FLOOR_FREQUENCY_CORE);
  uint32_t num_sockets = 2;
  amdsmi_processor_handle sockets[2]{};
  ASSERT_EQ(amdsmi_get_cpu_handles(&num_sockets, sockets), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_sockets, 2u);
  ASSERT_EQ(amdsmi_set_cpu_msr_floor_freq_limit(sockets[1], 1500), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(fixture.fake.range_socket, 1u);
  EXPECT_EQ(fixture.fake.floor_target, 1u);
  EXPECT_EQ(fixture.fake.floor_mode, SET_FLOOR_FREQUENCY_SOCKET);
}

TEST(SystemUnit, CpuCoreRoutingDoesNotGuessOnSparseTopology) {
  CpuRoutingFixture fixture(true);
  ASSERT_EQ(fixture.status, AMDSMI_STATUS_SUCCESS);
  uint32_t count = kPhysical;
  std::vector<amdsmi_processor_handle> cores(count);
  ASSERT_EQ(amdsmi_get_cpucore_handles(&count, cores.data()), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(count, kPhysical);
  uint64_t energy = UINT64_MAX;
  EXPECT_EQ(amdsmi_get_cpu_core_energy(cores.back(), &energy), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(energy, UINT64_MAX);
}
// A registered handle passed by the functional suite isn't necessarily a CPU core.
// Unsupported HSMP commands take precedence over an out-of-range ESMI index.
TEST(SystemUnit, CpuCoreRoutingPreservesLegacyHandleStatus) {
  CpuRoutingFixture fixture;
  ASSERT_EQ(fixture.status, AMDSMI_STATUS_SUCCESS);
  amd::smi::AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU);
  amd::smi::testing::ScopedProcessorRegistration registered(&gpu);
  amdsmi_processor_handle sockets[2]{};
  uint32_t n = 2;
  ASSERT_EQ(amdsmi_get_cpu_handles(&n, sockets), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(n, 2u);
  uint64_t energy = UINT64_MAX;
  EXPECT_EQ(amdsmi_get_cpu_core_energy(registered.handle(), &energy), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_cpu_core_energy(sockets[1], &energy), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(energy, 1u);

  for (const auto h : {registered.handle(), sockets[1]}) {
    for (const bool supported : {false, true}) {
      fixture.fake.hsmp_supported = supported;
      const auto expected =
          !supported ? AMDSMI_STATUS_NO_HSMP_MSG_SUP
                     : (h == registered.handle() ? AMDSMI_STATUS_INVAL : AMDSMI_STATUS_SUCCESS);
      uint32_t value = 0;
      EXPECT_EQ(amdsmi_get_cpu_core_current_freq_limit(h, &value), expected);
      EXPECT_EQ(amdsmi_get_cpu_core_boostlimit(h, &value), expected);
      EXPECT_EQ(amdsmi_set_cpu_core_boostlimit(h, 1500), expected);
      EXPECT_EQ(amdsmi_get_cpu_core_ccd_power(h, &value), expected);
      EXPECT_EQ(amdsmi_get_cpu_core_floor_freq_limit(h, &value), expected);
      EXPECT_EQ(amdsmi_get_cpu_core_eff_floor_freq_limit(h, &value), expected);
      EXPECT_EQ(amdsmi_set_cpu_core_floor_freq_limit(h, 1500), expected);
      EXPECT_EQ(amdsmi_set_cpu_core_msr_floor_freq_limit(h, 1500), expected);
      EXPECT_EQ(amdsmi_set_cpu_msr_floor_freq_limit(h, 1500), expected);
    }
  }
  // A core passed to the socket variant historically selected just one core.
  // Don't widen that operation to an entire socket. Use the corrected core ID.
  std::vector<amdsmi_processor_handle> cores(kPhysical);
  n = kPhysical;
  ASSERT_EQ(amdsmi_get_cpucore_handles(&n, cores.data()), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(amdsmi_set_cpu_msr_floor_freq_limit(cores.back(), 1500), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(fixture.fake.floor_mode, SET_FLOOR_FREQUENCY_CORE);
  EXPECT_EQ(fixture.fake.floor_target, kPhysical - 1);
  EXPECT_EQ(fixture.fake.range_socket, 1u);
}
#endif  // ENABLE_ESMI_LIB
