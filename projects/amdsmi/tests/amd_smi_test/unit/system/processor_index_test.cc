// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Processor index defaults. Only the CPU socket and core constructors pass an
// index. The CPU/Core APIs reuse that index as an ESMI socket or core number.
// An unset index must therefore stay an obvious out-of-range selection.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "amd_smi/impl/amd_smi_processor.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiProcessor;

// ESMI limits from e_smi.h. Both are counts, so the last valid index is one below.
constexpr uint32_t kMaxSockets = 8;           // MAX_SOCKET
constexpr uint32_t kMaxCoresPerSocket = 512;  // CPU_MAX_CORES_PER_SOCKET
constexpr uint32_t kMaxCoresInSystem = kMaxSockets * kMaxCoresPerSocket;
constexpr uint32_t kLastCoreInSocket = kMaxCoresPerSocket - 1;
constexpr uint32_t kLastCoreInSystem = kMaxCoresInSystem - 1;

// Spelled out rather than read from the header, so changing the sentinel fails here.
constexpr uint32_t kUnsetIndex = UINT32_MAX;
constexpr uint32_t kUnsetIndexLowByte = 255;

// Arbitrary in-range values. They only have to survive the round trip.
constexpr uint32_t kSampleSocketIndex = 3;
constexpr uint32_t kSampleCoreIndex = 191;

TEST(SystemUnit, ProcessorIndexUnsetDefaultsToInvalid) {
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU);
  EXPECT_EQ(gpu.get_processor_index(), kUnsetIndex)
      << "pindex_ lost its default initializer in amd_smi_processor.h. GPU, NIC and\n"
         "switch processors never pass an index, so they would hold indeterminate\n"
         "values again. Fix the initializer, not this expectation. See AILITOOLS-304.";
}

TEST(SystemUnit, ProcessorIndexIdentifierCtorDefaultsToInvalid) {
  AMDSmiProcessor by_id(std::string("0000:0c:00.0"));
  EXPECT_EQ(by_id.get_processor_index(), kUnsetIndex)
      << "The identifier constructor sets neither the type nor the index, so it\n"
         "depends entirely on the default initializer. See AILITOOLS-304.";
}

TEST(SystemUnit, ProcessorIndexPreservedWhenProvided) {
  AMDSmiProcessor socket(AMDSMI_PROCESSOR_TYPE_AMD_CPU, kSampleSocketIndex);
  EXPECT_EQ(socket.get_processor_index(), kSampleSocketIndex)
      << "CPU sockets must keep the index they were built with. Any other value\n"
         "means every CPU API would target the wrong socket.";

  AMDSmiProcessor core(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, kSampleCoreIndex);
  EXPECT_EQ(core.get_processor_index(), kSampleCoreIndex)
      << "CPU cores must keep the index they were built with.";
}

// ESMI allows CPU_MAX_CORES_PER_SOCKET (512) cores across MAX_SOCKET (8)
// sockets, so the index has to hold values well beyond a single byte. amd-smi
// assigns per-socket core indices today, so the system-wide ceiling is headroom
// rather than a value it currently builds.
TEST(SystemUnit, ProcessorIndexHoldsFullEsmiCoreRange) {
  AMDSmiProcessor last_in_socket(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, kLastCoreInSocket);
  EXPECT_EQ(last_in_socket.get_processor_index(), kLastCoreInSocket)
      << "Core " << kLastCoreInSocket << " is the last of " << kMaxCoresPerSocket
      << " cores ESMI allows on one socket. Narrowing pindex_ to uint8_t would\n"
         "fold it onto core "
      << (kLastCoreInSocket & 0xFFu) << ".";

  AMDSmiProcessor last_in_system(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, kLastCoreInSystem);
  EXPECT_EQ(last_in_system.get_processor_index(), kLastCoreInSystem)
      << "Core " << kLastCoreInSystem
      << " is the highest index ESMI can address: " << kMaxCoresPerSocket << " cores on each of "
      << kMaxSockets << " sockets.";
}

// The CPU/Core APIs narrow the index to uint8_t before passing it to ESMI.
// A low byte inside the socket range selects a real socket and returns its
// data. The previous uninitialized value did this non-deterministically.
TEST(SystemUnit, ProcessorIndexSentinelSurvivesUint8Narrowing) {
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU);
  const uint32_t narrowed = static_cast<uint8_t>(gpu.get_processor_index());
  EXPECT_EQ(narrowed, kUnsetIndexLowByte)
      << "The sentinel's low byte must stay outside the range of " << kMaxSockets
      << " sockets. UINT32_MAX narrows to " << kUnsetIndexLowByte
      << " and is rejected. A tidier\n"
         "looking value such as 0x100 narrows to 0 and would silently select socket\n"
         "0. Change the sentinel in amd_smi_processor.h, not this expectation.\n"
         "See AILITOOLS-304.";
}

#ifdef ENABLE_ESMI_LIB
TEST(SystemUnit, CpuModelNameRejectsGpuProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&gpu);
  amdsmi_cpu_info_t cpu_info{};
  cpu_info.model_name[0] = 'X';

  EXPECT_EQ(amdsmi_get_cpu_model_name(registration.handle(), &cpu_info),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(cpu_info.model_name[0], 'X');
}

TEST(SystemUnit, CpuCoreEnergyRejectsSocketProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor socket(AMDSMI_PROCESSOR_TYPE_AMD_CPU, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&socket);
  uint64_t energy = 42;

  EXPECT_EQ(amdsmi_get_cpu_core_energy(registration.handle(), &energy),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(energy, 42u);
}

TEST(SystemUnit, CpuCoreEnergyRejectsGpuProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&gpu);
  uint64_t energy = 42;

  EXPECT_EQ(amdsmi_get_cpu_core_energy(registration.handle(), &energy),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(energy, 42u);
}

TEST(SystemUnit, HsmpProtocolVersionRejectsGpuProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&gpu);
  uint32_t version = 42;

  EXPECT_EQ(amdsmi_get_cpu_hsmp_proto_ver(registration.handle(), &version),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(version, 42u);
}

TEST(SystemUnit, CpuDdrBandwidthRejectsGpuProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&gpu);
  amdsmi_ddr_bw_metrics_t bandwidth{};
  bandwidth.max_bw = 42;

  EXPECT_EQ(amdsmi_get_cpu_ddr_bw(registration.handle(), &bandwidth), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(bandwidth.max_bw, 42u);
}

TEST(SystemUnit, CpuGlobalVersionsRejectGpuProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&gpu);
  amdsmi_hsmp_driver_version_t driver{};
  amdsmi_smu_fw_version_t firmware{};
  uint32_t metrics_version = 42;
  driver.major = 42;
  firmware.major = 42;

  EXPECT_EQ(amdsmi_get_cpu_hsmp_driver_version(registration.handle(), &driver),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_get_cpu_smu_fw_version(registration.handle(), &firmware),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_get_hsmp_metrics_table_version(registration.handle(), &metrics_version),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(driver.major, 42u);
  EXPECT_EQ(firmware.major, 42u);
  EXPECT_EQ(metrics_version, 42u);
}

TEST(SystemUnit, CpuPowerEfficiencyRejectsGpuProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&gpu);
  uint32_t utilization = 1;
  uint32_t ppt_limit = 1;

  EXPECT_EQ(amdsmi_set_cpu_pwr_efficiency_mode(
                registration.handle(), AMDSMI_POWER_EFFICIENCY_MODE_4, &utilization, &ppt_limit),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(utilization, 1u);
  EXPECT_EQ(ppt_limit, 1u);
}

TEST(SystemUnit, CpuApisRejectOtherRegisteredProcessorTypes) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor core(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, 0);
  AMDSmiProcessor socket(AMDSMI_PROCESSOR_TYPE_AMD_CPU, 0);
  AMDSmiProcessor nic(AMDSMI_PROCESSOR_TYPE_AMD_NIC, 0);
  amd::smi::testing::ScopedProcessorRegistration core_registration(&core);
  amd::smi::testing::ScopedProcessorRegistration socket_registration(&socket);
  amd::smi::testing::ScopedProcessorRegistration nic_registration(&nic);
  uint32_t power = 42;
  uint64_t energy = 42;

  EXPECT_EQ(amdsmi_get_cpu_socket_power(core_registration.handle(), &power),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_get_cpu_socket_power(nic_registration.handle(), &power),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_get_cpu_core_energy(nic_registration.handle(), &energy),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_set_cpu_core_boostlimit(socket_registration.handle(), 1),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(amdsmi_set_cpu_socket_power_cap(core_registration.handle(), 1),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(power, 42u);
  EXPECT_EQ(energy, 42u);
}

TEST(SystemUnit, CpuGlobalVersionRejectsInvalidHandles) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor unregistered(AMDSMI_PROCESSOR_TYPE_AMD_CPU, 0);
  auto handle = reinterpret_cast<amdsmi_processor_handle>(&unregistered);
  uint32_t version = 42;

  EXPECT_EQ(amdsmi_get_cpu_hsmp_proto_ver(nullptr, &version), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_cpu_hsmp_proto_ver(handle, &version), AMDSMI_STATUS_NOT_FOUND);
  EXPECT_EQ(version, 42u);
}

TEST(SystemUnit, CpuModelNameAcceptsSocketProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor socket(AMDSMI_PROCESSOR_TYPE_AMD_CPU, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&socket);
  amdsmi_cpu_info_t cpu_info{};

  EXPECT_EQ(amdsmi_get_cpu_model_name(registration.handle(), &cpu_info), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(cpu_info.model_name[0], '\0');
}

TEST(SystemUnit, CpuCoreEnergyAcceptsCoreProcessor) {
  ScopedAmdsmiInit smi_init;
  AMDSmiProcessor core(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, 0);
  amd::smi::testing::ScopedProcessorRegistration registration(&core);
  uint64_t energy = 42;

  EXPECT_EQ(amdsmi_get_cpu_core_energy(registration.handle(), &energy), AMDSMI_STATUS_IO);
  EXPECT_EQ(energy, 42u);
}
#endif

}  // namespace
