/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <array>
#include <vector>

#include "hip_vmm_common.hh"

#if defined(_WIN32) && HT_AMD
#include <windows.h>
#endif

namespace {
constexpr size_t kReservationSize = size_t{16} << 30;
constexpr size_t kChunkSize = size_t{16} << 20;
constexpr size_t kChunkCount = 32;
enum class Lifecycle { Full, ReserveOnly, ReuseReservation };

// Track partial construction as well as the successful path. Nonfatal checks
// allow all releases to be attempted if a preceding operation failed.
struct Reservation {
  void* address = nullptr;
  std::array<hipMemGenericAllocationHandle_t, kChunkCount> handles{};
  size_t created = 0;
  size_t mapped = 0;

  void releaseMappings() {
    if (mapped != 0) {
      CHECK(hipMemUnmap(address, mapped * kChunkSize) == hipSuccess);
      mapped = 0;
    }
    for (size_t i = 0; i < created; ++i) {
      CHECK(hipMemRelease(handles[i]) == hipSuccess);
    }
    created = 0;
  }

  void releaseAddress() {
    if (address != nullptr) {
      CHECK(hipMemAddressFree(address, kReservationSize) == hipSuccess);
      address = nullptr;
    }
  }

  ~Reservation() {
    releaseMappings();
    releaseAddress();
  }
};

void runLifecycle(Lifecycle mode) {
  HIP_CHECK(hipSetDevice(0));
  checkVMMSupported(0);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity != 0);
  if (kChunkSize % granularity != 0) {
    HIP_SKIP_TEST("The 16 MiB regression fixture requires compatible VMM granularity.");
  }
  hipMemAccessDesc access{};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  std::vector<unsigned char> readback(64 * 1024);
  Reservation reservation;
  for (size_t cycle = 0; cycle < 40; ++cycle) {
    INFO("cycle " << cycle);
    if (reservation.address == nullptr) {
      HIP_CHECK(hipMemAddressReserve(&reservation.address, kReservationSize, 0, nullptr, 0));
    }
    if (mode != Lifecycle::ReserveOnly) {
      for (size_t chunk = 0; chunk < kChunkCount; ++chunk) {
        INFO("chunk " << chunk);
        HIP_CHECK(hipMemCreate(&reservation.handles[chunk], kChunkSize, &prop, 0));
        ++reservation.created;
        auto* address = static_cast<unsigned char*>(reservation.address) + chunk * kChunkSize;
        HIP_CHECK(hipMemMap(address, kChunkSize, 0, reservation.handles[chunk], 0));
        ++reservation.mapped;
        HIP_CHECK(hipMemSetAccess(address, kChunkSize, &access, 1));
        HIP_CHECK(hipMemset(address, (cycle + chunk) % 255 + 1, kChunkSize));
      }
      HIP_CHECK(hipDeviceSynchronize());
      for (size_t chunk = 0; chunk < kChunkCount; ++chunk) {
        const auto expected = static_cast<unsigned char>((cycle + chunk) % 255 + 1);
        for (size_t offset = 0; offset < kChunkSize; offset += readback.size()) {
          INFO("chunk " << chunk << ", byte offset " << offset);
          const auto* address =
              static_cast<unsigned char*>(reservation.address) + chunk * kChunkSize + offset;
          HIP_CHECK(hipMemcpy(readback.data(), address, readback.size(), hipMemcpyDeviceToHost));
          REQUIRE(std::all_of(readback.begin(), readback.end(),
                              [expected](unsigned char value) { return value == expected; }));
        }
      }
      reservation.releaseMappings();
    }
    if (mode != Lifecycle::ReuseReservation) {
      reservation.releaseAddress();
    }
  }
}

#if defined(_WIN32) && HT_AMD
void runWithComgr(Lifecycle mode) {
  // The normal test executable already imports HIP. This checks coexistence,
  // not COMGR-before-HIP DLL load order. Use a fresh test process per case.
  const char* path = std::getenv("HIP_TEST_COMGR_PATH");
  if (path == nullptr) {
    HIP_SKIP_TEST("Set HIP_TEST_COMGR_PATH to an absolute amd_comgr.dll path.");
  }
  REQUIRE(std::strlen(path) > 3);
  REQUIRE(((path[1] == ':' && (path[2] == '\\' || path[2] == '/')) ||
           (path[0] == '\\' && path[1] == '\\')));
  struct Module {
    HMODULE handle;
    ~Module() {
      if (handle != nullptr) CHECK(FreeLibrary(handle) != 0);
    }
  } module{LoadLibraryExA(path, nullptr,
                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)};
  INFO("COMGR path: " << path << "; loader error: " << GetLastError());
  REQUIRE(module.handle != nullptr);
  REQUIRE(GetProcAddress(module.handle, "amd_comgr_get_version") != nullptr);
  char loaded_path[32768]{};
  const DWORD length = GetModuleFileNameA(module.handle, loaded_path, sizeof(loaded_path));
  REQUIRE(length > 0);
  REQUIRE(length < sizeof(loaded_path));
  std::cout << "Loaded COMGR: " << loaded_path << std::endl;
  runLifecycle(mode);
}
#endif
}  // namespace

HIP_TEST_CASE(Unit_hipMemLifecycle_Full) {
  CTX_CREATE();
  runLifecycle(Lifecycle::Full);
  CTX_DESTROY();
}

HIP_TEST_CASE(Unit_hipMemLifecycle_ReserveOnly) {
  CTX_CREATE();
  runLifecycle(Lifecycle::ReserveOnly);
  CTX_DESTROY();
}

HIP_TEST_CASE(Unit_hipMemLifecycle_ReuseReservation) {
  CTX_CREATE();
  runLifecycle(Lifecycle::ReuseReservation);
  CTX_DESTROY();
}

#if defined(_WIN32) && HT_AMD
HIP_TEST_CASE(Unit_hipMemLifecycle_Full_Comgr) { runWithComgr(Lifecycle::Full); }
HIP_TEST_CASE(Unit_hipMemLifecycle_ReserveOnly_Comgr) { runWithComgr(Lifecycle::ReserveOnly); }
HIP_TEST_CASE(Unit_hipMemLifecycle_ReuseReservation_Comgr) {
  runWithComgr(Lifecycle::ReuseReservation);
}
#endif
