// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include "suites/functional/heap_reservation.h"

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

constexpr uint64_t kGiB = 1ULL << 30;

// Must match ReserveLocalHeapSpace()/ReserveSystemHeapSpace() in the DXG thunk
// (libhsakmt/src/dxg/openclose.cpp). If those change, the budgets below stop
// predicting which pool sizes fit.
constexpr uint64_t kLocalAlign = 1 * kGiB;
constexpr uint64_t kSystemAlign = 4 * kGiB;
constexpr uint64_t kSystemMin = 2 * kSystemAlign;  // retries stop here
constexpr uint64_t kSystemMax = 1024 * kGiB;

constexpr int kChildPass = 0;
constexpr int kChildFail = 1;
constexpr int kChildSkip = 77;

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Reserving `size` transiently costs size + 2*alignment: it is over-allocated
// to land on an alignment boundary, then trimmed. The peak is what must fit.
uint64_t Transient(uint64_t size, uint64_t alignment) {
  return size + 2 * alignment;
}

uint64_t HostRam() {
  int64_t page = sysconf(_SC_PAGESIZE);
  int64_t pages = sysconf(_SC_PHYS_PAGES);
  if (page <= 0 || pages <= 0) return 0;
  return static_cast<uint64_t>(page) * static_cast<uint64_t>(pages);
}

// Must be measured in the child before hsa_init(): measuring after counts the
// pools themselves and inflates every budget by their size.
uint64_t MappedAddressSpace() {
  FILE* maps = fopen("/proc/self/maps", "re");
  if (maps == nullptr) return 0;

  uint64_t total = 0;
  char line[512];
  while (fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long long lo = 0, hi = 0;
    if (sscanf(line, "%llx-%llx", &lo, &hi) == 2 && hi > lo) total += hi - lo;
  }
  fclose(maps);
  return total;
}

hsa_status_t SumDevicePool(hsa_amd_memory_pool_t pool, void* data) {
  rocrtst::pool_info_t info;
  hsa_status_t err = rocrtst::AcquirePoolInfo(pool, &info);
  if (err != HSA_STATUS_SUCCESS) return err;

  if (info.segment == HSA_AMD_SEGMENT_GLOBAL &&
      (info.global_flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
      !info.accessible_by_all) {
    *static_cast<uint64_t*>(data) += info.size;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t SumAgentVram(hsa_agent_t agent, void* data) {
  hsa_device_type_t type;
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;

  uint64_t agent_vram = 0;
  err = hsa_amd_agent_iterate_memory_pools(agent, SumDevicePool, &agent_vram);
  if (err != HSA_STATUS_SUCCESS) return err;

  if (agent_vram != 0) {
    *static_cast<uint64_t*>(data) += AlignUp(agent_vram, kLocalAlign);
  }
  return HSA_STATUS_SUCCESS;
}

uint64_t SmallestSystemAsk(uint64_t ram_aligned) {
  return std::max(ram_aligned / 4, kSystemMin);
}

// hsa_init() creates the async handler signal, the first system-memory
// allocation and so the first thing a missing host-RAM pool breaks.
int RunConstrained(uint64_t budget) {
  const uint64_t limit = MappedAddressSpace() + budget;

  struct rlimit rl;
  rl.rlim_cur = limit;
  rl.rlim_max = limit;
  if (setrlimit(RLIMIT_AS, &rl) != 0) return kChildSkip;

  hsa_status_t err = hsa_init();
  if (err != HSA_STATUS_SUCCESS) return kChildFail;

  hsa_shut_down();
  return kChildPass;
}

struct ChildResult {
  bool exited = false;  // false means it died by signal
  int status = 0;
  int signal = 0;
  std::string output;
};

ChildResult ForkAndRun(uint64_t budget) {
  ChildResult result;

  int fd = memfd_create("rocrtst-heap-reservation", MFD_CLOEXEC);

  pid_t pid = fork();
  if (pid < 0) {
    if (fd >= 0) close(fd);
    return result;
  }

  if (pid == 0) {
    if (fd >= 0) {
      dup2(fd, STDOUT_FILENO);
      dup2(fd, STDERR_FILENO);
    }
    _exit(RunConstrained(budget));
  }

  int wait_status = 0;
  if (waitpid(pid, &wait_status, 0) == pid) {
    if (WIFEXITED(wait_status)) {
      result.exited = true;
      result.status = WEXITSTATUS(wait_status);
    } else if (WIFSIGNALED(wait_status)) {
      result.signal = WTERMSIG(wait_status);
    }
  }

  if (fd >= 0) {
    lseek(fd, 0, SEEK_SET);
    char buf[4096];
    ssize_t bytes_read;
    while ((bytes_read = read(fd, buf, sizeof(buf))) > 0) {
      result.output.append(buf, bytes_read);
    }
    close(fd);
  }
  return result;
}

std::string Describe(const ChildResult& child) {
  char buf[64];
  if (child.exited) {
    snprintf(buf, sizeof(buf), "exit %d", child.status);
  } else if (child.signal != 0) {
    snprintf(buf, sizeof(buf), "killed by signal %d (%s)", child.signal,
             strsignal(child.signal));
  } else {
    snprintf(buf, sizeof(buf), "no result");
  }
  return std::string(buf);
}

}  // namespace

HeapReservationTest::HeapReservationTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("RocR Heap Reservation Test");
  set_description(
      "This test verifies that hsa_init() survives a constrained address "
      "space on the WSL/DXG backend, and still succeeds when a smaller "
      "host-RAM pool would fit.");
}

HeapReservationTest::~HeapReservationTest(void) {}

// Only the platform-filter half of TestBase::SetUp(). The rest of it calls
// InitAndSetupHSA(), and a runtime left open here would be inherited by every
// forked child: their hsa_init() would be a refcount bump that never reaches
// the reservation path, and the pools would already count against the budget
// they measure. Each child opens and closes the runtime itself.
void HeapReservationTest::SetUp(void) {
  if (!checkPlatformFiltering()) return;
  SetupPrint();
}

void HeapReservationTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) return;
  TestBase::Run();
}

// No TestBase::Close() for the same reason: there is no runtime of ours to
// tear down, each child having shut its own down already.
void HeapReservationTest::Close(void) { ClosePrint(); }

void HeapReservationTest::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) return;
  return;
}

void HeapReservationTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

// Every budget leaves the local pool room for its smallest size, so only the
// host-RAM pool is under test.
HeapReservationTest::Budgets HeapReservationTest::ComputeBudgets(
    uint64_t vram, uint64_t ram) const {
  Budgets budgets;
  if (vram == 0 || ram == 0) return budgets;

  const uint64_t ram_aligned = std::min(AlignUp(ram, kSystemMin), kSystemMax);

  const uint64_t local_floor_retained = 2 * vram;
  const uint64_t local_floor_transient =
      Transient(local_floor_retained, kLocalAlign);

  budgets.generous =
      local_floor_retained + Transient(ram_aligned, kSystemAlign) + 2 * kGiB;
  budgets.ladder = local_floor_retained +
                   Transient(ram_aligned / 2, kSystemAlign) + 2 * kGiB;
  budgets.tight = local_floor_retained + 4 * kGiB;

  const uint64_t ceiling =
      local_floor_retained +
      Transient(SmallestSystemAsk(ram_aligned), kSystemAlign);
  budgets.valid = budgets.tight > local_floor_transient &&
                  budgets.tight < ceiling && budgets.ladder > budgets.tight &&
                  budgets.generous > budgets.ladder;

  budgets.ladder_possible = ram_aligned / 2 >= kSystemMin;

  return budgets;
}

// ROCM-30547: a refused host-RAM VA pool left its allocator null while init
// reported success, and the next system allocation segfaulted. Checks that
// hsa_init() survives that, and still succeeds when a smaller pool would fit.
void HeapReservationTest::HeapReservationSurvivesConstrainedAddressSpace(void) {
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());
  vram_ = 0;
  hsa_status_t err = hsa_iterate_agents(SumAgentVram, &vram_);
  ram_ = HostRam();
  hsa_shut_down();
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  std::cout << "  vram " << (vram_ / kGiB) << " GiB, ram " << (ram_ / kGiB)
            << " GiB" << std::endl;

  const Budgets budgets = ComputeBudgets(vram_, ram_);
  if (!budgets.valid) {
    std::cout << "[ SKIPPED ] no usable RLIMIT_AS window on this hardware"
              << std::endl;
    return;
  }

  const bool verbose = verbosity() >= VERBOSE_PROGRESS;

  const struct {
    const char* name;
    uint64_t budget;
    bool expect_success;
    bool applicable;
  } cases[] = {
      {"generous (1x fits)", budgets.generous, true, true},
      {"ladder (1x refused)", budgets.ladder, true, budgets.ladder_possible},
      {"tight (nothing fits)", budgets.tight, false, true},
  };

  for (const auto& test_case : cases) {
    if (!test_case.applicable) {
      std::cout << "  skip " << test_case.name << " : 1/2 RAM is below the "
                << (kSystemMin / kGiB) << " GiB floor" << std::endl;
      continue;
    }

    const ChildResult child = ForkAndRun(test_case.budget);
    if (child.exited && child.status == kChildSkip) {
      std::cout << "  skip " << test_case.name
                << " : child could not set RLIMIT_AS" << std::endl;
      continue;
    }

    std::cout << "  " << test_case.name << " " << (test_case.budget / kGiB)
              << " GiB : " << Describe(child) << std::endl;

    const std::string already_shown;
    if (verbose && !child.output.empty()) {
      std::cout << child.output << std::flush;
    }
    const std::string& detail = verbose ? already_shown : child.output;

    ASSERT_TRUE(child.exited)
        << test_case.name << ": " << Describe(child) << "\n" << detail;

    if (test_case.expect_success) {
      ASSERT_EQ(kChildPass, child.status)
          << test_case.name << ": " << Describe(child) << "\n" << detail;
    }
  }
}
