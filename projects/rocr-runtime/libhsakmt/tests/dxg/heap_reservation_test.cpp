/*
 * hsaKmtOpenKFD() reserves two pools of virtual address space up front, one
 * sized from VRAM and one from host RAM. They are PROT_NONE/MAP_NORESERVE and
 * so cost no memory, but RLIMIT_AS - or valgrind - can still refuse them.
 *
 * ROCM-30547: when the system pool was refused, init reported success anyway
 * and left that pool's allocator null. The next system-memory allocation
 * dereferenced it and the process died with SIGSEGV. The fix retries at 1/2 and
 * 1/4 of RAM, and returns an error if nothing fits.
 *
 * This runs the thunk under three RLIMIT_AS values, allocates one page of
 * system memory under each, and checks that the process survives - and, where a
 * smaller pool would have fit, that the allocation is actually served.
 *
 * Needs a DXCore adapter; exits 77 (skip) without one.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

#include "hsakmt/hsakmt.h"

namespace {

constexpr int kPassExitCode = 0;
constexpr int kFailExitCode = 1;
constexpr int kSkipExitCode = 77;  // matches SKIP_RETURN_CODE in CMakeLists.txt

constexpr uint64_t kGiB = 1ULL << 30;

/* Must match ReserveLocalHeapSpace()/ReserveSystemHeapSpace() in openclose.cpp. */
constexpr uint64_t kLocalAlign = 1 * kGiB;
constexpr uint64_t kSystemAlign = 4 * kGiB;
constexpr uint64_t kSystemMin = 2 * kSystemAlign;  /* retries stop here */
constexpr uint64_t kSystemMax = 1024 * kGiB;

constexpr uint64_t kAllocSize = 4096;

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

/* Reserving `size` transiently costs size + 2*alignment: it is over-allocated
 * to land on an alignment boundary, then trimmed. The peak is what must fit. */
uint64_t Transient(uint64_t size, uint64_t alignment) {
  return size + 2 * alignment;
}

uint64_t HostRam() {
  long page = sysconf(_SC_PAGESIZE);
  long pages = sysconf(_SC_PHYS_PAGES);
  if (page <= 0 || pages <= 0) return 0;
  return static_cast<uint64_t>(page) * static_cast<uint64_t>(pages);
}

/* RLIMIT_AS bounds everything mapped, not just the pools, so each child adds
 * this to its budget. Measured in the child, where the thunk is not yet open:
 * measuring while it is open counts the pools themselves and inflates every
 * limit by their size. */
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

/* VRAM summed the way the local pool sizes itself. */
uint64_t SummedVram(const HsaSystemProperties& props) {
  uint64_t total = 0;

  for (HSAuint32 node = 0; node < props.NumNodes; node++) {
    HsaNodeProperties node_props = {};
    if (hsaKmtGetNodeProperties(node, &node_props) != HSAKMT_STATUS_SUCCESS) continue;
    if (node_props.NumMemoryBanks == 0) continue;

    std::vector<HsaMemoryProperties> banks(node_props.NumMemoryBanks);
    if (hsaKmtGetNodeMemoryProperties(node, node_props.NumMemoryBanks, banks.data()) !=
        HSAKMT_STATUS_SUCCESS)
      continue;

    uint64_t node_vram = 0;
    for (const HsaMemoryProperties& bank : banks) {
      if (bank.HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC ||
          bank.HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE)
        node_vram += bank.SizeInBytes;
    }
    if (node_vram != 0) total += AlignUp(node_vram, kLocalAlign);
  }

  return total;
}

/* Allocating from the system pool requires a CPU node (memory.cpp:218). */
bool FindCpuNode(const HsaSystemProperties& props, HSAuint32* cpu_node) {
  for (HSAuint32 node = 0; node < props.NumNodes; node++) {
    HsaNodeProperties node_props = {};
    if (hsaKmtGetNodeProperties(node, &node_props) != HSAKMT_STATUS_SUCCESS) continue;
    if (node_props.NumCPUCores > 0 && node_props.NumFComputeCores == 0) {
      *cpu_node = node;
      return true;
    }
  }
  return false;
}

/* Retries stop at kSystemMin, so this is not simply ram/4. */
uint64_t SmallestSystemAsk(uint64_t ram_aligned) {
  return std::max(ram_aligned / 4, kSystemMin);
}

/* Address space each child may use on top of what it already has mapped. */
struct Limits {
  uint64_t generous;  /* every size fits, including 1x RAM */
  uint64_t ladder;    /* 1x refused, a smaller one fits    */
  uint64_t tight;     /* nothing fits                      */
  bool valid = false;
  bool ladder_possible = false;  /* false when 1/2 RAM is already below the floor */
};

/* Every budget leaves the local pool room for its smallest size, so only the
 * system pool is under test. Too low and both fail, which is a different path. */
Limits ComputeLimits(uint64_t vram, uint64_t ram) {
  Limits limits;
  if (vram == 0 || ram == 0) return limits;

  const uint64_t ram_aligned = std::min(AlignUp(ram, kSystemMin), kSystemMax);

  const uint64_t local_floor_retained = 2 * vram;
  const uint64_t local_floor_transient = Transient(local_floor_retained, kLocalAlign);

  limits.generous = local_floor_retained + Transient(ram_aligned, kSystemAlign) + 2 * kGiB;
  limits.ladder = local_floor_retained + Transient(ram_aligned / 2, kSystemAlign) + 2 * kGiB;
  limits.tight = local_floor_retained + 4 * kGiB;

  /* Skip rather than quietly exercise the wrong path on hardware these
   * assumptions do not fit. */
  const uint64_t ceiling =
      local_floor_retained + Transient(SmallestSystemAsk(ram_aligned), kSystemAlign);
  limits.valid = limits.tight > local_floor_transient && limits.tight < ceiling &&
                 limits.ladder > limits.tight && limits.generous > limits.ladder;

  limits.ladder_possible = ram_aligned / 2 >= kSystemMin;

  return limits;
}

/* Runs in a forked child: RLIMIT_AS cannot be raised once lowered. `budget` is
 * what the thunk may map on top of what this process already has. */
int RunConstrained(uint64_t budget, HSAuint32 cpu_node) {
  const uint64_t limit = MappedAddressSpace() + budget;

  struct rlimit rl;
  rl.rlim_cur = limit;
  rl.rlim_max = limit;
  if (setrlimit(RLIMIT_AS, &rl) != 0) {
    perror("setrlimit(RLIMIT_AS)");
    return kFailExitCode;
  }

  if (hsaKmtOpenKFD() != HSAKMT_STATUS_SUCCESS) return kSkipExitCode;

  HsaSystemProperties props = {};
  if (hsaKmtAcquireSystemProperties(&props) != HSAKMT_STATUS_SUCCESS) {
    hsaKmtCloseKFD();
    return kFailExitCode;
  }

  /* HostAccess on a CPU node selects the system pool; NonPaged is what routes
   * the request through the code this test covers. Reading the topology alone
   * never touches that pool. */
  HsaMemFlags flags = {};
  flags.ui32.HostAccess = 1;
  flags.ui32.NonPaged = 1;

  void* mem = nullptr;
  HSAKMT_STATUS status = hsaKmtAllocMemory(cpu_node, kAllocSize, flags, &mem);
  if (status == HSAKMT_STATUS_SUCCESS) hsaKmtFreeMemory(mem, kAllocSize);

  hsaKmtReleaseSystemProperties();
  hsaKmtCloseKFD();

  /* An error is a legitimate outcome here; a signal is not. */
  return status == HSAKMT_STATUS_SUCCESS ? kPassExitCode : kFailExitCode;
}

struct ChildResult {
  bool exited = false;  /* false means it died by signal */
  int status = 0;
  int signal = 0;
};

ChildResult ForkAndRun(uint64_t budget, HSAuint32 cpu_node) {
  ChildResult result;

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return result;
  }

  if (pid == 0) _exit(RunConstrained(budget, cpu_node));

  int wait_status = 0;
  if (waitpid(pid, &wait_status, 0) != pid) {
    perror("waitpid");
    return result;
  }

  if (WIFEXITED(wait_status)) {
    result.exited = true;
    result.status = WEXITSTATUS(wait_status);
  } else if (WIFSIGNALED(wait_status)) {
    result.signal = WTERMSIG(wait_status);
  }
  return result;
}

const char* Describe(const ChildResult& r) {
  static char buf[64];
  if (r.exited)
    snprintf(buf, sizeof(buf), "exit %d", r.status);
  else if (r.signal != 0)
    snprintf(buf, sizeof(buf), "killed by signal %d (%s)", r.signal, strsignal(r.signal));
  else
    snprintf(buf, sizeof(buf), "no result");
  return buf;
}

}  // namespace

int main() {
  /* Measure the machine unconstrained first; the limits depend on it. */
  if (hsaKmtOpenKFD() != HSAKMT_STATUS_SUCCESS) {
    fprintf(stderr, "no DXCore adapter available; skipping\n");
    return kSkipExitCode;
  }

  HsaSystemProperties props = {};
  if (hsaKmtAcquireSystemProperties(&props) != HSAKMT_STATUS_SUCCESS) {
    fprintf(stderr, "hsaKmtAcquireSystemProperties failed unconstrained; skipping\n");
    hsaKmtCloseKFD();
    return kSkipExitCode;
  }

  const uint64_t vram = SummedVram(props);
  const uint64_t ram = HostRam();

  HSAuint32 cpu_node = 0;
  const bool have_cpu_node = FindCpuNode(props, &cpu_node);

  hsaKmtReleaseSystemProperties();
  hsaKmtCloseKFD();

  if (!have_cpu_node) {
    fprintf(stderr, "no CPU node in the topology; skipping\n");
    return kSkipExitCode;
  }

  printf("vram %.1f GiB, ram %.1f GiB, process overhead %.1f GiB\n",
         static_cast<double>(vram) / kGiB, static_cast<double>(ram) / kGiB,
         static_cast<double>(MappedAddressSpace()) / kGiB);

  const Limits limits = ComputeLimits(vram, ram);
  if (!limits.valid) {
    fprintf(stderr, "no usable RLIMIT_AS window on this hardware; skipping\n");
    return kSkipExitCode;
  }

  const struct {
    const char* name;
    uint64_t budget;
    bool expect_success;
    bool applicable;
  } cases[] = {
      {"generous (1x fits)", limits.generous, true, true},
      {"ladder (1x refused)", limits.ladder, true, limits.ladder_possible},
      {"tight (no rung fits)", limits.tight, false, true},
  };

  int failures = 0;

  for (const auto& c : cases) {
    if (!c.applicable) {
      printf("skip %-22s        not meaningful: 1/2 RAM is below the %llu GiB floor\n",
             c.name, static_cast<unsigned long long>(kSystemMin / kGiB));
      continue;
    }

    const ChildResult r = ForkAndRun(c.budget, cpu_node);
    const double gib = static_cast<double>(c.budget) / kGiB;

    if (r.exited && r.status == kSkipExitCode) {
      fprintf(stderr, "child skipped at %.1f GiB; skipping\n", gib);
      return kSkipExitCode;
    }

    /* Surviving is the point: this is where the unfixed thunk took a SIGSEGV. */
    if (!r.exited) {
      printf("FAIL %-22s %6.1f GiB  %s\n", c.name, gib, Describe(r));
      failures++;
      continue;
    }

    /* A served allocation under `ladder` means a smaller pool was reserved. */
    const bool succeeded = r.status == kPassExitCode;
    if (c.expect_success && !succeeded) {
      printf("FAIL %-22s %6.1f GiB  %s (expected success)\n", c.name, gib, Describe(r));
      failures++;
      continue;
    }

    printf("ok   %-22s %6.1f GiB  %s\n", c.name, gib, Describe(r));
  }

  return failures == 0 ? kPassExitCode : kFailExitCode;
}
