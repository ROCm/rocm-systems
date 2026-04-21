#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../raiser.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

struct KernelResult {
  std::string coFile;
  std::string kernelName;
  bool success = false;
  bool hasDivergentExec = false;
  int liftedCount = 0;
  int totalCount = 0;
  std::string failMnemonic;
  std::string failFormat;
};

static std::vector<std::string> collectCoFiles(const std::string &path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    fprintf(stderr, "ERROR: Cannot stat '%s'\n", path.c_str());
    return {};
  }

  if (S_ISREG(st.st_mode)) {
    if ((path.size() >= 3 && path.substr(path.size() - 3) == ".co") ||
        (path.size() >= 6 && path.substr(path.size() - 6) == ".hsaco"))
      return {path};
    fprintf(stderr, "WARNING: '%s' is not a .co/.hsaco file, skipping\n",
            path.c_str());
    return {};
  }

  std::vector<std::string> results;
  std::vector<std::string> dirs = {path};
  while (!dirs.empty()) {
    std::string dir = dirs.back();
    dirs.pop_back();
    DIR *d = opendir(dir.c_str());
    if (!d) continue;
    while (struct dirent *entry = readdir(d)) {
      if (entry->d_name[0] == '.') continue;
      std::string full = dir + "/" + entry->d_name;
      struct stat es;
      if (stat(full.c_str(), &es) != 0) continue;
      if (S_ISDIR(es.st_mode)) {
        dirs.push_back(full);
      } else if (S_ISREG(es.st_mode)) {
        if ((full.size() >= 3 && full.substr(full.size() - 3) == ".co") ||
            (full.size() >= 6 && full.substr(full.size() - 6) == ".hsaco"))
          results.push_back(full);
      }
    }
    closedir(d);
  }
  std::sort(results.begin(), results.end());
  return results;
}

static void runBatchRaise(const std::string &path, const std::string &isa) {
  auto coFiles = collectCoFiles(path);
  ASSERT_FALSE(coFiles.empty()) << "No .co files found at '" << path << "'";

  printf("=== Batch LLVM IR Raiser Test ===\n");
  printf("Target ISA: %s\n", isa.c_str());
  printf("Code objects: %zu\n\n", coFiles.size());

  std::map<std::string, int> failMnemonics;
  std::map<std::string, int> failFormats;
  int totalKernels = 0, successKernels = 0, failedKernels = 0;
  int totalFiles = 0, filesWithSuccess = 0;

  for (auto &coPath : coFiles) {
    auto coData = transpiler::readFile(coPath);
    if (coData.empty()) continue;

    auto kernelNames = transpiler::listKernelNames(coData);
    if (kernelNames.empty()) continue;

    totalFiles++;
    bool anySuccess = false;

    auto text = transpiler::extractTextSection(coData);
    if (!text.valid) continue;

    for (auto &kName : kernelNames) {
      totalKernels++;
      auto meta = transpiler::extractKernelMeta(coData, kName);
      auto raised = transpiler::raiseToIR(text.bytes, isa, kName, meta);

      if (raised.success) {
        successKernels++;
        anySuccess = true;
      } else {
        failedKernels++;
        std::string mn = raised.failure.mnemonic.empty()
                             ? "unknown"
                             : raised.failure.mnemonic;
        std::string fmt = raised.failure.format.empty()
                              ? "unknown"
                              : raised.failure.format;
        failMnemonics[mn]++;
        failFormats[fmt]++;
        EXPECT_TRUE(raised.success)
            << "Kernel '" << kName << "' in " << coPath
            << " failed on mnemonic: " << mn << " [" << fmt << "]";
      }
    }
    if (anySuccess) filesWithSuccess++;
  }

  printf("\nBatch Raiser Summary:\n");
  printf("  Files: %d, Kernels: %d, Succeeded: %d (%.1f%%), Failed: %d\n",
         totalFiles, totalKernels, successKernels,
         totalKernels ? 100.0 * successKernels / totalKernels : 0.0,
         failedKernels);

  if (!failMnemonics.empty()) {
    std::vector<std::pair<int, std::string>> sorted;
    for (auto &[mn, cnt] : failMnemonics)
      sorted.push_back({cnt, mn});
    std::sort(sorted.rbegin(), sorted.rend());
    printf("  Top failing mnemonics:\n");
    int shown = 0;
    for (auto &[cnt, mn] : sorted) {
      printf("    %-40s  %d\n", mn.c_str(), cnt);
      if (++shown >= 10) break;
    }
  }

  EXPECT_EQ(failedKernels, 0)
      << failedKernels << " of " << totalKernels << " kernels failed to raise";
}

// Fork-isolated batch raise: each code object is raised in a child process
// so that report_fatal_error in one kernel does not kill the entire test.
//
// The child writes a single-kernel outcome into a shared-memory struct so
// the parent can aggregate a top-level mnemonic histogram even across
// children that abort via report_fatal_error.  This mirrors the approach
// in corpus_test.cpp's SharedCorpusStats.
struct ForkRaiseStats {
  bool done;
  bool success;
  static constexpr int kMnemonicLen = 64;
  char failMnemonic[kMnemonicLen];
  char failFormat[kMnemonicLen];
};

// Passing `expectedFailures < 0` disables the regression check and turns the
// test into a pure coverage report (used by the --test-all full sweep).
static void runBatchRaiseIsolated(const std::vector<std::string> &coFiles,
                                  const std::string &isa,
                                  const std::string &label,
                                  int expectedFailures) {
  ASSERT_FALSE(coFiles.empty()) << "No .co files provided";

  printf("=== Batch LLVM IR Raiser Test (fork-isolated) ===\n");
  printf("Scope:  %s\n", label.c_str());
  printf("Target ISA: %s\n", isa.c_str());
  if (expectedFailures < 0) {
    printf("Code objects: %zu (coverage-only; regression check disabled)\n\n",
           coFiles.size());
  } else {
    printf("Code objects: %zu (expected failures: %d)\n\n", coFiles.size(),
           expectedFailures);
  }

  int totalKernels = 0, successKernels = 0, failedKernels = 0,
      crashedKernels = 0;
  std::map<std::string, int> failMnemonics;

  for (auto &coPath : coFiles) {
    auto coData = transpiler::readFile(coPath);
    if (coData.empty()) continue;
    auto kernelNames = transpiler::listKernelNames(coData);
    if (kernelNames.empty()) continue;
    auto text = transpiler::extractTextSection(coData);
    if (!text.valid) continue;

    for (auto &kName : kernelNames) {
      totalKernels++;

      auto *shm = static_cast<ForkRaiseStats *>(
          mmap(nullptr, sizeof(ForkRaiseStats), PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_ANONYMOUS, -1, 0));
      ASSERT_NE(shm, MAP_FAILED) << "mmap failed";
      memset(shm, 0, sizeof(ForkRaiseStats));

      pid_t pid = fork();
      if (pid == 0) {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        auto meta = transpiler::extractKernelMeta(coData, kName);
        auto raised = transpiler::raiseToIR(text.bytes, isa, kName, meta);
        shm->success = raised.success;
        if (!raised.success) {
          strncpy(shm->failMnemonic, raised.failure.mnemonic.c_str(),
                  ForkRaiseStats::kMnemonicLen - 1);
          strncpy(shm->failFormat, raised.failure.format.c_str(),
                  ForkRaiseStats::kMnemonicLen - 1);
        }
        shm->done = true;
        _exit(0);
      }

      int st = 0;
      waitpid(pid, &st, 0);

      if (shm->done && shm->success) {
        successKernels++;
      } else if (!shm->done || WIFSIGNALED(st)) {
        crashedKernels++;
        failMnemonics["<crash>"]++;
        printf("  CRASH: %s in %s\n", kName.c_str(), coPath.c_str());
      } else {
        failedKernels++;
        std::string mn = shm->failMnemonic[0] ? shm->failMnemonic : "unknown";
        std::string fmt = shm->failFormat[0] ? shm->failFormat : "unknown";
        failMnemonics[mn]++;
        printf("  FAIL:  %s in %s -> %s [%s]\n", kName.c_str(),
               coPath.c_str(), mn.c_str(), fmt.c_str());
      }

      munmap(shm, sizeof(ForkRaiseStats));
    }
  }

  int totalFailed = failedKernels + crashedKernels;
  printf("\nBatch Raiser Summary:\n");
  printf("  Kernels: %d, Succeeded: %d (%.1f%%), Failed: %d, Crashed: %d\n",
         totalKernels, successKernels,
         totalKernels ? 100.0 * successKernels / totalKernels : 0.0,
         failedKernels, crashedKernels);

  if (!failMnemonics.empty()) {
    std::vector<std::pair<int, std::string>> sorted;
    for (auto &[mn, cnt] : failMnemonics)
      sorted.push_back({cnt, mn});
    std::sort(sorted.rbegin(), sorted.rend());
    printf("  Top failing mnemonics:\n");
    int shown = 0;
    for (auto &[cnt, mn] : sorted) {
      printf("    %-40s  %d\n", mn.c_str(), cnt);
      if (++shown >= 10) break;
    }
  }

  if (expectedFailures < 0) return;  // coverage-only mode

  EXPECT_LE(totalFailed, expectedFailures)
      << "Regression: expected at most " << expectedFailures
      << " failures but got " << totalFailed;
  if (totalFailed < expectedFailures)
    printf("  NOTE: fewer failures than expected (%d < %d) — update the "
           "expected count.\n", totalFailed, expectedFailures);
}

TEST(BatchRaise, Gfx1250TestData) {
  std::string path = std::string(TEST_DATA_DIR) + "/gfx1250";
  if (!fileExists(path))
    GTEST_SKIP() << "Test data directory not found: " << path;
  runBatchRaise(path, "gfx1250");
}

// Deterministic "representative subset" selector for the AITER gfx950 corpus.
//
// Categories are the first path segment under AITER_CORPUS_DIR; loose files
// at the top level are binned under ".".  Per-category quotas mirror
// REPRESENTATIVE_SUBSET in hotswap/kernels/fetch_aiter_kernels.py so the
// local test subset agrees with what the fetcher itself considers
// representative.  Files are picked after an alphabetical sort, so the
// selection is stable across runs and checkouts.
//
// Unknown categories (ones that land locally but aren't in the quota map
// yet) fall through to kDefaultQuota, which is intentionally small — the
// goal is variety across opcode/encoding shapes, not depth within any one
// family.  New categories should be added to both this map and to
// fetch_aiter_kernels.py.
static std::vector<std::string> selectAiterRepresentativeSubset(
    const std::vector<std::string> &files, const std::string &corpusDir) {
  static const std::map<std::string, int> kQuota = {
      {"bf16gemm", 2},           {"f4gemm", 2},
      {"fmha_v3_fwd", 3},        {"fmha_v3_bwd", 2},
      {"fmoe", 2},               {"fmoe_2stages", 2},
      {"fp8gemm_blockscale", 2}, {"i8gemm", 2},
      {"mla", 2},                {"pa", 2},
      {"topksoftmax", 2},
      {".", 99},  // all loose top-level .co files (small set, ~4 today).
  };
  static constexpr int kDefaultQuota = 2;

  std::string prefix = corpusDir;
  while (!prefix.empty() && prefix.back() == '/')
    prefix.pop_back();

  std::vector<std::string> sorted = files;
  std::sort(sorted.begin(), sorted.end());

  std::map<std::string, int> taken;
  std::vector<std::string> out;
  for (const auto &f : sorted) {
    if (f.compare(0, prefix.size(), prefix) != 0) continue;
    std::string rel = f.substr(prefix.size());
    while (!rel.empty() && rel.front() == '/') rel.erase(0, 1);
    auto slash = rel.find('/');
    std::string category = (slash == std::string::npos) ? "." : rel.substr(0, slash);
    auto it = kQuota.find(category);
    int quota = (it != kQuota.end()) ? it->second : kDefaultQuota;
    if (taken[category] >= quota) continue;
    taken[category]++;
    out.push_back(f);
  }
  return out;
}

// AITER CK production kernels (gfx950): GEMM, FMHA, MoE, MLA, PA, TopK.
// Fork-isolated so that report_fatal_error in one kernel doesn't kill the
// test.
//
// By default, the test runs a deterministic *representative subset*
// (~27 kernels, 2–3 per category; selection matches
// hotswap/kernels/fetch_aiter_kernels.py's REPRESENTATIVE_SUBSET).  This
// keeps wall-time on CTest well under the 120 s timeout while still
// exercising every kernel family we ship against.
//
// Known-unsupported opcodes on the subset (tracked via expectedFailures):
//   v_permlane32_swap_b32 — gfx950 cross-lane swap used by
//     fmha_v3_fwd/fwd_hd128_bf16{,_causal,_causal_group}.co.  When raiser
//     support lands, drop kSubsetExpectedFailures back to 0 (the test
//     already warns when the actual count drops below the expectation).
//
// Pass `--test-all` to sweep the full corpus (1,300+ kernels today).  The
// full sweep is a pure coverage report (regression check disabled) and is
// intended for developers working on raiser coverage, not for every ctest
// invocation.
//
//   # default (subset, ~9 s, enforces expectedFailures):
//   ctest --test-dir build -R BatchRaise.AiterGfx950
//
//   # full sweep (coverage-only):
//   ./build/transpiler_tests --gtest_filter=BatchRaise.AiterGfx950 --test-all
TEST(BatchRaise, AiterGfx950) {
  // Current known-failing kernels on the representative subset.  Bump up
  // when a new mnemonic legitimately lands in the subset; bump down when
  // raiser support expands (the helper warns if actual < expected).
  constexpr int kSubsetExpectedFailures = 3;

  std::string path = AITER_CORPUS_DIR;
  if (!fileExists(path))
    GTEST_SKIP() << "AITER corpus not found: " << path
                 << " (run hotswap/kernels/fetch_aiter_kernels.py)";

  auto allCoFiles = collectCoFiles(path);
  ASSERT_FALSE(allCoFiles.empty())
      << "No .co files found at '" << path << "'";

  std::vector<std::string> selected;
  std::string label;
  int expectedFailures;
  if (g_config.testAll) {
    selected = allCoFiles;
    label = "full AITER gfx950 corpus (--test-all, coverage-only)";
    expectedFailures = -1;  // report only; don't enforce.
  } else {
    selected = selectAiterRepresentativeSubset(allCoFiles, path);
    ASSERT_FALSE(selected.empty())
        << "Representative subset is empty — check quota map vs. corpus "
           "layout at '" << path << "'";
    label = "representative AITER gfx950 subset (pass --test-all for full "
            "sweep)";
    expectedFailures = kSubsetExpectedFailures;
  }

  runBatchRaiseIsolated(selected, "gfx950", label, expectedFailures);
}

// Auto-detect ISA from a code-object path's filename (...gfx950...).
// Returns "gfx942" when no gfxNNNx token is present, matching the previous
// default behavior of BatchRaise.CustomDir.
static std::string isaFromCoPath(const std::string &coPath) {
  std::string isa;
  for (auto pos = coPath.find("gfx"); pos != std::string::npos;
       pos = coPath.find("gfx", pos + 1)) {
    size_t j = pos + 3;
    while (j < coPath.size() && coPath[j] >= '0' && coPath[j] <= '9') j++;
    if (j > pos + 3) {
      if (j < coPath.size() && coPath[j] >= 'a' && coPath[j] <= 'z') j++;
      isa = coPath.substr(pos, j - pos);
    }
  }
  if (isa.empty()) isa = "gfx942";
  return isa;
}

// Ad-hoc batch raise against user-supplied directories via --raise-dir=PATH.
// Pure coverage report: fork-isolated, prints mnemonic histogram, never
// asserts a specific failure count.
TEST(BatchRaise, CustomDir) {
  if (g_config.raiseDirs.empty())
    GTEST_SKIP() << "No --raise-dir= provided";

  for (auto &dir : g_config.raiseDirs) {
    auto coFiles = collectCoFiles(dir);
    ASSERT_FALSE(coFiles.empty()) << "No .co files found at '" << dir << "'";

    // All files in a single --raise-dir share an ISA in practice; use the
    // first one's filename to pick one.
    std::string isa = isaFromCoPath(coFiles.front());
    std::string label = "--raise-dir=" + dir;
    runBatchRaiseIsolated(coFiles, isa, label, /*expectedFailures=*/-1);
  }
}
