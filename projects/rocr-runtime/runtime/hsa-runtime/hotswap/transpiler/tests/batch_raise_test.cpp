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
struct ForkRaiseStats {
  int totalKernels;
  int successKernels;
  int failedKernels;
  int crashedKernels;
  bool done;
};

static void runBatchRaiseIsolated(const std::string &path,
                                  const std::string &isa,
                                  int expectedFailures) {
  auto coFiles = collectCoFiles(path);
  ASSERT_FALSE(coFiles.empty()) << "No .co files found at '" << path << "'";

  printf("=== Batch LLVM IR Raiser Test (fork-isolated) ===\n");
  printf("Target ISA: %s\n", isa.c_str());
  printf("Code objects: %zu (expected failures: %d)\n\n", coFiles.size(),
         expectedFailures);

  int totalKernels = 0, successKernels = 0, failedKernels = 0,
      crashedKernels = 0;

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
        shm->done = true;
        shm->totalKernels = raised.success ? 1 : 0;
        _exit(0);
      }

      int st = 0;
      waitpid(pid, &st, 0);

      if (shm->done && shm->totalKernels == 1) {
        successKernels++;
      } else if (!shm->done || WIFSIGNALED(st)) {
        crashedKernels++;
        printf("  CRASH: %s in %s\n", kName.c_str(), coPath.c_str());
      } else {
        failedKernels++;
        printf("  FAIL:  %s in %s\n", kName.c_str(), coPath.c_str());
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

// AITER CK production kernels (gfx950): GEMM, FMHA, MoE, MLA, PA, TopK.
// Fork-isolated so that report_fatal_error in one kernel doesn't kill the test.
// All 27 kernels raise successfully as of 2026-04-16.
TEST(BatchRaise, AiterGfx950) {
  std::string path = AITER_CORPUS_DIR;
  if (!fileExists(path))
    GTEST_SKIP() << "AITER corpus not found: " << path
                 << " (run hotswap/kernels/fetch_aiter_kernels.py)";
  runBatchRaiseIsolated(path, "gfx950", /*expectedFailures=*/0);
}

// Ad-hoc batch raise against user-supplied directories via --raise-dir=PATH.
// Uses fork isolation; reports all failures but does not assert a specific count.
TEST(BatchRaise, CustomDir) {
  if (g_config.raiseDirs.empty())
    GTEST_SKIP() << "No --raise-dir= provided";

  for (auto &dir : g_config.raiseDirs) {
    auto coFiles = collectCoFiles(dir);
    ASSERT_FALSE(coFiles.empty()) << "No .co files found at '" << dir << "'";

    printf("=== Batch raise: %s (%zu files) ===\n", dir.c_str(),
           coFiles.size());

    int totalKernels = 0, successKernels = 0, failedKernels = 0,
        crashedKernels = 0;

    for (auto &coPath : coFiles) {
      auto coData = transpiler::readFile(coPath);
      if (coData.empty()) continue;
      auto kernelNames = transpiler::listKernelNames(coData);
      if (kernelNames.empty()) continue;
      auto text = transpiler::extractTextSection(coData);
      if (!text.valid) continue;

      // Auto-detect ISA from the code object filename.
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

      for (auto &kName : kernelNames) {
        totalKernels++;

        auto *shm = static_cast<ForkRaiseStats *>(
            mmap(nullptr, sizeof(ForkRaiseStats), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0));
        ASSERT_NE(shm, MAP_FAILED);
        memset(shm, 0, sizeof(ForkRaiseStats));

        pid_t pid = fork();
        if (pid == 0) {
          int devnull = open("/dev/null", O_WRONLY);
          if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
          auto meta = transpiler::extractKernelMeta(coData, kName);
          auto raised = transpiler::raiseToIR(text.bytes, isa, kName, meta);
          shm->done = true;
          shm->totalKernels = raised.success ? 1 : 0;
          _exit(0);
        }

        int st = 0;
        waitpid(pid, &st, 0);

        if (shm->done && shm->totalKernels == 1) {
          successKernels++;
        } else if (!shm->done || WIFSIGNALED(st)) {
          crashedKernels++;
          printf("  CRASH: %s in %s\n", kName.c_str(), coPath.c_str());
        } else {
          failedKernels++;
          printf("  FAIL:  %s in %s\n", kName.c_str(), coPath.c_str());
        }

        munmap(shm, sizeof(ForkRaiseStats));
      }
    }

    printf("\nSummary for %s:\n", dir.c_str());
    printf("  Kernels: %d, Succeeded: %d (%.1f%%), Failed: %d, Crashed: %d\n",
           totalKernels, successKernels,
           totalKernels ? 100.0 * successKernels / totalKernels : 0.0,
           failedKernels, crashedKernels);
  }
}
