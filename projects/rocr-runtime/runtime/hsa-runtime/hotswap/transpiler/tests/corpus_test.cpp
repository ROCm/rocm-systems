// HSACO corpus raiser test: discovers rocblas/hipblaslt libraries on the system,
// groups them by ISA, and raises kernels to LLVM IR.  Each ISA group runs in a
// forked child process with shared memory so that partial results survive
// crashes from LLVM assertion failures.
//
// Only runs when --test-all is passed (gated via GTEST_SKIP).

#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../raiser.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <map>
#include <signal.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// ============================================================================
// Utilities
// ============================================================================

static std::string isaFromFilename(const std::string &path) {
  auto slash = path.rfind('/');
  std::string fname = slash != std::string::npos ? path.substr(slash + 1) : path;
  auto i = fname.find("gfx");
  if (i == std::string::npos) return "";
  size_t j = i + 3;
  while (j < fname.size() && fname[j] >= '0' && fname[j] <= '9') j++;
  if (j == i + 3) return "";
  if (j < fname.size() && fname[j] >= 'a' && fname[j] <= 'z') j++;
  return fname.substr(i, j - i);
}

static void collectFiles(const std::string &dir, std::vector<std::string> &out) {
  DIR *d = opendir(dir.c_str());
  if (!d) return;
  while (struct dirent *e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    std::string name = e->d_name;
    std::string full = dir + "/" + name;
    struct stat st;
    if (stat(full.c_str(), &st) != 0) continue;
    if (S_ISDIR(st.st_mode)) {
      collectFiles(full, out);
    } else if (S_ISREG(st.st_mode)) {
      if ((name.size() >= 6 && name.substr(name.size() - 6) == ".hsaco") ||
          (name.size() >= 3 && name.substr(name.size() - 3) == ".co"))
        out.push_back(full);
    }
  }
  closedir(d);
}

static bool isELF64(const std::vector<uint8_t> &data) {
  return data.size() >= 5 && data[0] == 0x7f &&
         data[1] == 'E' && data[2] == 'L' && data[3] == 'F' &&
         data[4] == 2;
}

// ============================================================================
// Shared memory structure for fork-based crash isolation
// ============================================================================

struct SharedCorpusStats {
  int files;
  int kernels;
  int raised;
  int failed;
  bool done;
  static constexpr int kMaxMnemonics = 64;
  static constexpr int kMnemonicLen = 48;
  struct FailEntry { char mnemonic[kMnemonicLen]; int count; };
  int numFails;
  FailEntry fails[kMaxMnemonics];
};

static void raiseCorpusToShm(const std::vector<std::string> &files,
                             const std::string &isa,
                             SharedCorpusStats *shm,
                             int maxKernels) {
  std::map<std::string, int> failMap;

  for (auto &path : files) {
    auto data = transpiler::readFile(path);
    if (data.empty() || !isELF64(data)) continue;
    auto names = transpiler::listKernelNames(data);
    if (names.empty()) continue;
    shm->files++;
    auto text = transpiler::extractTextSection(data);
    if (!text.valid) continue;

    for (auto &kn : names) {
      if (maxKernels > 0 && shm->kernels >= maxKernels) break;
      shm->kernels++;
      auto meta = transpiler::extractKernelMeta(data, kn);
      auto res = transpiler::raiseToIR(text.bytes, isa, kn, meta);
      if (res.success) {
        shm->raised++;
      } else {
        shm->failed++;
        std::string mn = res.failMnemonic.empty() ? "unknown" : res.failMnemonic;
        failMap[mn]++;
      }
    }
    if (maxKernels > 0 && shm->kernels >= maxKernels) break;
  }

  int idx = 0;
  for (auto &[mn, cnt] : failMap) {
    if (idx >= SharedCorpusStats::kMaxMnemonics) break;
    strncpy(shm->fails[idx].mnemonic, mn.c_str(),
            SharedCorpusStats::kMnemonicLen - 1);
    shm->fails[idx].count = cnt;
    idx++;
  }
  shm->numFails = idx;
  shm->done = true;
}

// ============================================================================
// Discover HSACO corpus from system rocm installations
// ============================================================================

struct CorpusGroup {
  std::string isa;
  std::vector<std::string> files;
};

static std::vector<CorpusGroup> discoverCorpus(
    const std::vector<std::string> &extraDirs) {
  std::map<std::string, std::vector<std::string>> byIsa;

  auto scanDir = [&](const std::string &dir) {
    std::vector<std::string> files;
    collectFiles(dir, files);
    for (auto &f : files) {
      std::string isa = isaFromFilename(f);
      if (!isa.empty()) byIsa[isa].push_back(f);
    }
  };

  DIR *opt = opendir("/opt");
  if (opt) {
    while (struct dirent *e = readdir(opt)) {
      if (strncmp(e->d_name, "rocm", 4) != 0) continue;
      std::string base = std::string("/opt/") + e->d_name;
      std::string rocblasLib = base + "/lib/rocblas/library";
      if (fileExists(rocblasLib)) scanDir(rocblasLib);
      std::string hipblasltLib = base + "/lib/hipblaslt/library";
      if (fileExists(hipblasltLib)) scanDir(hipblasltLib);
    }
    closedir(opt);
  }

  for (auto &d : extraDirs) scanDir(d);

  std::vector<CorpusGroup> groups;
  for (auto &[isa, files] : byIsa) {
    auto sortedFiles = files;
    std::sort(sortedFiles.begin(), sortedFiles.end());
    sortedFiles.erase(std::unique(sortedFiles.begin(), sortedFiles.end()),
                      sortedFiles.end());
    groups.push_back({isa, std::move(sortedFiles)});
  }
  std::sort(groups.begin(), groups.end(),
            [](auto &a, auto &b) { return a.isa < b.isa; });
  return groups;
}

// ============================================================================
// Test
// ============================================================================

TEST(Corpus, RaiseHsacoByIsa) {
  if (!g_config.testAll)
    GTEST_SKIP() << "Skipped: pass --test-all to run HSACO corpus tests";

  auto corpus = discoverCorpus(g_config.corpusDirs);
  ASSERT_FALSE(corpus.empty()) << "No HSACO corpus found on this system";

  static const int kCorpusTimeoutSecs = 90;

  for (auto &g : corpus) {
    std::string name = "corpus_" + g.isa;
    printf("\n--- %s (%zu files) ---\n", name.c_str(), g.files.size());

    auto *shm = static_cast<SharedCorpusStats *>(
        mmap(nullptr, sizeof(SharedCorpusStats),
             PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(shm, MAP_FAILED) << "mmap failed";
    memset(shm, 0, sizeof(SharedCorpusStats));

    auto t0 = std::chrono::steady_clock::now();
    pid_t pid = fork();

    if (pid == 0) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
      raiseCorpusToShm(g.files, g.isa, shm, g_config.corpusLimit);
      _exit(0);
    }

    int st = 0;
    bool corpusTimedOut = false;
    while (true) {
      pid_t w = waitpid(pid, &st, WNOHANG);
      if (w == pid) break;
      double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count();
      if (elapsed > kCorpusTimeoutSecs) {
        corpusTimedOut = true;
        kill(pid, SIGKILL);
        waitpid(pid, &st, 0);
        break;
      }
      usleep(50000);
    }
    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();

    int files = shm->files;
    int kernels = shm->kernels;
    int raised = shm->raised;
    int failed = shm->failed;
    bool completed = shm->done;

    std::map<std::string, int> failMnemonics;
    for (int i = 0; i < shm->numFails && i < SharedCorpusStats::kMaxMnemonics; i++)
      failMnemonics[shm->fails[i].mnemonic] = shm->fails[i].count;

    munmap(shm, sizeof(SharedCorpusStats));

    bool childCrashed = !corpusTimedOut && WIFSIGNALED(st);

    if (kernels > 0) {
      double pct = 100.0 * raised / kernels;
      printf("  %d/%d raised (%.1f%%) from %d files in %.1fs", raised, kernels,
             pct, files, elapsed);
      if (corpusTimedOut) printf(" [TIMEOUT %ds]", kCorpusTimeoutSecs);
      if (childCrashed) printf(" [CRASHED sig %d]", WTERMSIG(st));
      if (!completed) printf(" [INCOMPLETE]");
      printf("\n");
    } else {
      printf("  no kernels found in %zu files (%.1fs)\n", g.files.size(), elapsed);
    }

    if (!failMnemonics.empty()) {
      std::vector<std::pair<int, std::string>> sorted;
      for (auto &[mn, cnt] : failMnemonics)
        sorted.push_back({cnt, mn});
      std::sort(sorted.rbegin(), sorted.rend());
      printf("  Top failing mnemonics:\n");
      int shown = 0;
      for (auto &[cnt, mn] : sorted) {
        printf("    %-35s  %d\n", mn.c_str(), cnt);
        if (++shown >= 10) break;
      }
    }

    EXPECT_FALSE(corpusTimedOut)
        << name << " timed out after " << kCorpusTimeoutSecs << "s"
        << " (partial: " << raised << "/" << kernels << " raised)";
    EXPECT_FALSE(childCrashed)
        << name << " child crashed with signal "
        << (childCrashed ? WTERMSIG(st) : 0)
        << " (partial: " << raised << "/" << kernels << " raised)";
    EXPECT_TRUE(completed || corpusTimedOut || childCrashed)
        << name << " child exited without completing and without crashing";
    EXPECT_GT(kernels, 0)
        << name << ": no kernels found in " << g.files.size() << " files";
    if (kernels > 0) {
      EXPECT_GT(raised, 0)
          << name << ": zero kernels raised out of " << kernels;
      EXPECT_EQ(failed, 0)
          << name << ": " << failed << " of " << kernels
          << " kernels failed to raise";
    }
  }
}
