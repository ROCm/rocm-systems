#include "../code_object_utils.hpp"
#include "../raiser.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <map>
#include <string>
#include <sys/stat.h>
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


int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: %s <path-to-co-files> [--isa=gfx950] [--verbose]\n"
            "  <path> can be a single .co file or a directory (recursive)\n",
            argv[0]);
    return 1;
  }

  std::string path = argv[1];
  std::string isa = "gfx942";
  bool verbose = false;
  for (int i = 2; i < argc; i++) {
    if (strncmp(argv[i], "--isa=", 6) == 0)
      isa = argv[i] + 6;
    else if (strcmp(argv[i], "--verbose") == 0)
      verbose = true;
  }

  auto coFiles = collectCoFiles(path);
  if (coFiles.empty()) {
    fprintf(stderr, "ERROR: No .co files found at '%s'\n", path.c_str());
    return 1;
  }

  printf("=== Batch LLVM IR Raiser Test ===\n");
  printf("Target ISA: %s\n", isa.c_str());
  printf("Code objects: %zu\n\n", coFiles.size());

  std::vector<KernelResult> results;
  std::map<std::string, int> failMnemonics;
  std::map<std::string, int> failFormats;
  int totalKernels = 0, successKernels = 0, failedKernels = 0;
  int totalFiles = 0, filesWithSuccess = 0;

  for (auto &coPath : coFiles) {
    auto coData = transpiler::readFile(coPath);
    if (coData.empty()) {
      fprintf(stderr, "WARNING: Cannot read '%s', skipping\n", coPath.c_str());
      continue;
    }

    auto kernelNames = transpiler::listKernelNames(coData);
    if (kernelNames.empty()) {
      if (verbose)
        printf("  [%s] no kernels found in metadata\n", coPath.c_str());
      continue;
    }

    totalFiles++;
    bool anySuccess = false;

    auto text = transpiler::extractTextSection(coData);
    if (!text.valid) {
      fprintf(stderr, "WARNING: No .text in '%s', skipping\n", coPath.c_str());
      continue;
    }

    for (auto &kName : kernelNames) {
      totalKernels++;
      auto meta = transpiler::extractKernelMeta(coData, kName);

      auto raised = transpiler::raiseToIR(text.bytes, isa, kName, meta);

      KernelResult kr;
      kr.coFile = coPath;
      kr.kernelName = kName;
      kr.success = raised.success;
      kr.hasDivergentExec = raised.hasDivergentExec;
      kr.liftedCount = raised.liftedCount;
      kr.totalCount = raised.totalCount;

      if (raised.success) {
        successKernels++;
        anySuccess = true;
        if (verbose)
          printf("  OK  %-60s %d/%d insts%s\n", kName.c_str(),
                 raised.liftedCount, raised.totalCount,
                 raised.hasDivergentExec ? " [EXEC divergent]" : "");
      } else {
        failedKernels++;
        kr.failMnemonic = raised.failMnemonic.empty() ? "unknown" : raised.failMnemonic;
        kr.failFormat = raised.failFormat.empty() ? "unknown" : raised.failFormat;
        failMnemonics[kr.failMnemonic]++;
        failFormats[kr.failFormat]++;
        if (verbose)
          printf("  FAIL %-60s -> %s [%s]\n", kName.c_str(),
                 kr.failMnemonic.c_str(), kr.failFormat.c_str());
      }
      results.push_back(kr);
    }

    if (anySuccess) filesWithSuccess++;
  }

  // Summary
  printf("\n");
  printf("================================================================\n");
  printf("                    BATCH RAISER SUMMARY\n");
  printf("================================================================\n");
  printf("\n");
  printf("Code objects scanned:   %d\n", totalFiles);
  printf("  with >= 1 success:    %d\n", filesWithSuccess);
  printf("\n");
  int divergentKernels = 0;
  for (auto &kr : results)
    if (kr.success && kr.hasDivergentExec)
      divergentKernels++;

  printf("Kernels attempted:      %d\n", totalKernels);
  printf("  Succeeded:            %d  (%.1f%%)\n", successKernels,
         totalKernels ? 100.0 * successKernels / totalKernels : 0.0);
  printf("    with EXEC divergence: %d\n", divergentKernels);
  printf("  Failed:               %d  (%.1f%%)\n", failedKernels,
         totalKernels ? 100.0 * failedKernels / totalKernels : 0.0);
  printf("\n");

  if (!failMnemonics.empty()) {
    // Sort by frequency
    std::vector<std::pair<int, std::string>> sorted;
    for (auto &[mn, cnt] : failMnemonics)
      sorted.push_back({cnt, mn});
    std::sort(sorted.rbegin(), sorted.rend());

    printf("Top failing mnemonics:\n");
    printf("  %-40s  %s\n", "Mnemonic", "Count");
    printf("  %-40s  %s\n", "----------------------------------------", "-----");
    int shown = 0;
    for (auto &[cnt, mn] : sorted) {
      printf("  %-40s  %d\n", mn.c_str(), cnt);
      if (++shown >= 30) break;
    }
    printf("\n");

    printf("Failures by format:\n");
    printf("  %-20s  %s\n", "Format", "Count");
    printf("  %-20s  %s\n", "--------------------", "-----");
    std::vector<std::pair<int, std::string>> fmtSorted;
    for (auto &[fmt, cnt] : failFormats)
      fmtSorted.push_back({cnt, fmt});
    std::sort(fmtSorted.rbegin(), fmtSorted.rend());
    for (auto &[cnt, fmt] : fmtSorted)
      printf("  %-20s  %d\n", fmt.c_str(), cnt);
  }

  printf("\n");
  return (failedKernels > 0) ? 1 : 0;
}
