// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_pretranslate.cpp
/// @brief Populate the shared translation tier ahead of time.
///
/// @details Translating a large device library takes minutes. A process that
/// does it at load time pays that once per process, and a container that starts,
/// runs one job and exits pays it every single time. This tool moves the work to
/// image-build or post-install time: it translates the objects it is given and
/// writes them where the runtime hook looks first.
///
/// It links the translator and the store, and nothing else -- in particular not
/// the hook, which contributes nothing to translation. Both sides derive their
/// keys and their tree location from the translator itself, so agreement is a
/// property of linking against the same shared object rather than of two
/// programs being kept in step.

#include "rocjitsu/code/dbt/gfx1250_b0_a0_cache.h"
#include "rocjitsu/code/dbt/translation_store.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

/// @brief Refuse anything too large to be a code object.
/// @details Bounds the allocation from a path that turns out to be something
/// else entirely; the store applies its own limit to what it will accept.
constexpr uint64_t kMaxInputBytes = 4ull << 30;

[[nodiscard]] bool read_file(const char *path, std::vector<uint8_t> &out) {
  const int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return false;
  struct stat info {};
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) ||
      static_cast<uint64_t>(info.st_size) > kMaxInputBytes) {
    close(fd);
    return false;
  }
  out.resize(static_cast<size_t>(info.st_size));
  size_t done = 0;
  while (done < out.size()) {
    const ssize_t got = read(fd, out.data() + done, out.size() - done);
    if (got <= 0) {
      close(fd);
      return false;
    }
    done += static_cast<size_t>(got);
  }
  close(fd);
  return true;
}

void usage(const char *program) {
  std::fprintf(stderr,
               "usage: %s [--store-root DIR] [--force] [--portable] "
               "[--fail-on-skipped] FILE...\n"
               "\n"
               "Translate each gfx1250 code object and record the result in the\n"
               "shared translation tier, where the runtime hook finds it without\n"
               "translating again.\n"
               "\n"
               "  --store-root DIR  Write here instead of the location derived from\n"
               "                    the translator's own install prefix.\n"
               "  --force           Translate even objects already recorded.\n"
               "  --portable        Accepted for packaging compatibility; this\n"
               "                    domain always uses its packaged-cache contract.\n"
               "  --fail-on-skipped Exit nonzero if any input is rejected.\n"
               "\n"
               "Each input must be a standalone AMDGPU code object. Extracting\n"
               "those from fat binaries, bundles and archives is the caller's job;\n"
               "see rocjitsu-pretranslate.py, installed alongside this tool.\n",
               program);
}

/// @brief What became of one input.
enum class Outcome {
  kWritten,     ///< Translated and recorded.
  kAlreadyHeld, ///< The tier already had it; nothing to do.
  kRejected,    ///< Not something this translator handles. Expected in bulk runs.
  kFailed,      ///< Should have worked and did not.
};

struct Totals {
  size_t written = 0;
  size_t already_held = 0;
  size_t rejected = 0;
  size_t failed = 0;
};

Outcome pretranslate_one(const char *path, rocjitsu::TranslationStore &store, bool force) {
  std::vector<uint8_t> source;
  if (!read_file(path, source) || source.empty()) {
    std::fprintf(stderr, "%s: cannot read\n", path);
    return Outcome::kFailed;
  }

  const rocjitsu::CacheKey key = store.key_for(source, rocjitsu::kGfx1250B0A0Identity);
  if (!key.valid) {
    std::fprintf(stderr, "%s: no cache key could be derived\n", path);
    return Outcome::kFailed;
  }

  // Re-running over an unchanged tree is the normal case -- an image build that
  // adds one library should not retranslate the rest -- so an entry that is
  // already present is a success, not work to redo.
  if (!force && !store.lookup(key, rocjitsu::kGfx1250B0A0Identity).empty()) {
    std::printf("held    %s\n", path);
    return Outcome::kAlreadyHeld;
  }

  uint8_t *translated = nullptr;
  size_t translated_size = 0;
  // The info block is required by the entry point rather than optional, and this
  // tool has no use for what it reports: the store keys on the source bytes it
  // already holds. Diagnostics are declined for the same reason -- a scan over an
  // install tree meets many objects this translator will refuse, and the refusal
  // itself is the answer.
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  const rj_status_t status = rj_gfx1250_b0_to_a0_translate(
      source.data(), source.size(), &translated, &translated_size, &info, nullptr, nullptr);
  if (status != ROCJITSU_STATUS_SUCCESS || translated == nullptr || translated_size == 0) {
    rj_gfx1250_b0_to_a0_free(translated);
    // A verdict on the bytes, not a malfunction: a scan over an install tree
    // will hand this tool plenty of objects for other targets.
    if (status == ROCJITSU_STATUS_INVALID_CODE_OBJECT ||
        status == ROCJITSU_STATUS_INVALID_ARGUMENT) {
      std::printf("skip    %s (not a translatable gfx1250 object)\n", path);
      return Outcome::kRejected;
    }
    std::fprintf(stderr, "%s: translation failed with status %d\n", path, static_cast<int>(status));
    return Outcome::kFailed;
  }

  store.store(key, {translated, translated_size}, rocjitsu::kGfx1250B0A0Identity);
  rj_gfx1250_b0_to_a0_free(translated);

  // Read it back. The store is best effort by design: it declines quietly when
  // an entry is too large or the filesystem is too full, which is right for a
  // runtime that can translate instead but wrong for a tool whose only product
  // is the entry. Without this check the tool's whole failure mode is to report
  // success and leave start-up paying full cost.
  if (store.lookup(key, rocjitsu::kGfx1250B0A0Identity).empty()) {
    std::fprintf(stderr, "%s: translated but the store did not keep it\n", path);
    return Outcome::kFailed;
  }

  std::printf("write   %s (%zu -> %zu bytes)\n", path, source.size(), translated_size);
  return Outcome::kWritten;
}

} // namespace

int main(int argc, char **argv) {
  std::string root;
  bool force = false;
  bool fail_on_skipped = false;
  std::vector<const char *> inputs;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    }
    if (arg == "--force") {
      force = true;
    } else if (arg == "--portable") {
      // Compatibility with packaging recipes written before the domain owned
      // its key policy. There is deliberately no mode override: an entry this
      // product's runtime cannot read is not a useful output.
    } else if (arg == "--fail-on-skipped") {
      fail_on_skipped = true;
    } else if (arg == "--store-root") {
      if (++i == argc) {
        std::fprintf(stderr, "--store-root needs a directory\n");
        return 2;
      }
      root = argv[i];
    } else if (arg.starts_with("--store-root=")) {
      root = arg.substr(std::strlen("--store-root="));
    } else if (arg.starts_with("-")) {
      std::fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    } else {
      inputs.push_back(argv[i]);
    }
  }

  if (inputs.empty()) {
    usage(argv[0]);
    return 2;
  }

  if (root.empty())
    root = rocjitsu::shared_translation_root(rocjitsu::gfx1250_b0_a0_translator_anchor());
  if (root.empty()) {
    std::fprintf(stderr, "cannot locate the translator's install prefix; pass --store-root\n");
    return 1;
  }

  rocjitsu::TranslationStore store(
      rocjitsu::kGfx1250B0A0Domain, rocjitsu::gfx1250_b0_a0_translator_anchor(), root,
      rocjitsu::TranslationStore::Access::kReadWrite, rocjitsu::kGfx1250B0A0KeyMode);
  // Establish this before translating anything. Discovering an unusable root
  // after several minutes of work would be the same result reached expensively.
  if (!store.available()) {
    std::fprintf(stderr,
                 "%s is not usable as a translation store: it must be owned by root or by "
                 "this user, and writable by neither group nor other\n",
                 root.c_str());
    return 1;
  }
  std::printf("store   %s\n", root.c_str());

  Totals totals;
  for (const char *path : inputs) {
    switch (pretranslate_one(path, store, force)) {
    case Outcome::kWritten:
      ++totals.written;
      break;
    case Outcome::kAlreadyHeld:
      ++totals.already_held;
      break;
    case Outcome::kRejected:
      ++totals.rejected;
      break;
    case Outcome::kFailed:
      ++totals.failed;
      break;
    }
    std::fflush(stdout);
  }

  std::printf("summary written=%zu held=%zu skipped=%zu failed=%zu\n", totals.written,
              totals.already_held, totals.rejected, totals.failed);
  return totals.failed == 0 && (!fail_on_skipped || totals.rejected == 0) ? 0 : 1;
}
