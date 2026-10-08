/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/tuner/csv_tuner.cc, covering the configs
// compiled into librccl.so from the source tree's tuner/ directory.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "comm.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

#include CSV_TUNER_CC_PATH

namespace {

// Two valid configs, one of each optional-field length.
constexpr const char* kTwoConfigCsv =
    "# comment\n"
    "allreduce,0,1023,tree,ll,4,1,8,-1,-1\n"
    "\n"
    "allgather,1024,4095,ring,simple,8,-1,-1\n";

class CsvTunerMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetEnvFakes();
    // Point the ROCM_PATH probe at an empty dir so the suite cannot pick up an
    // installed /opt/rocm/share/rccl/tuner/*.csv from the host it runs on.
    emptyDir_ = MakeTempDir();
    SetMicroEnv("ROCM_PATH", emptyDir_.c_str());
    SetMicroEnvAbsent("NCCL_TUNER_CONFIG_FILE");
    rcclCsvTunerResetConfigPath();
  }

  void TearDown() override {
    for (const std::string& path : tempFiles_) unlink(path.c_str());
    tempFiles_.clear();
    // Deepest first, so nested probe directories come out before their parents.
    for (std::vector<std::string>::reverse_iterator it = tempDirs_.rbegin(); it != tempDirs_.rend(); ++it) {
      rmdir(it->c_str());
    }
    tempDirs_.clear();
    if (!emptyDir_.empty()) rmdir(emptyDir_.c_str());
    rcclCsvTunerResetConfigPath();
    ResetEnvFakes();
    ResetNcclFakes();
  }

  std::string MakeTempDir() {
    char tmpl[] = "/tmp/rccl_csv_tuner_dir_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    EXPECT_NE(nullptr, dir);
    return dir ? std::string(dir) : std::string();
  }

  // mkdir -p under base, recording each level for teardown.
  void MakeDirsUnder(const std::string& base, const std::string& relative) {
    std::string path = base;
    size_t start = 0;
    while (start <= relative.size()) {
      size_t slash = relative.find('/', start);
      const size_t end = (slash == std::string::npos) ? relative.size() : slash;
      path += "/" + relative.substr(start, end - start);
      if (mkdir(path.c_str(), 0755) == 0) tempDirs_.push_back(path);
      if (slash == std::string::npos) break;
      start = slash + 1;
    }
  }

  std::string WriteCsvAt(const std::string& path, const std::string& contents) {
    FILE* f = fopen(path.c_str(), "w");
    EXPECT_NE(nullptr, f) << path;
    if (!f) return std::string();
    EXPECT_EQ(contents.size(), fwrite(contents.data(), 1, contents.size(), f));
    fclose(f);
    tempFiles_.push_back(path);
    return path;
  }

  std::string WriteTempCsv(const std::string& contents) {
    char tmpl[] = "/tmp/rccl_csv_tuner_XXXXXX";
    int fd = mkstemp(tmpl);
    EXPECT_NE(-1, fd);
    if (fd == -1) return std::string();
    EXPECT_EQ((ssize_t)contents.size(), write(fd, contents.data(), contents.size()));
    close(fd);
    tempFiles_.push_back(tmpl);
    return tmpl;
  }

  // Arch of the first embedded rccl_tuner_<arch>.csv, empty if only a generic
  // config ships. Derived rather than hardcoded so renaming or re-targeting the
  // shipped CSV fails the arch cases instead of silently skipping them.
  static std::string FirstEmbeddedArch() {
    const std::string prefix = "rccl_tuner_";
    const std::string suffix = ".csv";
    for (const auto& entry : rcclCsvTunerEmbeddedConfigs()) {
      const std::string& name = entry.first;
      if (name.size() > prefix.size() + suffix.size() && name.compare(0, prefix.size(), prefix) == 0 &&
          name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
      }
    }
    return std::string();
  }

  static std::string ArchCsvName(const std::string& arch) { return "rccl_tuner_" + arch + ".csv"; }

  static bool ReadWholeFile(const std::string& path, std::string* out) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char chunk[4096];
    size_t n;
    out->clear();
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) out->append(chunk, n);
    fclose(f);
    return true;
  }

  // A context with no logger; the tests assert on parsed state, not log text.
  static CsvTunerContext MakeContext(size_t nRanks = 8, size_t nNodes = 1) {
    CsvTunerContext ctx{};
    ctx.nRanks = nRanks;
    ctx.nNodes = nNodes;
    return ctx;
  }

  static void FreeContext(CsvTunerContext* ctx) {
    free(ctx->configs);
    ctx->configs = nullptr;
  }

  // Field-wise, not memcmp: the structs come from malloc, so their padding
  // bytes are whatever the allocator last left there.
  static void ExpectSameConfig(const CsvTuningConfig& a, const CsvTuningConfig& b, const std::string& what) {
    EXPECT_EQ(a.collType, b.collType) << what;
    EXPECT_EQ(a.minBytes, b.minBytes) << what;
    EXPECT_EQ(a.maxBytes, b.maxBytes) << what;
    EXPECT_EQ(a.algorithm, b.algorithm) << what;
    EXPECT_EQ(a.protocol, b.protocol) << what;
    EXPECT_EQ(a.nChannels, b.nChannels) << what;
    EXPECT_EQ(a.nNodes, b.nNodes) << what;
    EXPECT_EQ(a.nRanks, b.nRanks) << what;
    EXPECT_EQ(a.numPipeOps, b.numPipeOps) << what;
    EXPECT_EQ(a.regBuff, b.regBuff) << what;
  }

  std::vector<std::string> tempFiles_;
  std::vector<std::string> tempDirs_;
  std::string emptyDir_;
};

// The embedded map is the whole point of the feature: it must be populated at
// build time from tuner/*.csv.
TEST_F(CsvTunerMicrotest, EmbeddedMapIsPopulated) {
  const std::map<std::string, std::string>& embedded = rcclCsvTunerEmbeddedConfigs();
  ASSERT_FALSE(embedded.empty()) << "tuner/*.csv was not compiled in";

  for (const auto& entry : embedded) {
    EXPECT_EQ(0u, entry.first.rfind("rccl_tuner", 0)) << entry.first;
    EXPECT_NE(std::string::npos, entry.first.find(".csv")) << entry.first;
    EXPECT_FALSE(entry.second.empty()) << entry.first;
  }
}

// The generator must embed each CSV byte for byte. Without this the rest of the
// suite only proves the parser is self-consistent, not that what shipped in the
// binary is what is in tuner/.
TEST_F(CsvTunerMicrotest, EmbeddedBytesMatchSourceCsv) {
  const std::map<std::string, std::string>& embedded = rcclCsvTunerEmbeddedConfigs();
  ASSERT_FALSE(embedded.empty());

  for (const auto& entry : embedded) {
    const std::string path = std::string(RCCL_TUNER_CSV_SOURCE_DIR) + "/" + entry.first;
    std::string onDisk;
    ASSERT_TRUE(ReadWholeFile(path, &onDisk)) << "cannot read " << path;
    EXPECT_EQ(onDisk, entry.second) << entry.first << " was not embedded verbatim";
  }
}

// Parsing the embedded text must produce exactly what parsing the same bytes
// off disk produces -- the claim the whole change rests on.
TEST_F(CsvTunerMicrotest, EmbeddedAndFileParseIdentically) {
  const std::map<std::string, std::string>& embedded = rcclCsvTunerEmbeddedConfigs();
  ASSERT_FALSE(embedded.empty());

  for (const auto& entry : embedded) {
    CsvTunerContext fromBuffer = MakeContext();
    ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&fromBuffer, entry.second.c_str(), entry.first.c_str()));

    const std::string path = WriteTempCsv(entry.second);
    ASSERT_FALSE(path.empty());
    CsvTunerContext fromFile = MakeContext();
    ASSERT_EQ(ncclSuccess, loadConfig(&fromFile, path.c_str()));

    ASSERT_EQ(fromBuffer.numConfigs, fromFile.numConfigs) << entry.first;
    ASSERT_GT(fromBuffer.numConfigs, 0) << entry.first << " parsed to nothing";
    for (int i = 0; i < fromBuffer.numConfigs; i++) {
      ExpectSameConfig(fromBuffer.configs[i], fromFile.configs[i],
                       entry.first + " row " + std::to_string(i));
    }

    FreeContext(&fromBuffer);
    FreeContext(&fromFile);
  }
}

TEST_F(CsvTunerMicrotest, ResolvesArchSpecificEmbeddedConfig) {
  const std::string arch = FirstEmbeddedArch();
  if (arch.empty()) GTEST_SKIP() << "no arch-specific config embedded";

  const char* source = rcclCsvTunerFindConfig(arch.c_str());
  ASSERT_NE(nullptr, source);
  EXPECT_EQ(std::string(RCCL_CSV_TUNER_EMBEDDED_PREFIX) + ArchCsvName(arch), std::string(source));
}

// Arch unknown is the one arm that deliberately hands back another
// architecture's tuning, so it needs a case of its own.
TEST_F(CsvTunerMicrotest, UnknownArchUsesFirstEmbeddedConfig) {
  const std::map<std::string, std::string>& embedded = rcclCsvTunerEmbeddedConfigs();
  ASSERT_FALSE(embedded.empty());
  const std::string expected = std::string(RCCL_CSV_TUNER_EMBEDDED_PREFIX) + embedded.begin()->first;

  const char* nullArch = rcclCsvTunerFindConfig(nullptr);
  ASSERT_NE(nullptr, nullArch);
  EXPECT_EQ(expected, std::string(nullArch));

  rcclCsvTunerResetConfigPath();
  const char* emptyArch = rcclCsvTunerFindConfig("");
  ASSERT_NE(nullptr, emptyArch);
  EXPECT_EQ(expected, std::string(emptyArch));
}

// A known arch with no entry must fall through to a generic rccl_tuner.csv or
// to no tuner -- never to another arch's tuning.
TEST_F(CsvTunerMicrotest, KnownArchWithNoEntryDoesNotBorrowAnotherArch) {
  const std::map<std::string, std::string>& embedded = rcclCsvTunerEmbeddedConfigs();
  ASSERT_FALSE(embedded.count("rccl_tuner_gfx000.csv"));

  const char* source = rcclCsvTunerFindConfig("gfx000");
  if (embedded.count("rccl_tuner.csv")) {
    ASSERT_NE(nullptr, source);
    EXPECT_STREQ(RCCL_CSV_TUNER_EMBEDDED_PREFIX "rccl_tuner.csv", source);
  } else {
    EXPECT_EQ(nullptr, source);
  }
}

TEST_F(CsvTunerMicrotest, ConfigFileEnvOverridesEmbedded) {
  const std::string path = WriteTempCsv(kTwoConfigCsv);
  ASSERT_FALSE(path.empty());
  SetMicroEnv("NCCL_TUNER_CONFIG_FILE", path.c_str());

  const char* source = rcclCsvTunerFindConfig("gfx950");
  ASSERT_NE(nullptr, source);
  EXPECT_STREQ(path.c_str(), source);
}

// The embedded map is last, not first: a CSV in the installed share directory
// still wins. Without this only the env-var step, which this PR did not touch,
// was covered.
TEST_F(CsvTunerMicrotest, SharePathCsvOverridesEmbedded) {
  const std::string arch = FirstEmbeddedArch();
  if (arch.empty()) GTEST_SKIP() << "no arch-specific config embedded";

  ASSERT_FALSE(emptyDir_.empty());
  MakeDirsUnder(emptyDir_, "share/rccl/tuner");
  const std::string path =
      WriteCsvAt(emptyDir_ + "/share/rccl/tuner/" + ArchCsvName(arch), kTwoConfigCsv);
  ASSERT_FALSE(path.empty());

  const char* source = rcclCsvTunerFindConfig(arch.c_str());
  ASSERT_NE(nullptr, source);
  EXPECT_STREQ(path.c_str(), source);
}

// Same directory, generic name, on an arch the embedded map does carry: a
// generic disk CSV still beats an arch-specific embedded entry. Using an arch
// with no embedded entry would pass even with the embedded map probed first.
TEST_F(CsvTunerMicrotest, GenericSharePathCsvOverridesEmbedded) {
  const std::string arch = FirstEmbeddedArch();
  if (arch.empty()) GTEST_SKIP() << "no arch-specific config embedded";

  ASSERT_FALSE(emptyDir_.empty());
  MakeDirsUnder(emptyDir_, "share/rccl/tuner");
  const std::string path = WriteCsvAt(emptyDir_ + "/share/rccl/tuner/rccl_tuner.csv", kTwoConfigCsv);
  ASSERT_FALSE(path.empty());

  const char* source = rcclCsvTunerFindConfig(arch.c_str());
  ASSERT_NE(nullptr, source);
  EXPECT_STREQ(path.c_str(), source);
}

TEST_F(CsvTunerMicrotest, EmbeddedDisabledByParam) {
  g_loadParam = [](const char* env, int64_t deftVal) -> int64_t {
    if (strcmp(env, "RCCL_TUNER_EMBEDDED_CONFIG") == 0) return 0;
    return deftVal;
  };

  EXPECT_EQ(nullptr, rcclCsvTunerFindConfig("gfx950"));
}

TEST_F(CsvTunerMicrotest, SkipsMalformedLines) {
  std::string csv = kTwoConfigCsv;
  csv += "allreduce,0,1023,,ll,4,1,8\n"; // empty field
  csv += "allreduce,0,1023,bogus,ll,4,1,8\n"; // unknown algorithm
  csv += "allreduce,0,1023,tree\n"; // too few fields
  csv += "garbage\n";

  CsvTunerContext ctx = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&ctx, csv.c_str(), "malformed"));
  EXPECT_EQ(2, ctx.numConfigs);
  FreeContext(&ctx);
}

// Buffer-backed parsing truncates over-long lines exactly like the fgets-based
// file path did, so an embedded config can never parse differently from a file.
// The tail after the 255-byte cut is a valid row, so discarding the remainder
// instead of resuming it drops the config and fails the case.
TEST_F(CsvTunerMicrotest, OverLongLineChunksLikeFgets) {
  const std::string overLong = "#" + std::string(RCCL_CSV_TUNER_MAX_LINE_LENGTH - 2, 'x');
  const std::string csv = overLong + "allreduce,0,1023,tree,ll,4,1,8,-1,-1\n";

  CsvTunerContext fromBuffer = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&fromBuffer, csv.c_str(), "overlong"));

  const std::string path = WriteTempCsv(csv);
  ASSERT_FALSE(path.empty());
  CsvTunerContext fromFile = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfig(&fromFile, path.c_str()));

  ASSERT_EQ(fromBuffer.numConfigs, fromFile.numConfigs);
  ASSERT_EQ(1, fromBuffer.numConfigs) << "the resumed remainder was not parsed";
  ExpectSameConfig(fromBuffer.configs[0], fromFile.configs[0], "overlong row 0");

  FreeContext(&fromBuffer);
  FreeContext(&fromFile);
}

// CRLF config text parses the same as LF. The row ends on a trailing comma so
// the stray CR would become its own token: without the strip, numPipeOps parses
// as atoi("\r") == 0 instead of the -1 that a missing field means.
TEST_F(CsvTunerMicrotest, ParsesCrlf) {
  CsvTunerContext lf = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&lf, "allreduce,0,1023,tree,ll,4,1,8,\n", "lf"));

  CsvTunerContext crlf = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&crlf, "allreduce,0,1023,tree,ll,4,1,8,\r\n", "crlf"));

  ASSERT_EQ(1, lf.numConfigs);
  ASSERT_EQ(1, crlf.numConfigs);
  EXPECT_EQ(-1, crlf.configs[0].numPipeOps) << "trailing CR leaked into numPipeOps";
  ExpectSameConfig(lf.configs[0], crlf.configs[0], "crlf row 0");

  FreeContext(&lf);
  FreeContext(&crlf);
}

// A buffer whose last line has no newline: the embedded text ends that way if
// the source CSV does.
TEST_F(CsvTunerMicrotest, ParsesBufferWithNoTrailingNewline) {
  CsvTunerContext ctx = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&ctx, "allreduce,0,1023,tree,ll,4,1,8,-1,-1", "no-newline"));

  ASSERT_EQ(1, ctx.numConfigs);
  EXPECT_EQ(4, ctx.configs[0].nChannels);
  FreeContext(&ctx);
}

// A config past the size cap is ignored rather than read into memory. Without
// the cap a non-regular path such as /dev/zero grows the allocation unbounded.
TEST_F(CsvTunerMicrotest, IgnoresOversizedConfigFile) {
  const std::string row = "allreduce,0,1023,tree,ll,4,1,8,-1,-1\n";
  std::string oversized;
  oversized.reserve(RCCL_CSV_TUNER_MAX_CONFIG_BYTES + row.size());
  while (oversized.size() <= RCCL_CSV_TUNER_MAX_CONFIG_BYTES) oversized += row;

  const std::string path = WriteTempCsv(oversized);
  ASSERT_FALSE(path.empty());

  CsvTunerContext ctx = MakeContext();
  EXPECT_EQ(ncclSuccess, loadConfig(&ctx, path.c_str()));
  EXPECT_EQ(0, ctx.numConfigs);
  FreeContext(&ctx);

  // Just under the cap still loads, so the cap is not rejecting everything.
  const std::string underCap = oversized.substr(0, RCCL_CSV_TUNER_MAX_CONFIG_BYTES - row.size());
  CsvTunerContext under = MakeContext();
  EXPECT_EQ(ncclSuccess, loadConfigFromBuffer(&under, underCap.c_str(), "under-cap"));
  EXPECT_GT(under.numConfigs, 0);
  FreeContext(&under);
}

// End to end: resolve the embedded source, init through the tuner entry point,
// and confirm the config lands in the cost table.
TEST_F(CsvTunerMicrotest, AppliesEmbeddedConfigToCostTable) {
  const std::string arch = FirstEmbeddedArch();
  if (arch.empty()) GTEST_SKIP() << "no arch-specific config embedded";

  ASSERT_NE(nullptr, rcclCsvTunerFindConfig(arch.c_str()));

  void* context = nullptr;
  // 1 node / 8 ranks matches the single-node rows of the shipped config.
  ASSERT_EQ(ncclSuccess, csvTunerInit(&context, /*commId=*/0, /*nRanks=*/8, /*nNodes=*/1,
                                      /*logFunction=*/nullptr, /*nvlDomainInfo=*/nullptr,
                                      /*constants=*/nullptr));
  ASSERT_NE(nullptr, context);

  CsvTunerContext* ctx = (CsvTunerContext*)context;
  ASSERT_GT(ctx->numConfigs, 0);

  // Drive the first matching config rather than hard-coding a size band, so this
  // stays valid when the shipped CSV is retuned. The predicate must mirror every
  // field csvTunerGetCollInfo gates on, including the numPipeOps and regBuff
  // passed to the probe below, or a retune leaves target matched here and
  // nothing matched there.
  const int kProbePipeOps = 1;
  const int kProbeRegBuff = 0;
  const CsvTuningConfig* target = nullptr;
  for (int i = 0; i < ctx->numConfigs; i++) {
    const CsvTuningConfig* c = &ctx->configs[i];
    if ((c->nNodes == -1 || c->nNodes == 1) && (c->nRanks == -1 || c->nRanks == 8) &&
        (c->numPipeOps == -1 || c->numPipeOps == kProbePipeOps) &&
        (c->regBuff == -1 || c->regBuff == kProbeRegBuff)) {
      target = c;
      break;
    }
  }
  ASSERT_NE(nullptr, target) << "shipped config has no row matching the probe";

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) table[a][p] = 1.0f;
  }

  int nChannels = -1;
  ASSERT_EQ(ncclSuccess, csvTunerGetCollInfo(context, target->collType, target->minBytes, kProbePipeOps,
                                             (float**)table, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                                             kProbeRegBuff, &nChannels));

  EXPECT_FLOAT_EQ(0.0f, table[target->algorithm][target->protocol]);
  EXPECT_EQ(target->nChannels >= 1 ? target->nChannels : 0, nChannels);

  EXPECT_EQ(ncclSuccess, csvTunerFinalize(context));
}

}  // namespace
