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
  const std::map<std::string, std::string>& embedded = rcclCsvTunerEmbeddedConfigs();
  if (!embedded.count("rccl_tuner_gfx950.csv")) GTEST_SKIP() << "no gfx950 config embedded";

  const char* source = rcclCsvTunerFindConfig("gfx950");
  ASSERT_NE(nullptr, source);
  EXPECT_STREQ(RCCL_CSV_TUNER_EMBEDDED_PREFIX "rccl_tuner_gfx950.csv", source);
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

TEST_F(CsvTunerMicrotest, EmbeddedDisabledByParam) {
  g_loadParam = [](const char* env, int64_t deftVal) -> int64_t {
    if (strcmp(env, "RCCL_TUNER_EMBEDDED_CONFIG") == 0) return 0;
    return deftVal;
  };

  EXPECT_EQ(nullptr, rcclCsvTunerFindConfig("gfx950"));
}

TEST_F(CsvTunerMicrotest, SkipsMalformedLines) {
  std::string csv = kTwoConfigCsv;
  csv += "allreduce,0,1023,,ll,4,1,8\n";       // empty field
  csv += "allreduce,0,1023,bogus,ll,4,1,8\n";  // unknown algorithm
  csv += "allreduce,0,1023,tree\n";            // too few fields
  csv += "garbage\n";

  CsvTunerContext ctx = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&ctx, csv.c_str(), "malformed"));
  EXPECT_EQ(2, ctx.numConfigs);
  FreeContext(&ctx);
}

// Buffer-backed parsing truncates over-long lines exactly like the fgets-based
// file path did, so an embedded config can never parse differently from a file.
TEST_F(CsvTunerMicrotest, OverLongLineChunksLikeFgets) {
  const std::string overLong(RCCL_CSV_TUNER_MAX_LINE_LENGTH + 64, 'x');
  const std::string csv = overLong + "\nallgather,1024,4095,ring,simple,8,-1,-1\n";

  CsvTunerContext fromBuffer = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&fromBuffer, csv.c_str(), "overlong"));

  const std::string path = WriteTempCsv(csv);
  ASSERT_FALSE(path.empty());
  CsvTunerContext fromFile = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfig(&fromFile, path.c_str()));

  ASSERT_EQ(fromBuffer.numConfigs, fromFile.numConfigs);
  EXPECT_EQ(1, fromBuffer.numConfigs);
  ExpectSameConfig(fromBuffer.configs[0], fromFile.configs[0], "overlong row 0");

  FreeContext(&fromBuffer);
  FreeContext(&fromFile);
}

// CRLF config text parses the same as LF.
TEST_F(CsvTunerMicrotest, ParsesCrlf) {
  CsvTunerContext lf = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&lf, "allreduce,0,1023,tree,ll,4,1,8,-1,-1\n", "lf"));

  CsvTunerContext crlf = MakeContext();
  ASSERT_EQ(ncclSuccess, loadConfigFromBuffer(&crlf, "allreduce,0,1023,tree,ll,4,1,8,-1,-1\r\n", "crlf"));

  ASSERT_EQ(1, lf.numConfigs);
  ASSERT_EQ(1, crlf.numConfigs);
  ExpectSameConfig(lf.configs[0], crlf.configs[0], "crlf row 0");

  FreeContext(&lf);
  FreeContext(&crlf);
}

// End to end: resolve the embedded source, init through the tuner entry point,
// and confirm the config lands in the cost table.
TEST_F(CsvTunerMicrotest, AppliesEmbeddedConfigToCostTable) {
  const std::map<std::string, std::string>& embedded = rcclCsvTunerEmbeddedConfigs();
  if (!embedded.count("rccl_tuner_gfx950.csv")) GTEST_SKIP() << "no gfx950 config embedded";

  ASSERT_NE(nullptr, rcclCsvTunerFindConfig("gfx950"));

  void* context = nullptr;
  // 1 node / 8 ranks matches the single-node rows of the shipped gfx950 config.
  ASSERT_EQ(ncclSuccess, csvTunerInit(&context, /*commId=*/0, /*nRanks=*/8, /*nNodes=*/1,
                                      /*logFunction=*/nullptr, /*nvlDomainInfo=*/nullptr,
                                      /*constants=*/nullptr));
  ASSERT_NE(nullptr, context);

  CsvTunerContext* ctx = (CsvTunerContext*)context;
  ASSERT_GT(ctx->numConfigs, 0);

  // Drive the first single-node config rather than hard-coding a size band, so
  // this stays valid when the shipped CSV is retuned.
  const CsvTuningConfig* target = nullptr;
  for (int i = 0; i < ctx->numConfigs; i++) {
    const CsvTuningConfig* c = &ctx->configs[i];
    if ((c->nNodes == -1 || c->nNodes == 1) && (c->nRanks == -1 || c->nRanks == 8)) {
      target = c;
      break;
    }
  }
  ASSERT_NE(nullptr, target) << "shipped gfx950 config has no 1-node/8-rank row";

  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) table[a][p] = 1.0f;
  }

  int nChannels = -1;
  ASSERT_EQ(ncclSuccess, csvTunerGetCollInfo(context, target->collType, target->minBytes,
                                             /*numPipeOps=*/1, (float**)table, NCCL_NUM_ALGORITHMS,
                                             NCCL_NUM_PROTOCOLS, /*regBuff=*/0, &nChannels));

  EXPECT_FLOAT_EQ(0.0f, table[target->algorithm][target->protocol]);
  EXPECT_EQ(target->nChannels >= 1 ? target->nChannels : 0, nChannels);

  EXPECT_EQ(ncclSuccess, csvTunerFinalize(context));
}

}  // namespace
