/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*
 * Library configuration sources, in increasing precedence:
 *   1. Built-in defaults (the inline initializers in reshard_internal.h).
 *   2. The ncclM2nConfig_t copied into the first ncclM2nHandle_t in
 *      a process-lifetime init/finalize epoch.
 *   3. Environment variables (always win when set; honors the
 *      "env-overrides-everything" convention used elsewhere in NCCL).
 *
 * resetReshardRuntimeConfig(), applyReshardConfig(), and applyReshardEnv() are
 * called during the first runtime initialization in each epoch, in that order.
 */

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "m2n_checks.h"
#include "m2n_env_parse.h"
#include "reshard_internal.h"
#include "m2n_log.h"
#include "reshard_types.h"

namespace {

bool parseAlgorithmEnv(const char* s, ReshardAlgorithm* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "AUTO") == 0) {
    *out = RESHARD_ALGO_AUTO;
    return true;
  }
  if (strcasecmp(s, "RING") == 0) {
    *out = RESHARD_ALGO_RING;
    return true;
  }
  if (strcasecmp(s, "DIRECT") == 0) {
    *out = RESHARD_ALGO_DIRECT;
    return true;
  }
  return false;
}

bool parseCopyAlgorithmEnv(const char* s, ReshardCopyAlgorithm* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "PIPE") == 0) {
    *out = RESHARD_COPY_ALGO_PIPE;
    return true;
  }
  if (strcasecmp(s, "DIRECT") == 0) {
    *out = RESHARD_COPY_ALGO_DIRECT;
    return true;
  }
  if (strcasecmp(s, "PACK") == 0) {
    *out = RESHARD_COPY_ALGO_PACK;
    return true;
  }
  return false;
}

bool parsePipeNetModeEnv(const char* s, ReshardPipeNetMode* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "DEVICE") == 0 || strcasecmp(s, "DEV") == 0 || strcasecmp(s, "KERNEL") == 0) {
    *out = RESHARD_PIPE_NET_DEVICE;
    return true;
  }
  if (strcasecmp(s, "HOST_RMA") == 0 || strcasecmp(s, "HOST-RMA") == 0 || strcasecmp(s, "HOSTRMA") == 0) {
    *out = RESHARD_PIPE_NET_HOST_RMA;
    return true;
  }
  return false;
}

bool parseLbModeEnv(const char* s, ReshardLoadBalanceMode* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "UNIFORM") == 0) {
    *out = RESHARD_LB_UNIFORM;
    return true;
  }
  if (strcasecmp(s, "NODE_AWARE") == 0) {
    *out = RESHARD_LB_NODE_AWARE;
    return true;
  }
  return false;
}

bool parseBoolEnv(const char* s, bool* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "1") == 0 || strcasecmp(s, "true") == 0 || strcasecmp(s, "yes") == 0 || strcasecmp(s, "on") == 0) {
    *out = true;
    return true;
  }
  if (strcasecmp(s, "0") == 0 || strcasecmp(s, "false") == 0 || strcasecmp(s, "no") == 0 || strcasecmp(s, "off") == 0) {
    *out = false;
    return true;
  }
  return false;
}

bool parseStagingBucketSize(const char* s, const char** endOut, size_t* sizeOut) {
  if (s == nullptr || endOut == nullptr || sizeOut == nullptr || !isM2nEnvDigit(*s)) return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long value = strtoull(s, &end, 10);
  if (errno == ERANGE || end == s || value == 0) return false;
  size_t multiplier = 1;
  if (*end == 'k' || *end == 'K') {
    multiplier = 1024;
    end++;
  } else if (*end == 'm' || *end == 'M') {
    multiplier = 1024 * 1024;
    end++;
  }
  if (value > std::numeric_limits<size_t>::max() / multiplier) return false;
  *sizeOut = static_cast<size_t>(value) * multiplier;
  *endOut = skipM2nEnvSpaces(end);
  return true;
}

} // namespace

void applyReshardConfig(const ncclM2nConfig_t* config) {
  if (config == nullptr) return;

  if (config->maxCta != NCCL_M2N_CONFIG_UNDEF_INT) {
    if (config->maxCta <= 0) RESHARD_WARN(-1, "ncclM2nInit: ignoring config.maxCta=%d (must be > 0).", config->maxCta);
    else gReshardMaxCta = config->maxCta;
  }
}

void resetReshardRuntimeConfig() {
  gReshardGpusPerNode = DEFAULT_GPUS_PER_NODE;
  gReshardSrcDomainSize = 0;
  gReshardDstDomainSize = 0;
  gReshardAlgorithm = RESHARD_ALGO_AUTO;
  gReshardLbMode = RESHARD_LB_UNIFORM;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PACK;
  gReshardPipeNetMode = RESHARD_PIPE_NET_DEVICE;
  gReshardStagingRuntimeConfig = {};
  gReshardAutoUniformBcast = true;
  gReshardAutoUniformBcastSet = false;
  gReshardSplitComm = true;
  gReshardSplitCommSet = false;
  gReshardSplitSingleRepInject = false;
  gReshardSplitSingleRepInjectSet = false;
  gReshardSplitAutoParentThreshold = 200;
  gReshardCommBForceRail = false;
  gReshardLbModeSet = false;
  gReshardAdaptiveCallConfig = {};
  gReshardAdaptiveCallConfigValid = false;
  gReshardMaxCta = 0;
  gReshardNumCtasOverride = 0;
  gReshardNumCtas = DEFAULT_NUM_CTAS;
  gReshardElementsPerChunk = DEFAULT_ELEMENTS_PER_CHUNK;
  gReshardGinContextCount = DEFAULT_GIN_CONTEXT_COUNT;
  gReshardPipeGinPeersPerSlot = STAGING_PIPE_GIN_PEERS_PER_SLOT;
  gReshardPipeGinChannelsPerPeer = STAGING_PIPE_GIN_CHANNELS_PER_PEER;
  gReshardPipeLaunchOrder = true;
  gReshardCommBConcurrency = kDefaultStagingBucketSlots;
  gReshardPipeCommBParentCapacity = RESHARD_PIPE_DEFAULT_PARENT_CAPACITY;
  gReshardUseInternalStreams = true;
  gReshardDevCommCacheCapacity = MAX_DEVCOMM_CACHE_ENTRIES;
  gReshardChunkSizeBytes = 0;
  for (int i = 0; i < kMaxStagingBuckets; i++) {
    gReshardStagingBuckets[i] = {};
  }
  gReshardStagingBuckets[0] = {kDefaultStagingBucketBytes, kDefaultStagingBucketSlots};
  gReshardStagingBucketCount = 1;
  gReshardStagingBucketsImplicitDefault = true;
}

void resolveReshardStagingRuntimeConfig() {
  const bool hostRma = gReshardCopyAlgorithm == RESHARD_COPY_ALGO_PIPE &&
                       gReshardPipeNetMode == RESHARD_PIPE_NET_HOST_RMA;
  ReshardStagingRuntimeConfig config = {};
  if (hostRma) {
    config.numChannels = STAGING_PIPE_HOST_RMA_DEFAULT_NUM_CHANNELS;
    config.channelDataSize = STAGING_PIPE_HOST_RMA_DEFAULT_CHANNEL_DATA_SIZE;
    config.chunkSize = STAGING_PIPE_HOST_RMA_DEFAULT_CHUNK_SIZE;
    config.hostRmaDefault = true;
  }

  int value = 0;
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_STAGING_NUM_CHANNELS"), &value)) {
    config.numChannels = value;
    config.numChannelsExplicit = true;
    config.numChannelsFixed = true;
  }
  size_t sizeValue = 0;
  if (parseM2nEnvSize(getenv("NCCL_RESHARD_STAGING_CHANNEL_SIZE"), &sizeValue, false)) {
    config.channelDataSize = sizeValue;
    config.channelDataSizeExplicit = true;
  }
  if (parseM2nEnvSize(getenv("NCCL_RESHARD_STAGING_CHUNK_SIZE"), &sizeValue, false)) {
    config.chunkSize = sizeValue;
  }
  if (parseM2nNonNegativeEnvInt(getenv("NCCL_RESHARD_STAGING_PEERS_PER_CHANNEL"), &value)) {
    config.peersPerChannel = value;
  }
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_STAGING_TARGET_CTAS"), &value)) {
    config.targetCtas = value;
    config.targetCtasExplicit = true;
  }

  gReshardStagingRuntimeConfig = config;
}

ncclResult_t validateReshardConfigHeader(const ncclM2nConfig_t* config) {
  if (config == nullptr) return ncclSuccess;

  if (config->size != sizeof(ncclM2nConfig_t) || config->magic != NCCL_M2N_API_MAGIC) {
    NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                  "NCCL M2N config: rejecting malformed ncclM2nConfig_t "
                  "(size=%zu, magic=0x%x, version=%u). Use NCCL_M2N_CONFIG_INITIALIZER.",
                  config->size, config->magic, config->version);
  }

  if (config->version != NCCL_M2N_API_VERSION) {
    RESHARD_WARN(-1,
                 "ncclM2nInit: ncclM2nConfig_t.version=%u, library API_VERSION=%u; "
                 "behavior may differ across versions.",
                 config->version, NCCL_M2N_API_VERSION);
  }

  return ncclSuccess;
}

ncclResult_t resolveReshardDomainSizes(int worldRank, ReshardAlgorithm algo, int srcLsaSize, int dstLsaSize,
                                       int* srcGpusPerDomain, int* dstGpusPerDomain) {
  NCCL_M2N_CHECK_ARG(srcGpusPerDomain != nullptr && dstGpusPerDomain != nullptr, worldRank,
                     "resolveReshardDomainSizes: output pointers must be non-null");

  const int srcOverride = reshardGetSrcDomainSize();
  const int dstOverride = reshardGetDstDomainSize();
  if (algo == RESHARD_ALGO_RING) {
    NCCL_M2N_CHECK_ARG(dstOverride <= 0 || dstLsaSize <= 0 || dstOverride <= dstLsaSize, worldRank,
                       "NCCL_RESHARD_DST_DOMAIN_SIZE=%d exceeds the NCCL DevComm LSA team size %d", dstOverride,
                       dstLsaSize);
  }

  const int gpusPerNode = reshardGetGpusPerNode();
  *srcGpusPerDomain =
    srcOverride > 0 ? srcOverride : (srcLsaSize > 0 ? srcLsaSize : (gpusPerNode > 0 ? gpusPerNode : 1));
  *dstGpusPerDomain =
    dstOverride > 0 ? dstOverride : (dstLsaSize > 0 ? dstLsaSize : (gpusPerNode > 0 ? gpusPerNode : 1));
  return ncclSuccess;
}

// `getenv` is the only POSIX path to read process env vars; there is no portable
// thread-safe alternative (`secure_getenv` is glibc-only). Library init runs
// once on the first calling thread before ncclM2nInit returns, so concurrent
// env mutation by user code is the caller's problem — scope the
// concurrency-mt-unsafe suppression to just this function.
//
// NOLINTBEGIN(concurrency-mt-unsafe)
void applyReshardEnv() {
  ReshardLogLevel lvl;
  if (reshardLogLevelFromStr(getenv("NCCL_RESHARD_LOG_LEVEL"), &lvl)) reshardSetLogLevel(lvl);

  const char* cacheCapacity = getenv("NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY");
  if (cacheCapacity != nullptr && !parseM2nPositiveEnvInt(cacheCapacity, &gReshardDevCommCacheCapacity)) {
    RESHARD_WARN(-1, "Ignoring invalid NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY=\"%s\"; expected a positive int",
                   cacheCapacity);
  }

  ReshardAlgorithm algo;
  if (parseAlgorithmEnv(getenv("NCCL_RESHARD_ALGORITHM"), &algo)) gReshardAlgorithm = algo;

  const char* copyAlgorithm = getenv("NCCL_RESHARD_COPY_ALGORITHM");
  ReshardCopyAlgorithm copyAlgo;
  if (copyAlgorithm != nullptr) {
    if (parseCopyAlgorithmEnv(copyAlgorithm, &copyAlgo)) {
      gReshardCopyAlgorithm = copyAlgo;
    } else {
      RESHARD_WARN(-1,
                   "NCCL_RESHARD_COPY_ALGORITHM=\"%s\" is not supported; accepted values are DIRECT, PIPE, and PACK",
                   copyAlgorithm);
    }
  }

  ReshardPipeNetMode pipeNetMode;
  if (parsePipeNetModeEnv(getenv("NCCL_RESHARD_PIPE_NET_MODE"), &pipeNetMode)) {
    gReshardPipeNetMode = pipeNetMode;
  }

  ReshardLoadBalanceMode lb;
  if (parseLbModeEnv(getenv("NCCL_RESHARD_LB_MODE"), &lb)) {
    gReshardLbMode = lb;
    gReshardLbModeSet = true;
  }

  int n;
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_NUM_CTAS"), &n)) {
    gReshardNumCtasOverride = n;
    gReshardNumCtas = n;
  }
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_GIN_CONTEXT_COUNT"), &n)) {
    gReshardGinContextCount = n;
  }
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_PIPE_GIN_PEERS_PER_SLOT"), &n)) {
    gReshardPipeGinPeersPerSlot = n;
  }
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_PIPE_GIN_CHANNELS_PER_PEER"), &n)) {
    gReshardPipeGinChannelsPerPeer = n;
  }
  // Zero means no explicit PACK cap; PIPE keeps its default of four lanes.
  int genCommConcurrency = 0;
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_GEN_COMM_CONCURRENCY"), &n)) {
    if (n <= MAX_SPLIT_CONCURRENCY) {
      gReshardCommBConcurrency = n;
      genCommConcurrency = n;
    } else {
      RESHARD_WARN(-1, "ignoring NCCL_RESHARD_GEN_COMM_CONCURRENCY=%d; maximum supported value is %d", n,
                   MAX_SPLIT_CONCURRENCY);
    }
  }

  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_GEN_COMM_PARENT_CAPACITY"), &n)) {
    if (n <= RESHARD_PIPE_MAX_PARENT_NAMESPACES) {
      gReshardPipeCommBParentCapacity = n;
    } else {
      RESHARD_WARN(-1, "ignoring NCCL_RESHARD_GEN_COMM_PARENT_CAPACITY=%d; maximum supported value is %d", n,
                   RESHARD_PIPE_MAX_PARENT_NAMESPACES);
    }
  }

  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_SRC_DOMAIN_SIZE"), &n)) gReshardSrcDomainSize = n;
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_DST_DOMAIN_SIZE"), &n)) gReshardDstDomainSize = n;

  size_t sizeValue;
  if (parseM2nEnvSize(getenv("NCCL_RESHARD_ELEMENTS_PER_CHUNK"), &sizeValue, false)) {
    gReshardElementsPerChunk = sizeValue;
  }

  /* Cache chunk-size override so prepareReshardParams doesn't touch
   * getenv on the hot path. */
  if (parseM2nEnvSize(getenv("NCCL_RESHARD_CHUNK_SIZE"), &sizeValue, false)) {
    gReshardChunkSizeBytes = sizeValue;
  }

  bool useInternalStreams;
  if (parseBoolEnv(getenv("NCCL_RESHARD_USE_INTERNAL_STREAMS"), &useInternalStreams)) {
    gReshardUseInternalStreams = useInternalStreams;
  }

  bool pipeLaunchOrder;
  if (parseBoolEnv(getenv("NCCL_RESHARD_PIPE_LAUNCH_ORDER"), &pipeLaunchOrder)) {
    gReshardPipeLaunchOrder = pipeLaunchOrder;
  }

  bool splitComm;
  if (parseBoolEnv(getenv("NCCL_RESHARD_SPLIT_COMM"), &splitComm)) {
    gReshardSplitComm = splitComm;
    gReshardSplitCommSet = true;
  }

  bool singleRepInject;
  if (parseBoolEnv(getenv("NCCL_RESHARD_SPLIT_SINGLE_REP_INJECT"), &singleRepInject)) {
    gReshardSplitSingleRepInject = singleRepInject;
    gReshardSplitSingleRepInjectSet = true;
  }

  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_SPLIT_AUTO_PARENT_THRESHOLD"), &n)) {
    gReshardSplitAutoParentThreshold = n;
  }

  bool commBForceRail;
  if (parseBoolEnv(getenv("NCCL_RESHARD_COMMB_FORCE_RAIL"), &commBForceRail)) {
    gReshardCommBForceRail = commBForceRail;
  }

  bool autoUniformBcast;
  if (parseBoolEnv(getenv("NCCL_RESHARD_AUTO_UNIFORM_BCAST"), &autoUniformBcast)) {
    gReshardAutoUniformBcast = autoUniformBcast;
    gReshardAutoUniformBcastSet = true;
  }

  /* Bucketed staging pool: "size[:slots],size[:slots],...". Bare sizes use
   * one slot; sizes accept bytes or binary K/M suffixes.
   * Unset keeps the built-in 2-GiB:4 profile. Invalid explicit values also
   * retain that default so PACK always has at least one bucket. Parsed
   * buckets are kept ascending by size so best-fit is a forward scan. */
  const char* buckets = getenv("NCCL_RESHARD_PACK_BUFFSIZES");
  if (buckets == nullptr) {
    RESHARD_INFO(-1, "staging-bucket pool enabled: %d buckets, %d total slots (implicit default)",
                 gReshardStagingBucketCount, reshardStagingTotalSlots());
  } else {
    const char* value = skipM2nEnvSpaces(buckets);
    ReshardStagingBucketCfg parsed[kMaxStagingBuckets] = {};
    int parsedCount = 0;
    bool valid = *value != '\0';
    int totalSlots = 0;
    const char* p = value;
    while (valid && *p != '\0') {
      if (parsedCount >= kMaxStagingBuckets) {
        valid = false;
        break;
      }
      p = skipM2nEnvSpaces(p);
      if (!isM2nEnvDigit(*p)) {
        valid = false;
        break;
      }
      size_t size = 0;
      const char* sizeEnd = nullptr;
      if (!parseStagingBucketSize(p, &sizeEnd, &size)) {
        valid = false;
        break;
      }
      long slots = 1;
      const char* itemEnd = sizeEnd;
      if (*sizeEnd == ':') {
        p = skipM2nEnvSpaces(sizeEnd + 1);
        if (!isM2nEnvDigit(*p)) {
          valid = false;
          break;
        }
        char* slotsEnd = nullptr;
        errno = 0;
        slots = strtol(p, &slotsEnd, 10);
        if (errno == ERANGE || slotsEnd == p) {
          valid = false;
          break;
        }
        itemEnd = skipM2nEnvSpaces(slotsEnd);
      }
      if (slots <= 0 || slots > MAX_SPLIT_CONCURRENCY ||
          totalSlots > MAX_SPLIT_CONCURRENCY - slots) {
        valid = false;
        break;
      }
      parsed[parsedCount++] = {size, static_cast<int>(slots)};
      totalSlots += static_cast<int>(slots);
      p = itemEnd;
      if (*p == ',') {
        p++;
        if (*p == '\0') valid = false;
      } else if (*p != '\0') {
        valid = false;
        break;
      }
    }
    for (int i = 1; i < parsedCount; i++) {
      ReshardStagingBucketCfg key = parsed[i];
      int j = i - 1;
      while (j >= 0 && parsed[j].size > key.size) {
        parsed[j + 1] = parsed[j];
        j--;
      }
      parsed[j + 1] = key;
    }
    for (int i = 1; i < parsedCount; i++) {
      if (parsed[i - 1].size == parsed[i].size) {
        valid = false;
        break;
      }
    }
    /* Every PACK request reserves at least 2 KiB. A profile below that
     * ceiling can never service a call, so reject it at configuration time. */
    if (parsedCount == 0 || parsed[parsedCount - 1].size < kMinPackStagingBytes) valid = false;
    if (valid) {
      for (int i = 0; i < kMaxStagingBuckets; i++) {
        gReshardStagingBuckets[i] = i < parsedCount ? parsed[i] : ReshardStagingBucketCfg{};
      }
      gReshardStagingBucketCount = parsedCount;
      gReshardStagingBucketsImplicitDefault = false;
      RESHARD_INFO(-1, "staging-bucket pool enabled: %d buckets, %d total slots", gReshardStagingBucketCount,
                   reshardStagingTotalSlots());
    } else {
      for (int i = 0; i < kMaxStagingBuckets; i++) gReshardStagingBuckets[i] = {};
      gReshardStagingBuckets[0] = {kDefaultStagingBucketBytes, kDefaultStagingBucketSlots};
      gReshardStagingBucketCount = 1;
      gReshardStagingBucketsImplicitDefault = true;
      RESHARD_WARN(-1, "ignoring invalid NCCL_RESHARD_PACK_BUFFSIZES=\"%s\"; using built-in default %zu:%d",
                   buckets, kDefaultStagingBucketBytes, kDefaultStagingBucketSlots);
    }
  }
  // Both physical-slot assignment and commB sizing read this resolved profile.
  // Cap it before first pool use instead of aliasing distinct physical slots
  // onto fewer resource lanes, which would bypass their existing event order.
  if (gReshardCopyAlgorithm == RESHARD_COPY_ALGO_PACK && genCommConcurrency > 0) {
    for (int i = 0; i < gReshardStagingBucketCount; i++) {
      auto& bucket = gReshardStagingBuckets[i];
      if (bucket.numSlots > genCommConcurrency) bucket.numSlots = genCommConcurrency;
    }
    RESHARD_INFO(-1, "PACK generator concurrency cap=%d: %d active staging slots", genCommConcurrency,
                   reshardStagingTotalSlots());
  }
  if (gReshardCopyAlgorithm == RESHARD_COPY_ALGO_PACK && gReshardStagingBucketCount > 1) {
    RESHARD_WARN(-1, "PACK split comm is disabled for multiple size buckets; using the parent DevComm path");
  }
  resolveReshardStagingRuntimeConfig();
}
// NOLINTEND(concurrency-mt-unsafe)
