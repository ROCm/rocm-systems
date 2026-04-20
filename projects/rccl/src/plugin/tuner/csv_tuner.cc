/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Built-in CSV-based tuner for RCCL.
 *
 * This tuner reads tuning configurations from a CSV file and overrides
 * RCCL's default algorithm/protocol selection based on message size,
 * collective type, and topology.
 *
 * The CSV config file is searched in the following order:
 * 1. NCCL_TUNER_CONFIG_FILE environment variable
 * 2. <librccl.so dir>/tuner/rccl_tuner_<arch>.csv (for development builds)
 * 3. <librccl.so dir>/tuner/rccl_tuner.csv (for development builds)
 * 4. <librccl.so dir>/../share/rccl/tuner/rccl_tuner_<arch>.csv (installed RCCL)
 * 5. <librccl.so dir>/../share/rccl/tuner/rccl_tuner.csv (installed RCCL)
 * 6. ${ROCM_PATH}/share/rccl/tuner/rccl_tuner_<arch>.csv (fallback)
 * 7. ${ROCM_PATH}/share/rccl/tuner/rccl_tuner.csv (fallback)
 *
 * At each location, if gpuArch is unknown, the directory is scanned for any
 * rccl_tuner*.csv file. If no config file is found, the tuner is not activated.
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>
#include <dlfcn.h>
#include <link.h>
#include <dirent.h>

#include "checks.h"
#include "debug.h"
#include "tuner.h"
#include "nccl_tuner.h"

#define RCCL_CSV_TUNER_MAX_LINE_LENGTH 256

// CSV field indices for configuration parsing
// Format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
#define CONFIG_FIELD_COLLTYPE     0
#define CONFIG_FIELD_MINBYTES     1
#define CONFIG_FIELD_MAXBYTES     2
#define CONFIG_FIELD_ALGORITHM    3
#define CONFIG_FIELD_PROTOCOL     4
#define CONFIG_FIELD_CHANNELS     5
#define CONFIG_FIELD_NNODES       6
#define CONFIG_FIELD_NRANKS       7
#define CONFIG_FIELD_PIPEOPS      8  // Optional field
#define CONFIG_FIELD_REGBUFF      9  // Optional field

// Field count constants
#define CONFIG_FIELDS_REQUIRED    8   // Minimum required fields (up to nRanks)
#define CONFIG_FIELDS_WITH_PIPEOPS 9  // Fields including numPipeOps
#define CONFIG_FIELDS_WITH_REGBUFF 10 // Fields including both numPipeOps and regBuff
#define CONFIG_FIELDS_MAX         10  // Maximum number of fields supported

// Global state for CSV config file path discovery
static std::mutex csvTunerMutex;
static char csvTunerConfigPath[512] = {0};
static bool csvTunerConfigPathSet = false;

// Forward declaration of tuner symbol (defined at end of file)
extern ncclTuner_t rcclCsvTuner;

struct CsvTuningConfig {
  ncclFunc_t collType;
  size_t minBytes;
  size_t maxBytes;
  int algorithm;
  int protocol;
  int nChannels;
  int nNodes;
  int nRanks;
  int numPipeOps;
  int regBuff;
};

struct CsvTunerContext {
  CsvTuningConfig* configs;
  int numConfigs;
  int maxConfigs;
  size_t nRanks;
  size_t nNodes;
  ncclDebugLogger_t logFunction;
  ncclNvlDomainInfo_t nvlDomainInfo;
};

// Parse collective type from string; sets *valid=false and returns a placeholder if unknown
static ncclFunc_t parseCollType(const char* str, bool* valid) {
  *valid = true;
  if (strcmp(str, "broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "allgather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "reducescatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "allreduce") == 0) return ncclFuncAllReduce;
  *valid = false;
  return ncclFuncAllReduce; // placeholder, caller must check *valid
}

// Convert collective type to string (for logging)
static const char* collTypeToString(ncclFunc_t collType) {
  switch (collType) {
    case ncclFuncBroadcast: return "broadcast";
    case ncclFuncReduce: return "reduce";
    case ncclFuncAllGather: return "allgather";
    case ncclFuncReduceScatter: return "reducescatter";
    case ncclFuncAllReduce: return "allreduce";
    default: return "unknown";
  }
}

// Parse algorithm from string; sets *valid=false and returns a placeholder if unknown
static int parseAlgorithm(const char* str, bool* valid) {
  *valid = true;
  if (strcmp(str, "tree") == 0) return NCCL_ALGO_TREE;
  if (strcmp(str, "ring") == 0) return NCCL_ALGO_RING;
  if (strcmp(str, "collnet_direct") == 0) return NCCL_ALGO_COLLNET_DIRECT;
  if (strcmp(str, "collnet_chain") == 0) return NCCL_ALGO_COLLNET_CHAIN;
  if (strcmp(str, "nvls") == 0) return NCCL_ALGO_NVLS;
  if (strcmp(str, "nvls_tree") == 0) return NCCL_ALGO_NVLS_TREE;
  if (strcmp(str, "pat") == 0) return NCCL_ALGO_PAT;
  *valid = false;
  return NCCL_ALGO_RING; // placeholder, caller must check *valid
}

// Convert algorithm to string (for logging)
static const char* algorithmToString(int algorithm) {
  switch (algorithm) {
    case NCCL_ALGO_TREE: return "tree";
    case NCCL_ALGO_RING: return "ring";
    case NCCL_ALGO_COLLNET_DIRECT: return "collnet_direct";
    case NCCL_ALGO_COLLNET_CHAIN: return "collnet_chain";
    case NCCL_ALGO_NVLS: return "nvls";
    case NCCL_ALGO_NVLS_TREE: return "nvls_tree";
    case NCCL_ALGO_PAT: return "pat";
    default: return "unknown";
  }
}

// Parse protocol from string; sets *valid=false and returns a placeholder if unknown
static int parseProtocol(const char* str, bool* valid) {
  *valid = true;
  if (strcmp(str, "ll") == 0) return NCCL_PROTO_LL;
  if (strcmp(str, "ll128") == 0) return NCCL_PROTO_LL128;
  if (strcmp(str, "simple") == 0) return NCCL_PROTO_SIMPLE;
  *valid = false;
  return NCCL_PROTO_SIMPLE; // placeholder, caller must check *valid
}

// Convert protocol to string (for logging)
static const char* protocolToString(int protocol) {
  switch (protocol) {
    case NCCL_PROTO_LL: return "ll";
    case NCCL_PROTO_LL128: return "ll128";
    case NCCL_PROTO_SIMPLE: return "simple";
    default: return "unknown";
  }
}

// Helper function to count valid configuration lines in file
static int countConfigLines(const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    return 0;
  }

  char line[RCCL_CSV_TUNER_MAX_LINE_LENGTH];
  int count = 0;

  while (fgets(line, sizeof(line), file)) {
    const char* trimmed = line + strspn(line, " \t\r\n\v\f");

    // Skip comments and empty/whitespace-only lines
    if (*trimmed == '#' || *trimmed == '\0') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Check if line has content
    if (strlen(trimmed) > 0) {
      count++;
    }
  }

  fclose(file);
  return count;
}

// Load configuration from file
static ncclResult_t loadConfig(CsvTunerContext* ctx, const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/CsvTuner: Config file %s not found", filename);
    }
    return ncclSuccess; // Not finding config file is not an error
  }

  // First pass: count valid configuration lines
  int configCount = countConfigLines(filename);
  if (configCount == 0) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/CsvTuner: No valid configurations found in %s", filename);
    }
    fclose(file);
    return ncclSuccess;
  }

  // Allocate memory for configurations based on actual count
  ctx->configs = (CsvTuningConfig*)malloc(configCount * sizeof(CsvTuningConfig));
  if (!ctx->configs) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/CsvTuner: Failed to allocate memory for %d configurations", configCount);
    }
    fclose(file);
    return ncclSystemError;
  }

  ctx->maxConfigs = configCount;
  ctx->numConfigs = 0;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/CsvTuner: Allocated memory for %d configurations", configCount);
  }

  // Reset file pointer to beginning
  fseek(file, 0, SEEK_SET);

  char line[RCCL_CSV_TUNER_MAX_LINE_LENGTH];
  int lineNum = 0;

  while (fgets(line, sizeof(line), file) && ctx->numConfigs < ctx->maxConfigs) {
    lineNum++;

    // Skip comments and empty lines, allowing for leading whitespace
    char* trimmedLine = line;
    while (*trimmedLine == ' ' || *trimmedLine == '\t') trimmedLine++;
    if (trimmedLine[0] == '#' || trimmedLine[0] == '\n' || trimmedLine[0] == '\0') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Parse CSV format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
    char* token;
    char* tokens[CONFIG_FIELDS_MAX];
    int tokenCount = 0;
    bool lineValid = true;

    // Make a copy of the line for tokenizing
    char lineCopy[RCCL_CSV_TUNER_MAX_LINE_LENGTH];
    strncpy(lineCopy, line, sizeof(lineCopy));
    lineCopy[sizeof(lineCopy) - 1] = '\0';

    // Tokenize by comma
    token = strtok(lineCopy, ",");
    while (token != NULL && tokenCount < CONFIG_FIELDS_MAX) {
      // Trim leading whitespace
      while (*token == ' ' || *token == '\t') token++;
      // Trim trailing whitespace (safe for empty tokens)
      size_t tlen = strlen(token);
      if (tlen > 0) {
        char* end = token + tlen - 1;
        while (end > token && (*end == ' ' || *end == '\t')) {
          *end = '\0';
          end--;
        }
      }
      // Reject lines with empty fields after trimming
      if (*token == '\0') {
        if (ctx->logFunction) {
          ctx->logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/CsvTuner: Skipping malformed line %d with empty field: %s",
                           lineNum, line);
        }
        lineValid = false;
        break;
      }
      tokens[tokenCount++] = token;
      token = strtok(NULL, ",");
    }

    if (!lineValid) continue;

    // Validate field count: support required fields (8), with pipeOps (9), or with regBuff (10)
    if (tokenCount >= CONFIG_FIELDS_REQUIRED && tokenCount <= CONFIG_FIELDS_MAX) {
      bool collTypeValid, algoValid, protoValid;
      CsvTuningConfig* config = &ctx->configs[ctx->numConfigs];
      config->collType = parseCollType(tokens[CONFIG_FIELD_COLLTYPE], &collTypeValid);
      config->minBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MINBYTES], NULL, 10);
      config->maxBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MAXBYTES], NULL, 10);
      config->algorithm = parseAlgorithm(tokens[CONFIG_FIELD_ALGORITHM], &algoValid);
      config->protocol = parseProtocol(tokens[CONFIG_FIELD_PROTOCOL], &protoValid);
      config->nChannels = atoi(tokens[CONFIG_FIELD_CHANNELS]);
      config->nNodes = atoi(tokens[CONFIG_FIELD_NNODES]);
      config->nRanks = atoi(tokens[CONFIG_FIELD_NRANKS]);

      // Log and skip lines with unknown colltype, algorithm, or protocol
      if (!collTypeValid || !algoValid || !protoValid) {
        if (ctx->logFunction) {
          ctx->logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/CsvTuner: Skipping line %d with unknown %s%s%s value: %s",
                           lineNum,
                           !collTypeValid ? "colltype " : "",
                           !algoValid ? "algorithm " : "",
                           !protoValid ? "protocol" : "",
                           line);
        }
        continue;
      }

      // numPipeOps is optional (9th field, index 8)
      if (tokenCount >= CONFIG_FIELDS_WITH_PIPEOPS) {
        config->numPipeOps = atoi(tokens[CONFIG_FIELD_PIPEOPS]);
      } else {
        config->numPipeOps = -1; // -1 means match any numPipeOps
      }

      // regBuff is optional (10th field, index 9)
      if (tokenCount >= CONFIG_FIELDS_WITH_REGBUFF) {
        config->regBuff = atoi(tokens[CONFIG_FIELD_REGBUFF]);
      } else {
        config->regBuff = -1; // -1 means match any regBuff value
      }

      ctx->numConfigs++;

      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/CsvTuner: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d",
                         collTypeToString(config->collType), config->minBytes, config->maxBytes,
                         algorithmToString(config->algorithm), protocolToString(config->protocol),
                         config->nChannels, config->nNodes, config->nRanks);
      }
    }
  }

  fclose(file);
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/CsvTuner: Loaded %d tuning configurations from %s", ctx->numConfigs, filename);
  }
  return ncclSuccess;
}

// Check if a file exists
static bool fileExists(const char* path) {
  FILE* f = fopen(path, "r");
  if (f) {
    fclose(f);
    return true;
  }
  return false;
}

// Find first rccl_tuner*.csv file in directory (for when gpuArch is not available)
static bool findTunerFileInDir(const char* dirPath, char* outPath, size_t outPathSize) {
  DIR* dp = opendir(dirPath);
  if (!dp) return false;

  struct dirent* entry;
  while ((entry = readdir(dp))) {
    if (entry->d_type != DT_LNK && entry->d_type != DT_REG) continue;

    const char* name = entry->d_name;
    // Match rccl_tuner*.csv pattern
    if (strncmp(name, "rccl_tuner", 10) == 0) {
      size_t len = strlen(name);
      if (len > 4 && strcmp(name + len - 4, ".csv") == 0) {
        snprintf(outPath, outPathSize, "%s/%s", dirPath, name);
        closedir(dp);
        return true;
      }
    }
  }
  closedir(dp);
  return false;
}

// Get the directory containing librccl.so using dladdr1 (same pattern as MSCCL)
static bool getLibraryDirectory(std::string& libDir) {
  Dl_info dl_info;
  struct link_map *link_map_ptr = nullptr;
  if (!dladdr1((void *)&rcclCsvTuner, &dl_info, (void **)&link_map_ptr, RTLD_DL_LINKMAP)) {
    return false;
  }
  std::string selfLibPath = link_map_ptr->l_name;
  size_t lastSlash = selfLibPath.find_last_of("/\\");
  if (lastSlash != std::string::npos) {
    libDir = selfLibPath.substr(0, lastSlash);
    return true;
  }
  return false;
}

// Find CSV config file path - called before tuner init to determine if we should use CSV tuner
const char* rcclCsvTunerFindConfig(const char* gpuArch) {
  std::lock_guard<std::mutex> lock(csvTunerMutex);

  if (csvTunerConfigPathSet) {
    return csvTunerConfigPath[0] ? csvTunerConfigPath : nullptr;
  }

  csvTunerConfigPathSet = true;

  // 1. Check NCCL_TUNER_CONFIG_FILE environment variable (highest priority)
  const char* envConfig = getenv("NCCL_TUNER_CONFIG_FILE");
  if (envConfig && fileExists(envConfig)) {
    strncpy(csvTunerConfigPath, envConfig, sizeof(csvTunerConfigPath) - 1);
    csvTunerConfigPath[sizeof(csvTunerConfigPath) - 1] = '\0';
    return csvTunerConfigPath;
  }

  // 2. Check library directory (e.g., build/release/tuner/) - useful for development
  std::string libDir;
  if (getLibraryDirectory(libDir)) {
    std::string tunerDir = libDir + "/tuner";
    if (gpuArch && gpuArch[0]) {
      snprintf(csvTunerConfigPath, sizeof(csvTunerConfigPath),
               "%s/rccl_tuner_%s.csv", tunerDir.c_str(), gpuArch);
      if (fileExists(csvTunerConfigPath)) {
        return csvTunerConfigPath;
      }
    }
    snprintf(csvTunerConfigPath, sizeof(csvTunerConfigPath),
             "%s/rccl_tuner.csv", tunerDir.c_str());
    if (fileExists(csvTunerConfigPath)) {
      return csvTunerConfigPath;
    }
    // Only scan directory when gpuArch is unknown/empty (avoids nondeterministic cross-arch selection)
    if (!gpuArch || !gpuArch[0]) {
      if (findTunerFileInDir(tunerDir.c_str(), csvTunerConfigPath, sizeof(csvTunerConfigPath))) {
        return csvTunerConfigPath;
      }
    }

    // 3. Check relative share path: <libdir>/../share/rccl/tuner/ (for installed RCCL)
    std::string tunerShareDir = libDir + "/../share/rccl/tuner";
    if (gpuArch && gpuArch[0]) {
      snprintf(csvTunerConfigPath, sizeof(csvTunerConfigPath),
               "%s/rccl_tuner_%s.csv", tunerShareDir.c_str(), gpuArch);
      if (fileExists(csvTunerConfigPath)) {
        return csvTunerConfigPath;
      }
    }
    snprintf(csvTunerConfigPath, sizeof(csvTunerConfigPath),
             "%s/rccl_tuner.csv", tunerShareDir.c_str());
    if (fileExists(csvTunerConfigPath)) {
      return csvTunerConfigPath;
    }
    // Only scan directory when gpuArch is unknown/empty (avoids nondeterministic cross-arch selection)
    if (!gpuArch || !gpuArch[0]) {
      if (findTunerFileInDir(tunerShareDir.c_str(), csvTunerConfigPath, sizeof(csvTunerConfigPath))) {
        return csvTunerConfigPath;
      }
    }
  }

  // Get ROCM_PATH (default to /opt/rocm)
  const char* rocmPath = getenv("ROCM_PATH");
  if (!rocmPath) {
    rocmPath = "/opt/rocm";
  }

  // 4. Check GPU arch-specific config: ${ROCM_PATH}/share/rccl/tuner/rccl_tuner_<arch>.csv
  if (gpuArch && gpuArch[0]) {
    snprintf(csvTunerConfigPath, sizeof(csvTunerConfigPath),
             "%s/share/rccl/tuner/rccl_tuner_%s.csv", rocmPath, gpuArch);
    if (fileExists(csvTunerConfigPath)) {
      return csvTunerConfigPath;
    }
  }

  // 5. Check generic config: ${ROCM_PATH}/share/rccl/tuner/rccl_tuner.csv
  snprintf(csvTunerConfigPath, sizeof(csvTunerConfigPath),
           "%s/share/rccl/tuner/rccl_tuner.csv", rocmPath);
  if (fileExists(csvTunerConfigPath)) {
    return csvTunerConfigPath;
  }

  // 6. Only scan ROCM_PATH tuner directory when gpuArch is unknown/empty
  if (!gpuArch || !gpuArch[0]) {
    char tunerShareDir[512];
    snprintf(tunerShareDir, sizeof(tunerShareDir), "%s/share/rccl/tuner", rocmPath);
    if (findTunerFileInDir(tunerShareDir, csvTunerConfigPath, sizeof(csvTunerConfigPath))) {
      return csvTunerConfigPath;
    }
  }

  // No config file found
  csvTunerConfigPath[0] = '\0';
  return nullptr;
}

// Reset config path discovery (for testing)
void rcclCsvTunerResetConfigPath() {
  std::lock_guard<std::mutex> lock(csvTunerMutex);
  csvTunerConfigPath[0] = '\0';
  csvTunerConfigPathSet = false;
}

// Tuner init function
static ncclResult_t csvTunerInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                  ncclDebugLogger_t logFunction, ncclNvlDomainInfo_t* nvlDomainInfo,
                                  ncclTunerConstants_t* constants) {
  CsvTunerContext* ctx = (CsvTunerContext*)malloc(sizeof(CsvTunerContext));
  if (!ctx) return ncclSystemError;

  ctx->configs = nullptr;
  ctx->numConfigs = 0;
  ctx->maxConfigs = 0;
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  ctx->logFunction = logFunction;
  if (nvlDomainInfo) {
    ctx->nvlDomainInfo = *nvlDomainInfo;
  } else {
    memset(&ctx->nvlDomainInfo, 0, sizeof(ncclNvlDomainInfo_t));
  }

  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "TUNER/CsvTuner: Initializing built-in CSV tuner for %zu nodes, %zu ranks",
                nNodes, nRanks);
  }

  // Use the config path that was discovered earlier
  const char* configFile = nullptr;
  {
    std::lock_guard<std::mutex> lock(csvTunerMutex);
    if (csvTunerConfigPathSet && csvTunerConfigPath[0]) {
      configFile = csvTunerConfigPath;
    }
  }

  if (!configFile) {
    // Fallback: try environment variable or default
    configFile = getenv("NCCL_TUNER_CONFIG_FILE");
    if (!configFile) {
      configFile = "rccl_tuner.csv";
    }
  }

  ncclResult_t result = loadConfig(ctx, configFile);
  if (result != ncclSuccess) {
    if (ctx->configs) {
      free(ctx->configs);
    }
    free(ctx);
    return result;
  }

  *context = ctx;
  return ncclSuccess;
}

// Tuner getCollInfo function
static ncclResult_t csvTunerGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                                         int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                                         int regBuff, int* nChannels) {
  CsvTunerContext* ctx = (CsvTunerContext*)context;
  if (!ctx) return ncclInternalError;

  // Set default channels to 0 to ensure RCCL uses its default channel selection logic
  *nChannels = 0;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/CsvTuner: getCollInfo - collType=%s, nBytes=%zu, numPipeOps=%d, regBuff=%d",
                     collTypeToString(collType), nBytes, numPipeOps, regBuff);
  }

  // Cast the collCostTable pointer to a 2D array
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  // Look for matching configuration
  for (int i = 0; i < ctx->numConfigs; i++) {
    CsvTuningConfig* config = &ctx->configs[i];

    // Check if this config matches the current collective, size range, topology, pipeline ops, and regBuff
    if (config->collType == collType &&
        nBytes >= config->minBytes &&
        nBytes <= config->maxBytes &&
        (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes) &&
        (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks) &&
        (config->numPipeOps == -1 || config->numPipeOps == numPipeOps) &&
        (config->regBuff == -1 || config->regBuff == regBuff)) {

      // Check bounds
      if (config->algorithm < numAlgo && config->protocol < numProto) {
        if (table[config->algorithm][config->protocol] != NCCL_ALGO_PROTO_IGNORE) {
          table[config->algorithm][config->protocol] = 0.0; // Set low cost to prefer this configuration

          // Only override channels for valid explicit values; -1 keeps the default.
          if (config->nChannels >= 1) {
            *nChannels = config->nChannels;
          } else if (config->nChannels != -1) {
            if (ctx->logFunction) {
              ctx->logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/CsvTuner: Ignoring invalid channel count %d for %s, bytes=%zu",
                               config->nChannels, collTypeToString(config->collType), nBytes);
            }
          }

          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/CsvTuner: Applied config for %s, bytes=%zu: algo=%s, proto=%s, channels=%d",
                             collTypeToString(config->collType), nBytes,
                             algorithmToString(config->algorithm), protocolToString(config->protocol),
                             config->nChannels);
          }
          return ncclSuccess;
        }
      }
    }
  }

  // No matching config found
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/CsvTuner: No matching config for %s, bytes=%zu",
                     collTypeToString(collType), nBytes);
  }

  return ncclSuccess;
}

// Tuner finalize function
static ncclResult_t csvTunerFinalize(void* context) {
  if (context) {
    CsvTunerContext* ctx = (CsvTunerContext*)context;
    if (ctx->configs) {
      free(ctx->configs);
    }
    free(context);
  }
  return ncclSuccess;
}


ncclTuner_t rcclCsvTuner = {
  .name = "RcclCsvTuner",
  .init = csvTunerInit,
  .getCollInfo = csvTunerGetCollInfo,
  .finalize = csvTunerFinalize
};
