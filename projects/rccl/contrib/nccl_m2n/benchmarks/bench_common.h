/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_BENCH_COMMON_H_
#define NCCL_M2N_BENCH_COMMON_H_

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

// ============================================================================
// CLI parsing helpers
// ============================================================================

// Pre-MPI_Init env propagation (CLI → library init env).
static inline void benchSetEnv(const char* name, const char* value) {
  setenv(name, value, 1);
}

enum class BenchParseResult {
  Success,
  Help,
  Error,
};

enum class ReshardApiMode {
  Window,
  Default,
};

static inline void benchConfigureCopyAlgorithm(const char* copyAlgorithm) {
  if (copyAlgorithm != nullptr) {
    benchSetEnv("NCCL_RESHARD_COPY_ALGORITHM", copyAlgorithm);
  }
}

static inline const char* benchResolvedCopyAlgorithm() {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — benchmark configuration is resolved before worker threads start.
  const char* copyAlgorithm = getenv("NCCL_RESHARD_COPY_ALGORITHM");
  return copyAlgorithm != nullptr ? copyAlgorithm : "PACK";
}

static inline bool benchArgIs(const char* value, const char* expected) {
  return value != nullptr && strcmp(value, expected) == 0;
}

// ============================================================================
// Error Checking Macros
// ============================================================================

#define MPICHECK(cmd)                                                           \
  do {                                                                          \
    int e = cmd;                                                                \
    if (e != MPI_SUCCESS) {                                                     \
      fprintf(stderr, "Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      abort();                                                                  \
    }                                                                           \
  } while (0)

#define CUDACHECK(cmd)                                                                               \
  do {                                                                                               \
    cudaError_t e = cmd;                                                                             \
    if (e != cudaSuccess) {                                                                          \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      abort();                                                                                       \
    }                                                                                                \
  } while (0)

#define NCCLCHECK(cmd)                                                                               \
  do {                                                                                               \
    ncclResult_t r = cmd;                                                                            \
    if (r != ncclSuccess && r != ncclInProgress) {                                                   \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      abort();                                                                                       \
    }                                                                                                \
  } while (0)

// Use public NCCL APIs to finish communicator work before consuming its outputs.
static inline ncclResult_t benchWaitCommReady(ncclComm_t comm) {
  ncclResult_t state = ncclInProgress;
  while (state == ncclInProgress) {
    ncclResult_t result = ncclCommGetAsyncError(comm, &state);
    if (result != ncclSuccess) return result;
    if (state == ncclInProgress) std::this_thread::yield();
  }
  return state;
}

// ============================================================================
// Argument Parsing Helpers
// ============================================================================

static inline bool benchParseInt(const char* value, int* out) {
  if (value == nullptr || out == nullptr) return false;
  char* end = nullptr;
  errno = 0;
  const long parsed = strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) return false;
  *out = static_cast<int>(parsed);
  return true;
}

static inline bool benchParseSize(const char* value, size_t* out) {
  if (value == nullptr || out == nullptr) return false;
  char* end = nullptr;
  errno = 0;
  const double parsed = strtod(value, &end);
  if (errno != 0 || end == value || !std::isfinite(parsed) || parsed < 0.0) return false;

  double multiplier = 1.0;
  if (*end != '\0') {
    if (end[1] != '\0') return false;
    switch (*end) {
    case 'k':
    case 'K':
      multiplier = 1024.0;
      break;
    case 'm':
    case 'M':
      multiplier = 1024.0 * 1024.0;
      break;
    case 'g':
    case 'G':
      multiplier = 1024.0 * 1024.0 * 1024.0;
      break;
    default:
      return false;
    }
  }

  const long double scaled = static_cast<long double>(parsed) * multiplier;
  long double integral = 0.0L;
  if (std::modf(scaled, &integral) != 0.0L ||
      integral > static_cast<long double>(std::numeric_limits<size_t>::max())) {
    return false;
  }
  *out = static_cast<size_t>(integral);
  return true;
}

static inline bool benchParseNonNegativeIntList(const char* value, std::vector<int>* out) {
  if (value == nullptr || out == nullptr || *value == '\0') return false;
  std::vector<int> parsed;
  const std::string input(value);
  size_t begin = 0;
  while (begin < input.size()) {
    const size_t end = input.find(',', begin);
    const std::string item = input.substr(begin, end - begin);
    int parsedItem = 0;
    if (item.empty() || !benchParseInt(item.c_str(), &parsedItem) || parsedItem < 0) return false;
    parsed.push_back(parsedItem);
    if (end == std::string::npos) {
      *out = std::move(parsed);
      return true;
    }
    begin = end + 1;
  }
  return false;
}

static inline bool benchParseMeshDims(const char* value, int dims[2]) {
  if (value == nullptr || dims == nullptr) return false;
  const char* split = strpbrk(value, ",x");
  if (split == nullptr || split == value || split[1] == '\0' || strpbrk(split + 1, ",x") != nullptr) return false;

  int parsed[2];
  const std::string first(value, split);
  if (!benchParseInt(first.c_str(), &parsed[0]) || !benchParseInt(split + 1, &parsed[1])) return false;
  dims[0] = parsed[0];
  dims[1] = parsed[1];
  return true;
}

static inline bool benchParseTensorDims(const char* value, size_t dims[3], int* nDims) {
  if (value == nullptr || dims == nullptr || nDims == nullptr || *value == '\0') return false;

  size_t parsed[3];
  int count = 0;
  const char* token = value;
  while (true) {
    const char* split = strpbrk(token, ",x");
    const char* end = split == nullptr ? token + strlen(token) : split;
    if (count == 3 || end == token) return false;
    const std::string item(token, end);
    if (!benchParseSize(item.c_str(), &parsed[count])) return false;
    count++;
    if (split == nullptr) break;
    token = split + 1;
  }

  std::copy(parsed, parsed + count, dims);
  *nDims = count;
  return true;
}

struct BenchOptionSpec {
  const char* name;
  bool bRequiresValue;
  std::function<BenchParseResult(const char*)> handler;
};

// Option names match exactly, so `--opt=value` is not accepted. Repeated
// options are processed in order and the last value wins. Diagnostics are
// emitted by rank 0 only.
class BenchArgParser {
 public:
  BenchArgParser(int argc, char* argv[], int mpiRank) : argc_(argc), argv_(argv), mpiRank_(mpiRank) {}

  BenchArgParser& value(const char* name, std::function<BenchParseResult(const char*)> handler) {
    options_.push_back({name, true, std::move(handler)});
    return *this;
  }

  BenchArgParser& flag(const char* name, const std::function<void()>& handler) {
    options_.push_back({name, false, [handler](const char*) {
                          handler();
                          return BenchParseResult::Success;
                        }});
    return *this;
  }

  BenchArgParser& help(const char* name, const std::function<void()>& handler) {
    options_.push_back({name, false, [handler](const char*) {
                          handler();
                          return BenchParseResult::Help;
                        }});
    return *this;
  }

  BenchArgParser& help(const std::function<void(const char*)>& printUsage) {
    const char* prog = argv_[0];
    const int mpiRank = mpiRank_;
    options_.push_back({"--help", false, [printUsage, prog, mpiRank](const char*) {
                          if (mpiRank == 0) {
                            printUsage(prog);
                          }
                          return BenchParseResult::Help;
                        }});
    options_.push_back({"-h", false, [printUsage, prog, mpiRank](const char*) {
                          if (mpiRank == 0) {
                            printUsage(prog);
                          }
                          return BenchParseResult::Help;
                        }});
    return *this;
  }

  BenchArgParser& integer(const char* name, int* out) {
    return integer(name, name, out);
  }

  BenchArgParser& integer(const char* name, const char* what, int* out) {
    const int mpiRank = mpiRank_;
    return value(name, [what, out, mpiRank](const char* value) {
      if (!benchParseInt(value, out)) {
        if (mpiRank == 0) {
          printf("[bench] %s: invalid integer '%s'\n", what, value);
        }
        return BenchParseResult::Error;
      }
      return BenchParseResult::Success;
    });
  }

  BenchArgParser& meshDims(const char* name, int dims[2]) {
    const int mpiRank = mpiRank_;
    return value(name, [name, dims, mpiRank](const char* value) {
      if (!benchParseMeshDims(value, dims)) {
        if (mpiRank == 0) {
          printf("ERROR: invalid %s value '%s' (expected d0,d1)\n", name, value);
        }
        return BenchParseResult::Error;
      }
      return BenchParseResult::Success;
    });
  }

  BenchArgParser& tensorDims(const char* name, size_t dims[3], int* nDims) {
    const int mpiRank = mpiRank_;
    return value(name, [name, dims, nDims, mpiRank](const char* value) {
      if (!benchParseTensorDims(value, dims, nDims)) {
        if (mpiRank == 0) {
          printf("ERROR: invalid %s value '%s' (expected d0[,d1[,d2]])\n", name, value);
        }
        return BenchParseResult::Error;
      }
      return BenchParseResult::Success;
    });
  }

  BenchArgParser& apiMode(const char* name, ReshardApiMode* out) {
    const int mpiRank = mpiRank_;
    return value(name, [name, out, mpiRank](const char* value) {
      if (benchArgIs(value, "window")) {
        *out = ReshardApiMode::Window;
        return BenchParseResult::Success;
      }
      if (benchArgIs(value, "default")) {
        *out = ReshardApiMode::Default;
        return BenchParseResult::Success;
      }
      if (mpiRank == 0) {
        printf("ERROR: unknown %s value '%s' (use 'window' or 'default')\n", name, value);
      }
      return BenchParseResult::Error;
    });
  }

  // Map an enum-style option to a canonical string. `mapping` pairs each accepted
  // token with the value stored into *out; `errorFmt` is a printf format with a
  // single %s for the rejected token (kept per-call so each benchmark's existing
  // error wording is preserved verbatim).
  BenchArgParser& enumValue(const char* name, const char** out,
      std::initializer_list<std::pair<const char*, const char*>> mapping, const char* errorFmt) {
    std::vector<std::pair<const char*, const char*>> table(mapping);
    const int mpiRank = mpiRank_;
    return value(name, [out, table, mpiRank, errorFmt](const char* value) {
      for (const std::pair<const char*, const char*>& entry : table) {
        if (benchArgIs(value, entry.first)) {
          *out = entry.second;
          return BenchParseResult::Success;
        }
      }
      if (mpiRank == 0) {
        printf(errorFmt, value); // NOLINT(clang-diagnostic-format-nonliteral)
      }
      return BenchParseResult::Error;
    });
  }

  BenchParseResult parse() const {
    for (int i = 1; i < argc_; i++) {
      const BenchOptionSpec* option = nullptr;
      for (const BenchOptionSpec& candidate : options_) {
        if (strcmp(argv_[i], candidate.name) == 0) {
          option = &candidate;
          break;
        }
      }
      if (option == nullptr) {
        if (mpiRank_ == 0) {
          printf("ERROR: unknown option '%s'\n", argv_[i]);
        }
        return BenchParseResult::Error;
      }

      const char* value = nullptr;
      if (option->bRequiresValue) {
        value = (i + 1 < argc_) ? argv_[i + 1] : nullptr;
        if (value == nullptr || strncmp(value, "--", 2) == 0) {
          if (mpiRank_ == 0) {
            printf("ERROR: %s requires a value\n", option->name);
          }
          return BenchParseResult::Error;
        }
        ++i;
      }

      BenchParseResult result = option->handler(value);
      if (result != BenchParseResult::Success) {
        return result;
      }
    }
    return BenchParseResult::Success;
  }

 private:
  int argc_;
  char** argv_;
  int mpiRank_;
  std::vector<BenchOptionSpec> options_;
};

// Finalize MPI and map a parse result to a process exit code, or -1 when parsing
// succeeded and the caller should continue. Usage:
//   int rc = benchParseExitCode(parser.parse());
//   if (rc >= 0) return rc;
static inline int benchParseExitCode(BenchParseResult result) {
  if (result == BenchParseResult::Help) {
    MPI_Finalize();
    return 0;
  }
  if (result == BenchParseResult::Error) {
    MPI_Finalize();
    return 1;
  }
  return -1;
}

static inline MPI_Comm benchMpiWorld() {
  return MPI_COMM_WORLD; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Datatype benchMpiByte() {
  return MPI_BYTE; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Datatype benchMpiInt() {
  return MPI_INT; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Datatype benchMpiDouble() {
  return MPI_DOUBLE; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Op benchMpiMin() {
  return MPI_MIN; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Op benchMpiMax() {
  return MPI_MAX; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Op benchMpiSum() {
  return MPI_SUM; // NOLINT(bugprone-casting-through-void)
}

#endif // NCCL_M2N_BENCH_COMMON_H_
