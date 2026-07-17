/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Reference NCCL/RCCL environment plugin that reads parameters from a
 * JSON file instead of (or in addition to) process-level env vars.
 *
 * Usage:
 *   export NCCL_ENV_PLUGIN=/path/to/librccl-env-json.so
 *   export NCCL_ENV_JSON_FILE=/path/to/config.json
 *
 * The JSON file is a flat object mapping NCCL/RCCL variable names to
 * string values, e.g.:
 *   { "NCCL_DEBUG": "INFO", "NCCL_ALGO": "Ring" }
 *
 * Lookup precedence: JSON file value > process environment (getenv).
 * If NCCL_ENV_JSON_FILE is unset or the file cannot be read, the
 * plugin falls back to getenv() for all lookups.
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "nccl/env.h"

#define MAX_ENTRIES  256
#define MAX_KEY_LEN  256
#define MAX_VAL_LEN  4096
#define MAX_FILE_SIZE (1 << 20)

typedef struct {
  char key[MAX_KEY_LEN];
  char value[MAX_VAL_LEN];
} EnvEntry;

static EnvEntry entries[MAX_ENTRIES];
static int      numEntries = 0;
static int      jsonLoaded = 0;

static void skipWhitespace(const char **p) {
  while (**p && isspace((unsigned char)**p)) (*p)++;
}

static int parseString(const char **p, char *out, int maxLen) {
  skipWhitespace(p);
  if (**p != '"') return -1;
  (*p)++;
  int i = 0;
  while (**p && **p != '"' && i < maxLen - 1) {
    if (**p == '\\') {
      (*p)++;
      if (!**p) return -1;
    }
    out[i++] = **p;
    (*p)++;
  }
  out[i] = '\0';
  if (**p == '"') (*p)++;
  else return -1;
  return 0;
}

static int loadJsonFile(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;

  char *buf = (char *)malloc(MAX_FILE_SIZE);
  if (!buf) { fclose(f); return -1; }

  size_t n = fread(buf, 1, MAX_FILE_SIZE - 1, f);
  int readErr = ferror(f);
  int truncated = !feof(f);
  fclose(f);
  if (readErr || truncated) { free(buf); return -1; }
  buf[n] = '\0';

  numEntries = 0;
  const char *p = buf;
  skipWhitespace(&p);
  if (*p != '{') { free(buf); return -1; }
  p++;

  while (numEntries < MAX_ENTRIES) {
    skipWhitespace(&p);
    if (*p == '}') break;
    if (*p == ',' && numEntries > 0) { p++; continue; }

    if (parseString(&p, entries[numEntries].key, MAX_KEY_LEN) != 0) {
      free(buf);
      return -1;
    }

    skipWhitespace(&p);
    if (*p != ':') { free(buf); return -1; }
    p++;

    if (parseString(&p, entries[numEntries].value, MAX_VAL_LEN) != 0) {
      free(buf);
      return -1;
    }

    numEntries++;
  }

  int ok = (*p == '}') ? 0 : -1;
  free(buf);
  return ok;
}

static ncclResult_t ncclEnvJsonInit(uint8_t ncclMajor, uint8_t ncclMinor,
                                     uint8_t ncclPatch, const char *suffix) {
  const char *jsonPath = getenv("NCCL_ENV_JSON_FILE");
  if (jsonPath && strlen(jsonPath) > 0) {
    if (loadJsonFile(jsonPath) == 0) {
      jsonLoaded = 1;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclEnvJsonFinalize(void) {
  numEntries = 0;
  jsonLoaded = 0;
  return ncclSuccess;
}

static const char *ncclEnvJsonGetEnv(const char *name) {
  if (jsonLoaded) {
    for (int i = 0; i < numEntries; i++) {
      if (strcmp(entries[i].key, name) == 0) {
        return entries[i].value;
      }
    }
  }
  return getenv(name);
}

const ncclEnv_v1_t ncclEnvPlugin_v1 = {
    .name     = "ncclEnvJson",
    .init     = ncclEnvJsonInit,
    .finalize = ncclEnvJsonFinalize,
    .getEnv   = ncclEnvJsonGetEnv,
};
