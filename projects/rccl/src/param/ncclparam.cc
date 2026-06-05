/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"
#include "nccl.h"
#include "param/c_api.h"

int main(int argc, char* argv[]) {
  bool longFormat = false;
  bool showAll = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-l") == 0) {
      longFormat = true;
    } else if (strcmp(argv[i], "-a") == 0) {
      showAll = true;
    } else {
      fprintf(stderr, "Usage: %s [-l] [-a]\n", argv[0]);
      fprintf(stderr, "  -l  long format (show values and descriptions)\n");
      fprintf(stderr, "  -a  show all parameters including private ones\n");
      return EXIT_FAILURE;
    }
  }

  if (showAll) {
    setenv("NCCL_PARAM_DUMP_ALL", "TRUE", 1);
  }

  bool initialized = false;
  ncclComm_t comm;
  int dev = 0;

  cudaError_t ce = cudaSetDevice(dev);
  if (ce == cudaSuccess) {
    ncclResult_t nr = ncclCommInitAll(&comm, 1, &dev);
    if (nr == ncclSuccess) {
      initialized = true;
    }
  }

  if (longFormat) {
    ncclParamDumpAll();
  } else {
    const char** keys = nullptr;
    int nkeys = 0;
    ncclParamGetAllParameterKeys(&keys, &nkeys);
    for (int i = 0; i < nkeys; i++) {
      printf("%s\n", keys[i]);
    }
  }

  if (initialized) {
    ncclCommDestroy(comm);
  }
  return 0;
}
