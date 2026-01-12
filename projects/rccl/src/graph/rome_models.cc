/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
Copyright (c) 2024 GigaIO Networks, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "core.h"
#include "comm.h"
#include "graph.h"
#include "topo.h"
#include "xml.h"
#include <math.h>
#include <sys/time.h>
#include <algorithm>
#include <vector>
#include <string.h>
#include "rome_models.h"

struct rcclRomeModel {
  int nGpus;
  int nCpus;
  int nNics;
  int nLinks;
  int64_t gpuIds[NCCL_TOPO_MAX_NODES];
  int64_t nicIds[NCCL_TOPO_MAX_NODES];
  int64_t gpuNuma[NCCL_TOPO_MAX_NODES];
  int64_t nicNuma[NCCL_TOPO_MAX_NODES];
  uint8_t connMatrix[NCCL_TOPO_MAX_NODES*NCCL_TOPO_MAX_NODES];
  uint8_t gdrLevel[NCCL_TOPO_MAX_NODES*NCCL_TOPO_MAX_NODES];
  const char *pattern;
  const char *ringBase;
  const char *ringTail2;  // Lines to use for node N-2 if the total number of nodes is odd
  const char *ringTail1;  // Lines to use for node N-1 if the total number of nodes is odd
  const char *options;
  const char *treeBase;
  const char *treeRail;
};

static struct rcclRomeModel rome_model_22 = {
  .nGpus = 8, .nCpus = 4, .nNics = 1, .nLinks = 2,
  .gpuIds = { 0x3000, 0x43000, 0x26000, 0xc3000, 0x83000, 0x23000, 0xc6000, 0xa3000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 1, 0, 1, 2, 3, 1, 2, 3, },
  .nicNuma = { 2, },
  .connMatrix = { 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, },
  .gdrLevel = { PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_SYS, PATH_SYS, PATH_PHB, PATH_SYS, },
  .pattern = "10302120",
  .ringBase = "7 4 5 3 1 0 6 2|4 7 3 5 0 1 2 6",
  .options = "",
};

static struct rcclRomeModel rome_model_25 = {
  .nGpus = 8, .nCpus = 4, .nNics = 2, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { 0x61000, 0xa1000, },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 2, 3, },
  .nicNuma = { 0, 3, },
  .connMatrix = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, },
  .gdrLevel = { PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, },
  .pattern = "11303011",
  .ringBase = "2 1 0 3 6 7 5 4|7 6 4 5 1 2 3 0",
  .options = "",
};

static struct rcclRomeModel rome_model_27 = {
  .nGpus = 8, .nCpus = 4, .nNics = 2, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { 0x61000, 0xa1000, },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 2, 3, },
  .nicNuma = { 0, 3, },
  .connMatrix = { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, },
  .gdrLevel = { PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, },
  .pattern = "11303011",
  .ringBase = "0 6 2 3 1 7 5 4|7 1 4 5 6 0 3 2",
  .options = "",
};

static struct rcclRomeModel rome_model_29 = {
  .nGpus = 8, .nCpus = 4, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { 2, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, },
  .pattern = "10302120",
  .ringBase = "6 5 7 4 0 1 3 2|6 4 7 5 2 3 1 0",
  .options = "",
};

static struct rcclRomeModel rome_model_31 = {
  .nGpus = 8, .nCpus = 8, .nNics = 2, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { 0x61000, 0xa1000, },
  .gpuNuma = { 1, 2, 2, 3, 4, 5, 5, 7, },
  .nicNuma = { 0, 6, },
  .connMatrix = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, },
  .gdrLevel = { PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, },
  .pattern = "0110201010200110",
  .ringBase = "1 2 3 0 6 4 5 7|4 6 7 5 2 1 0 3",
  .options = "",
};

static struct rcclRomeModel rome_model_33 = {
  .nGpus = 8, .nCpus = 8, .nNics = 2, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { 0x61000, 0xa1000, },
  .gpuNuma = { 1, 2, 2, 3, 4, 5, 5, 7, },
  .nicNuma = { 0, 6, },
  .connMatrix = { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, },
  .gdrLevel = { PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, },
  .pattern = "0110201010200110",
  .ringBase = "1 4 5 7 0 3 2 6|4 1 7 5 6 2 3 0",
  .options = "",
};

static struct rcclRomeModel rome_model_30 = {
  .nGpus = 8, .nCpus = 8, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 1, 2, 2, 3, 4, 5, 5, 7, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "0010201010200010",
  .ringBase = "3 0 1 2 6 7 5 4|2 1 0 3 7 6 4 5",
  .options = "",
};

static struct rcclRomeModel rome_model_32 = {
  .nGpus = 8, .nCpus = 8, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 1, 2, 2, 3, 4, 5, 5, 7, },
  .nicNuma = { },
  .connMatrix = { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, },
  .gdrLevel = { },
  .pattern = "0010201010200010",
  .ringBase = "0 6 2 3 4 5 7 1|3 2 6 0 1 7 5 4",
  .options = "",
};

static struct rcclRomeModel rome_model_24 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 2, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "10303010",
  .ringBase = "0 1 2 3 5 7 6 4|1 0 3 2 7 5 4 6",
  .options = "",
};

static struct rcclRomeModel rome_model_26 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xe3000, 0xc3000, 0xc6000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 2, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, },
  .gdrLevel = { },
  .pattern = "10303010",
  .ringBase = "4 5 7 1 0 3 2 6|3 0 6 2 1 7 5 4",
  .options = "",
};

static struct rcclRomeModel rome_model_23 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, },
  .gdrLevel = { },
  .pattern = "10302020",
  .ringBase = "1 7 6 4 5 2 0 3|2 5 3 0 4 6 7 1",
  .options = "",
};

static struct rcclRomeModel rome_model_38 = {
  .nGpus = 8, .nCpus = 7, .nNics = 0, .nLinks = 2,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { },
  .connMatrix = { 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, },
  .gdrLevel = { },
  .pattern = "10201000201010",
  .ringBase = "6 7 1 4 3 5 2 0|0 2 5 3 4 1 7 6",
  .options = "",
};

static struct rcclRomeModel rome_model_28 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "10302020",
  .ringBase = "0 3 2 1 4 5 6 7|7 6 5 4 1 2 3 0|0 2 5 7 4 6 3 1|1 3 6 4 7 5 2 0",
  .options = "",
};

static struct rcclRomeModel rome_model_40 = {
  .nGpus = 8, .nCpus = 4, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 0, 1, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { 2, },
  .connMatrix = { 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, },
  .gdrLevel = { PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, },
  .pattern = "10302120",
  .ringBase = "6 7 1 4 0 5 3 2|7 6 4 1 0 2 3 5",
  .options = "",
};

static struct rcclRomeModel rome_model_42 = {
  .nGpus = 8, .nCpus = 7, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { 4, },
  .connMatrix = { 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, },
  .gdrLevel = { PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, },
  .pattern = "10201001201010",
  .ringBase = "7 4 6 1 3 0 2 5|6 4 7 1 3 2 5 0",
  .options = "",
};

static struct rcclRomeModel rome_model_44 = {
  .nGpus = 8, .nCpus = 4, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x63000, 0x43000, 0x27000, 0x3000, 0xe3000, 0xc3000, 0xa3000, 0x83000, },
  .nicIds = { 0xc4000, },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { 2, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, },
  .pattern = "20202120",
  .ringBase = "5 4 7 6 2 1 3 0|5 6 7 4 1 0 2 3",
  .options = "",
};

static struct rcclRomeModel rome_model_45 = {
  .nGpus = 8, .nCpus = 7, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "10201000201010",
  .ringBase = "0 1 2 3 4 5 6 7|0 2 5 7 4 6 1 3|0 3 1 6 4 7 5 2|0 7 6 5 4 3 2 1",
  .options = "",
};

static struct rcclRomeModel rome_model_46 = {
  .nGpus = 8, .nCpus = 7, .nNics = 1, .nLinks = 3,
  .gpuIds = { 0x43000, 0x23000, 0x26000, 0x3000, 0xc3000, 0xc6000, 0xa3000, 0x83000, },
  .nicIds = { 0xe1000, },
  .gpuNuma = { 1, 2, 2, 3, 5, 5, 6, 7, },
  .nicNuma = { 4, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, },
  .pattern = "10201001201010",
  .ringBase = "6 5 7 4 1 2 3 0|7 4 6 5 1 0 3 2",
  .options = "",
};

static struct rcclRomeModel rome_model_48 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x4a000, 0x50000, 0xa000, 0xf000, 0xcb000, 0xd1000, 0x8a000, 0x90000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "20202020",
  .ringBase = "0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0|0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0",
  .options = "",
};

static struct rcclRomeModel rome_model_49 = {
  .nGpus = 8, .nCpus = 4, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0x4a000, 0x50000, 0xa000, 0xf000, 0xcb000, 0xd1000, 0x8a000, 0x90000, },
  .nicIds = { 0x45000, 0x13000, 0xc6000, 0x85000, },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { 0, 1, 2, 3, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, },
  .pattern = "21212121",
  .ringBase = "N0 0 1 2 3 4 5 6 7 N3|N3 7 6 5 4 3 2 1 0 N0|N1 2 3 0 1 6 7 4 5 N2|N2 5 4 7 6 1 0 3 2 N1",
  .options = "",
};

static struct rcclRomeModel rome_model_52 = {
  .nGpus = 8, .nCpus = 1, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0xc1000, 0xc5000, 0xc9000, 0xcd000, 0xd1000, 0xd5000, 0xd9000, 0xdd000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 0, 0, 0, 0, 0, 0, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "80",
  .ringBase = "0 1 3 2 4 5 7 6|6 7 5 4 2 3 1 0|0 1 5 4 6 7 3 2|2 3 7 6 4 5 1 0",
  .options = "",
};

static struct rcclRomeModel rome_model_53 = {
  .nGpus = 8, .nCpus = 4, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0x4a000, 0x50000, 0xa000, 0xf000, 0xcb000, 0xd1000, 0x8a000, 0x90000, },
  .nicIds = { 0x45000, 0x13000, 0xc6000, 0x85000, },
  .gpuNuma = { 1, 1, 3, 3, 5, 5, 7, 7, },
  .nicNuma = { 1, 3, 5, 7, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, },
  .pattern = "21212121",
  .ringBase = "N0 0 1 2 3 4 5 6 7 N3|N3 7 6 5 4 3 2 1 0 N0|N1 2 3 0 1 6 7 4 5 N2|N2 5 4 7 6 1 0 3 2 N1",
  .options = "",
};

static struct rcclRomeModel rome_model_43 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x63000, 0x43000, 0x27000, 0x3000, 0xe3000, 0xc3000, 0xa3000, 0x83000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "20202020",
  .ringBase = "0 1 2 3 4 5 6 7|0 2 5 7 4 6 1 3|0 3 1 6 4 7 5 2|0 7 6 5 4 3 2 1|0 1 2 3 4 5 6 7|0 2 5 7 4 6 1 3|0 3 1 6 4 7 5 2|0 7 6 5 4 3 2 1|0 1 2 3 4 5 6 7|0 2 5 7 4 6 1 3|0 3 1 6 4 7 5 2|0 7 6 5 4 3 2 1",
  .options = "treeDefined=1",
  .treeBase = "(2(5(6(7(4))))(3(0(1))))|(2(5(7(6(4))))(0(1(3))))|(2(5(7(4(6))))(1(3(0))))|(6(1(0(2(3))))(7(4(5))))|(6(1(2(0(3))))(4(5(7))))|(6(1(0(3(2))))(5(7(4))))|(1(6(7(5(4))))(2(3(0))))|(1(6(4(7(5))))(3(2(0))))|(1(6(5(4(7))))(3(0(2))))|(5(2(3(1(0))))(4(6(7))))|(5(2(0(3(1))))(6(4(7))))|(5(2(1(0(3))))(4(7(6))))",
};

static struct rcclRomeModel rome_model_55 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x100000, 0x200000, 0x300000, 0x400000, 0x500000, 0x600000, 0x700000, 0x800000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "20202020",
  .ringBase = "0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0|2 3 0 1 6 7 4 5|5 4 7 6 1 0 3 2",
  .options = "",
};

static struct rcclRomeModel rome_model_56 = {
  .nGpus = 16, .nCpus = 4, .nNics = 0, .nLinks = 4,
  .gpuIds = { 0x4e000, 0x51000, 0x56000, 0x59000, 0xe000, 0x11000, 0x16000, 0x19000, 0xcf000, 0xd2000, 0xd7000, 0xda000, 0x8f000, 0x92000, 0x97000, 0x9a000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 4, 0, },
  .gdrLevel = { },
  .pattern = "40404040",
  .ringBase = "0 1 3 2 6 7 15 14 10 11 9 8 12 13 5 4|0 1 2 3 7 6 13 12 8 9 10 11 15 14 5 4|0 2 3 7 6 14 15 11 10 8 9 13 12 4 5 1|4 5 13 12 8 9 11 10 14 15 7 6 2 3 1 0|4 5 14 15 11 10 9 8 12 13 6 7 3 2 1 0|1 5 4 12 13 9 8 10 11 15 14 6 7 3 2 0",
  .options = "pivotA2AEnabled=1,pivotA2ANumBiRings=3,tuning=1,mscclEnabled=1,treeDefined=1",
  .treeBase= "(0(1(3(2(6(7(15(14(10))))))))(4(5(13(12(8(9(11))))))))|(2(3(7(6(13(12(8(9(10))))))))(1(0(4(5(14(15(11))))))))|(14(15(11(10(8(9(13(12(4))))))))(6(7(3(2(0(1(5))))))))|(10(11(9(8(12(13(5(4(0))))))))(14(15(7(6(2(3(1))))))))|(10(11(15(14(5(4(0(1(2))))))))(9(8(12(13(6(7(3))))))))|(4(5(1(0(2(3(7(6(14))))))))(12(13(9(8(10(11(15))))))))|(6(7(15(14(10(11(9(8(12))))))))(2(3(1(0(4(5(13))))))))|(13(12(8(9(10(11(15(14(5))))))))(6(7(3(2(1(0(4))))))))|(8(9(13(12(4(5(1(0(2))))))))(10(11(15(14(6(7(3))))))))|(12(13(5(4(0(1(3(2(6))))))))(8(9(11(10(14(15(7))))))))|(5(4(0(1(2(3(7(6(13))))))))(14(15(11(10(9(8(12))))))))|(2(3(7(6(14(15(11(10(8))))))))(0(1(5(4(12(13(9))))))))",
};

static struct rcclRomeModel rome_model_58 = {
  .nGpus = 8, .nCpus = 3, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0xc1000, 0xc6000, 0xc9000, 0xce000, 0xd1000, 0xd6000, 0xd9000, 0xde000, },
  .nicIds = { },
  .gpuNuma = { 3, 3, 1, 1, 0, 0, 0, 0, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "402020",
  .ringBase = "0 1 3 2 4 5 7 6|6 7 5 4 2 3 1 0|0 1 5 4 6 7 3 2|2 3 7 6 4 5 1 0",
  .options = "",
};

static struct rcclRomeModel rome_model_59 = {
  .nGpus = 16, .nCpus = 4, .nNics = 8, .nLinks = 4,
  .gpuIds = { 0x4e000, 0x51000, 0x56000, 0x59000, 0xe000, 0x11000, 0x16000, 0x19000, 0xcf000, 0xd2000, 0xd7000, 0xda000, 0x8f000, 0x92000, 0x97000, 0x9a000, },
  .nicIds = { 0x4b000, 0x5a000, 0xb000, 0x1a000, 0xcc000, 0xdb000, 0x8c000, 0x9b000, },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, },
  .nicNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .connMatrix = { 0, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 4, 0, },
  .gdrLevel = { PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, },
  .pattern = "42424242",
  .ringBase = "N4 9 8 12 13 5 4 0 1 3 2 6 7 15 14 10 11 N5|N1 3 2 6 7 15 14 10 11 9 8 12 13 5 4 0 1 N0|N3 7 6 2 3 1 0 4 5 13 12 8 9 11 10 14 15 N7|N7 15 14 10 11 9 8 12 13 5 4 0 1 3 2 6 7 N3|N5 11 10 14 15 7 6 2 3 1 0 4 5 13 12 8 9 N4|N0 1 0 4 5 13 12 8 9 11 10 14 15 7 6 2 3 N1|N3 6 7 3 2 1 0 4 5 14 15 11 10 9 8 12 13 N6|N7 14 15 11 10 9 8 12 13 6 7 3 2 1 0 4 5 N2|N2 5 4 0 1 2 3 7 6 13 12 8 9 10 11 15 14 N7|N6 13 12 8 9 10 11 15 14 5 4 0 1 2 3 7 6 N3|N4 8 9 13 12 4 5 1 0 2 3 7 6 14 15 11 10 N5|N5 10 11 15 14 6 7 3 2 0 1 5 4 12 13 9 8 N4|N6 12 13 9 8 10 11 15 14 6 7 3 2 0 1 5 4 N2|N2 4 5 1 0 2 3 7 6 14 15 11 10 8 9 13 12 N6|N1 2 3 7 6 14 15 11 10 8 9 13 12 4 5 1 0 N0|N0 0 1 5 4 12 13 9 8 10 11 15 14 6 7 3 2 N1|N5 10 11 9 8 12 13 5 4 0 1 3 2 6 7 15 14 N7|N3 6 7 15 14 10 11 9 8 12 13 5 4 0 1 3 2 N1|N1 2 3 1 0 4 5 13 12 8 9 11 10 14 15 7 6 N3|N7 14 15 7 6 2 3 1 0 4 5 13 12 8 9 11 10 N5|N0 0 1 2 3 7 6 13 12 8 9 10 11 15 14 5 4 N2|N4 8 9 10 11 15 14 5 4 0 1 2 3 7 6 13 12 N6|N3 7 6 13 12 8 9 10 11 15 14 5 4 0 1 2 3 N1|N1 3 2 1 0 4 5 14 15 11 10 9 8 12 13 6 7 N3|N6 12 13 6 7 3 2 1 0 4 5 14 15 11 10 9 8 N4|N2 4 5 14 15 11 10 9 8 12 13 6 7 3 2 1 0 N0|N0 1 0 2 3 7 6 14 15 11 10 8 9 13 12 4 5 N2|N6 13 12 4 5 1 0 2 3 7 6 14 15 11 10 8 9 N4|N5 11 10 8 9 13 12 4 5 1 0 2 3 7 6 14 15 N7|N2 5 4 12 13 9 8 10 11 15 14 6 7 3 2 0 1 N0|N7 15 14 6 7 3 2 0 1 5 4 12 13 9 8 10 11 N5|N4 9 8 10 11 15 14 6 7 3 2 0 1 5 4 12 13 N6",
  .options = "tuning=4,ll128Enabled=1,baseBw=161.4",
};

static struct rcclRomeModel rome_model_62 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0xc1000, 0xc6000, 0xc9000, 0xce000, 0xd1000, 0xd6000, 0xd9000, 0xde000, },
  .nicIds = { },
  .gpuNuma = { 3, 3, 1, 1, 0, 0, 2, 2, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "20202020",
  .ringBase = "0 1 3 2 4 5 7 6|6 7 5 4 2 3 1 0|0 1 5 4 6 7 3 2|2 3 7 6 4 5 1 0",
  .options = "",
};

static struct rcclRomeModel rome_model_63 = {
  .nGpus = 8, .nCpus = 4, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0xc1000, 0xc6000, 0xc9000, 0xce000, 0xd1000, 0xd6000, 0xd9000, 0xde000, },
  .nicIds = { 0xc5000, 0xcd000, 0xd5000, 0xdd000, },
  .gpuNuma = { 3, 3, 1, 1, 0, 0, 2, 2, },
  .nicNuma = { 3, 1, 0, 2, },
  .connMatrix = { 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, },
  .gdrLevel = { PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, },
  .pattern = "21212121",
  .ringBase = "N0 0 1 5 4 6 7 3 2 N1|N1 2 3 7 6 4 5 1 0 N0|N3 7 6 0 1 3 2 4 5 N2|N2 5 4 2 3 1 0 6 7 N3|N0 0 1 5 4 6 7 3 2 N1|N1 2 3 7 6 4 5 1 0 N0|N3 7 6 0 1 3 2 4 5 N2|N2 5 4 2 3 1 0 6 7 N3",
  .options = "tuning=3",
};

static struct rcclRomeModel rome_model_65 = {
  .nGpus = 16, .nCpus = 4, .nNics = 8, .nLinks = 4,
  .gpuIds = { 0x4e000, 0x51000, 0x56000, 0x59000, 0xe000, 0x11000, 0x16000, 0x19000, 0xcf000, 0xd2000, 0xd7000, 0xda000, 0x8f000, 0x92000, 0x97000, 0x9a000, },
  .nicIds = { 0x4b000, 0x5a000, 0xb000, 0x1a000, 0xcc000, 0xdb000, 0x8c000, 0x9b000, },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, },
  .nicNuma = { 0, 0, 1, 1, 2, 2, 3, 3, },
  .connMatrix = { 0, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 4, 0, },
  .gdrLevel = { PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, },
  .pattern = "42424242",
  .ringBase = "N4 9 8 12 13 5 4 0 1 3 2 6 7 15 14 10 11 N5|N1 3 2 6 7 15 14 10 11 9 8 12 13 5 4 0 1 N0|N3 7 6 2 3 1 0 4 5 13 12 8 9 11 10 14 15 N7|N7 15 14 10 11 9 8 12 13 5 4 0 1 3 2 6 7 N3|N5 11 10 14 15 7 6 2 3 1 0 4 5 13 12 8 9 N4|N0 1 0 4 5 13 12 8 9 11 10 14 15 7 6 2 3 N1|N3 6 7 3 2 1 0 4 5 14 15 11 10 9 8 12 13 N6|N7 14 15 11 10 9 8 12 13 6 7 3 2 1 0 4 5 N2|N2 5 4 0 1 2 3 7 6 13 12 8 9 10 11 15 14 N7|N6 13 12 8 9 10 11 15 14 5 4 0 1 2 3 7 6 N3|N4 8 9 13 12 4 5 1 0 2 3 7 6 14 15 11 10 N5|N5 10 11 15 14 6 7 3 2 0 1 5 4 12 13 9 8 N4|N6 12 13 9 8 10 11 15 14 6 7 3 2 0 1 5 4 N2|N2 4 5 1 0 2 3 7 6 14 15 11 10 8 9 13 12 N6|N1 2 3 7 6 14 15 11 10 8 9 13 12 4 5 1 0 N0|N0 0 1 5 4 12 13 9 8 10 11 15 14 6 7 3 2 N1|N5 10 11 9 8 12 13 5 4 0 1 3 2 6 7 15 14 N7|N3 6 7 15 14 10 11 9 8 12 13 5 4 0 1 3 2 N1|N1 2 3 1 0 4 5 13 12 8 9 11 10 14 15 7 6 N3|N7 14 15 7 6 2 3 1 0 4 5 13 12 8 9 11 10 N5|N0 0 1 2 3 7 6 13 12 8 9 10 11 15 14 5 4 N2|N4 8 9 10 11 15 14 5 4 0 1 2 3 7 6 13 12 N6|N3 7 6 13 12 8 9 10 11 15 14 5 4 0 1 2 3 N1|N1 3 2 1 0 4 5 14 15 11 10 9 8 12 13 6 7 N3|N6 12 13 6 7 3 2 1 0 4 5 14 15 11 10 9 8 N4|N2 4 5 14 15 11 10 9 8 12 13 6 7 3 2 1 0 N0|N0 1 0 2 3 7 6 14 15 11 10 8 9 13 12 4 5 N2|N6 13 12 4 5 1 0 2 3 7 6 14 15 11 10 8 9 N4|N5 11 10 8 9 13 12 4 5 1 0 2 3 7 6 14 15 N7|N2 5 4 12 13 9 8 10 11 15 14 6 7 3 2 0 1 N0|N7 15 14 6 7 3 2 0 1 5 4 12 13 9 8 10 11 N5|N4 9 8 10 11 15 14 6 7 3 2 0 1 5 4 12 13 N6",
  .options = "tuning=4,ll128Enabled=1,baseBw=161.4",
};

static struct rcclRomeModel rome_model_66 = {
  .nGpus = 8, .nCpus = 2, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x29000, 0x2c000, 0x2f000, 0x32000, 0xad000, 0xb0000, 0xb3000, 0xb6000, },
  .nicIds = { },
  .gpuNuma = { 1, 1, 1, 1, 3, 3, 3, 3, },
  .nicNuma = { },
  .connMatrix = { 0, 4, 0, 0, 2, 0, 1, 0, 4, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 1, 0, 2, 0, 0, 1, 4, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 1, 4, 0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 1, 4, 0, },
  .gdrLevel = { },
  .pattern = "4040",
  .ringBase = "0 6 7 5 4 2 3 1|1 3 2 4 5 7 6 0|0 1 7 6 2 3 5 4|4 5 3 2 6 7 1 0",
  .options = "disableNumaMatching=1,tuning=2",
};

static struct rcclRomeModel rome_model_67 = {
  .nGpus = 8, .nCpus = 2, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0x29000, 0x2c000, 0x2f000, 0x32000, 0xad000, 0xb0000, 0xb3000, 0xb6000, },
  .nicIds = { 0x1d000, 0x1e000, 0xa1000, 0xa2000, },
  .gpuNuma = { 1, 1, 1, 1, 3, 3, 3, 3, },
  .nicNuma = { 1, 1, 3, 3, },
  .connMatrix = { 0, 4, 0, 0, 2, 0, 1, 0, 4, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 1, 0, 2, 0, 0, 1, 4, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 1, 4, 0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 1, 4, 0, },
  .gdrLevel = { PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, },
  .pattern = "4242",
  .ringBase = "N3 7 6 0 1 3 2 4 5 N2|N2 5 4 2 3 1 0 6 7 N3|N1 2 3 5 4 0 1 7 6 N3|N2 4 5 3 2 6 7 1 0 N0|N1 3 2 4 5 7 6 0 1 N0|N0 1 0 6 7 5 4 2 3 N1|N0 0 1 7 6 2 3 5 4 N2|N3 6 7 1 0 4 5 3 2 N1",
  .options = "disableNumaMatching=1,tuning=2",
};

static struct rcclRomeModel rome_model_68 = {
  .nGpus = 16, .nCpus = 1, .nNics = 16, .nLinks = 3,
  .gpuIds = { 0xcf000, 0xd4000, 0xd5000, 0xd6000, 0xd0000, 0xd1000, 0xd2000, 0xd3000, 0xf0000, 0xf1000, 0xf2000, 0xf3000, 0xf4000, 0xf5000, 0xf6000, 0xf7000, },
  .nicIds = { 0xcd000, 0xc8000, 0xc9000, 0xcb000, 0xcc000, 0xce000, 0xc7000, 0xca000, 0xe8000, 0xe9000, 0xea000, 0xeb000, 0xec000, 0xed000, 0xee000, 0xef000, },
  .gpuNuma = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, },
  .nicNuma = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, },
  .connMatrix = { 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, },
  .gdrLevel = { PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PXB, PATH_PIX, },
  .pattern = "@@",
  .ringBase = "N0 0 1 2 3 N3 N4 4 5 6 7 N7 N8 8 9 10 11 N11 N12 12 13 14 15 N15|N15 15 14 13 12 N12 N11 11 10 9 8 N8 N7 7 6 5 4 N4 N3 3 2 1 0 N0|N1 1 3 0 2 N2 N5 5 7 4 6 N6 N9 9 11 8 10 N10 N13 13 15 12 14 N14|N14 14 12 15 13 N13 N10 10 8 11 9 N9 N6 6 4 7 5 N5 N2 2 0 3 1 N1|N0 0 1 2 3 N3 N4 4 5 6 7 N7 N8 8 9 10 11 N11 N12 12 13 14 15 N15|N15 15 14 13 12 N12 N11 11 10 9 8 N8 N7 7 6 5 4 N4 N3 3 2 1 0 N0|N1 1 3 0 2 N2 N5 5 7 4 6 N6 N9 9 11 8 10 N10 N13 13 15 12 14 N14|N14 14 12 15 13 N13 N10 10 8 11 9 N9 N6 6 4 7 5 N5 N2 2 0 3 1 N1",
  .options = "",
};

static struct rcclRomeModel rome_model_71 = {
  .nGpus = 8, .nCpus = 2, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0x32000, 0x35000, 0x11000, 0x14000, 0xae000, 0xb3000, 0x8e000, 0x93000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .nicNuma = { },
  .connMatrix = { 0, 4, 1, 0, 0, 0, 2, 0, 4, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 4, 2, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 4, 1, 0, 0, 1, 0, 0, 4, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 1, 0, 1, 4, 0, },
  .gdrLevel = { },
  .pattern = "4040",
  .ringBase = "0 1 3 2 4 5 7 6|6 7 5 4 2 3 1 0|0 1 5 4 2 3 7 6|6 7 3 2 4 5 1 0",
  .options = "disableNumaMatching=1,tuning=2",
};

static struct rcclRomeModel rome_model_72 = {
  .nGpus = 8, .nCpus = 2, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0x32000, 0x35000, 0x11000, 0x14000, 0xae000, 0xb3000, 0x8e000, 0x93000, },
  .nicIds = { 0x1d000, 0x1e000, 0xa0000, 0xa1000, },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .nicNuma = { 0, 0, 1, 1, },
  .connMatrix = { 0, 4, 1, 0, 0, 0, 2, 0, 4, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 4, 2, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 4, 1, 0, 0, 1, 0, 0, 4, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 1, 0, 1, 4, 0, },
  .gdrLevel = { PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, },
  .pattern = "4242",
  .ringBase = "N0 0 1 3 2 4 5 7 6 N3|N1 2 3 1 0 6 7 5 4 N2|N3 7 6 0 1 5 4 2 3 N1|N0 1 0 6 7 3 2 4 5 N2|N2 4 5 7 6 0 1 3 2 N1|N3 6 7 5 4 2 3 1 0 N0|N2 5 4 2 3 7 6 0 1 N0|N1 3 2 4 5 1 0 6 7 N3",
  .options = "disableNumaMatching=1,tuning=2",
};

static struct rcclRomeModel rome_model_73 = {
  .nGpus = 8, .nCpus = 4, .nNics = 0, .nLinks = 3,
  .gpuIds = { 0xc1000, 0xc6000, 0xc9000, 0xce000, 0xd1000, 0xd6000, 0xd9000, 0xde000, },
  .nicIds = { },
  .gpuNuma = { 3, 3, 1, 1, 0, 0, 2, 2, },
  .nicNuma = { },
  .connMatrix = { 0, 4, 1, 0, 0, 0, 2, 0, 4, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 4, 2, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 4, 1, 0, 0, 1, 0, 0, 4, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 1, 0, 1, 4, 0, },
  .gdrLevel = { },
  .pattern = "20202020",
  .ringBase = "0 1 3 2 4 5 7 6|6 7 5 4 2 3 1 0|0 1 5 4 6 7 3 2|2 3 7 6 4 5 1 0",
  .options = "",
};

static struct rcclRomeModel rome_model_74 = {
  .nGpus = 8, .nCpus = 4, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0xc1000, 0xc6000, 0xc9000, 0xce000, 0xd1000, 0xd6000, 0xd9000, 0xde000, },
  .nicIds = { 0xc5000, 0xcd000, 0xd5000, 0xdd000, },
  .gpuNuma = { 3, 3, 1, 1, 0, 0, 2, 2, },
  .nicNuma = { 3, 1, 0, 2, },
  .connMatrix = { 0, 4, 1, 0, 0, 0, 2, 0, 4, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 4, 2, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 4, 1, 0, 0, 1, 0, 0, 4, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 1, 0, 1, 4, 0, },
  .gdrLevel = { PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, },
  .pattern = "21212121",
  .ringBase = "N0 0 1 5 4 6 7 3 2 N1|N1 2 3 7 6 4 5 1 0 N0|N3 7 6 0 1 3 2 4 5 N2|N2 5 4 2 3 1 0 6 7 N3|N0 0 1 5 4 6 7 3 2 N1|N1 2 3 7 6 4 5 1 0 N0|N3 7 6 0 1 3 2 4 5 N2|N2 5 4 2 3 1 0 6 7 N3",
  .options = "tuning=3",
};

static struct rcclRomeModel rome_model_76 = {
  .nGpus = 8, .nCpus = 2, .nNics = 8, .nLinks = 3,
  .gpuIds = { 0x32000, 0x35000, 0x11000, 0x14000, 0xae000, 0xb3000, 0x8e000, 0x93000, },
  .nicIds = { 0x26000, 0x2d000, 0x5000, 0xc000, 0xab000, 0xb4000, 0x8b000, 0x94000, },
  .gpuNuma = { 1, 1, 1, 1, 3, 3, 3, 3, },
  .nicNuma = { 1, 1, 1, 1, 3, 3, 3, 3, },
  .connMatrix = { 0, 4, 1, 0, 0, 0, 2, 0, 4, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 4, 2, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 4, 1, 0, 0, 1, 0, 0, 4, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 1, 0, 1, 4, 0, },
  .gdrLevel = { PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, },
  .pattern = "4444",
  .ringBase = "N0 0 1 3 2 4 5 7 6 N6|N2 2 3 1 0 6 7 5 4 N4|N5 5 4 2 3 7 6 0 1 N1|N1 1 0 6 7 3 2 4 5 N5|N4 4 5 7 6 0 1 3 2 N2|N2 2 3 1 0 6 7 5 4 N4|N0 0 1 5 4 2 3 7 6 N6|N3 3 2 4 5 1 0 6 7 N7|N4 4 5 7 6 0 1 3 2 N2|N6 6 7 5 4 2 3 1 0 N0|N7 7 6 0 1 5 4 2 3 N3|N6 6 7 3 2 4 5 1 0 N0|N3 3 2 0 1 5 4 6 7 N7|N1 1 0 2 3 7 6 4 5 N5|N5 5 4 6 7 3 2 0 1 N1|N7 7 6 4 5 1 0 2 3 N3",
  .options = "disableNumaMatching=1,tuning=3",
};

static struct rcclRomeModel rome_model_79 = {
  .nGpus = 8, .nCpus = 2, .nNics = 0, .nLinks = 7,
  .gpuIds = { 0x1d000, 0x2e000, 0x3f000, 0x61000, 0x9f000, 0xaf000, 0xbf000, 0xdf000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "4040",
  .ringBase = "0 1 2 3 4 5 6 7|0 1 2 3 4 5 7 6|0 2 4 1 3 6 5 7|0 2 4 6 1 7 3 5|0 3 1 5 2 7 4 6|0 3 5 1 6 2 7 4|0 4 1 7 3 6 2 5|7 6 5 4 3 2 1 0|6 7 5 4 3 2 1 0|7 5 6 3 1 4 2 0|5 3 7 1 6 4 2 0|6 4 7 2 5 1 3 0|4 7 2 6 1 5 3 0|5 2 6 3 7 1 4 0|0 1 2 3 4 5 6 7|0 1 2 3 4 5 7 6|0 2 4 1 3 6 5 7|0 2 4 6 1 7 3 5|0 3 1 5 2 7 4 6|0 3 5 1 6 2 7 4|0 4 1 7 3 6 2 5|7 6 5 4 3 2 1 0|6 7 5 4 3 2 1 0|7 5 6 3 1 4 2 0|5 3 7 1 6 4 2 0|6 4 7 2 5 1 3 0|4 7 2 6 1 5 3 0|5 2 6 3 7 1 4 0",
  .options = "noCpuCheck=1,mscclEnabled=1,tuning=5,disableNumaMatching=1",
};

static struct rcclRomeModel rome_model_80 = {
  .nGpus = 4, .nCpus = 4, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0x82000, 0xc2000, 0x2000, 0x42000, },
  .nicIds = { 0x81000, 0xc1000, 0x1000, 0x41000, },
  .gpuNuma = { 2, 3, 0, 1, },
  .nicNuma = { 2, 3, 0, 1, },
  .connMatrix = { 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, },
  .gdrLevel = { PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, },
  .pattern = "11111111",
  .ringBase = "N2 2 3 0 1 N1|N0 0 1 3 2 N2|N0 0 2 1 3 N3|N3 3 1 0 2 N2|N3 3 1 2 0 N0|N1 1 0 3 2 N2|N1 1 2 3 0 N0|N2 2 0 1 3 N3|N3 3 0 2 1 N1|N2 2 3 1 0 N0|N1 1 2 0 3 N3|N0 0 3 2 1 N1",
  .options = "",
};

static struct rcclRomeModel rome_model_81 = {
  .nGpus = 8, .nCpus = 2, .nNics = 8, .nLinks = 7,
  .gpuIds     = { 0xc000, 0x22000, 0x38000, 0x5c000, 0x9f000, 0xaf000, 0xbf000, 0xdf000, },
  .nicIds     = { 0x7000, 0x1d000, 0x33000, 0x57000, 0x9a000, 0xaa000, 0xba000, 0xda000, },
  .gpuNuma    = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .nicNuma    = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .connMatrix = { 0, 1, 1, 1, 1, 1, 1, 1,
                  1, 0, 1, 1, 1, 1, 1, 1,
                  1, 1, 0, 1, 1, 1, 1, 1,
                  1, 1, 1, 0, 1, 1, 1, 1,
                  1, 1, 1, 1, 0, 1, 1, 1,
                  1, 1, 1, 1, 1, 0, 1, 1,
                  1, 1, 1, 1, 1, 1, 0, 1,
                  1, 1, 1, 1, 1, 1, 1, 0, },
  .gdrLevel   = {PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS,
                 PATH_PHB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS,
                 PATH_PHB, PATH_PHB, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS,
                 PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS,
                 PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB,
                 PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PXB, PATH_PHB, PATH_PHB,
                 PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PHB,
                 PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, },
  .pattern    = "4444",
  .ringBase   = "N0 0 1 2 3 4 5 6 7 N7|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N2 2 5 0 3 6 1 7 4 N4|"
                "N3 3 7 0 4 2 1 6 5 N5|"
                "N4 4 6 2 7 3 0 5 1 N1|"
                "N5 5 4 7 1 3 2 6 0 N0|"
                "N6 6 3 1 4 0 7 5 2 N2|"
                "N7 7 2 0 6 4 1 5 3 N3|"

                "N1 1 0 2 4 3 5 7 6 N6|"
                "N2 2 5 0 3 6 1 7 4 N4|"
                "N3 3 7 0 4 2 1 6 5 N5|"
                "N4 4 6 2 7 3 0 5 1 N1|"
                "N5 5 4 7 1 3 2 6 0 N0|"
                "N6 6 3 1 4 0 7 5 2 N2|"
                "N7 7 2 0 6 4 1 5 3 N3|"
                "N0 0 1 2 3 4 5 6 7 N7|"

                "N2 2 5 0 3 6 1 7 4 N4|"
                "N3 3 7 0 4 2 1 6 5 N5|"
                "N4 4 6 2 7 3 0 5 1 N1|"
                "N5 5 4 7 1 3 2 6 0 N0|"
                "N6 6 3 1 4 0 7 5 2 N2|"
                "N7 7 2 0 6 4 1 5 3 N3|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N1 1 0 2 4 3 5 7 6 N6|"

                "N3 3 7 0 4 2 1 6 5 N5|"
                "N4 4 6 2 7 3 0 5 1 N1|"
                "N5 5 4 7 1 3 2 6 0 N0|"
                "N6 6 3 1 4 0 7 5 2 N2|"
                "N7 7 2 0 6 4 1 5 3 N3|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N2 2 5 0 3 6 1 7 4 N4|"

                "N4 4 6 2 7 3 0 5 1 N1|"
                "N5 5 4 7 1 3 2 6 0 N0|"
                "N6 6 3 1 4 0 7 5 2 N2|"
                "N7 7 2 0 6 4 1 5 3 N3|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N2 2 5 0 3 6 1 7 4 N4|"
                "N3 3 7 0 4 2 1 6 5 N5|"

                "N5 5 4 7 1 3 2 6 0 N0|"
                "N6 6 3 1 4 0 7 5 2 N2|"
                "N7 7 2 0 6 4 1 5 3 N3|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N2 2 5 0 3 6 1 7 4 N4|"
                "N3 3 7 0 4 2 1 6 5 N5|"
                "N4 4 6 2 7 3 0 5 1 N1|"

                "N6 6 3 1 4 0 7 5 2 N2|"
                "N7 7 2 0 6 4 1 5 3 N3|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N2 2 5 0 3 6 1 7 4 N4|"
                "N3 3 7 0 4 2 1 6 5 N5|"
                "N4 4 6 2 7 3 0 5 1 N1|"
                "N5 5 4 7 1 3 2 6 0 N0|"

                "N7 7 2 0 6 4 1 5 3 N3|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N2 2 5 0 3 6 1 7 4 N4|"
                "N3 3 7 0 4 2 1 6 5 N5|"
                "N4 4 6 2 7 3 0 5 1 N1|"
                "N5 5 4 7 1 3 2 6 0 N0|"
                "N6 6 3 1 4 0 7 5 2 N2|",

  .ringTail2  = "N7 7 4 1 3 2 0 6 5 N5|"
                "N6 6 3 0 7 5 1 4 2 N2|"
                "N4 4 6 2 1 7 0 5 3 N3|"
                "N5 5 2 7 3 1 6 0 4 N4|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N2 2 5 0 3 6 4 7 1 N1|"
                "N3 3 7 2 6 1 5 4 0 N0|"

                "N6 6 3 0 7 5 1 4 2 N2|"
                "N4 4 6 2 1 7 0 5 3 N3|"
                "N5 5 2 7 3 1 6 0 4 N4|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N2 2 5 0 3 6 4 7 1 N1|"
                "N3 3 7 2 6 1 5 4 0 N0|"
                "N7 7 4 1 3 2 0 6 5 N5|"

                "N4 4 6 2 1 7 0 5 3 N3|"
                "N5 5 2 7 3 1 6 0 4 N4|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N2 2 5 0 3 6 4 7 1 N1|"
                "N3 3 7 2 6 1 5 4 0 N0|"
                "N7 7 4 1 3 2 0 6 5 N5|"
                "N6 6 3 0 7 5 1 4 2 N2|"

                "N5 5 2 7 3 1 6 0 4 N4|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N2 2 5 0 3 6 4 7 1 N1|"
                "N3 3 7 2 6 1 5 4 0 N0|"
                "N7 7 4 1 3 2 0 6 5 N5|"
                "N6 6 3 0 7 5 1 4 2 N2|"
                "N4 4 6 2 1 7 0 5 3 N3|"

                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N2 2 5 0 3 6 4 7 1 N1|"
                "N3 3 7 2 6 1 5 4 0 N0|"
                "N7 7 4 1 3 2 0 6 5 N5|"
                "N6 6 3 0 7 5 1 4 2 N2|"
                "N4 4 6 2 1 7 0 5 3 N3|"
                "N5 5 2 7 3 1 6 0 4 N4|"

                "N0 0 1 2 3 4 5 6 7 N7|"
                "N2 2 5 0 3 6 4 7 1 N1|"
                "N3 3 7 2 6 1 5 4 0 N0|"
                "N7 7 4 1 3 2 0 6 5 N5|"
                "N6 6 3 0 7 5 1 4 2 N2|"
                "N4 4 6 2 1 7 0 5 3 N3|"
                "N5 5 2 7 3 1 6 0 4 N4|"
                "N1 1 0 2 4 3 5 7 6 N6|"

                "N2 2 5 0 3 6 4 7 1 N1|"
                "N3 3 7 2 6 1 5 4 0 N0|"
                "N7 7 4 1 3 2 0 6 5 N5|"
                "N6 6 3 0 7 5 1 4 2 N2|"
                "N4 4 6 2 1 7 0 5 3 N3|"
                "N5 5 2 7 3 1 6 0 4 N4|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"

                "N3 3 7 2 6 1 5 4 0 N0|"
                "N7 7 4 1 3 2 0 6 5 N5|"
                "N6 6 3 0 7 5 1 4 2 N2|"
                "N4 4 6 2 1 7 0 5 3 N3|"
                "N5 5 2 7 3 1 6 0 4 N4|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N2 2 5 0 3 6 4 7 1 N1|",


  .ringTail1  = "N5 5 4 2 7 1 6 3 0 N0|"
                "N2 2 5 0 3 7 4 6 1 N1|"
                "N3 3 6 4 0 5 1 7 2 N2|"
                "N4 4 7 0 6 5 2 1 3 N3|"
                "N6 6 2 0 7 5 3 1 4 N4|"
                "N7 7 3 2 6 0 4 1 5 N5|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"

                "N2 2 5 0 3 7 4 6 1 N1|"
                "N3 3 6 4 0 5 1 7 2 N2|"
                "N4 4 7 0 6 5 2 1 3 N3|"
                "N6 6 2 0 7 5 3 1 4 N4|"
                "N7 7 3 2 6 0 4 1 5 N5|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N5 5 4 2 7 1 6 3 0 N0|"

                "N3 3 6 4 0 5 1 7 2 N2|"
                "N4 4 7 0 6 5 2 1 3 N3|"
                "N6 6 2 0 7 5 3 1 4 N4|"
                "N7 7 3 2 6 0 4 1 5 N5|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N5 5 4 2 7 1 6 3 0 N0|"
                "N2 2 5 0 3 7 4 6 1 N1|"

                "N4 4 7 0 6 5 2 1 3 N3|"
                "N6 6 2 0 7 5 3 1 4 N4|"
                "N7 7 3 2 6 0 4 1 5 N5|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N5 5 4 2 7 1 6 3 0 N0|"
                "N2 2 5 0 3 7 4 6 1 N1|"
                "N3 3 6 4 0 5 1 7 2 N2|"

                "N6 6 2 0 7 5 3 1 4 N4|"
                "N7 7 3 2 6 0 4 1 5 N5|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N5 5 4 2 7 1 6 3 0 N0|"
                "N2 2 5 0 3 7 4 6 1 N1|"
                "N3 3 6 4 0 5 1 7 2 N2|"
                "N4 4 7 0 6 5 2 1 3 N3|"

                "N7 7 3 2 6 0 4 1 5 N5|"
                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N5 5 4 2 7 1 6 3 0 N0|"
                "N2 2 5 0 3 7 4 6 1 N1|"
                "N3 3 6 4 0 5 1 7 2 N2|"
                "N4 4 7 0 6 5 2 1 3 N3|"
                "N6 6 2 0 7 5 3 1 4 N4|"

                "N1 1 0 2 4 3 5 7 6 N6|"
                "N0 0 1 2 3 4 5 6 7 N7|"
                "N5 5 4 2 7 1 6 3 0 N0|"
                "N2 2 5 0 3 7 4 6 1 N1|"
                "N3 3 6 4 0 5 1 7 2 N2|"
                "N4 4 7 0 6 5 2 1 3 N3|"
                "N6 6 2 0 7 5 3 1 4 N4|"
                "N7 7 3 2 6 0 4 1 5 N5|"

                "N0 0 1 2 3 4 5 6 7 N7|"
                "N5 5 4 2 7 1 6 3 0 N0|"
                "N2 2 5 0 3 7 4 6 1 N1|"
                "N3 3 6 4 0 5 1 7 2 N2|"
                "N4 4 7 0 6 5 2 1 3 N3|"
                "N6 6 2 0 7 5 3 1 4 N4|"
                "N7 7 3 2 6 0 4 1 5 N5|"
                "N1 1 0 2 4 3 5 7 6 N6|",

  .options    = "noCpuCheck=1,tuning=5,ll128Enabled=1,disableNumaMatching=1",

  .treeRail   = "N0 0 1 2 4 3 6 5 7 N1|"
                "N1 1 0 4 7 3 5 2 6 N0|"
                "N2 2 3 0 5 6 1 7 4 N3|"
                "N3 3 2 7 0 6 4 1 5 N2|"
                "N4 4 5 1 6 0 3 7 2 N5|"
                "N5 5 4 6 2 0 7 1 3 N4|"
                "N6 6 7 5 3 4 0 2 1 N7|"
                "N7 7 6 3 1 4 2 5 0 N6|"

                "N2 2 3 0 5 6 1 7 4 N3|"
                "N3 3 2 7 0 6 4 1 5 N2|"
                "N4 4 5 1 6 0 3 7 2 N5|"
                "N5 5 4 6 2 0 7 1 3 N4|"
                "N6 6 7 5 3 4 0 2 1 N7|"
                "N7 7 6 3 1 4 2 5 0 N6|"
                "N0 0 1 2 4 3 6 5 7 N1|"
                "N1 1 0 4 7 3 5 2 6 N0|"

                "N4 4 5 1 6 0 3 7 2 N5|"
                "N5 5 4 6 2 0 7 1 3 N4|"
                "N6 6 7 5 3 4 0 2 1 N7|"
                "N7 7 6 3 1 4 2 5 0 N6|"
                "N0 0 1 2 4 3 6 5 7 N1|"
                "N1 1 0 4 7 3 5 2 6 N0|"
                "N2 2 3 0 5 6 1 7 4 N3|"
                "N3 3 2 7 0 6 4 1 5 N2|"

                "N6 6 7 5 3 4 0 2 1 N7|"
                "N7 7 6 3 1 4 2 5 0 N6|"
                "N0 0 1 2 4 3 6 5 7 N1|"
                "N1 1 0 4 7 3 5 2 6 N0|"
                "N2 2 3 0 5 6 1 7 4 N3|"
                "N3 3 2 7 0 6 4 1 5 N2|"
                "N4 4 5 1 6 0 3 7 2 N5|"
                "N5 5 4 6 2 0 7 1 3 N4|"

                "N0 0 1 2 4 3 6 5 7 N1|"
                "N1 1 0 4 7 3 5 2 6 N0|"
                "N2 2 3 0 5 6 1 7 4 N3|"
                "N3 3 2 7 0 6 4 1 5 N2|"
                "N4 4 5 1 6 0 3 7 2 N5|"
                "N5 5 4 6 2 0 7 1 3 N4|"
                "N6 6 7 5 3 4 0 2 1 N7|"
                "N7 7 6 3 1 4 2 5 0 N6|"

                "N2 2 3 0 5 6 1 7 4 N3|"
                "N3 3 2 7 0 6 4 1 5 N2|"
                "N4 4 5 1 6 0 3 7 2 N5|"
                "N5 5 4 6 2 0 7 1 3 N4|"
                "N6 6 7 5 3 4 0 2 1 N7|"
                "N7 7 6 3 1 4 2 5 0 N6|"
                "N0 0 1 2 4 3 6 5 7 N1|"
                "N1 1 0 4 7 3 5 2 6 N0|"

                "N4 4 5 1 6 0 3 7 2 N5|"
                "N5 5 4 6 2 0 7 1 3 N4|"
                "N6 6 7 5 3 4 0 2 1 N7|"
                "N7 7 6 3 1 4 2 5 0 N6|"
                "N0 0 1 2 4 3 6 5 7 N1|"
                "N1 1 0 4 7 3 5 2 6 N0|"
                "N2 2 3 0 5 6 1 7 4 N3|"
                "N3 3 2 7 0 6 4 1 5 N2|"

                "N6 6 7 5 3 4 0 2 1 N7|"
                "N7 7 6 3 1 4 2 5 0 N6|"
                "N0 0 1 2 4 3 6 5 7 N1|"
                "N1 1 0 4 7 3 5 2 6 N0|"
                "N2 2 3 0 5 6 1 7 4 N3|"
                "N3 3 2 7 0 6 4 1 5 N2|"
                "N4 4 5 1 6 0 3 7 2 N5|"
                "N5 5 4 6 2 0 7 1 3 N4|",
};

static struct rcclRomeModel rome_model_84 = {
  .nGpus = 8, .nCpus = 2, .nNics = 16, .nLinks = 7,
  .gpuIds = { 0x11000, 0x2f000, 0x46000, 0x5d000, 0x8b000, 0xaa000, 0xc2000, 0xda000, },
  .nicIds = { 0xc000, 0xc000, 0x2a000, 0x2a000, 0x41000, 0x41000, 0x58000, 0x58000, 0x86000, 0x86000, 0xa5000, 0xa5000, 0xbd000, 0xbd000, 0xd5000, 0xd5000, },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .nicNuma = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, },
  .connMatrix = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, },
  .gdrLevel = { PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, },
  .pattern = "4848",
  .ringBase = "N0 0 1 2 3 4 5 6 7 N14|N2 1 0 2 4 3 5 7 6 N12|N4 2 5 0 3 7 1 6 4 N8|N6 3 6 1 5 2 7 4 0 N0|N8 4 7 0 6 5 1 3 2 N4|N10 5 4 6 3 0 7 2 1 N2|N12 6 2 0 4 1 7 5 3 N6|N14 7 3 1 4 2 6 0 5 N10|N1 0 1 2 3 4 5 6 7 N15|N3 1 0 2 4 3 5 7 6 N13|N5 2 5 0 3 7 1 6 4 N9|N7 3 6 1 5 2 7 4 0 N1|N9 4 7 0 6 5 1 3 2 N5|N11 5 4 6 3 0 7 2 1 N3|N13 6 2 0 4 1 7 5 3 N7|N15 7 3 1 4 2 6 0 5 N11",
  .options = "noCpuCheck=1,tuning=5",
};

static struct rcclRomeModel rome_model_85 = {
  .nGpus = 8, .nCpus = 2, .nNics = 4, .nLinks = 3,
  .gpuIds = { 0x32000, 0x35000, 0x11000, 0x14000, 0xae000, 0xb3000, 0x8e000, 0x93000, },
  .nicIds = { 0x2d000, 0x5000, 0xab000, 0x94000, },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .nicNuma = { 0, 0, 1, 1, },
  .connMatrix = { 0, 4, 1, 0, 0, 0, 2, 0, 4, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 4, 2, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 4, 1, 0, 0, 1, 0, 0, 4, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 1, 0, 1, 4, 0, },
  .gdrLevel = { PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PXB, },
  .pattern = "4242",
  .ringBase = "N0 0 1 3 2 4 5 7 6 N3|N1 2 3 1 0 6 7 5 4 N2|N3 7 6 0 1 5 4 2 3 N1|N0 1 0 6 7 3 2 4 5 N2|N2 4 5 7 6 0 1 3 2 N1|N3 6 7 5 4 2 3 1 0 N0|N2 5 4 2 3 7 6 0 1 N0|N1 3 2 4 5 1 0 6 7 N3",
  .options = "tuning=2",
};

static struct rcclRomeModel rome_model_87 = {
  .nGpus = 8, .nCpus = 2, .nNics = 4, .nLinks = 7,
  .gpuIds = { 0xa000, 0x80000, 0xa4000, 0xc8000, 0x10b000, 0x181000, 0x1a5000, 0x1c9000, },
  .nicIds = { 0xc9000, 0x1a2000, 0x108000, 0x81000, },
  .gpuNuma = { 0, 0, 0, 0, 1, 1, 1, 1, },
  .nicNuma = { 0, 1, 1, 0, },
  .connMatrix = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, },
  .gdrLevel = { PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, PATH_PXB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PHB, PATH_PXB, PATH_PHB, PATH_PHB, PATH_SYS, PATH_SYS, PATH_SYS, PATH_SYS, },
  .pattern = "4242",
  .ringBase = "N3 0 1 2 3 4 5 6 7 N1|"
              "N3 1 0 2 4 3 5 7 6 N1|"
              "N0 2 5 0 3 6 1 7 4 N2|"
              "N0 3 7 0 4 2 1 6 5 N2|"
              "N2 4 6 2 7 3 0 5 1 N3|"
              "N2 5 4 7 1 3 2 6 0 N3|"
              "N1 6 3 1 4 0 7 5 2 N0|"
              "N1 7 2 0 6 4 1 5 3 N0|"

              "N3 1 0 2 4 3 5 7 6 N1|"
              "N0 2 5 0 3 6 1 7 4 N2|"
              "N0 3 7 0 4 2 1 6 5 N2|"
              "N2 4 6 2 7 3 0 5 1 N3|"
              "N2 5 4 7 1 3 2 6 0 N3|"
              "N1 6 3 1 4 0 7 5 2 N0|"
              "N1 7 2 0 6 4 1 5 3 N0|"
              "N3 0 1 2 3 4 5 6 7 N1|"

              "N0 2 5 0 3 6 1 7 4 N2|"
              "N0 3 7 0 4 2 1 6 5 N2|"
              "N2 4 6 2 7 3 0 5 1 N3|"
              "N2 5 4 7 1 3 2 6 0 N3|"
              "N1 6 3 1 4 0 7 5 2 N0|"
              "N1 7 2 0 6 4 1 5 3 N0|"
              "N3 0 1 2 3 4 5 6 7 N1|"
              "N3 1 0 2 4 3 5 7 6 N1|"

              "N0 3 7 0 4 2 1 6 5 N2|"
              "N2 4 6 2 7 3 0 5 1 N3|"
              "N2 5 4 7 1 3 2 6 0 N3|"
              "N1 6 3 1 4 0 7 5 2 N0|"
              "N1 7 2 0 6 4 1 5 3 N0|"
              "N3 0 1 2 3 4 5 6 7 N1|"
              "N3 1 0 2 4 3 5 7 6 N1|"
              "N0 2 5 0 3 6 1 7 4 N2|"

              "N2 4 6 2 7 3 0 5 1 N3|"
              "N2 5 4 7 1 3 2 6 0 N3|"
              "N1 6 3 1 4 0 7 5 2 N0|"
              "N1 7 2 0 6 4 1 5 3 N0|"
              "N3 0 1 2 3 4 5 6 7 N1|"
              "N3 1 0 2 4 3 5 7 6 N1|"
              "N0 2 5 0 3 6 1 7 4 N2|"
              "N0 3 7 0 4 2 1 6 5 N2|"

              "N2 5 4 7 1 3 2 6 0 N3|"
              "N1 6 3 1 4 0 7 5 2 N0|"
              "N1 7 2 0 6 4 1 5 3 N0|"
              "N3 0 1 2 3 4 5 6 7 N1|"
              "N3 1 0 2 4 3 5 7 6 N1|"
              "N0 2 5 0 3 6 1 7 4 N2|"
              "N0 3 7 0 4 2 1 6 5 N2|"
              "N2 4 6 2 7 3 0 5 1 N3|"

              "N1 6 3 1 4 0 7 5 2 N0|"
              "N1 7 2 0 6 4 1 5 3 N0|"
              "N3 0 1 2 3 4 5 6 7 N1|"
              "N3 1 0 2 4 3 5 7 6 N1|"
              "N0 2 5 0 3 6 1 7 4 N2|"
              "N0 3 7 0 4 2 1 6 5 N2|"
              "N2 4 6 2 7 3 0 5 1 N3|"
              "N2 5 4 7 1 3 2 6 0 N3|"

              "N1 7 2 0 6 4 1 5 3 N0|"
              "N3 0 1 2 3 4 5 6 7 N1|"
              "N3 1 0 2 4 3 5 7 6 N1|"
              "N0 2 5 0 3 6 1 7 4 N2|"
              "N0 3 7 0 4 2 1 6 5 N2|"
              "N2 4 6 2 7 3 0 5 1 N3|"
              "N2 5 4 7 1 3 2 6 0 N3|"
              "N1 6 3 1 4 0 7 5 2 N0|",

  .ringTail2 = "N1 6 3 1 4 0 7 5 2 N0|"
               "N1 7 2 0 6 4 1 5 3 N0|"
               "N2 4 6 2 7 3 0 5 1 N3|"
               "N2 5 4 7 1 3 2 6 0 N3|"
               "N3 0 1 2 3 4 5 6 7 N1|"
               "N3 1 0 2 4 3 5 7 6 N1|"
               "N0 2 5 0 3 6 1 7 4 N2|"
               "N0 3 7 0 4 2 1 6 5 N2|"

               "N1 7 2 0 6 4 1 5 3 N0|"
               "N2 4 6 2 7 3 0 5 1 N3|"
               "N2 5 4 7 1 3 2 6 0 N3|"
               "N3 0 1 2 3 4 5 6 7 N1|"
               "N3 1 0 2 4 3 5 7 6 N1|"
               "N0 2 5 0 3 6 1 7 4 N2|"
               "N0 3 7 0 4 2 1 6 5 N2|"
               "N1 6 3 1 4 0 7 5 2 N0|"

               "N2 4 6 2 7 3 0 5 1 N3|"
               "N2 5 4 7 1 3 2 6 0 N3|"
               "N3 0 1 2 3 4 5 6 7 N1|"
               "N3 1 0 2 4 3 5 7 6 N1|"
               "N0 2 5 0 3 6 1 7 4 N2|"
               "N0 3 7 0 4 2 1 6 5 N2|"
               "N1 6 3 1 4 0 7 5 2 N0|"
               "N1 7 2 0 6 4 1 5 3 N0|"

               "N2 5 4 7 1 3 2 6 0 N3|"
               "N3 0 1 2 3 4 5 6 7 N1|"
               "N3 1 0 2 4 3 5 7 6 N1|"
               "N0 2 5 0 3 6 1 7 4 N2|"
               "N0 3 7 0 4 2 1 6 5 N2|"
               "N1 6 3 1 4 0 7 5 2 N0|"
               "N1 7 2 0 6 4 1 5 3 N0|"
               "N2 4 6 2 7 3 0 5 1 N3|"

               "N3 0 1 2 3 4 5 6 7 N1|"
               "N3 1 0 2 4 3 5 7 6 N1|"
               "N0 2 5 0 3 6 1 7 4 N2|"
               "N0 3 7 0 4 2 1 6 5 N2|"
               "N1 6 3 1 4 0 7 5 2 N0|"
               "N1 7 2 0 6 4 1 5 3 N0|"
               "N2 4 6 2 7 3 0 5 1 N3|"
               "N2 5 4 7 1 3 2 6 0 N3|"

               "N3 1 0 2 4 3 5 7 6 N1|"
               "N0 2 5 0 3 6 1 7 4 N2|"
               "N0 3 7 0 4 2 1 6 5 N2|"
               "N1 6 3 1 4 0 7 5 2 N0|"
               "N1 7 2 0 6 4 1 5 3 N0|"
               "N2 4 6 2 7 3 0 5 1 N3|"
               "N2 5 4 7 1 3 2 6 0 N3|"
               "N3 0 1 2 3 4 5 6 7 N1|"

               "N0 2 5 0 3 6 1 7 4 N2|"
               "N0 3 7 0 4 2 1 6 5 N2|"
               "N1 6 3 1 4 0 7 5 2 N0|"
               "N1 7 2 0 6 4 1 5 3 N0|"
               "N2 4 6 2 7 3 0 5 1 N3|"
               "N2 5 4 7 1 3 2 6 0 N3|"
               "N3 0 1 2 3 4 5 6 7 N1|"
               "N3 1 0 2 4 3 5 7 6 N1|"

               "N0 3 7 0 4 2 1 6 5 N2|"
               "N1 6 3 1 4 0 7 5 2 N0|"
               "N1 7 2 0 6 4 1 5 3 N0|"
               "N2 4 6 2 7 3 0 5 1 N3|"
               "N2 5 4 7 1 3 2 6 0 N3|"
               "N3 0 1 2 3 4 5 6 7 N1|"
               "N3 1 0 2 4 3 5 7 6 N1|"
               "N0 2 5 0 3 6 1 7 4 N2|",

  .ringTail1 = "N0 2 1 3 4 5 6 7 0 N3|"
               "N0 3 0 2 4 6 5 7 1 N3|"
               "N3 0 1 4 7 5 3 6 2 N0|"
               "N3 1 0 5 2 7 6 4 3 N0|"
               "N1 6 1 2 3 5 0 7 4 N2|"
               "N1 7 3 2 6 0 4 1 5 N2|"
               "N2 4 0 3 7 2 5 1 6 N1|"
               "N2 5 4 2 0 6 3 1 7 N1|"

               "N0 3 0 2 4 6 5 7 1 N3|"
               "N3 0 1 4 7 5 3 6 2 N0|"
               "N3 1 0 5 2 7 6 4 3 N0|"
               "N1 6 1 2 3 5 0 7 4 N2|"
               "N1 7 3 2 6 0 4 1 5 N2|"
               "N2 4 0 3 7 2 5 1 6 N1|"
               "N2 5 4 2 0 6 3 1 7 N1|"
               "N0 2 1 3 4 5 6 7 0 N3|"

               "N3 0 1 4 7 5 3 6 2 N0|"
               "N3 1 0 5 2 7 6 4 3 N0|"
               "N1 6 1 2 3 5 0 7 4 N2|"
               "N1 7 3 2 6 0 4 1 5 N2|"
               "N2 4 0 3 7 2 5 1 6 N1|"
               "N2 5 4 2 0 6 3 1 7 N1|"
               "N0 2 1 3 4 5 6 7 0 N3|"
               "N0 3 0 2 4 6 5 7 1 N3|"

               "N3 1 0 5 2 7 6 4 3 N0|"
               "N1 6 1 2 3 5 0 7 4 N2|"
               "N1 7 3 2 6 0 4 1 5 N2|"
               "N2 4 0 3 7 2 5 1 6 N1|"
               "N2 5 4 2 0 6 3 1 7 N1|"
               "N0 2 1 3 4 5 6 7 0 N3|"
               "N0 3 0 2 4 6 5 7 1 N3|"
               "N3 0 1 4 7 5 3 6 2 N0|"

               "N1 6 1 2 3 5 0 7 4 N2|"
               "N1 7 3 2 6 0 4 1 5 N2|"
               "N2 4 0 3 7 2 5 1 6 N1|"
               "N2 5 4 2 0 6 3 1 7 N1|"
               "N0 2 1 3 4 5 6 7 0 N3|"
               "N0 3 0 2 4 6 5 7 1 N3|"
               "N3 0 1 4 7 5 3 6 2 N0|"
               "N3 1 0 5 2 7 6 4 3 N0|"

               "N1 7 3 2 6 0 4 1 5 N2|"
               "N2 4 0 3 7 2 5 1 6 N1|"
               "N2 5 4 2 0 6 3 1 7 N1|"
               "N0 2 1 3 4 5 6 7 0 N3|"
               "N0 3 0 2 4 6 5 7 1 N3|"
               "N3 0 1 4 7 5 3 6 2 N0|"
               "N3 1 0 5 2 7 6 4 3 N0|"
               "N1 6 1 2 3 5 0 7 4 N2|"

               "N2 4 0 3 7 2 5 1 6 N1|"
               "N2 5 4 2 0 6 3 1 7 N1|"
               "N0 2 1 3 4 5 6 7 0 N3|"
               "N0 3 0 2 4 6 5 7 1 N3|"
               "N3 0 1 4 7 5 3 6 2 N0|"
               "N3 1 0 5 2 7 6 4 3 N0|"
               "N1 6 1 2 3 5 0 7 4 N2|"
               "N1 7 3 2 6 0 4 1 5 N2|"

               "N2 5 4 2 0 6 3 1 7 N1|"
               "N0 2 1 3 4 5 6 7 0 N3|"
               "N0 3 0 2 4 6 5 7 1 N3|"
               "N3 0 1 4 7 5 3 6 2 N0|"
               "N3 1 0 5 2 7 6 4 3 N0|"
               "N1 6 1 2 3 5 0 7 4 N2|"
               "N1 7 3 2 6 0 4 1 5 N2|"
               "N2 4 0 3 7 2 5 1 6 N1|",

  .options = "noCpuCheck=1,netOverride=1,tuning=5,ll128Enabled=1",

  .treeRail =  "N3 0 1 2 4 3 6 5 7 N1|"
               "N3 1 0 4 7 3 5 2 6 N1|"
               "N0 2 3 0 5 6 1 7 4 N2|"
               "N0 3 2 7 0 6 4 1 5 N2|"
               "N2 4 5 1 6 0 3 7 2 N3|"
               "N2 5 4 6 2 0 7 1 3 N3|"
               "N1 6 7 5 3 4 0 2 1 N0|"
               "N1 7 6 3 1 4 2 5 0 N0|"

               "N0 2 3 0 5 6 1 7 4 N2|"
               "N0 3 2 7 0 6 4 1 5 N2|"
               "N2 4 5 1 6 0 3 7 2 N3|"
               "N2 5 4 6 2 0 7 1 3 N3|"
               "N1 6 7 5 3 4 0 2 1 N0|"
               "N1 7 6 3 1 4 2 5 0 N0|"
               "N3 0 1 2 4 3 6 5 7 N1|"
               "N3 1 0 4 7 3 5 2 6 N1|"

               "N2 4 5 1 6 0 3 7 2 N3|"
               "N2 5 4 6 2 0 7 1 3 N3|"
               "N1 6 7 5 3 4 0 2 1 N0|"
               "N1 7 6 3 1 4 2 5 0 N0|"
               "N3 0 1 2 4 3 6 5 7 N1|"
               "N3 1 0 4 7 3 5 2 6 N1|"
               "N0 2 3 0 5 6 1 7 4 N2|"
               "N0 3 2 7 0 6 4 1 5 N2|"

               "N1 6 7 5 3 4 0 2 1 N0|"
               "N1 7 6 3 1 4 2 5 0 N0|"
               "N3 0 1 2 4 3 6 5 7 N1|"
               "N3 1 0 4 7 3 5 2 6 N1|"
               "N0 2 3 0 5 6 1 7 4 N2|"
               "N0 3 2 7 0 6 4 1 5 N2|"
               "N2 4 5 1 6 0 3 7 2 N3|"
               "N2 5 4 6 2 0 7 1 3 N3|"

               "N3 0 1 2 4 3 6 5 7 N1|"
               "N3 1 0 4 7 3 5 2 6 N1|"
               "N0 2 3 0 5 6 1 7 4 N2|"
               "N0 3 2 7 0 6 4 1 5 N2|"
               "N2 4 5 1 6 0 3 7 2 N3|"
               "N2 5 4 6 2 0 7 1 3 N3|"
               "N1 6 7 5 3 4 0 2 1 N0|"
               "N1 7 6 3 1 4 2 5 0 N0|"

               "N0 2 3 0 5 6 1 7 4 N2|"
               "N0 3 2 7 0 6 4 1 5 N2|"
               "N2 4 5 1 6 0 3 7 2 N3|"
               "N2 5 4 6 2 0 7 1 3 N3|"
               "N1 6 7 5 3 4 0 2 1 N0|"
               "N1 7 6 3 1 4 2 5 0 N0|"
               "N3 0 1 2 4 3 6 5 7 N1|"
               "N3 1 0 4 7 3 5 2 6 N1|"

               "N2 4 5 1 6 0 3 7 2 N3|"
               "N2 5 4 6 2 0 7 1 3 N3|"
               "N1 6 7 5 3 4 0 2 1 N0|"
               "N1 7 6 3 1 4 2 5 0 N0|"
               "N3 0 1 2 4 3 6 5 7 N1|"
               "N3 1 0 4 7 3 5 2 6 N1|"
               "N0 2 3 0 5 6 1 7 4 N2|"
               "N0 3 2 7 0 6 4 1 5 N2|"

               "N1 6 7 5 3 4 0 2 1 N0|"
               "N1 7 6 3 1 4 2 5 0 N0|"
               "N3 0 1 2 4 3 6 5 7 N1|"
               "N3 1 0 4 7 3 5 2 6 N1|"
               "N0 2 3 0 5 6 1 7 4 N2|"
               "N0 3 2 7 0 6 4 1 5 N2|"
               "N2 4 5 1 6 0 3 7 2 N3|"
               "N2 5 4 6 2 0 7 1 3 N3|",
};

static struct rcclRomeModel romeTopoModels[] = {
  rome_model_22, /*  0 */
  rome_model_25, /*  1 */
  rome_model_27, /*  2 */
  rome_model_29, /*  3 */
  rome_model_31, /*  4 */
  rome_model_33, /*  5 */
  rome_model_30, /*  6 */
  rome_model_32, /*  7 */
  rome_model_24, /*  8 */
  rome_model_26, /*  9 */
  rome_model_23, /* 10 */
  rome_model_38, /* 11 */
  rome_model_28, /* 12 */
  rome_model_40, /* 13 */
  rome_model_42, /* 14 */
  rome_model_44, /* 15 */
  rome_model_45, /* 16 */
  rome_model_46, /* 17 */
  rome_model_48, /* 18 */
  rome_model_49, /* 19 */
  rome_model_52, /* 20 */
  rome_model_53, /* 21 */
  rome_model_43, /* 22 */
  rome_model_55, /* 23 */
  rome_model_56, /* 24 */
  rome_model_58, /* 25 */
  rome_model_59, /* 26 */
  rome_model_62, /* 27 */
  rome_model_63, /* 28 */
  rome_model_65, /* 29 */
  rome_model_66, /* 30 */
  rome_model_67, /* 31 */
  rome_model_68, /* 32 */
  rome_model_71, /* 33 */
  rome_model_72, /* 34 */
  rome_model_73, /* 35 */
  rome_model_74, /* 36 */
  rome_model_76, /* 37 */
  rome_model_79, /* 38 */
  rome_model_80, /* 39 */
  rome_model_81, /* 40 */
  rome_model_84, /* 41 */
  rome_model_85, /* 42 */
  rome_model_87, /* 43 */
};

// This environment variable allows disabling of the reversal the graph parsing
RCCL_PARAM(ModelReversalDisable, "MODEL_REVERSAL_DISABLE", 0);

/* Parse user defined rings. Format is like :
 * "0 1|1 0|0 1 2 3|3 2 1 0|N0 0 2 3 1 N1|1 3 2 0|0 1 2 3 4 5 6 7|N2 7 6 5 4 3 2 1 0 N1"
 * GPUs are numberd from 0 to ngpus, represening GPU-index for the current communicator
 * gpu_map translates between GPU-index and rocm-smi index
 * Network interfaces can be optionally specified by N prefix.
 * Rings with a non-matching number of gpus are ignored so we can provide
 * rings for multiple cases.
 */
ncclResult_t parseGraph(const char* str, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* gpu_map, int* net_map, int reverse) {
  int gpus[NCCL_TOPO_MAX_NODES];
  int nChannels = 0;
  int gpu = 0;
  int offset = 0;
  int status = 0; // 0 : between numbers, 1 : inside number, 2: start NET, 3: inside NET
  int nets[NCCL_TOPO_MAX_NODES*2];
  int net_offset = 0, net_count = 0;
  int ngpus = system->nodes[GPU].count;
  int nnets = system->nodes[NET].count;

  if (rcclParamModelReversalDisable()) reverse = 0;

  do {
    if (str[offset] == 'N') {
      if (status == 0) {
        status = 2;
      }
    } else {
      int digit = str[offset] - '0';
      if (digit >= 0 && digit <= 9) {
        switch (status) {
          case 0:
            gpus[gpu] = digit;
            status = 1;
            break;
          case 1:
            gpus[gpu] = gpus[gpu]*10+digit;
            break;
          case 2:
            nets[net_offset] = digit+'N';
            status = 3;
            break;
          case 3:
            nets[net_offset] = (nets[net_offset]-'N')*10+digit+'N';
            break;
        }
      } else {
        if (status == 1) {
          gpu++;
          net_offset = 2*gpu-1;
          if (gpu > NCCL_TOPO_MAX_NODES) goto end;
        } else if (status == 2 || status == 3) {
          net_offset++;
          net_count++;
          if (net_offset > ngpus*2) goto end;
        }
        status = 0;
        if (str[offset] == '|' || str[offset] == '\0') {
          // Ignore if ngpus doesn't match
          if (gpu != ngpus) goto newchannel;
          // Ignore if net_count is not 0 or odd number
          if (net_count && net_count%2) goto newchannel;

          for (int r=0; r<ngpus; r++) {
            int g = gpus[r];
            // Ignore if gpus are out of bounds
            if (g < 0 || g >= ngpus) goto newchannel;
            // Ignore if gpus are duplicate
            for (int i=0; i<r; i++)
              if (gpus[i] == g) goto newchannel;
            // find rocm-smi id if needed
            if (gpu_map) g = gpu_map[g];
            // Translate gpu numbers into ranks
            int j = 0;
            for (j = 0; j < ngpus; j++)
              if (g == system->nodes[GPU].nodes[j].gpu.dev)
                break;
            if (j < ngpus) {
              int idx = (nChannels*ngpus) + (reverse ? ngpus - 1 - r : r);
              graph->intra[idx] = system->nodes[GPU].nodes[j].gpu.rank;
            }
            else
              return ncclInternalError;
          }

          if (net_count) {
            for (int i = 0; net_map && i < ngpus*2; i++) {
              if (nets[i]-'N' < 0 || nets[i]-'N' >= nnets) continue;
              nets[i] = net_map[nets[i]-'N']+'N';
            }
            // Swap input/output NICs if reversed
            if (reverse) {
              int t = nets[0];
              nets[0] = nets[ngpus*2-1];
              nets[ngpus*2-1] = t;
            }
            memcpy(&graph->intraNets[ngpus*nChannels*2], nets, ngpus*2*sizeof(int));
            graph->nIntraChannels++;
            if (nets[0]-'N' >= nnets || nets[ngpus*2-1]-'N' >= nnets) goto newchannel;
            graph->inter[nChannels*2] = nets[0]-'N';
            graph->inter[nChannels*2+1] = nets[ngpus*2-1]-'N';
          } else if (nnets) {
            graph->inter[nChannels*2] = system->nodes[NET].nodes[nChannels%nnets].id;
            graph->inter[nChannels*2+1] = system->nodes[NET].nodes[(nChannels+1)%nnets].id;
          }
          nChannels++;
newchannel:
          gpu = 0;
          net_offset = 0;
          net_count = 0;
        }
      }
    }
  } while (str[offset++] != 0);
end:
  graph->nChannels = nChannels;
  graph->bwIntra = graph->bwInter = system->totalBw/nChannels;
  if (graph->id == 1) {
    for (int i=0; i<graph->nChannels; i++) {
      int64_t netId;
      ncclTopoGetLocalNet(system, graph->intra[i*ngpus+1], i, &netId, nullptr);
      graph->inter[i*2+1] = netId;
    }
  }

#if 0
  for (int i=0; i<graph->nChannels; i++) {
    printf("Channel %02d: ", i);
    printf ("NET/%02d ", graph->inter[i*2]);
    for (int j=0; j<ngpus; j++) printf("GPU/%02d ", graph->intra[i*ngpus+j]);
    printf ("NET/%d ", graph->inter[i*2+1]);
    printf("\n");
  }
#endif
  return ncclSuccess;
}


/* Parse user defined treeBase for complicated trees. Format is like :
 * "(4(2(3)(1))(6(5)))"
 *
 * Rings with a non-matching number of gpus are ignored so we can provide
 * rings for multiple cases.
 */
ncclResult_t parseGraphLight(const char* str, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* gpu_map) {
  int gpus[NCCL_TOPO_MAX_NODES]; //transcribe/change according to gpu_map
  int gpu = 0;
  int offset = 0;
  int start_offset = offset;
  if (str[0] == 0) {
    graph->treeBase[0][0] = 0;
    return ncclSuccess;
  }
  int status = 0; // 0 : between numbers, 1 : inside number
  int ngpus = system->nodes[GPU].count;
  int x=0;
  do {
    int digit = str[offset] - '0';
    if (digit >= 0 && digit <= 9) {
      switch (status) {
        case 0:
          gpus[gpu] = digit;
          status = 1;
          break;
        case 1:
          gpus[gpu] = gpus[gpu]*10+digit;
          break;
      }
    } else {
      if (status == 1) {
        gpu++;
      }
      status = 0;
      if (str[offset] == '|' || str[offset] == 0) {
        int r = 0, y = 0;
        while(start_offset < offset) {
        // for (int r=0; r<gpu; r++) {
          if (str[start_offset] == '(' || str[start_offset] == ')') {
            graph->treeBase[x][y] = str[start_offset];
            y++;
            start_offset++;
          }
          else {
            int g = gpus[r];
            // remap if needed
            if (gpu_map) g = gpu_map[g];
            r++;
            int j = 0;
            // Translate gpu numbers into ranks
            for (j = 0; j < ngpus; j++)
              if (g == system->nodes[GPU].nodes[j].gpu.dev)
                break;
            if (j < ngpus)
            {
              while (str[start_offset] != '(' && str[start_offset] != ')') start_offset++;
              char number_str[10];
              sprintf(number_str, "%d", g);
              int k=0;
              while (number_str[k] != 0) {
                graph->treeBase[x][y]=number_str[k];
                y++;
                k++;
              }
            }
            else
              return ncclInternalError;
          }

        }
        graph->treeBase[x][y] = 0;
        x++;
        gpu=0;
        start_offset = offset + 1;
      }
    }
  } while (str[offset++] != 0);
  graph->treeBase[x][0] = 0;
  return ncclSuccess;
}



#define MAX_OPT_TOKENS 10
extern const char* topoPathTypeStr[];

static void parseOptions(struct ncclTopoSystem* system, const char *options) {
  if (strcmp(options, "")) {
    char *str_temp = (char *)malloc(strlen(options) + 1);
    strcpy(str_temp, options);
    char* tokens[MAX_OPT_TOKENS];
    int numTokens = 0;
    char* state;
    tokens[numTokens] = strtok_r(str_temp, "=, ", &state);
    numTokens++;
    while (tokens[numTokens-1] != NULL && numTokens < MAX_OPT_TOKENS)
        tokens[numTokens++] = strtok_r(NULL, "=, ", &state);
    for (int i = 0; i < numTokens/2; i++) {
      if (strcmp(tokens[i*2], "netGdrLevel") == 0) {
        int j;
        for (j = 0; j <= PATH_SYS; j++) {
          if (strcmp(tokens[i*2+1], topoPathTypeStr[j]) == 0)
            break;
        }
        if (j <= PATH_SYS)
          system->netGdrLevel = j;
        else {
          system->netGdrLevel = -2;
          WARN("invalid netGdrLevel: %s", tokens[i*2+1]);
        }
      } else if (strcmp(tokens[i*2], "pivotA2AEnabled") == 0) {
        system->pivotA2AEnabled = (bool)atol(tokens[i*2+1]);
      } else if (strcmp(tokens[i*2], "pivotA2ANumBiRings") == 0) {
        system->pivotA2ANumBiRings = atol(tokens[i*2+1]);
      } else if (strcmp(tokens[i*2], "tuning") == 0) {
        system->tuning = atol(tokens[i*2+1]);
      } else if (strcmp(tokens[i*2], "ll128Enabled") == 0) {
        system->ll128Enabled = (bool)atol(tokens[i*2+1]);
      } else if (strcmp(tokens[i*2], "baseBw") == 0) {
        system->baseBw = std::stof(tokens[i*2+1]);
      } else if (strcmp(tokens[i*2], "mscclEnabled") == 0) {
        system->mscclEnabled = (bool)atol(tokens[i*2+1]);
      } else if (strcmp(tokens[i*2], "treeDefined") == 0) {
        system->treeDefined = (bool)atol(tokens[i*2+1]);
      }
    }
    free(str_temp);
  }
}

static bool checkOption(const char *options, const char *name) {
  if (strcmp(options, "")) {
    char *str_temp = (char *)malloc(strlen(options) + 1);
    strcpy(str_temp, options);
    char* tokens[MAX_OPT_TOKENS];
    int numTokens = 0;
    char* state;
    tokens[numTokens] = strtok_r(str_temp, "=, ", &state);
    numTokens++;
    while (tokens[numTokens-1] != NULL && numTokens < MAX_OPT_TOKENS)
        tokens[numTokens++] = strtok_r(NULL, "=, ", &state);

    bool result = false;
    for (int i = 0; i < numTokens/2; i++) {
      if (strcmp(tokens[i*2], name) == 0) {
        result = (bool)atol(tokens[i*2+1]);
        break;
      }
    }
    free(str_temp);
    return result;
  }
  return false;
}

ncclResult_t parseChordalRing(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  static const char *ringBase = "0 1 2 3 5 4 7 6|0 2 4 1 7 3 6 5|0 3 1 5 7 2 6 4|0 6 7 4 5 3 2 1|0 5 6 3 7 1 4 2|0 4 6 2 7 5 1 3";
  int id[8], dist[8];
  int i;

  int ngpus = system->nodes[GPU].count;
  if (ngpus != 8)
    return ncclSuccess;
  // validate chordal ring and calculate distance
  for (i=0; i<ngpus; i++) {
    struct ncclTopoNode* node = system->nodes[GPU].nodes+i;
    if (node->paths[GPU] == NULL) continue;
    int sum = ngpus*(ngpus-1)/2 - node->gpu.dev;
    int count = 0;
    for (int n = 0; n<ngpus; n++) {
      struct ncclTopoLink* link;
      for (link = node->links; link->remNode; link++) {
        if (link->remNode->gpu.dev == n) break;
      }
      if (!link->remNode) continue;
      if (link->type != LINK_NVL) continue;
      sum -= system->nodes[GPU].nodes[n].gpu.dev;
      count ++;
    }
    if(count != ngpus-2 || sum < 0 || sum > ngpus-1) {
      return ncclSuccess;
    }
    dist[i] = sum;
  }
  // remap GPU ids
  for (i = 0; i<ngpus; i++) id[i] = i;
  for (i = 0; i<ngpus; i++) {
    if (dist[i] == ngpus-1-i) continue;
    int j, m, n, temp;
    for (j=i+1; j < ngpus; j++)
      if(dist[j] == ngpus-1-i) break;
    m = dist[i]; n = dist[j]; dist[i] = n; dist[j] = m;
    temp = id[m]; id[m] = id[n]; id[n] = temp; temp =dist[m];
    dist[m] = dist[n]; dist[n] = temp;
  }
  // create chordal ring based on reference and remapped ids
  system->type |= RCCL_TOPO_CR8G;
  NCCLCHECK(parseGraph(ringBase, system, graph, id, NULL, false));
  if (system->nodes[NET].count && system->nodes[GPU].count != system->nRanks) {
    int *intra, *used;
    graph->nChannels = system->nodes[NET].count;
    NCCLCHECK(ncclCalloc(&intra, ngpus));
    NCCLCHECK(ncclCalloc(&used,system->nodes[NET].count));
    for (int n = 0; n < system->nodes[NET].count; n++) {
      graph->inter[n*2] = graph->inter[n*2+1] = n;
      struct ncclTopoNode* net = system->nodes[NET].nodes+n;
      struct ncclTopoLinkList* paths = net->paths[GPU];
      // find the first unsed GPU that is closest to NIC
      int f, m;
      for (f = 0; f < ngpus; f++) {
        int j = 0; for (j = 0; j < n; j++) if(used[j] == system->nodes[GPU].nodes[f].gpu.rank) break;
        if(j >= n) break;
      }
      for (int i = 0; i < ngpus; i++) {
        int j = 0; for (j = 0; j < n; j++) if(used[j] == system->nodes[GPU].nodes[i].gpu.rank) break;
        if (j < n) continue;
        if (paths[i].count < paths[f].count) f = i;
      }
      for (m = 0; m<ngpus; m++) if (graph->intra[n*ngpus+m] == system->nodes[GPU].nodes[f].gpu.rank) break;
      used[n] = graph->intra[n*ngpus+m];
      for (int i = 0; i < ngpus; i++) intra[i] = graph->intra[n*ngpus+((i+m)%ngpus)];
      for (int i = 0; i < ngpus; i++) graph->intra[n*ngpus+i] = intra[i];
    }
    free(used);
    free(intra);
  }
  return ncclSuccess;
}


static ncclResult_t parseRomeSystem(struct ncclTopoSystem* system, struct rcclRomeModel* romeTopo,
                                    char *pattern, int *devids) {
  pattern[0] = 0; // pattern will be NULL for invalid topology
  romeTopo->nGpus = system->nodes[GPU].count;
  romeTopo->nCpus = system->nodes[CPU].count;
  romeTopo->nNics = system->nodes[NET].count;
  romeTopo->nLinks = 0;

  struct ncclGpuIdHIP {
    int g;
    int dev;
  };

  auto cmpIds = [](const void * g1, const void * g2) {
    struct ncclGpuIdHIP *s1 = (struct ncclGpuIdHIP*)g1;
    struct ncclGpuIdHIP *s2 = (struct ncclGpuIdHIP*)g2;
    return s1->dev - s2->dev;
  };

  struct ncclCpuNuma {
    int c;
    uint64_t numa;
  };

  auto cmpNuma = [](const void * g1, const void * g2) {
    struct ncclCpuNuma *s1 = (struct ncclCpuNuma*)g1;
    struct ncclCpuNuma *s2 = (struct ncclCpuNuma*)g2;
    return (int)(s1->numa - s2->numa);
  };

  struct ncclNetId {
    int n;
    uint64_t id;
  };

  auto cmpNets = [](const void * g1, const void * g2) {
    struct ncclNetId *s1 = (struct ncclNetId*)g1;
    struct ncclNetId *s2 = (struct ncclNetId*)g2;
    return (int)(s1->id - s2->id);
  };

  // sort GPU devices by HIP device ID
  struct ncclGpuIdHIP gpu_scores[NCCL_TOPO_MAX_NODES];
  for (int i = 0; i < romeTopo->nGpus; i ++) {
    gpu_scores[i].g = i;
    gpu_scores[i].dev = system->nodes[GPU].nodes[i].gpu.dev;
  }
  qsort(gpu_scores, romeTopo->nGpus, sizeof(struct ncclGpuIdHIP), cmpIds);
  // sort CPU devices by NUMA id
  struct ncclCpuNuma cpu_scores[NCCL_TOPO_MAX_NODES];
  for (int i = 0; i < romeTopo->nCpus; i ++) {
    cpu_scores[i].c = i;
    cpu_scores[i].numa = system->nodes[CPU].nodes[i].id;
  }
  qsort(cpu_scores, romeTopo->nCpus, sizeof(struct ncclCpuNuma), cmpNuma);
  // sort NET devices by id
  struct ncclNetId net_scores[NCCL_TOPO_MAX_NODES];
  for (int i = 0; i < romeTopo->nNics; i ++) {
    net_scores[i].n = i;
    net_scores[i].id = system->nodes[NET].nodes[i].net.dev;
  }
  qsort(net_scores, romeTopo->nNics, sizeof(struct ncclNetId), cmpNets);

  for (int i = 0; i < romeTopo->nGpus; i ++) {
    int gpu, n, m, distance;
    gpu = gpu_scores[i].g;
    romeTopo->gpuIds[i] = system->nodes[GPU].nodes[gpu].id;
    m = 0;
    distance = system->nodes[GPU].nodes[gpu].paths[CPU][m].count;
    for (n = 1; n < romeTopo->nCpus; n++) {
      if (system->nodes[GPU].nodes[gpu].paths[CPU][n].count < distance) {
        distance = system->nodes[GPU].nodes[gpu].paths[CPU][n].count;
        m = n;
      }
    }
    if (m < romeTopo->nCpus) romeTopo->gpuNuma[i] = system->nodes[CPU].nodes[m].id;

    struct ncclTopoNode* node = system->nodes[GPU].nodes+gpu;
    if (node->paths[GPU] == NULL) continue;
    int count = 0;
    for (n = 0; n < romeTopo->nGpus; n++) {
      romeTopo->connMatrix[i*romeTopo->nGpus+n] = 0;
      struct ncclTopoLinkList *path = node->paths[GPU] + gpu_scores[n].g;
      if (path->type != LINK_NVL) continue;
      romeTopo->connMatrix[i*romeTopo->nGpus+n] = path->bw/ncclTopoXGMISpeed(node->gpu.gcn);
      count ++;
    }
    if (romeTopo->nLinks < count) romeTopo->nLinks = count;
  }

  for (int i = 0; i < romeTopo->nNics; i++) {
    int n, m, distance;
    m = 0;
    int net = net_scores[i].n;
    romeTopo->nicIds[i] = system->nodes[NET].nodes[net].net.busId;
    distance = system->nodes[NET].nodes[net].paths[CPU][m].count;
    for (n = 0; n < romeTopo->nCpus; n++)
      if (system->nodes[NET].nodes[net].paths[CPU][n].count < distance) {
        distance = system->nodes[NET].nodes[net].paths[CPU][n].count;
        m = n;
      }
    if (m < romeTopo->nCpus) romeTopo->nicNuma[i] = system->nodes[CPU].nodes[m].id;
    else return ncclSuccess;
  }

  // number of GPUs and NICs on each numa node is used as first screening pattern
  for (int i = 0; i < romeTopo->nCpus; i++) {
    uint64_t id = system->nodes[CPU].nodes[cpu_scores[i].c].id;
    int g = 0, n = 0;
    for (int j = 0; j < romeTopo->nGpus; j++)
      if (romeTopo->gpuNuma[j] == id) g++;
    for (int j = 0; j < romeTopo->nNics; j++)
      if (romeTopo->nicNuma[j] == id) n++;
    pattern[i*2] = '0' + g;
    pattern[i*2+1] = '0' + n;
  }
  pattern[romeTopo->nCpus*2] = 0;

  if (devids)
    for (int i = 0; i<romeTopo->nGpus; i++)
       devids[i] = gpu_scores[i].dev;

  // compute gdr level matrix
  for (int i = 0; i < romeTopo->nNics; i++) {
    int n = net_scores[i].n;
    for (int j = 0; j < romeTopo->nGpus; j++) {
      int g = gpu_scores[j].g;
      romeTopo->gdrLevel[i*romeTopo->nGpus+j] = system->nodes[GPU].nodes[g].paths[NET][n].type;
    }
  }

  const char* romeModelFile = getenv("RCCL_DUMP_ROME_MODEL_FILE");
  if (romeModelFile) {
    INFO(NCCL_ENV, "RCCL_DUMP_ROME_MODEL_FILE set by environment to %s", romeModelFile);
    FILE* file = fopen(romeModelFile, "w");
    if (file == NULL) {
      WARN("Unable to open %s, not dumping Rome model.", romeModelFile);
      return ncclSuccess;
    }
    fprintf(file, "static struct rcclRomeModel rome_model_ = {\n");
    fprintf(file, "  .nGpus = %d, .nCpus = %d, .nNics = %d, .nLinks = %d,\n", romeTopo->nGpus, romeTopo->nCpus, romeTopo->nNics, romeTopo->nLinks);
    fprintf(file, "  .gpuIds = { ");
    for (int i = 0; i < romeTopo->nGpus; i ++) fprintf(file, "0x%lx, ", romeTopo->gpuIds[i]);
    fprintf(file, "},\n");
    fprintf(file, "  .nicIds = { ");
    for (int i = 0; i < romeTopo->nNics; i ++) fprintf(file, "0x%lx, ", romeTopo->nicIds[i]);
    fprintf(file, "},\n");
    fprintf(file, "  .gpuNuma = { ");
    for (int i = 0; i < romeTopo->nGpus; i ++) fprintf(file, "%ld, ", romeTopo->gpuNuma[i]);
    fprintf(file, "},\n");
    fprintf(file, "  .nicNuma = { ");
    for (int i = 0; i < romeTopo->nNics; i ++) fprintf(file, "%ld, ", romeTopo->nicNuma[i]);
    fprintf(file, "},\n");
    fprintf(file, "  .connMatrix = { ");
    for (int i = 0; i < romeTopo->nGpus; i ++)
      for (int n = 0; n < romeTopo->nGpus; n++) fprintf(file, "%d, ", romeTopo->connMatrix[i*romeTopo->nGpus+n]);
    fprintf(file, "},\n");
    fprintf(file, "  .gdrLevel = { ");
    for (int i = 0; i < romeTopo->nNics; i ++)
      for (int n = 0; n < romeTopo->nGpus; n++) fprintf(file, "PATH_%s, ", topoPathTypeStr[romeTopo->gdrLevel[i*romeTopo->nGpus+n]]);
    fprintf(file, "},\n");
    fprintf(file, "  .pattern = \"%s\",\n", pattern);
    fprintf(file, "  .ringBase = \"\",\n");
    fprintf(file, "  .options = \"\",\n");
    fprintf(file, "};\n");
    fclose(file);
  }
  return ncclSuccess;
}

static bool permuteGpuIds(int *g, int n, int last, struct rcclRomeModel* ref, struct rcclRomeModel* topo, int* time, bool nbio, bool ignore_numa) {
  (*time) ++;
  if (n == last) {
    int i, j;
    // match GPU numa
    if (!ignore_numa) {
      for (i = 0; i < ref->nGpus; i++)
        if (ref->gpuNuma[i] != topo->gpuNuma[g[i]]) break;
      if (i < ref->nGpus) return false;
    }
    // match XGMI connection
    for (i = 0; i < ref->nGpus; i++) {
      for (j = 0; j < ref->nGpus; j++) {
        if (ref->connMatrix[i*ref->nGpus+j] != topo->connMatrix[g[i]*ref->nGpus+g[j]]) break;
        if ((ref->gpuIds[i]-ref->gpuIds[j])*(topo->gpuIds[g[i]]-topo->gpuIds[g[j]]) < 0) break;
      }
      if (j < ref->nGpus) break;
    }
    if (i < ref->nGpus) return false;
    // match NBIO
    if (nbio) {
      for (i = 0; i < ref->nGpus; i++) {
        for (j = 0; j < ref->nGpus; j++) {
          if (i == j) continue;
          bool nbio_ref = (ref->gpuIds[i]&0xf0000) == (ref->gpuIds[j]&0xf0000);
          bool nbio_topo = (topo->gpuIds[g[i]]&0xf0000) == (topo->gpuIds[g[j]]&0xf0000);
          if (nbio_ref != nbio_topo) break;
          if (nbio_ref && ((ref->gpuIds[i]-ref->gpuIds[j])*(topo->gpuIds[g[i]]-topo->gpuIds[g[j]]) < 0)) break;
        }
        if (j < ref->nGpus) break;
      }
      if (i < ref->nGpus) return false;
    }
    return true;
  } else {
    for (int i = n; i <= last; i++) {
      std::swap(g[n], g[i]);
      if (permuteGpuIds(g, n+1, last, ref, topo, time, nbio, ignore_numa)) return true;
      std::swap(g[n], g[i]);
    }
  }
  return false;
}

static bool permuteNetIds(int *n, int *g, int s, int last, struct rcclRomeModel* ref, struct rcclRomeModel* topo, int* time, bool ignore_numa) {
  (*time) ++;
  if (s == last) {
    int i, j;
    // match NET numa
    if (!ignore_numa) {
      for (i = 0; i < ref->nNics; i++) {
        if (ref->nicNuma[i] != topo->nicNuma[n[i]]) break;
      }
      if (i < ref->nNics) return false;
    }
    // match gdr level
    for (i = 0; i < ref->nNics; i++) {
      for (j = 0; j < ref->nGpus; j++) {
        // enabling PXN override paths over PHB and SYS
        if (topo->gdrLevel[n[i]*ref->nGpus+g[j]] == PATH_PXN) continue;
        // treat PIX and PXB as same
        if ((ref->gdrLevel[i*ref->nGpus+j] == PATH_PXB && topo->gdrLevel[n[i]*ref->nGpus+g[j]] == PATH_PIX) ||
          (ref->gdrLevel[i*ref->nGpus+j] == PATH_PIX && topo->gdrLevel[n[i]*ref->nGpus+g[j]] == PATH_PXB))
          continue;
        if (ref->gdrLevel[i*ref->nGpus+j] != topo->gdrLevel[n[i]*ref->nGpus+g[j]]) break;
      }
      if (j < ref->nGpus) break;
    }
    if (i < ref->nNics) return false;
    return true;
  } else {
    for (int i = s; i <= last; i++) {
      std::swap(n[s], n[i]);
      if (permuteNetIds(n, g, s+1, last, ref, topo, time, ignore_numa)) return true;
      std::swap(n[s], n[i]);
    }
  }
  return false;
}

int checkAlltoallWidth(struct rcclRomeModel *romeTopo) {
  int i, width = 0;
  for (i = 0; i < romeTopo->nGpus; i++) {
    int j;
    for (j = 0; j < romeTopo->nGpus; j++) {
      if (i == j) continue;
      if (romeTopo->connMatrix[i*romeTopo->nGpus+j] == 0)
        break;
      if (width == 0)
        width = romeTopo->connMatrix[i*romeTopo->nGpus+j];
      else if (width != romeTopo->connMatrix[i*romeTopo->nGpus+j])
        break;
    }
    if (j < romeTopo->nGpus) break;
  }
  if (i < romeTopo->nGpus)
    width = 0;
  return width;
}

RCCL_PARAM(DisableRailTrees, "DISABLE_RAIL_TREES", 0);

ncclResult_t parseA2a8P(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, const char *ringBase) {
  constexpr int NUMA_CPUS = 2;
  constexpr int NUMA_GPUS = 4;
  constexpr int NUMA_PERMUTE_COUNT = 24;
  constexpr int TOTAL_PERMUTE_COUNT = (NUMA_PERMUTE_COUNT*NUMA_PERMUTE_COUNT);

  int i;

  const int ngpus = system->nodes[GPU].count;
  const int ncpus = system->nodes[CPU].count;
  const int nnets = system->nodes[NET].count;

  // number of GPUs and NICs on each numa node is used as first screening pattern
  struct rcclRomeModel romeTopo;
  char pattern[256];
  int devids[NUMA_CPUS * NUMA_GPUS];
  NCCLCHECK(parseRomeSystem(system, &romeTopo, pattern, devids));

  // only match for system with 8 GPUs and 2 CPUs
  if (ngpus != 8 || ncpus != NUMA_CPUS) return ncclSuccess;

  // check if all GPUs are connected alltoall with equal width
  bool isAlltoall = checkAlltoallWidth(&romeTopo);
  if (!isAlltoall) return ncclSuccess;

  int *g8, n[NCCL_TOPO_MAX_NODES];
  int *all_gpu_permutations = (int *)malloc(TOTAL_PERMUTE_COUNT*NUMA_CPUS*NUMA_GPUS*sizeof(int));
  struct timeval tvs, tve;
  gettimeofday(&tvs, NULL);
  std::vector<int> r(ngpus), g(ngpus), rdm(ngpus);
  for (i = 0; i < sizeof(romeTopoModels)/sizeof(romeTopoModels[0]); i++) {
    if (romeTopo.nCpus != romeTopoModels[i].nCpus || romeTopo.nGpus != romeTopoModels[i].nGpus ||
      romeTopo.nNics != romeTopoModels[i].nNics || romeTopo.nLinks != romeTopoModels[i].nLinks) continue;
    if (strcmp(romeTopoModels[i].pattern, pattern)) continue;
    int j;
    int numa_gpu_permutations[NUMA_CPUS][NUMA_PERMUTE_COUNT][NUMA_GPUS];
    if (isAlltoall != checkAlltoallWidth(romeTopoModels+i))
      continue;
    // permute GPUs for each CPU NUMA nodes
    for (j = 0; j < ncpus; j++) {
      int ngpusPerNuma = 0, cnt = 0, npermute = 0;
      for (int k = 0; k < ngpus; k++) {
        if (romeTopoModels[i].gpuNuma[k] != j) continue;
        r[ngpusPerNuma++] = k;
      }
      if (ngpusPerNuma == 0) continue;
      if (ngpusPerNuma != NUMA_GPUS) break;
      // init GPU mapping
      for (int k = 0; k < ngpus; k++) {
        if (romeTopo.gpuNuma[k] != j) continue;
        g[(2+cnt++)%ngpusPerNuma] = k;
      }
      std::sort(g.begin(), g.begin()+ngpusPerNuma);
      do {
        for (int n = 0; n < ngpusPerNuma; n++)
          numa_gpu_permutations[j][npermute][n] = g[n];
        npermute++;
      } while (std::next_permutation(g.begin(), g.begin()+ngpusPerNuma));
      if (npermute != NUMA_PERMUTE_COUNT) break;
    }
    if (j < ncpus) continue;
    // permute GPUs for all CPU NUMA nodes
    for (int a = 0; a < NUMA_PERMUTE_COUNT; a++) {
      for (int b = 0; b < NUMA_PERMUTE_COUNT; b++) {
        uint64_t offset = a*NUMA_PERMUTE_COUNT+b;
        //offset = (offset+TOTAL_PERMUTE_COUNT/2)%TOTAL_PERMUTE_COUNT;
        offset *= (NUMA_CPUS*NUMA_GPUS);
        memcpy(all_gpu_permutations+offset, &numa_gpu_permutations[0][a][0], NUMA_GPUS*sizeof(int));
        memcpy(all_gpu_permutations+offset+NUMA_GPUS, &numa_gpu_permutations[1][b][0], NUMA_GPUS*sizeof(int));
      }
    }
    // match all GPUs' XGMI connection
    int p;
    for (p = 0; p < TOTAL_PERMUTE_COUNT; p++) {
      g8 = all_gpu_permutations+p*NUMA_CPUS*NUMA_GPUS;
      int k;
      for (k = 0; k < romeTopoModels[i].nGpus; k++) {
        int m;
        for (m = 0; m < romeTopoModels[i].nGpus; m++) {
          if (romeTopoModels[i].connMatrix[k*romeTopoModels[i].nGpus+m] != romeTopo.connMatrix[g8[k]*romeTopoModels[i].nGpus+g8[m]]) break;
        }
        if (m < romeTopoModels[i].nGpus) break;
      }
      if (k < romeTopoModels[i].nGpus) continue;
      //printf("found match %d: ", p); for (int n = 0; n < NUMA_CPUS*NUMA_GPUS; n++) printf("%d ", g8[n]); printf("\n");
      if (nnets > 1) {
        // permute NET IDs
        int time = 0;
        for (int m = 0; m < nnets; m++) n[m] = (m+2)%nnets;
        if (permuteNetIds(n, g8, 0, nnets-1, romeTopoModels+i, &romeTopo, &time, false)) break;
      } else break;
    }
    if (p < TOTAL_PERMUTE_COUNT) break;
  }
  gettimeofday(&tve, NULL);
  if (i >= sizeof(romeTopoModels)/sizeof(romeTopoModels[0])) {
    //printf("No solution in %.2fms\n", t);
    return ncclSuccess;
  }
  for (int m = 0; m<ngpus; m++) rdm[m] = devids[g8[m]];

  char line[1024];
  //sprintf(line, "Found matching Rome model index %d in %.2fms with GPU mapping: ", i, t);
  sprintf(line, "Found matching Rome model index %d with GPU mapping: ", i);
  int offset = strlen(line);
  for (int k = 0; k < ngpus; k++) {
    sprintf(line+offset, "%d ", rdm[k]);
    offset = strlen(line);
  }
  if (nnets > 1) {
    sprintf(line+offset, "NET mapping: ");
    offset = strlen(line);
    for (int k = 0; k < nnets; k++) {
      sprintf(line+offset, "%d ", n[k]);
      offset = strlen(line);
    }
  }
  INFO(NCCL_GRAPH, "%s", line);
  system->type |= RCCL_TOPO_4P2H_ROME;
  parseOptions(system, romeTopoModels[i].options);
  if(IsArchMatch(system->nodes[GPU].nodes[0].gpu.gcn, "gfx950")){
    system->tuning = 6;
  }
  // create 4P2H based on reference and remapped ids
  switch (graph->pattern) {
  case NCCL_TOPO_PATTERN_RING:
    // Attempt to use rail-optimized rings if they exist
    if (system->nHosts % 2 == 0) {
      // For even number of nodes, alternate forward/reverse on ringBase
      NCCLCHECK(parseGraph(ringBase != nullptr ? ringBase : romeTopoModels[i].ringBase, system, graph, rdm.data(), nnets > 1 ? n : NULL, system->hostIdx % 2));
    }
    else {
      // For odd number of nodes, check first to see if ringTail1 and ringTail2 are defined
      if (system->nHosts == 1 || romeTopoModels[i].ringTail1 == nullptr || romeTopoModels[i].ringTail2 == nullptr) {
        if (system->nHosts > 1)
          INFO(NCCL_GRAPH, "[WARN] Dropping back due to lack of support for odd-number of nodes for model index %d\n", i);
        NCCLCHECK(parseGraph(ringBase != nullptr ? ringBase : romeTopoModels[i].ringBase, system, graph, rdm.data(), nnets > 1 ? n : NULL, system->hostIdx % 2));
      }
      else
      {
        if (system->hostIdx == (system->nHosts - 1)) {
          NCCLCHECK(parseGraph(romeTopoModels[i].ringTail1, system, graph, rdm.data(), nnets > 1 ? n : NULL, 0));
        } else if (system->hostIdx == (system->nHosts - 2)) {
          NCCLCHECK(parseGraph(romeTopoModels[i].ringTail2, system, graph, rdm.data(), nnets > 1 ? n : NULL, 0));
        } else {
          NCCLCHECK(parseGraph(ringBase != nullptr ? ringBase : romeTopoModels[i].ringBase, system, graph, rdm.data(), nnets > 1 ? n : NULL, system->hostIdx % 2));
        }
      }
    }
    break;
  case NCCL_TOPO_PATTERN_BALANCED_TREE:

    // Check if rail-optimized trees have been defined
    system->useRailOptimizedTrees = false;
    if (romeTopoModels[i].treeRail != nullptr && !rcclParamDisableRailTrees()) {

      // If so, parse the lines in advanced
      // These lines will be modified appropriately during ncclTopoPostset
      NCCLCHECK(parseGraph(romeTopoModels[i].treeRail, system, graph, g8, nnets > 1 ? n : NULL, 0));
      if (graph->nChannels) {
        system->useRailOptimizedTrees = true;
        return ncclSuccess;
      }
    }

    // Fall back to looking for tree configuration from treeBase
    if (romeTopoModels[i].treeBase != nullptr) {
      NCCLCHECK(parseGraphLight(romeTopoModels[i].treeBase, system, graph, rdm.data()));
      if (graph->nChannels) return ncclSuccess;
    }

    // Fall back to tree from ringBase
    NCCLCHECK(parseGraph(ringBase != nullptr ? ringBase : romeTopoModels[i].ringBase, system, graph, rdm.data(), nnets > 1 ? n : NULL, 0));
    // Override GDR distance if requested
    if (checkOption(romeTopoModels[i].options, "netOverride")) {
      for (int i = 0; i < system->nodes[NET].count; i++) {
        for (int j = 0; j < system->nodes[GPU].count; j++) {
          if (system->nodes[GPU].nodes[j].paths[NET][i].type == PATH_PXB) {
            int k;
            for (k = 0; k < system->nodes[GPU].count; k++) {
              if (k != j &&
                system->nodes[GPU].nodes[k].gpu.dev/2 == system->nodes[GPU].nodes[j].gpu.dev/2)
                break;
            }
            if (k < system->nodes[GPU].count)
              system->nodes[GPU].nodes[k].paths[NET][i].type = PATH_PXB;
          }
        }
      }
    }
    break;
  }

  // clean up
  free(all_gpu_permutations);
  return ncclSuccess;
}

ncclResult_t parseRome4P2H(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, const char *ringBase) {
  int i;

  int ngpus = system->nodes[GPU].count;
  int nnets = system->nodes[NET].count;

  // Only support ring and tree graphs
  if (graph->pattern != NCCL_TOPO_PATTERN_RING &&
      graph->pattern != NCCL_TOPO_PATTERN_BALANCED_TREE)
    return ncclSuccess;

  if (ngpus > 8) return ncclSuccess;
  // only valid on Rome
  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));

  // number of GPUs and NICs on each numa node is used as first screening pattern
  struct rcclRomeModel romeTopo;
  char pattern[256];
  int devids[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(parseRomeSystem(system, &romeTopo, pattern, devids));

  // recognize system as Rome 4P2H even if no matching model
  if (ngpus > 4 && romeTopo.nLinks) system->type |= RCCL_TOPO_4P2H_ROME;

  // detect multiple NICs per GPU
  int nnetspergpu = (nnets%ngpus == 0) ? nnets/ngpus : 0;

  int g[NCCL_TOPO_MAX_NODES], n[NCCL_TOPO_MAX_NODES], rdm[NCCL_TOPO_MAX_NODES];
  int time = 0;
  struct timeval tvs, tve;
  gettimeofday(&tvs, NULL);

  // check if GPUs are directly connected to CPU
  bool match_nbio = true;
  for (i = 0; i < romeTopo.nGpus; i++) {
    int cpu, gpu;
    NCCLCHECK(ncclTopoIdToIndex(system, CPU,  romeTopo.gpuNuma[i], &cpu));
    NCCLCHECK(ncclTopoIdToIndex(system, GPU,  romeTopo.gpuIds[i], &gpu));
    if (system->nodes[GPU].nodes[gpu].paths[CPU][cpu].count > 2) break;
  }
  if (i < romeTopo.nGpus) match_nbio = false;

  for (i = 0; i < sizeof(romeTopoModels)/sizeof(romeTopoModels[0]); i++) {
    bool ignore_cpu = checkOption(romeTopoModels[i].options, "noCpuCheck");
    if (!ignore_cpu && (arch != NCCL_TOPO_CPU_ARCH_X86 || vendor != NCCL_TOPO_CPU_VENDOR_AMD || model != NCCL_TOPO_CPU_MODEL_AMD_ROME))
      continue;
    bool ignore_numa = checkOption(romeTopoModels[i].options, "disableNumaMatching");
    if (!ignore_numa && romeTopo.nCpus != romeTopoModels[i].nCpus) continue;
    if (romeTopo.nGpus != romeTopoModels[i].nGpus ||
      romeTopo.nNics != romeTopoModels[i].nNics || romeTopo.nLinks != romeTopoModels[i].nLinks) continue;
    if (!ignore_numa && strcmp(romeTopoModels[i].pattern, pattern)) continue;
    // permute GPU IDs
    for (int j = 0; j < ngpus; j++) g[j] = (j+2)%ngpus;
    if (!permuteGpuIds(g, 0, ngpus-1, romeTopoModels+i, &romeTopo, &time, ignore_cpu ? false : match_nbio, ignore_numa)) continue;
    if (nnetspergpu > 1) {
      int found = 0;
      // initialize nics mapping
      for (int j = 0; j < nnets; j++) n[j] = -1;
      // match nics
      for (int j = 0; j < nnets; j++) {
        int k;
        for (k = 0; k < nnets; k++) {
          int f;
          // skips nics that are matched
          for (f = 0; f < found; f++)
            if (k == n[f]) break;
          if (f < found) continue;
          // check NUMA
          if (!ignore_numa && romeTopoModels[i].nicNuma[j] != romeTopo.nicNuma[k]) continue;
          int g;
          // check GDR
          for (g = 0; g < ngpus; g++) {
            // enabling PXN override paths over PHB and SYS
            if (romeTopo.gdrLevel[k*ngpus+g] == PATH_PXN) continue;
            if (romeTopoModels[i].gdrLevel[j*ngpus+g] != romeTopo.gdrLevel[k*ngpus+g]) break;
          }
          if (g >= ngpus) break;
        }
        if (k < nnets) {
          // found match
          n[j] = k;
          found++;
        } else {
          return ncclSuccess;
        }
      }
      break;
    } else {
      if (nnets > 1) {
        // permute NET IDs
        for (int j = 0; j < nnets; j++) n[j] = (j+2)%nnets;
        if (permuteNetIds(n, g, 0, nnets-1, romeTopoModels+i, &romeTopo, &time, ignore_numa)) break;
      } else break;
    }
  }
  gettimeofday(&tve, NULL);
  if (i >= sizeof(romeTopoModels)/sizeof(romeTopoModels[0])) {
    //printf("No solution in %.2fms (%d iter)\n", t, time);
    return ncclSuccess;
  }
  for (int m = 0; m<ngpus; m++) rdm[m] = devids[g[m]];

  char line[1024];
  //sprintf(line, "Found matching Rome model index %d in %.2fms (%d iter) with GPU mapping: ", i, t, time);
  sprintf(line, "Found matching Rome model index %d with GPU mapping: ", i);
  int offset = strlen(line);
  for (int k = 0; k < ngpus; k++) {
    sprintf(line+offset, "%d ", rdm[k]);
    offset = strlen(line);
  }
  if (nnets > 1) {
    sprintf(line+offset, "NET mapping: ");
    offset = strlen(line);
    for (int k = 0; k < nnets; k++) {
      sprintf(line+offset, "%d ", n[k]);
      offset = strlen(line);
    }
  }
  INFO(NCCL_GRAPH, "%s", line);
  parseOptions(system, romeTopoModels[i].options);
  if(IsArchMatch(system->nodes[GPU].nodes[0].gpu.gcn, "gfx950")){
    system->tuning = 6;
  }
  // create 4P2H based on reference and remapped ids
  switch (graph->pattern) {
  case NCCL_TOPO_PATTERN_RING:
    // Attempt to use rail-optimized rings if they exist
    if (system->nHosts % 2 == 0) {
      // For even number of nodes, alternate forward/reverse on ringBase
      NCCLCHECK(parseGraph(ringBase != nullptr ? ringBase : romeTopoModels[i].ringBase, system, graph, rdm, nnets > 1 ? n : NULL, system->hostIdx % 2));
    }
    else {
      // For odd number of nodes, check first to see if ringTail1 and ringTail2 are defined
      if (system->nHosts == 1 || romeTopoModels[i].ringTail1 == nullptr || romeTopoModels[i].ringTail2 == nullptr) {
        if (system->nHosts > 1)
          INFO(NCCL_GRAPH, "[WARN] Dropping back due to lack of support for odd-number of nodes for model index %d\n", i);
        NCCLCHECK(parseGraph(ringBase != nullptr ? ringBase : romeTopoModels[i].ringBase, system, graph, rdm, nnets > 1 ? n : NULL, system->hostIdx % 2));
      }
      else
      {
        if (system->hostIdx == (system->nHosts - 1)) {
          NCCLCHECK(parseGraph(romeTopoModels[i].ringTail1, system, graph, rdm, nnets > 1 ? n : NULL, 0));
        } else if (system->hostIdx == (system->nHosts - 2)) {
          NCCLCHECK(parseGraph(romeTopoModels[i].ringTail2, system, graph, rdm, nnets > 1 ? n : NULL, 0));
        } else {
          NCCLCHECK(parseGraph(ringBase != nullptr ? ringBase : romeTopoModels[i].ringBase, system, graph, rdm, nnets > 1 ? n : NULL, system->hostIdx % 2));
        }
      }
    }
    break;
  case NCCL_TOPO_PATTERN_BALANCED_TREE:
    if (romeTopoModels[i].treeBase != nullptr) {
      NCCLCHECK(parseGraphLight(romeTopoModels[i].treeBase, system, graph, g));
      if (graph->nChannels) return ncclSuccess;
    }

    // Fall back to tree from ringBase
    NCCLCHECK(parseGraph(ringBase != nullptr ? ringBase : romeTopoModels[i].ringBase, system, graph, rdm, nnets > 1 ? n : NULL, 0));
    // Override GDR distance if requested
    if (checkOption(romeTopoModels[i].options, "netOverride")) {
      for (int i = 0; i < system->nodes[NET].count; i++) {
        for (int j = 0; j < system->nodes[GPU].count; j++) {
          if (system->nodes[GPU].nodes[j].paths[NET][i].type == PATH_PXB) {
            int k;
            for (k = 0; k < system->nodes[GPU].count; k++) {
              if (k != j &&
                system->nodes[GPU].nodes[k].gpu.dev/2 == system->nodes[GPU].nodes[j].gpu.dev/2)
                break;
            }
            if (k < system->nodes[GPU].count)
              system->nodes[GPU].nodes[k].paths[NET][i].type = PATH_PXB;
          }
        }
      }
    }
    break;
  }
  return ncclSuccess;
}

ncclResult_t parse1H16P(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  constexpr int NUMA_CPUS = 4;
  constexpr int NUMA_GPUS = 4;
  constexpr int NUMA_PERMUTE_COUNT = 24;
  constexpr int TOTAL_PERMUTE_COUNT = (NUMA_PERMUTE_COUNT*NUMA_PERMUTE_COUNT*NUMA_PERMUTE_COUNT*NUMA_PERMUTE_COUNT);

  int i;

  const int ngpus = system->nodes[GPU].count;
  const int ncpus = system->nodes[CPU].count;
  const int nnets = system->nodes[NET].count;

  // only valid on Rome
  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
  if (arch != NCCL_TOPO_CPU_ARCH_X86 || vendor != NCCL_TOPO_CPU_VENDOR_AMD || model != NCCL_TOPO_CPU_MODEL_AMD_ROME)
    return ncclSuccess;

  // number of GPUs and NICs on each numa node is used as first screening pattern
  struct rcclRomeModel romeTopo;
  char pattern[256];
  int devids[NUMA_CPUS*NUMA_GPUS];
  NCCLCHECK(parseRomeSystem(system, &romeTopo, pattern, devids));

  // only match for system with 16 GPUs
  if (ngpus != 16 || ncpus != NUMA_CPUS) return ncclSuccess;

  int *g16, n[NCCL_TOPO_MAX_NODES], rdm[NUMA_GPUS*NUMA_CPUS];
  int *all_gpu_permutations = (int *)malloc(TOTAL_PERMUTE_COUNT*NUMA_CPUS*NUMA_GPUS*sizeof(int));
  struct timeval tvs, tve;
  gettimeofday(&tvs, NULL);
  std::vector<int> r(ngpus), g(ngpus);
  for (i = 0; i < sizeof(romeTopoModels)/sizeof(romeTopoModels[0]); i++) {
    if (romeTopo.nCpus != romeTopoModels[i].nCpus || romeTopo.nGpus != romeTopoModels[i].nGpus ||
      romeTopo.nNics != romeTopoModels[i].nNics || romeTopo.nLinks != romeTopoModels[i].nLinks) continue;
    if (strcmp(romeTopoModels[i].pattern, pattern)) continue;
    int j;
    int numa_gpu_permutations[NUMA_CPUS][NUMA_PERMUTE_COUNT][NUMA_GPUS];
    // permute GPUs for each CPU NUMA nodes
    for (j = 0; j < ncpus; j++) {
      int ngpusPerNuma = 0, cnt = 0, npermute = 0;
      for (int k = 0; k < ngpus; k++) {
        if (romeTopoModels[i].gpuNuma[k] != j) continue;
        r[ngpusPerNuma++] = k;
      }
      if (ngpusPerNuma == 0) continue;
      if (ngpusPerNuma != NUMA_GPUS) break;
      // init GPU mapping
      for (int k = 0; k < ngpus; k++) {
        if (romeTopo.gpuNuma[k] != j) continue;
        g[(2+cnt++)%ngpusPerNuma] = k;
      }
      std::sort(g.begin(), g.begin()+ngpusPerNuma);
      do {
        for (int n = 0; n < ngpusPerNuma; n++)
          numa_gpu_permutations[j][npermute][n] = g[n];
        npermute++;
      } while (std::next_permutation(g.begin(), g.begin()+ngpusPerNuma));
      if (npermute != NUMA_PERMUTE_COUNT) break;
    }
    if (j < ncpus) continue;
    // permute GPUs for all CPU NUMA nodes
    for (int a = 0; a < NUMA_PERMUTE_COUNT; a++) {
      for (int b = 0; b < NUMA_PERMUTE_COUNT; b++) {
        for (int c = 0; c < NUMA_PERMUTE_COUNT; c++) {
          for (int d = 0; d < NUMA_PERMUTE_COUNT; d++) {
            uint64_t offset = ((a*NUMA_PERMUTE_COUNT+b)*NUMA_PERMUTE_COUNT+c)*NUMA_PERMUTE_COUNT+d;
            //offset = (offset+TOTAL_PERMUTE_COUNT/2)%TOTAL_PERMUTE_COUNT;
            offset *= (NUMA_CPUS*NUMA_GPUS);
            memcpy(all_gpu_permutations+offset, &numa_gpu_permutations[0][a][0], NUMA_GPUS*sizeof(int));
            memcpy(all_gpu_permutations+offset+NUMA_GPUS, &numa_gpu_permutations[1][b][0], NUMA_GPUS*sizeof(int));
            memcpy(all_gpu_permutations+offset+NUMA_GPUS*2, &numa_gpu_permutations[2][c][0], NUMA_GPUS*sizeof(int));
            memcpy(all_gpu_permutations+offset+NUMA_GPUS*3, &numa_gpu_permutations[3][d][0], NUMA_GPUS*sizeof(int));
          }
        }
      }
    }
    // match all GPUs' XGMI connection
    int p;
    for (p = 0; p < TOTAL_PERMUTE_COUNT; p++) {
      g16 = all_gpu_permutations+p*NUMA_CPUS*NUMA_GPUS;
      int k;
      for (k = 0; k < romeTopoModels[i].nGpus; k++) {
        int m;
        for (m = 0; m < romeTopoModels[i].nGpus; m++) {
          if (romeTopoModels[i].connMatrix[k*romeTopoModels[i].nGpus+m] != romeTopo.connMatrix[g16[k]*romeTopoModels[i].nGpus+g16[m]]) break;
        }
        if (m < romeTopoModels[i].nGpus) break;
      }
      if (k < romeTopoModels[i].nGpus) continue;
      //printf("found match %d: ", p); for (int n = 0; n < NUMA_CPUS*NUMA_GPUS; n++) printf("%d ", g16[n]); printf("\n");
      if (nnets > 1) {
        // permute NET IDs
        int time = 0;
        for (int m = 0; m < nnets; m++) n[m] = (m+2)%nnets;
        if (permuteNetIds(n, g16, 0, nnets-1, romeTopoModels+i, &romeTopo, &time, false)) break;
      } else break;
    }
    if (p < TOTAL_PERMUTE_COUNT) break;
  }
  gettimeofday(&tve, NULL);
  if (i >= sizeof(romeTopoModels)/sizeof(romeTopoModels[0])) {
    //printf("No solution in %.2fms\n", t);
    return ncclSuccess;
  }
  for (int m = 0; m<ngpus; m++) rdm[m] = devids[g16[m]];

  char line[1024];
  //sprintf(line, "Found matching Rome model index %d in %.2fms with GPU mapping: ", i, t);
  sprintf(line, "Found matching Rome model index %d with GPU mapping: ", i);
  int offset = strlen(line);
  for (int k = 0; k < ngpus; k++) {
    sprintf(line+offset, "%d ", rdm[k]);
    offset = strlen(line);
  }
  if (nnets > 1) {
    sprintf(line+offset, "NET mapping: ");
    offset = strlen(line);
    for (int k = 0; k < nnets; k++) {
      sprintf(line+offset, "%d ", n[k]);
      offset = strlen(line);
    }
  }
  INFO(NCCL_GRAPH, "%s", line);
  system->type |= RCCL_TOPO_16P1H;
  parseOptions(system, romeTopoModels[i].options);
  if(IsArchMatch(system->nodes[GPU].nodes[0].gpu.gcn, "gfx950")){
    system->tuning = 6;
  }

  // create 16P1H based on reference and remapped ids
  NCCLCHECK(parseGraph(romeTopoModels[i].ringBase, system, graph, rdm, nnets > 1 ? n : NULL, false));

  if (romeTopoModels[i].treeBase != nullptr) NCCLCHECK(parseGraphLight(romeTopoModels[i].treeBase, system, graph, rdm));
  // clean up
  free(all_gpu_permutations);
  return ncclSuccess;
}

ncclResult_t find_gpu_hives(int *g_hives, int *ng_hives, struct rcclRomeModel *romeTopo) {
  const int ngpus = romeTopo->nGpus;
  const int gpus_per_hive = romeTopo->nLinks + 1;
  const int nhives = ngpus / gpus_per_hive;

  int i;
  for (i = 0; i < ngpus; i++) {
    int j, h;
    for (j = 0; j < nhives; j++) {
      if (ng_hives[j]) {
        if (romeTopo->connMatrix[i*ngpus+g_hives[j*gpus_per_hive]]) {
          g_hives[j*gpus_per_hive+ng_hives[j]] = i;
          ng_hives[j] ++;
          break;
        }
      }
    }
    if (j >= nhives) {
      for (h = 0; h < nhives; h++) {
        if (ng_hives[h] == 0) {
          g_hives[h*gpus_per_hive] = i;
          ng_hives[h]++;
          break;
        }
      }
      if (h >= nhives)
        return ncclInternalError;
    }
  }
  for (int i = 0; i < nhives; i++) {
    if (ng_hives[i] != gpus_per_hive)
      return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t parse4H4P(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  constexpr int NUM_HIVES = 4;
  constexpr int HIVE_GPUS = 4;

  const int ngpus = system->nodes[GPU].count;
  const int nnets = system->nodes[NET].count;

  // only valid on Rome
  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
  if (arch != NCCL_TOPO_CPU_ARCH_X86 || vendor != NCCL_TOPO_CPU_VENDOR_AMD || model != NCCL_TOPO_CPU_MODEL_AMD_ROME)
    return ncclSuccess;

  // number of GPUs and NICs on each numa node is used as first screening pattern
  struct rcclRomeModel romeTopo;
  char pattern[256];
  int devids[HIVE_GPUS*NUM_HIVES];
  NCCLCHECK(parseRomeSystem(system, &romeTopo, pattern, devids));

  // only match for system with 16 GPUs
  if (ngpus != NUM_HIVES*HIVE_GPUS || nnets != NUM_HIVES*HIVE_GPUS) return ncclSuccess;

  std::vector<int> g_hives(ngpus), n_hives(nnets);
  int ng_hives[NUM_HIVES];
  int rdm[NUM_HIVES*HIVE_GPUS];

  // try to sort GPUs into hives
  for (int i = 0; i < NUM_HIVES; i++)
    ng_hives[i] = 0;
  for (int i = 0; i < nnets; i++)
    n_hives[i] = -1;
  for (int i = 0; i < ngpus; i++)
    g_hives[i] = -1;

  if (find_gpu_hives(g_hives.data(), ng_hives, &romeTopo))
    return ncclSuccess;

  for (int i = 0; i < NUM_HIVES; i++)
    if (ng_hives[i] != 4) return ncclSuccess;
  // remap NET ids
  for (int i = 0; i < nnets; i++) {
    int j;
    for (j = 0; j < ngpus; j++) {
      if(romeTopo.gdrLevel[i*nnets+g_hives[j]] == PATH_PIX) {
        n_hives[j] = i;
        break;
      }
    }
    if (j >= ngpus) return ncclSuccess;
  }
  // validation
  for (int i = 0; i < nnets; i++)
    if (n_hives[i] == -1) return ncclSuccess;
  for (int i = 0; i < ngpus; i++)
    if (g_hives[i] == -1) return ncclSuccess;

  for (int m = 0; m<ngpus; m++) rdm[m] = devids[g_hives[m]];

  char line[1024];
  sprintf(line, "Found matching Rome model 4P4H with GPU mapping: ");
  int offset = strlen(line);
  for (int k = 0; k < ngpus; k++) {
    sprintf(line+offset, "%d ", rdm[k]);
    offset = strlen(line);
  }
  if (nnets > 1) {
    sprintf(line+offset, "NET mapping: ");
    offset = strlen(line);
    for (int k = 0; k < nnets; k++) {
      sprintf(line+offset, "%d ", n_hives[k]);
      offset = strlen(line);
    }
  }
  INFO(NCCL_GRAPH, "%s", line);
  if (arch == NCCL_TOPO_CPU_ARCH_X86 && vendor == NCCL_TOPO_CPU_VENDOR_AMD && model == NCCL_TOPO_CPU_MODEL_AMD_ROME)
    system->type |= RCCL_TOPO_4P2H_ROME;
  parseOptions(system, rome_model_68.options);
  // create 4P4H based on reference and remapped ids
  NCCLCHECK(parseGraph(rome_model_68.ringBase, system, graph, rdm, n_hives.data(), false));
  return ncclSuccess;
}

static struct rcclRomeModel gio16gColumbaModel = {
  .nGpus = 16, .nCpus = 2, .nNics = 0, .nLinks = 7,
  .gpuIds = { 0x24000, 0x2a000, 0x37000, 0x3d000, 0x4a000, 0x50000, 0x5d000, 0x63000, 0xaf000, 0xb5000, 0xc2000, 0xc8000, 0xd5000, 0xdb000, 0xe8000, 0xee000, },
  .nicIds = { },
  .gpuNuma = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, },
  .nicNuma = { },
  .connMatrix = { 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, },
  .gdrLevel = { },
  .pattern = "8080",
  .ringBase = "0 1 2 3 4 5 7 6 12 13 14 15 8 9 11 10"
            "| 2 7 3 5 1 6 0 4 14 9 15 10 13 11 12 8"
            "| 1 3 0 7 4 6 2 5 13 15 11 8 10 14 12 9"
            "| 3 6 5 0 2 4 1 7 15 12 10 9 13 8 14 11"
            "| 10 11 9 8 15 14 13 12 6 7 5 4 3 2 1 0"
            "| 8 12 11 13 10 15 9 14 4 0 6 1 5 3 7 2"
            "| 9 12 14 10 8 11 15 13 5 2 6 4 7 0 3 1"
            "| 11 14 8 13 9 10 12 15 7 1 4 2 0 5 6 3",
  .options = "",
};

ncclResult_t parseGIOTopos(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  ncclResult_t r;
  constexpr int NGPUS = 16;
  constexpr int NHIVES = 2;
  constexpr int GPUS_PER_HIVE = NGPUS/NHIVES;
  const int ngpus = system->nodes[GPU].count;
  const int ncpus = system->nodes[CPU].count;
  const int nnets = system->nodes[NET].count;

  if (ngpus < NGPUS || ncpus > 2 || nnets != 0)
    return ncclSuccess;

  struct rcclRomeModel romeTopo;
  char pattern[256];
  int devids[NGPUS];
  NCCLCHECK(parseRomeSystem(system, &romeTopo, pattern, devids));

  if (romeTopo.nLinks + 1 != GPUS_PER_HIVE)
    return ncclSuccess;

  int g_hives[NGPUS], ng_hives[NHIVES], rdm[NGPUS];
  memset(ng_hives, 0, sizeof(ng_hives));
  memset(g_hives, -1, sizeof(g_hives));
  NCCLCHECKGOTO(find_gpu_hives(g_hives, ng_hives, &romeTopo), r, exit);

  for (int m = 0; m<ngpus; m++) rdm[m] = devids[g_hives[m]];

  char line[1024];
  int offset;
  sprintf(line, "Found GigaIO rome topo, gpu_mapping: ");
  offset = strlen(line);
  for (int k = 0; k < ngpus; k++) {
    sprintf(line+offset, "%d ", g_hives[k]);
    offset = strlen(line);
  }
  INFO(NCCL_GRAPH,"%s", line);

  system->type |= RCCL_TOPO_4P2H_ROME;

  NCCLCHECKGOTO(parseGraph(gio16gColumbaModel.ringBase, system, graph, rdm, NULL, false), r, exit);

exit:
  return ncclSuccess;
}
