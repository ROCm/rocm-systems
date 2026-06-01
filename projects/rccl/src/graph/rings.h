/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next);
ncclResult_t rcclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next);