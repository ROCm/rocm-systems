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
#ifndef RCCL_ROME_MODELS_H_
#define RCCL_ROME_MODELS_H_

ncclResult_t parseGraph(const char* str, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* gpu_map, int* net_map, int reverse);
ncclResult_t parseGraphLight(const char* str, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* gpu_map);
ncclResult_t parseRome4P2H(struct ncclTopoSystem* system, struct ncclTopoGraph* graph2, const char *ringBase);
ncclResult_t parseChordalRing(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t parse1H16P(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t parse4H4P(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t parseA2a8P(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, const char *ringBase);
ncclResult_t parseGIOTopos(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

#endif
