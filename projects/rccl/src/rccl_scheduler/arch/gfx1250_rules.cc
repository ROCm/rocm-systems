/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

// gfx1250 (MI455). Fabric DDA (LL, LL128, VMM) and the symk-withdrawn reroutes.
// SPX, DPX and CPX share these rows; their windows differ through the arch table.

#include "rccl_scheduler/rccl_scheduler.h"

namespace gfx1250 {

// CTAPolicy ZERO prefers CE; RCCL_FORCE_SYMMETRIC and RCCL_FORCE_DDA prefer those.
const rcclStep kArPrologue[] = {Eval(RCCL_FACT_CAPTURING), Eval(RCCL_FACT_CE_AR_GRAPH_OK),
                                SkipRulesIf(RCCL_FACT_ALGO_ENV)};
const rcclRule kArRules[] = {
  Rule(kGinSdma).Not(kInGroup),
  Rule(kCe2Shot).If(kCeArGraphOk | kCeAr2ShotOk).Not(kInGroup | kSymkRequested),
  Rule(kDdaFabricLL128).If(kRerouteCeReg),  // symk withdrawn: DDA replaces CE registered
  Rule(kDdaFabricVmm, RCCL_LIMIT_DDA_AR_VMM_FIRST).Not(kSymkVeto | kRerouteCeReg),
  Rule(kDdaFabricTiers).Not(kSymkVeto | kRerouteCeReg),
  Rule(kCeRegistered, RCCL_LIMIT_CE_REG).Not(kSysmem).YieldTo(kSymkEligible),
  Rule(kCeStaged, RCCL_LIMIT_CE_REG).If(kCeArGraphOk | kCeAr2ShotOk).Not(kInGroup | kSysmem | kSymkRequested)
    .YieldTo(kSymkEligible),
  Rule(kSymmetricKernel, RCCL_LIMIT_SYMK).If(kSymkRequested),
};
const rcclBackend* const kArFallbacks[] = {&kGeneralRing};

const rcclStep kAgPrologue[] = {Eval(RCCL_FACT_CAPTURING), SkipRulesIf(RCCL_FACT_ALGO_ENV)};
const rcclRule kAgRules[] = {
  Rule(kGinSdma),
  Rule(kDdaFabricTiers).Not(kSymkVeto),
  Rule(kGeneralHier).Not(kInGroup | kSymkVeto),
  Rule(kCeScratch, RCCL_LIMIT_CE_NONREG).Not(kCapturing | kSysmem).AnySizeWhen(kCtaZero),
  Rule(kCeRegistered, RCCL_LIMIT_CE_REG_AG).Not(kCapturing | kSysmem).AnySizeWhen(kCtaZero).YieldTo(kSymkEligible),
  Rule(kCeHier).If(kCtaZero).Not(kCapturing | kSysmem),
  Rule(kSymmetricKernel).If(kSymkEligible),
};
const rcclBackend* const kAgFallbacks[] = {&kGeneralDirect, &kGeneralRing};

// symk is chosen last but vetoes DDA, hierarchical and Direct.
const rcclStep kRsPrologue[] = {SkipRulesIf(RCCL_FACT_ALGO_ENV), Eval(RCCL_FACT_SYMK_ELIGIBLE),
                                Eval(RCCL_FACT_CE_AR_GRAPH_OK), Eval(RCCL_FACT_CE_RS_OK)};
const rcclRule kRsRules[] = {
  Rule(kCe2Shot).If(kCeRsOk).Not(kSymkEligible),
  Rule(kCeStagingInit).If(kCeRsOk).Not(kSymkEligible),
  Rule(kDdaFabricTiers).Not(kSymkVeto),
  Rule(kGeneralHier).Not(kInGroup | kSymkVeto),
  Rule(kGeneralDirect).Not(kInGroup | kSymkVeto),
  Rule(kCeRegistered).If(kCeRsOk).Not(kSysmem),
  Rule(kSymmetricKernel).If(kSymkEligible),
};
const rcclBackend* const kRsFallbacks[] = {&kGeneralRing};

// NCCL_ALGO does not apply. Fabric LL and LL128 launch directly, so they skip
// in-place calls and the enqueue-only path. The capture probe runs at the first CE rule.
const rcclRule kA2aRules[] = {
  Rule(kGinGda),
  Rule(kGinSdma),
  Rule(kDdaFabricLL, RCCL_LIMIT_DDA_LL).Not(kMustEnqueue | kInPlace | kSymkVeto),
  Rule(kDdaFabricLL128, RCCL_LIMIT_DDA_LL128).Not(kMustEnqueue | kInPlace | kSymkVeto),
  Rule(kDdaFabricVmm, RCCL_LIMIT_DDA_VMM).Not(kInPlace | kSymkVeto),
  Rule(kCeRegistered, RCCL_LIMIT_CE_REG).Not(kCapturing | kSysmem).AnySizeWhen(kCtaZero),
  Rule(kCeHier).If(kCtaZero).Not(kCapturing | kSysmem),
  Rule(kCeScratch, RCCL_LIMIT_CE_NONREG).Not(kCapturing | kSysmem).AnySizeWhen(kCtaZero),
};
const rcclBackend* const kA2aFallbacks[] = {&kGeneralDirect};

// Without a DDA tier the enqueue path picks the native algorithm.
const rcclStep kBcPrologue[] = {SkipRulesIf(RCCL_FACT_ALGO_ENV)};
const rcclRule kBcRules[] = {
  Rule(kDdaFabricLL, RCCL_LIMIT_DDA_LL),
  Rule(kDdaFabricLL128, RCCL_LIMIT_DDA_LL128),
};
const rcclBackend* const kBcFallbacks[] = {&kGeneralNative};

const rcclRow kAllReduce = Row("AR", kArPrologue, kArRules, kArFallbacks, rcclAllReducePreferredBackends);
const rcclRow kAllGather = Row("AG", kAgPrologue, kAgRules, kAgFallbacks);
const rcclRow kReduceScatter = Row("RS", kRsPrologue, kRsRules, kRsFallbacks);
const rcclRow kAlltoAll = Row("A2A", kA2aRules, kA2aFallbacks);
const rcclRow kBroadcast = Row("BC", kBcPrologue, kBcRules, kBcFallbacks);

}  // namespace gfx1250

const rcclPolicy kRcclGfx1250Policy = {"gfx1250", &gfx1250::kAllReduce, &gfx1250::kAllGather,
                                       &gfx1250::kReduceScatter, &gfx1250::kAlltoAll, &gfx1250::kBroadcast};
