/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include "staging_profile.h"
#include "m2n_log.h"

#include <atomic>
#include <new>

static std::atomic<int> gStagingProfileCallCount{0};

static double stagingProfileMs(StagingProfileTimePoint start, StagingProfileTimePoint end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

StagingProfile::StagingProfile(int callIndex) : callIndex_(callIndex), callStart_(StagingProfileClock::now()) {
  for (int phase = 0; phase < STAGING_PROFILE_PHASE_COUNT; phase++) {
    phaseStart_[phase] = callStart_;
    phaseEnd_[phase] = callStart_;
  }
}

void StagingProfile::begin(StagingProfilePhase phase) {
  phaseStart_[phase] = StagingProfileClock::now();
}

void StagingProfile::end(StagingProfilePhase phase) {
  phaseEnd_[phase] = StagingProfileClock::now();
}

void StagingProfile::log(int rank) const {
  if (rank != 0) {
    return;
  }

  const StagingProfileTimePoint callEnd = StagingProfileClock::now();
  RESHARD_DEBUG(rank,
                "[STAGING-PROFILE] call=%d "
                "ensure_buffer=%.1fms register_window=%.1fms "
                "probe_dev_comm=%.1fms build_descriptor=%.1fms "
                "prepare_params=%.1fms resolve_gin_counts=%.1fms "
                "get_dev_comm=%.1fms launch_kernel=%.1fms total=%.1fms",
                callIndex_, phaseMs(STAGING_PROFILE_ENSURE_BUFFER), phaseMs(STAGING_PROFILE_REGISTER_WINDOW),
                phaseMs(STAGING_PROFILE_PROBE_DEV_COMM), phaseMs(STAGING_PROFILE_BUILD_DESCRIPTOR),
                phaseMs(STAGING_PROFILE_PREPARE_PARAMS), phaseMs(STAGING_PROFILE_RESOLVE_GIN_COUNTS),
                phaseMs(STAGING_PROFILE_GET_DEV_COMM), phaseMs(STAGING_PROFILE_LAUNCH_KERNEL),
                stagingProfileMs(callStart_, callEnd));
}

double StagingProfile::phaseMs(StagingProfilePhase phase) const {
  return stagingProfileMs(phaseStart_[phase], phaseEnd_[phase]);
}

std::unique_ptr<StagingProfile> stagingProfileCreate() {
  const int callIndex = gStagingProfileCallCount.fetch_add(1, std::memory_order_relaxed);
  return std::unique_ptr<StagingProfile>(new (std::nothrow) StagingProfile(callIndex));
}
