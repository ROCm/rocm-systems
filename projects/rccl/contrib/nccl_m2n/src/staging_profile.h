/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_STAGING_PROFILE_H_
#define NCCL_STAGING_PROFILE_H_

#include <chrono>
#include <memory>

/* DEBUG-only host-side profiling helpers for ncclReshard staging setup. */
enum StagingProfilePhase {
  STAGING_PROFILE_ENSURE_BUFFER = 0,
  STAGING_PROFILE_REGISTER_WINDOW,
  STAGING_PROFILE_PROBE_DEV_COMM,
  STAGING_PROFILE_BUILD_DESCRIPTOR,
  STAGING_PROFILE_PREPARE_PARAMS,
  STAGING_PROFILE_RESOLVE_GIN_COUNTS,
  STAGING_PROFILE_GET_DEV_COMM,
  STAGING_PROFILE_LAUNCH_KERNEL,
  STAGING_PROFILE_PHASE_COUNT
};

using StagingProfileClock = std::chrono::steady_clock;
using StagingProfileTimePoint = StagingProfileClock::time_point;

class StagingProfile {
public:
  explicit StagingProfile(int callIndex);

  void begin(StagingProfilePhase phase);
  void end(StagingProfilePhase phase);
  void log(int rank) const;

private:
  double phaseMs(StagingProfilePhase phase) const;

  int callIndex_;
  StagingProfileTimePoint callStart_;
  StagingProfileTimePoint phaseStart_[STAGING_PROFILE_PHASE_COUNT];
  StagingProfileTimePoint phaseEnd_[STAGING_PROFILE_PHASE_COUNT];
};

class StagingProfileScope {
public:
  StagingProfileScope(StagingProfile* profile, StagingProfilePhase phase) : profile_(profile), phase_(phase) {
    if (profile_ != nullptr) {
      profile_->begin(phase_);
    }
  }

  ~StagingProfileScope() {
    if (profile_ != nullptr) {
      profile_->end(phase_);
    }
  }

  StagingProfileScope(const StagingProfileScope&) = delete;
  StagingProfileScope& operator=(const StagingProfileScope&) = delete;

private:
  StagingProfile* profile_;
  StagingProfilePhase phase_;
};

std::unique_ptr<StagingProfile> stagingProfileCreate();

#endif /* NCCL_STAGING_PROFILE_H_ */
