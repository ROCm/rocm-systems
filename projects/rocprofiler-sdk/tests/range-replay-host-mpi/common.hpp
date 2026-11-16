#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Shared by main.cpp (the MPI application) and client.cpp (the tool that replays its range).

constexpr uint64_t kRangeId = 0x4D5049;

constexpr int kSteps = 4;

// Scenarios, selected for every rank by RR_MPI_MODE. The rank that misbehaves is
// RR_MPI_SPECIAL_RANK (default 1).
//   replay    every rank takes part: all passes run on all ranks
//   leak      the special rank leaks a device allocation inside the range, so its range declines
//             ALLOCATION_CHANGED_IN_RANGE; its peers must decline with PEER_DECLINED
//   unarmed   the special rank opens the range without arming the jump, so it declines
//             HOST_JMP_NOT_SET; its peers must decline with PEER_DECLINED
// RR_MPI_PASSES_SPECIAL, when set, is the pass count the special rank asks for, while the others
// ask for RR_MPI_PASSES: the job must run the smaller of the two.
inline int
env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    return (v == nullptr || *v == '\0') ? fallback : std::atoi(v);
}

inline std::string
env_str(const char* name, const char* fallback)
{
    const char* v = std::getenv(name);
    return (v == nullptr || *v == '\0') ? std::string{fallback} : std::string{v};
}
