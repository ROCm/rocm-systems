// Minimal bootstrap.h stub for CPU-only RCCL unit tests.
// Only declares bootstrapBidirEnabled — the sole function used by BootstrapBidirTests.
#pragma once

bool bootstrapBidirEnabled(int nranks, int kind);
