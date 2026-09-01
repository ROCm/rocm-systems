/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams whose definitions live in transport_stubs.cc. Declared in a header, not
// re-declared by each consumer, so a signature change is a compile error at
// every use rather than a silent link-time mismatch.

#ifndef RCCL_TEST_HOST_TRANSPORT_STUBS_H_
#define RCCL_TEST_HOST_TRANSPORT_STUBS_H_

// rcclUseAinic (src/transport/net.cc:343) queries whether an AINIC is present.
// A host-only binary has no device, so `false` is the honest answer rather than
// a steering choice; override it to exercise the AINIC arm.
extern bool g_rcclUseAinic;

void ResetTransportStubs();

#endif  // RCCL_TEST_HOST_TRANSPORT_STUBS_H_
