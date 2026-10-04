// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// Optional runtime binding to an eXpressive Tuning Profile (XTP) reader.
//
// XTP is a versioned file format -- a bundle of tuning profiles plus a
// manifest saying which hardware each profile covers -- and a small C ABI for
// querying it. RCCL depends on that contract, not on any particular library
// implementing it: the reader is dlopen'd on demand when RCCL_XTP names a
// bundle directory, so RCCL takes no build-time dependency on it and builds
// and ships identically whether or not it is installed.
//
// Every entry point below fails closed when the reader is absent, unloadable,
// or carries no value for this comm. Callers keep their built-in behavior in
// all of those cases, so a profile is never a hard dependency.
//
// This is the only RCCL-side code that knows the reader exists, and it holds
// no decision logic of its own: it supplies facts (the comm's topology
// descriptor) and hands back values the profile already resolved.

#ifndef XTP_WRAP_H_
#define XTP_WRAP_H_

#include <cstdint>

struct ncclComm;

// Comm-level ("Class A") knob lookups, keyed by the names the XTP schema
// uses: "PXN", "P2P_NET_CHUNKSIZE", "UNROLL_FACTOR", "THREADS_PER_BLOCK".
//
// Both bind the comm on first use and cache the binding for its lifetime, so
// the caller does not have to order an explicit bind against init. They return
// false and leave *out untouched whenever no profile supplies the knob.
//
// Note these do NOT consult the environment: env-wins is the caller's to
// enforce, because only the caller knows which variable overrides which knob.
bool rcclXtpClassAInt(struct ncclComm* comm, const char* key, int64_t* out);
bool rcclXtpClassAStr(struct ncclComm* comm, const char* key, const char** out);

// Release the binding held for comm. Safe on a comm that was never bound.
void rcclXtpCommFree(struct ncclComm* comm);

#endif // XTP_WRAP_H_
