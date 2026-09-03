// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <sys/types.h>

// Library-local test seam: swaps the read() used by the CPER reader so unit
// tests can drive zero/partial/error returns the way an empty debugfs ring does.
// Not amdsmi_-prefixed, so the linker version script keeps it out of
// libamd_smi.so (no public ABI); tests reach it through the static archive.
// Passing nullptr restores POSIX read(). Not thread-safe: only the
// single-threaded tests call it; production never does.
//
// Shared by the definition (src/amd_smi/amd_smi_cper.cc) and the tests
// (tests/amd_smi_test/functional/cper_read.cc) so the signature stays in sync.
void cper_set_read_fn_for_testing(ssize_t (*read_fn)(int, void*, size_t));
