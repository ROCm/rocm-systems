// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translator_build_probe.cpp
/// @brief Loadable module standing in for a translator, built twice.
///
/// @details The translation store keys every entry on the GNU build id of the
/// module that produced it, so a translator rebuilt with different output cannot
/// serve entries written by the previous one. That is the property protecting
/// every deployment from a stale hit, and it cannot be exercised from a single
/// binary: it needs two modules that differ in nothing but identity.
///
/// This source is therefore linked into two targets whose only difference is an
/// explicit --build-id. Same code, same exported symbol, different identity --
/// exactly what a rebuild produces.

extern "C" __attribute__((visibility("default"))) void rj_probe_translator_anchor() {}
