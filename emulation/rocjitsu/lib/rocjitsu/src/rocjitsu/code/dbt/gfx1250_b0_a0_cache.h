// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt/gfx1250_b0_a0_cache.h
/// @brief What every consumer of the fixed-profile B0-to-A0 translator must
///        agree on to share cached translations.
///
/// @details Two programs share entries only if they derive the same key and look
/// in the same tree. Nothing reports a disagreement: a tool writing under one
/// domain and a runtime reading under another both work perfectly, produce no
/// error, and never exchange a single entry. The pre-translation tool would
/// appear to succeed while start-up kept paying full translation cost.
///
/// So these are stated once and included by both sides rather than declared
/// alongside each use.

#ifndef ROCJITSU_CODE_DBT_GFX1250_B0_A0_CACHE_H_
#define ROCJITSU_CODE_DBT_GFX1250_B0_A0_CACHE_H_

#include "rocjitsu/code/dbt/translation_store.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include <string_view>

namespace rocjitsu {

/// @brief Tree holding this translator's entries, in either tier.
constexpr std::string_view kGfx1250B0A0Domain = "gfx1250-b0-a0";

/// @brief Key contract shared by the packaged producer and runtime consumer.
/// @details These entries are built and shipped as one deployment artifact, so
/// their explicit profile and format revisions, rather than one producer ELF's
/// build ID, define compatibility.
constexpr TranslationStore::KeyMode kGfx1250B0A0KeyMode =
    TranslationStore::KeyMode::kPortablePrebuilt;

/// @brief The single configuration this translator runs under.
///
/// @details Its entry point takes no options, so every request shares one
/// identity. It is still recorded and re-checked, so an entry produced under a
/// future profile can never satisfy a request made under this one.
constexpr TranslationIdentity kGfx1250B0A0Identity{
    .profile_id = 1,
    .input_revision = 1,
    .output_revision = 0,
    .target_isa = "gfx1250",
};

/// @brief An address inside the translator, not inside the caller.
///
/// @details Passed to the store, whose keys name whatever produced an entry, and
/// to shared_translation_root(), which locates the shared tree relative to that
/// same object. Both callers resolve this to the one translator they are linked
/// against, so they agree by construction rather than by convention.
inline const void *gfx1250_b0_a0_translator_anchor() {
  return reinterpret_cast<const void *>(&rj_gfx1250_b0_to_a0_translate);
}

} // namespace rocjitsu

#endif // ROCJITSU_CODE_DBT_GFX1250_B0_A0_CACHE_H_
