// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H
#define ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H

#include "rocjitsu/code/patch/consan/consan.h"

#include <span>
#include <unordered_map>

namespace rocjitsu {

struct SyncEventSemanticIdHash {
  [[nodiscard]] size_t operator()(const SemanticSiteId &identity) const noexcept;
};

using SyncEventSemanticIndex =
    std::unordered_map<SemanticSiteId, const ConSanSyncEvent *, SyncEventSemanticIdHash>;

[[nodiscard]] SyncEventSemanticIndex
build_sync_event_semantic_index(std::span<const ConSanSyncEvent> sync_events);

[[nodiscard]] const ConSanSyncEvent *find_sequence_member_event(const SyncEventSemanticIndex &index,
                                                                SemanticSiteId identity);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H
