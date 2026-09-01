// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_sync_metadata.h"

#include <algorithm>
#include <ranges>
#include <span>

namespace rocjitsu {
namespace {

[[nodiscard]] bool same_execution_owners(std::span<const ConSanExecutionOwner> lhs,
                                         std::span<const ConSanExecutionOwner> rhs) {
  return !lhs.empty() && lhs.size() == rhs.size() &&
         std::ranges::equal(
             lhs, rhs, [](const ConSanExecutionOwner &left, const ConSanExecutionOwner &right) {
               return left.descriptor_file_offset == right.descriptor_file_offset &&
                      left.proof == right.proof;
             });
}

} // namespace

bool is_release_cache_event(const ConSanSyncEvent *event) {
  return event != nullptr && event->cache_operation == ConSanCacheOperation::Release;
}

bool is_acquire_cache_event(const ConSanSyncEvent *event) {
  // A split cache acquire is admitted only by the exact ordered-pair matcher
  // in semantic association. A lone completion operation is workgroup-local
  // cache maintenance, not a complete addressed acquire.
  return event != nullptr && event->cache_operation == ConSanCacheOperation::Acquire;
}

bool consan_ordinary_acquire_metadata_compatible(const ConSanSyncEvent &load,
                                                 const ConSanSyncSequence &load_sequence,
                                                 const ConSanSyncEvent &cache,
                                                 const ConSanSyncSequence &cache_sequence,
                                                 ConSanOrdinaryAcquireMetadataPolicy policy) {
  const bool require_same_block =
      policy == ConSanOrdinaryAcquireMetadataPolicy::SameBlockSingleFence ||
      policy == ConSanOrdinaryAcquireMetadataPolicy::SameBlockCachePairMember;
  const bool allow_cache_pair_member =
      policy == ConSanOrdinaryAcquireMetadataPolicy::SameBlockCachePairMember ||
      policy == ConSanOrdinaryAcquireMetadataPolicy::BoundedPathCachePairMember;
  return load.kind == ConSanSyncEventKind::OrdinaryMemory &&
         load.operation == ConSanSyncOperation::OrdinaryLoad && load.width_bits == 32u &&
         load.confidence == ConSanSemanticConfidence::Conservative && load.raw_scope &&
         (*load.raw_scope >= 1u && *load.raw_scope <= 3u) &&
         cache.kind == ConSanSyncEventKind::Fence &&
         cache.operation == ConSanSyncOperation::Fence &&
         (cache.cache_operation == ConSanCacheOperation::Acquire ||
          (*load.raw_scope == 1u &&
           cache.cache_operation == ConSanCacheOperation::AcquirePairCompletion) ||
          (allow_cache_pair_member &&
           (cache.cache_operation == ConSanCacheOperation::AcquirePairPrefix ||
            cache.cache_operation == ConSanCacheOperation::AcquirePairCompletion))) &&
         cache.confidence == ConSanSemanticConfidence::Conservative &&
         load.code_object_fingerprint == cache.code_object_fingerprint &&
         load.container_name == cache.container_name && load.in_kernel == cache.in_kernel &&
         load_sequence.kind == ConSanSyncSequenceKind::OrdinaryMemory &&
         load_sequence.operation == ConSanSyncOperation::OrdinaryLoad &&
         cache_sequence.kind == ConSanSyncSequenceKind::Fence &&
         cache_sequence.operation == ConSanSyncOperation::Fence &&
         load_sequence.basic_block_index && cache_sequence.basic_block_index &&
         (!require_same_block ||
          load_sequence.basic_block_index == cache_sequence.basic_block_index) &&
         same_execution_owners(load.execution_owners, cache.execution_owners) &&
         same_execution_owners(load_sequence.execution_owners, cache_sequence.execution_owners) &&
         same_execution_owners(load.execution_owners, load_sequence.execution_owners);
}

bool consan_ordinary_acquire_metadata_compatible(const ConSanSyncEvent &load,
                                                 const ConSanSyncSequence &load_sequence,
                                                 const ConSanSyncEvent &cache,
                                                 const ConSanSyncSequence &cache_sequence) {
  return consan_ordinary_acquire_metadata_compatible(
      load, load_sequence, cache, cache_sequence,
      ConSanOrdinaryAcquireMetadataPolicy::SameBlockSingleFence);
}

bool consan_ordinary_release_metadata_compatible(const ConSanSyncEvent &cache,
                                                 const ConSanSyncSequence &cache_sequence,
                                                 const ConSanSyncEvent &store,
                                                 const ConSanSyncSequence &store_sequence) {
  return cache.kind == ConSanSyncEventKind::Fence &&
         cache.operation == ConSanSyncOperation::Fence &&
         cache.cache_operation == ConSanCacheOperation::Release &&
         cache.confidence == ConSanSemanticConfidence::Conservative &&
         store.kind == ConSanSyncEventKind::OrdinaryMemory &&
         store.operation == ConSanSyncOperation::OrdinaryStore && store.width_bits != 0u &&
         store.width_bits <= 128u && store.confidence == ConSanSemanticConfidence::Conservative &&
         store.raw_scope && (*store.raw_scope == 2u || *store.raw_scope == 3u) &&
         cache.code_object_fingerprint == store.code_object_fingerprint &&
         cache.container_name == store.container_name && cache.in_kernel == store.in_kernel &&
         cache_sequence.kind == ConSanSyncSequenceKind::Fence &&
         cache_sequence.operation == ConSanSyncOperation::Fence &&
         store_sequence.kind == ConSanSyncSequenceKind::OrdinaryMemory &&
         store_sequence.operation == ConSanSyncOperation::OrdinaryStore &&
         cache_sequence.basic_block_index &&
         cache_sequence.basic_block_index == store_sequence.basic_block_index &&
         same_execution_owners(cache.execution_owners, store.execution_owners) &&
         same_execution_owners(cache_sequence.execution_owners, store_sequence.execution_owners) &&
         same_execution_owners(store.execution_owners, store_sequence.execution_owners);
}

} // namespace rocjitsu
