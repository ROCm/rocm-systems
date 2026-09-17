// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_sync_metadata.h"

#include <algorithm>
#include <ranges>
#include <span>

namespace rocjitsu::consan {
bool is_release_cache_event(std::span<const ProgramSite> program_sites, const SyncEvent *event) {
  const FenceSite *fence =
      event == nullptr ? nullptr : program_site<FenceSite>(program_sites, *event);
  return fence != nullptr && fence->cache_operation == CacheOperation::Release;
}

bool is_acquire_cache_event(std::span<const ProgramSite> program_sites, const SyncEvent *event) {
  // A split cache acquire is admitted only by the exact ordered-pair matcher
  // in semantic association. A lone completion operation is workgroup-local
  // cache maintenance, not a complete addressed acquire.
  const FenceSite *fence =
      event == nullptr ? nullptr : program_site<FenceSite>(program_sites, *event);
  return fence != nullptr && fence->cache_operation == CacheOperation::Acquire;
}

bool ordinary_acquire_metadata_compatible(std::span<const ProgramSite> program_sites,
                                          const SyncEvent &load, const SyncSequence &load_sequence,
                                          const SyncEvent &cache,
                                          const SyncSequence &cache_sequence,
                                          OrdinaryAcquireMetadataPolicy policy) {
  const OrdinaryMemorySite *load_source = program_site<OrdinaryMemorySite>(program_sites, load);
  const FenceSite *cache_source = program_site<FenceSite>(program_sites, cache);
  const ProgramSite *load_decoded = program_site(program_sites, load);
  const ProgramSite *cache_decoded = program_site(program_sites, cache);
  if (load_source == nullptr || cache_source == nullptr || load_decoded == nullptr ||
      cache_decoded == nullptr)
    return false;
  const bool require_same_block = policy == OrdinaryAcquireMetadataPolicy::SameBlockSingleFence ||
                                  policy == OrdinaryAcquireMetadataPolicy::SameBlockCachePairMember;
  const bool allow_cache_pair_member =
      policy == OrdinaryAcquireMetadataPolicy::SameBlockCachePairMember ||
      policy == OrdinaryAcquireMetadataPolicy::BoundedPathCachePairMember;
  return load.kind == SyncKind::OrdinaryMemory && load.operation == SyncOperation::OrdinaryLoad &&
         load_source->width_bits == 32u && load.confidence == SemanticConfidence::Conservative &&
         load.scope && memory_scope_is_supported(*load.scope) &&
         *load.scope != MemoryScope::Wavefront && cache.kind == SyncKind::Fence &&
         cache.operation == SyncOperation::Fence &&
         (cache_source->cache_operation == CacheOperation::Acquire ||
          (*load.scope == MemoryScope::Workgroup &&
           cache_source->cache_operation == CacheOperation::AcquirePairCompletion) ||
          (allow_cache_pair_member &&
           (cache_source->cache_operation == CacheOperation::AcquirePairPrefix ||
            cache_source->cache_operation == CacheOperation::AcquirePairCompletion))) &&
         cache.confidence == SemanticConfidence::Conservative &&
         load.semantic_id.physical.code_object == cache.semantic_id.physical.code_object &&
         load_decoded->container == cache_decoded->container &&
         load_sequence.kind == SyncKind::OrdinaryMemory &&
         load_sequence.operation == SyncOperation::OrdinaryLoad &&
         cache_sequence.kind == SyncKind::Fence &&
         cache_sequence.operation == SyncOperation::Fence && load_sequence.basic_block_index &&
         cache_sequence.basic_block_index &&
         (!require_same_block ||
          load_sequence.basic_block_index == cache_sequence.basic_block_index) &&
         nonempty_execution_owners_equal(load_decoded->execution_owners,
                                         cache_decoded->execution_owners);
}

bool ordinary_acquire_metadata_compatible(std::span<const ProgramSite> program_sites,
                                          const SyncEvent &load, const SyncSequence &load_sequence,
                                          const SyncEvent &cache,
                                          const SyncSequence &cache_sequence) {
  return ordinary_acquire_metadata_compatible(program_sites, load, load_sequence, cache,
                                              cache_sequence,
                                              OrdinaryAcquireMetadataPolicy::SameBlockSingleFence);
}

bool ordinary_release_metadata_compatible(std::span<const ProgramSite> program_sites,
                                          const SyncEvent &cache,
                                          const SyncSequence &cache_sequence,
                                          const SyncEvent &store,
                                          const SyncSequence &store_sequence) {
  const FenceSite *cache_source = program_site<FenceSite>(program_sites, cache);
  const OrdinaryMemorySite *store_source = program_site<OrdinaryMemorySite>(program_sites, store);
  const ProgramSite *cache_decoded = program_site(program_sites, cache);
  const ProgramSite *store_decoded = program_site(program_sites, store);
  return cache_source != nullptr && store_source != nullptr && cache_decoded != nullptr &&
         store_decoded != nullptr && cache.kind == SyncKind::Fence &&
         cache.operation == SyncOperation::Fence &&
         cache_source->cache_operation == CacheOperation::Release &&
         cache.confidence == SemanticConfidence::Conservative &&
         store.kind == SyncKind::OrdinaryMemory &&
         store.operation == SyncOperation::OrdinaryStore && store_source->width_bits != 0u &&
         store_source->width_bits <= 128u && store.confidence == SemanticConfidence::Conservative &&
         store.scope && memory_scope_is_agent_or_system(*store.scope) &&
         cache.semantic_id.physical.code_object == store.semantic_id.physical.code_object &&
         cache_decoded->container == store_decoded->container &&
         cache_sequence.kind == SyncKind::Fence &&
         cache_sequence.operation == SyncOperation::Fence &&
         store_sequence.kind == SyncKind::OrdinaryMemory &&
         store_sequence.operation == SyncOperation::OrdinaryStore &&
         cache_sequence.basic_block_index &&
         cache_sequence.basic_block_index == store_sequence.basic_block_index &&
         nonempty_execution_owners_equal(cache_decoded->execution_owners,
                                         store_decoded->execution_owners);
}

} // namespace rocjitsu::consan
