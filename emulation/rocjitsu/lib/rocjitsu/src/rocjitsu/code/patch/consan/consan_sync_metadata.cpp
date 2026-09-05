// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_sync_metadata.h"

#include <algorithm>
#include <ranges>
#include <span>

namespace rocjitsu {
bool is_release_cache_event(std::span<const ConSanProgramSite> program_sites,
                            const ConSanSyncEvent *event) {
  const ConSanFenceSite *fence =
      event == nullptr ? nullptr : consan_program_site<ConSanFenceSite>(program_sites, *event);
  return fence != nullptr && fence->cache_operation == ConSanCacheOperation::Release;
}

bool is_acquire_cache_event(std::span<const ConSanProgramSite> program_sites,
                            const ConSanSyncEvent *event) {
  // A split cache acquire is admitted only by the exact ordered-pair matcher
  // in semantic association. A lone completion operation is workgroup-local
  // cache maintenance, not a complete addressed acquire.
  const ConSanFenceSite *fence =
      event == nullptr ? nullptr : consan_program_site<ConSanFenceSite>(program_sites, *event);
  return fence != nullptr && fence->cache_operation == ConSanCacheOperation::Acquire;
}

bool consan_ordinary_acquire_metadata_compatible(std::span<const ConSanProgramSite> program_sites,
                                                 const ConSanSyncEvent &load,
                                                 const ConSanSyncSequence &load_sequence,
                                                 const ConSanSyncEvent &cache,
                                                 const ConSanSyncSequence &cache_sequence,
                                                 ConSanOrdinaryAcquireMetadataPolicy policy) {
  const ConSanOrdinaryMemorySite *load_source =
      consan_program_site<ConSanOrdinaryMemorySite>(program_sites, load);
  const ConSanFenceSite *cache_source = consan_program_site<ConSanFenceSite>(program_sites, cache);
  const ConSanProgramSite *load_decoded = consan_program_site(program_sites, load);
  const ConSanProgramSite *cache_decoded = consan_program_site(program_sites, cache);
  if (load_source == nullptr || cache_source == nullptr || load_decoded == nullptr ||
      cache_decoded == nullptr)
    return false;
  const bool require_same_block =
      policy == ConSanOrdinaryAcquireMetadataPolicy::SameBlockSingleFence ||
      policy == ConSanOrdinaryAcquireMetadataPolicy::SameBlockCachePairMember;
  const bool allow_cache_pair_member =
      policy == ConSanOrdinaryAcquireMetadataPolicy::SameBlockCachePairMember ||
      policy == ConSanOrdinaryAcquireMetadataPolicy::BoundedPathCachePairMember;
  return load.kind == ConSanSyncKind::OrdinaryMemory &&
         load.operation == ConSanSyncOperation::OrdinaryLoad && load_source->width_bits == 32u &&
         load.confidence == ConSanSemanticConfidence::Conservative && load.scope &&
         consan_memory_scope_is_supported(*load.scope) &&
         *load.scope != ConSanMemoryScope::Wavefront && cache.kind == ConSanSyncKind::Fence &&
         cache.operation == ConSanSyncOperation::Fence &&
         (cache_source->cache_operation == ConSanCacheOperation::Acquire ||
          (*load.scope == ConSanMemoryScope::Workgroup &&
           cache_source->cache_operation == ConSanCacheOperation::AcquirePairCompletion) ||
          (allow_cache_pair_member &&
           (cache_source->cache_operation == ConSanCacheOperation::AcquirePairPrefix ||
            cache_source->cache_operation == ConSanCacheOperation::AcquirePairCompletion))) &&
         cache.confidence == ConSanSemanticConfidence::Conservative &&
         load.semantic_id.physical.code_object == cache.semantic_id.physical.code_object &&
         load_decoded->container == cache_decoded->container &&
         load_sequence.kind == ConSanSyncKind::OrdinaryMemory &&
         load_sequence.operation == ConSanSyncOperation::OrdinaryLoad &&
         cache_sequence.kind == ConSanSyncKind::Fence &&
         cache_sequence.operation == ConSanSyncOperation::Fence &&
         load_sequence.basic_block_index && cache_sequence.basic_block_index &&
         (!require_same_block ||
          load_sequence.basic_block_index == cache_sequence.basic_block_index) &&
         consan_nonempty_execution_owners_equal(load_decoded->execution_owners,
                                                cache_decoded->execution_owners);
}

bool consan_ordinary_acquire_metadata_compatible(std::span<const ConSanProgramSite> program_sites,
                                                 const ConSanSyncEvent &load,
                                                 const ConSanSyncSequence &load_sequence,
                                                 const ConSanSyncEvent &cache,
                                                 const ConSanSyncSequence &cache_sequence) {
  return consan_ordinary_acquire_metadata_compatible(
      program_sites, load, load_sequence, cache, cache_sequence,
      ConSanOrdinaryAcquireMetadataPolicy::SameBlockSingleFence);
}

bool consan_ordinary_release_metadata_compatible(std::span<const ConSanProgramSite> program_sites,
                                                 const ConSanSyncEvent &cache,
                                                 const ConSanSyncSequence &cache_sequence,
                                                 const ConSanSyncEvent &store,
                                                 const ConSanSyncSequence &store_sequence) {
  const ConSanFenceSite *cache_source = consan_program_site<ConSanFenceSite>(program_sites, cache);
  const ConSanOrdinaryMemorySite *store_source =
      consan_program_site<ConSanOrdinaryMemorySite>(program_sites, store);
  const ConSanProgramSite *cache_decoded = consan_program_site(program_sites, cache);
  const ConSanProgramSite *store_decoded = consan_program_site(program_sites, store);
  return cache_source != nullptr && store_source != nullptr && cache_decoded != nullptr &&
         store_decoded != nullptr && cache.kind == ConSanSyncKind::Fence &&
         cache.operation == ConSanSyncOperation::Fence &&
         cache_source->cache_operation == ConSanCacheOperation::Release &&
         cache.confidence == ConSanSemanticConfidence::Conservative &&
         store.kind == ConSanSyncKind::OrdinaryMemory &&
         store.operation == ConSanSyncOperation::OrdinaryStore && store_source->width_bits != 0u &&
         store_source->width_bits <= 128u &&
         store.confidence == ConSanSemanticConfidence::Conservative && store.scope &&
         consan_memory_scope_is_agent_or_system(*store.scope) &&
         cache.semantic_id.physical.code_object == store.semantic_id.physical.code_object &&
         cache_decoded->container == store_decoded->container &&
         cache_sequence.kind == ConSanSyncKind::Fence &&
         cache_sequence.operation == ConSanSyncOperation::Fence &&
         store_sequence.kind == ConSanSyncKind::OrdinaryMemory &&
         store_sequence.operation == ConSanSyncOperation::OrdinaryStore &&
         cache_sequence.basic_block_index &&
         cache_sequence.basic_block_index == store_sequence.basic_block_index &&
         consan_nonempty_execution_owners_equal(cache_decoded->execution_owners,
                                                store_decoded->execution_owners);
}

} // namespace rocjitsu
