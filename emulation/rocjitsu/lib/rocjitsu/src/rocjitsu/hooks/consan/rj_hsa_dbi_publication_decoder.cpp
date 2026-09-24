// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rj_hsa_dbi_publication.h"
#include <algorithm>
#include <limits>

namespace rocjitsu::consan::hook {
DecodedPublications decode_publications(const ReportHeader &header,
                                        std::span<const PublicationRecord> records) {
  using Status = PublicationDecodeStatus;
  constexpr uint32_t flags = kPublicationTraceEnabled | kPublicationTraceComplete;
  if (header.publication_flags & ~flags)
    return {.status = Status::Malformed};
  if (!(header.publication_flags & kPublicationTraceEnabled)) {
    if (header.publication_flags || header.publication_event_count ||
        header.publication_dropped_count)
      return {.status = Status::Malformed};
    // Access sequence collection may precede atomic capture. A disabled trace
    // never supplies ordering evidence, regardless of its clock value.
    return {};
  }
  if (records.size() != header.publication_event_capacity ||
      header.publication_event_count > header.publication_event_capacity ||
      header.publication_dropped_count)
    return {.status = Status::Incomplete};
  DecodedPublications result{.status = (header.publication_flags & kPublicationTraceComplete)
                                           ? Status::Complete
                                           : Status::Incomplete};
  std::vector<uint64_t> sequences;
  for (uint32_t i = 0; i < header.publication_event_count; ++i) {
    const auto &record = records[i];
    const bool opaque = record.operation == PublicationRecordOperation::OpaqueModification;
    if (record.state != kPublicationReady)
      return {.status = Status::Incomplete};
    if (record.generation != header.generation || !record.dispatch_id || !record.sequence ||
        record.sequence > header.publication_clock || record.lane_id >= 64 || !record.byte_count ||
        record.byte_count > (opaque ? 128u : 8u) ||
        (!opaque && (record.byte_count & (record.byte_count - 1))) ||
        record.address > std::numeric_limits<uint64_t>::max() - record.byte_count ||
        (!opaque && record.scope == 0) || record.scope > 5 ||
        (record.roles & ~(kPublicationRelease | kPublicationAcquire | kPublicationObserved)) ||
        (record.operation != PublicationRecordOperation::Read &&
         record.operation != PublicationRecordOperation::Rmw &&
         record.operation != PublicationRecordOperation::Store && !opaque) ||
        (record.operation == PublicationRecordOperation::Read &&
         (record.roles & kPublicationRelease)) ||
        (record.operation == PublicationRecordOperation::Store &&
         (record.roles & kPublicationAcquire)))
      return {.status = Status::Malformed};
    if (opaque && (record.roles || record.observed || record.written))
      return {.status = Status::Malformed};
    if (!opaque && !(record.roles & kPublicationObserved))
      return {.status = Status::Incomplete};
    const uint64_t mask =
        record.byte_count >= 8 ? ~uint64_t{0} : (uint64_t{1} << (record.byte_count * 8)) - 1;
    if ((record.observed & ~mask) || (record.written & ~mask))
      return {.status = Status::Malformed};
    sequences.push_back(record.sequence);
    result.events.push_back({.point = {.domain = {.generation = record.generation,
                                                  .dispatch = record.dispatch_id,
                                                  .workgroup_x = record.workgroup_x,
                                                  .workgroup_y = record.workgroup_y,
                                                  .workgroup_z = record.workgroup_z,
                                                  .cluster_workgroup = record.cluster_workgroup_id},
                                       .owner = record.owner_id,
                                       .sequence = record.sequence,
                                       .lane = record.lane_id},
                             .address = record.address,
                             .bytes = record.byte_count,
                             .observed = record.observed,
                             .written = record.written,
                             .operation = opaque ? PublicationOperation::OpaqueModification
                                          : record.operation == PublicationRecordOperation::Read
                                              ? PublicationOperation::Read
                                          : record.operation == PublicationRecordOperation::Store
                                              ? PublicationOperation::Store
                                              : PublicationOperation::Rmw,
                             .release = (record.roles & kPublicationRelease) != 0,
                             .acquire = (record.roles & kPublicationAcquire) != 0,
                             .covers_workgroup = record.scope >= 2,
                             .observation_valid = !opaque});
  }
  std::ranges::sort(sequences);
  if (std::adjacent_find(sequences.begin(), sequences.end()) != sequences.end())
    return {.status = Status::Malformed};
  return result;
}

} // namespace rocjitsu::consan::hook
