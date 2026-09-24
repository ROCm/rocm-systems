// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rj_hsa_dbi_publication.h"

#include <algorithm>
#include <limits>
#include <map>
#include <vector>

namespace rocjitsu::consan::hook {
namespace {
bool modifies(const PublicationEvent &event) {
  return event.operation == PublicationOperation::Rmw ||
         event.operation == PublicationOperation::Store;
}
PublicationOrdering publication_orders_validated(const PublicationPoint &before,
                                                 const PublicationPoint &after,
                                                 std::span<const PublicationEvent> input,
                                                 bool complete) {
  using Result = PublicationOrdering;
  if (!(before.domain == after.domain))
    return Result::Unordered;
  if (!complete || before.sequence == 0 || after.sequence == 0 || before.lane >= 64 ||
      after.lane >= 64)
    return Result::Incomplete;
  std::vector<const PublicationEvent *> events;
  for (const auto &event : input) {
    if (!(event.point.domain == before.domain))
      continue;
    if (!event.observation_valid || event.point.sequence == 0 || event.point.lane >= 64 ||
        event.bytes == 0 || event.bytes > 8 || (event.bytes & (event.bytes - 1)) != 0 ||
        event.address > std::numeric_limits<uint64_t>::max() - event.bytes ||
        (event.operation != PublicationOperation::Read &&
         event.operation != PublicationOperation::Rmw &&
         event.operation != PublicationOperation::Store) ||
        (event.operation == PublicationOperation::Read && event.release) ||
        (event.operation == PublicationOperation::Store && event.acquire))
      return Result::Incomplete;
    const uint64_t mask = event.bytes == 8 ? ~uint64_t{0} : (uint64_t{1} << (event.bytes * 8)) - 1;
    if ((event.observed & ~mask) || (event.written & ~mask))
      return Result::Incomplete;
    events.push_back(&event);
  }
  const size_t count = events.size();
  const size_t source = count, sink = count + 1;
  std::vector<PublicationPoint> points;
  for (const auto *event : events)
    points.push_back(event->point);
  points.push_back(before);
  points.push_back(after);
  std::vector<std::vector<size_t>> edges(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      if (points[i].owner != points[j].owner || points[i].lane != points[j].lane)
        continue;
      if (points[i].sequence == points[j].sequence)
        return Result::Incomplete;
      if (points[i].sequence < points[j].sequence)
        edges[i].push_back(j);
      else
        edges[j].push_back(i);
    }
  }
  const auto same_object = [&](size_t a, size_t b) {
    return events[a]->address == events[b]->address && events[a]->bytes == events[b]->bytes;
  };
  std::vector<size_t> predecessor(count, count);
  for (size_t i = 0; i < count; ++i) {
    const auto &event = *events[i];
    if (modifies(event) && event.observed == event.written)
      return Result::Incomplete;
    for (size_t j = 0; j < count; ++j) {
      if (i == j)
        continue;
      const auto &other = *events[j];
      if (!same_object(i, j)) {
        if (event.address < other.address + other.bytes &&
            other.address < event.address + event.bytes)
          return Result::Incomplete;
        continue;
      }
      if (modifies(event) && modifies(other) &&
          (event.written == other.written || event.observed == other.observed))
        return Result::Incomplete;
      if (modifies(other) && other.written == event.observed) {
        if (predecessor[i] != count)
          return Result::Incomplete;
        predecessor[i] = j;
      }
    }
  }
  // Reject disconnected modification chains: an omitted transition cannot be
  // repaired by guessing which same-address publication an acquire observed.
  for (size_t i = 0; i < count; ++i) {
    if (!modifies(*events[i]) || predecessor[i] != count)
      continue;
    for (size_t j = i + 1; j < count; ++j) {
      if (same_object(i, j) && modifies(*events[j]) && predecessor[j] == count)
        return Result::Incomplete;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    if (events[i]->operation != PublicationOperation::Read || predecessor[i] != count)
      continue;
    for (size_t j = 0; j < count; ++j) {
      if (!same_object(i, j) || !modifies(*events[j]))
        continue;
      if ((predecessor[j] == count && events[i]->observed != events[j]->observed) ||
          (events[i]->point.owner == events[j]->point.owner &&
           events[i]->point.lane == events[j]->point.lane &&
           events[j]->point.sequence < events[i]->point.sequence))
        return Result::Incomplete;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    size_t current = predecessor[i];
    size_t traversed = 0;
    bool scopes_cover = events[i]->covers_workgroup;
    bool release_sequence = true;
    while (current != count) {
      if (++traversed > count)
        return Result::Incomplete;
      if (events[current]->point.owner == events[i]->point.owner &&
          events[current]->point.lane == events[i]->point.lane &&
          events[current]->point.sequence >= events[i]->point.sequence)
        return Result::Incomplete;
      scopes_cover &= events[current]->covers_workgroup;
      if (release_sequence && events[i]->acquire && scopes_cover && events[current]->release)
        edges[current].push_back(i);
      // A store may head its own release sequence, but cannot relay a
      // preceding release through the observation RMW used to instrument it.
      if (events[current]->operation == PublicationOperation::Store)
        release_sequence = false;
      current = predecessor[current];
    }
  }
  // Conflicting observation/program-order evidence must not turn a cycle into
  // arbitrary happens-before. Validate the whole graph before answering.
  std::vector<size_t> indegree(points.size());
  for (const auto &next : edges)
    for (size_t j : next)
      ++indegree[j];
  std::vector<size_t> ready;
  for (size_t i = 0; i < indegree.size(); ++i)
    if (!indegree[i])
      ready.push_back(i);
  size_t visited = 0;
  while (!ready.empty()) {
    size_t i = ready.back();
    ready.pop_back();
    ++visited;
    for (size_t j : edges[i])
      if (--indegree[j] == 0)
        ready.push_back(j);
  }
  if (visited != points.size())
    return Result::Incomplete;
  std::vector<bool> reached(points.size(), false);
  ready.push_back(source);
  reached[source] = true;
  while (!ready.empty()) {
    size_t i = ready.back();
    ready.pop_back();
    for (size_t j : edges[i])
      if (!reached[j]) {
        reached[j] = true;
        ready.push_back(j);
      }
  }
  return reached[sink] ? Result::Ordered : Result::Unordered;
}
} // namespace

PublicationOrdering publication_orders(const PublicationPoint &before,
                                       const PublicationPoint &after,
                                       std::span<const PublicationEvent> input, bool complete) {
  using Result = PublicationOrdering;
  if (!(before.domain == after.domain))
    return Result::Unordered;
  if (!complete || !before.sequence || !after.sequence || before.lane >= 64 || after.lane >= 64)
    return Result::Incomplete;

  // A usable object must have an unambiguous observed modification chain.
  // Preserve independent good objects, including transitive publication across
  // them, even when unrelated bookkeeping objects have opaque modifications.
  using Object = std::pair<uint64_t, uint32_t>;
  std::map<Object, std::vector<PublicationEvent>> objects;
  std::vector<const PublicationEvent *> modifications;
  for (const auto &event : input) {
    if (event.point.domain.generation != before.domain.generation)
      continue;
    const bool opaque = event.operation == PublicationOperation::OpaqueModification;
    if (!event.point.sequence || event.point.lane >= 64 || !event.bytes ||
        event.bytes > (opaque ? 128u : 8u) || (!opaque && (event.bytes & (event.bytes - 1))) ||
        event.address > std::numeric_limits<uint64_t>::max() - event.bytes ||
        (opaque && (event.observation_valid || event.observed || event.written || event.release ||
                    event.acquire)))
      return Result::Incomplete;
    modifications.push_back(&event);
    if (!opaque && event.point.domain == before.domain)
      objects[{event.address, event.bytes}].push_back(event);
  }

  bool incomplete_object = false;
  std::vector<PublicationEvent> usable;
  for (auto &[object, events] : objects) {
    const auto [address, bytes] = object;
    bool invalidated = false;
    for (const auto *other : modifications) {
      if (address >= other->address + other->bytes || other->address >= address + bytes)
        continue;
      // Foreign workgroup/dispatch transitions must not disappear when the
      // workgroup-local proof filters its events. In particular, that would
      // hide an ABA or a modification breaking the release sequence.
      if (other->operation == PublicationOperation::OpaqueModification ||
          !(other->point.domain == before.domain) || other->address != address ||
          other->bytes != bytes) {
        invalidated = true;
        break;
      }
    }
    // An observed identity RMW (add 0, OR of already-set bits) preserves
    // the incoming release sequence. Its acquire half may read that sequence
    // just like a load. Its own release cannot be placed among other equal-
    // value operations, so never infer an outgoing release edge from it.
    // Keep narrow-scope operations conservative rather than eliding them.
    for (auto &event : events) {
      if (event.operation == PublicationOperation::Rmw && event.observation_valid &&
          event.covers_workgroup && event.observed == event.written) {
        incomplete_object |= event.release;
        event.operation = PublicationOperation::Read;
        event.release = false;
      }
    }
    if (invalidated ||
        publication_orders_validated(before, after, events, true) == Result::Incomplete) {
      incomplete_object = true;
      continue;
    }
    usable.insert(usable.end(), events.begin(), events.end());
  }
  const auto result = publication_orders_validated(before, after, usable, true);
  if (result == Result::Ordered)
    return result; // This path uses only independently validated objects.
  return incomplete_object ? Result::Incomplete : result;
}
} // namespace rocjitsu::consan::hook
