// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rj_hsa_dbi_publication.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace rocjitsu::consan::hook {
PublicationOrdering publication_orders(const PublicationPoint &before,
                                       const PublicationPoint &after,
                                       std::span<const PublicationEvent> input, bool complete) {
  using Result = PublicationOrdering;
  if (!(before.domain == after.domain))
    return Result::Unordered;
  if (!complete || before.sequence == 0 || after.sequence == 0)
    return Result::Incomplete;
  std::vector<const PublicationEvent *> events;
  for (const auto &event : input) {
    if (!(event.point.domain == before.domain))
      continue;
    if (!event.observation_valid || event.point.sequence == 0 || event.bytes == 0 ||
        event.bytes > 8 || (event.bytes & (event.bytes - 1)) != 0 ||
        event.address > std::numeric_limits<uint64_t>::max() - event.bytes ||
        (event.operation != PublicationOperation::Read &&
         event.operation != PublicationOperation::Rmw) ||
        (event.operation == PublicationOperation::Read && event.release))
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
      if (points[i].owner != points[j].owner)
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
    if (event.operation == PublicationOperation::Rmw && event.observed == event.written)
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
      if (event.operation == PublicationOperation::Rmw &&
          other.operation == PublicationOperation::Rmw &&
          (event.written == other.written || event.observed == other.observed))
        return Result::Incomplete;
      if (other.operation == PublicationOperation::Rmw && other.written == event.observed) {
        if (predecessor[i] != count)
          return Result::Incomplete;
        predecessor[i] = j;
      }
    }
  }
  // Reject disconnected modification chains: an omitted transition cannot be
  // repaired by guessing which same-address publication an acquire observed.
  for (size_t i = 0; i < count; ++i) {
    if (events[i]->operation != PublicationOperation::Rmw || predecessor[i] != count)
      continue;
    for (size_t j = i + 1; j < count; ++j) {
      if (same_object(i, j) && events[j]->operation == PublicationOperation::Rmw &&
          predecessor[j] == count)
        return Result::Incomplete;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    if (events[i]->operation != PublicationOperation::Read || predecessor[i] != count)
      continue;
    for (size_t j = 0; j < count; ++j) {
      if (!same_object(i, j) || events[j]->operation != PublicationOperation::Rmw)
        continue;
      if ((predecessor[j] == count && events[i]->observed != events[j]->observed) ||
          (events[i]->point.owner == events[j]->point.owner &&
           events[j]->point.sequence < events[i]->point.sequence))
        return Result::Incomplete;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    size_t current = predecessor[i];
    size_t traversed = 0;
    bool scopes_cover = events[i]->covers_workgroup;
    while (current != count) {
      if (++traversed > count)
        return Result::Incomplete;
      if (events[current]->point.owner == events[i]->point.owner &&
          events[current]->point.sequence >= events[i]->point.sequence)
        return Result::Incomplete;
      scopes_cover &= events[current]->covers_workgroup;
      if (events[i]->acquire && scopes_cover && events[current]->release)
        edges[current].push_back(i);
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
} // namespace rocjitsu::consan::hook
