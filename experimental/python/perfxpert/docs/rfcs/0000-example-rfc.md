---
status: accepted
author: aelwazir
start_date: 2026-04-17
pr: https://example.com/pr/0
implements: []
supersedes: []
---

# RFC-0000: Canonical example (non-binding)

## Summary

This file is a worked example demonstrating the RFC format. It is not a
live proposal and no implementation follows from it.

## Motivation

Contributors benefit from seeing a completed RFC before writing their first one.

## Design

(Illustrative; not implemented.) Add a "thermal-throttle detection" signal
to the `latency` bottleneck classifier that reads temperature counters
when available.

### Public-facing change

`perfxpert analyze` would emit a new recommendation category
`thermal_throttling` when `TEMPERATURE_EDGE > 95 C` correlates with
kernel slowdown.

### Internal change

- New tool `tools/thermal.py:classify_throttle`
- New knowledge entry `thermal_thresholds.yaml`
- New bottleneck sub-type in `bottleneck_types.yaml`

### Backward compatibility

Non-breaking. If thermal counters absent, classifier skips silently.

## Alternatives considered

(1) OS-level thermal probe — rejected: requires root.
(2) Ignore thermal entirely — rejected: leaves a real class of slowdowns invisible.

## Unresolved questions

- What per-arch threshold? Differs MI300X vs MI350X.

## Test plan

Tool unit tests, integration test with a thermal-throttled fixture,
proven-optimization entry with fixture-pair.

## Migration

n/a (additive)

## Prior art

AMD SMI / ROCm SMI thermal telemetry and existing ROCm monitoring flows.

## Approval checklist

- [x] Example (non-binding)
