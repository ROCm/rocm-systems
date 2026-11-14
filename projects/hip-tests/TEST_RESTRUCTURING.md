<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# HIP Test Restructuring

This document describes the planned, incremental restructuring of the HIP
Catch2 test suite. It is a proposal for upcoming work and does not change the
current test-layer guidance in [`TESTING.md`](TESTING.md); the tiers, gates,
and workflows described there remain authoritative today.

The HIP Catch2 suite is being incrementally reorganized to make its behavioral
coverage easier to understand, maintain, and extend. This work complements the
existing contract-test tier: contract tests continue to define small, portable
public-API guarantees, while the restructured suites retain deeper functional,
lifecycle, backend, platform, integration, stress, and performance coverage.

## Why this change

The legacy `catch/unit/` suite contains large files with mixed responsibilities,
duplicated setup, inconsistent cleanup, and overlapping coverage. As a result, it
can be difficult to determine:

- which behavior a test actually validates;
- whether coverage is already provided elsewhere;
- whether a failure is a runtime defect or an unsupported capability;
- which tests require a specific backend, platform, GPU, or system configuration; and
- whether a legacy test can be safely retired.

The restructuring addresses these issues by organizing tests around focused
behavioral domains and separating coverage according to its purpose.

## Scope

Restructured coverage may be organized into the following tiers:

| Tier | Location | Purpose |
|---|---|---|
| Unit | `catch/restructured/unit/<domain>/` | Focused API and feature behavior |
| Integration | `catch/restructured/integration/<domain>/` | Cross-component, interop, loader, RTC, and multi-device behavior |
| Stress | `catch/restructured/stress/<domain>/` | Long-running and high-load scenarios |
| Performance | `catch/restructured/performance/<domain>/` | Throughput and latency scenarios |

The work includes:

- decomposing broad legacy sources into focused behavioral cases;
- preserving unique coverage while removing redundant tests;
- reusing existing contract or restructured coverage where behavior is equivalent;
- adding targeted tests for meaningful gaps discovered during restructuring;
- improving resource-lifecycle and capability handling;
- documenting backend, platform, and hardware-specific behavior; and
- retaining integration, stress, and performance scenarios in the tier most appropriate for their purpose.

The restructuring is incremental. Legacy tests remain available while replacement
coverage is reviewed and validated; moving coverage into a restructured domain
does not by itself imply that the legacy test can be removed.

## Expected benefits

This work is expected to provide:

- smaller tests with clearer intent and easier failure diagnosis;
- more consistent resource management;
- less duplicated setup and overlapping coverage;
- clearer separation between contract, unit, integration, stress, and performance testing;
- improved traceability from APIs and behaviors to their tests;
- safer handling of unsupported capabilities and platform differences;
- better visibility into hardware-dependent coverage gaps; and
- a more maintainable foundation for future HIP runtime and backend changes.
