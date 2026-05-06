# Contributing: new proven optimization case

## What you're adding

A before/after profiling case with measurable speedup and a citation. The
case becomes both (a) a knowledge-base precedent specialists cite and (b) a
test fixture that must pass all 5 correctness gates.

## File locations

- Entry: `perfxpert/knowledge/proven_optimizations.yaml`
- Fixture pair: `tests/fixtures/proven_optimizations/<id>.baseline.db` + `.optimized.db` + `.md`

## Template

Append to `proven_optimizations.yaml`:

```yaml
- id: <lowercase_snake_case>
  bottleneck_type: <compute|memory_transfer|latency|api_overhead|mixed>
  technique: "<short-phrase>"
  description: >
    2-4 sentence description of the pattern.
  measured_speedup_range: [<low>, <high>]      # e.g. [1.25, 1.40]
  source_citation: "<paper/doc/experiment-id>"
  preconditions:
    - {metric: "<name>", op: ">", threshold: <num>}
  failure_modes:
    - "<pitfall 1>"
  fixture_pair:
    baseline_db:    "tests/fixtures/proven_optimizations/<id>.baseline.db"
    optimized_db:   "tests/fixtures/proven_optimizations/<id>.optimized.db"
    description_md: "tests/fixtures/proven_optimizations/<id>.md"
```

## Concrete example

```yaml
- id: hip_stream_overlap_async_h2d
  bottleneck_type: memory_transfer
  technique: "overlap H2D copies with compute via hipMemcpyAsync"
  description: >
    The baseline trace serialized host-to-device transfers and kernel launches
    on the default stream. The optimized variant moved transfers onto a
    dedicated stream, inserted the required synchronization once at the true
    dependency boundary, and reduced API-visible idle time while preserving
    wall-clock correctness.
  measured_speedup_range: [1.12, 1.28]
  source_citation: "amd-internal:perfxpert-phase10-async-overlap-example"
  preconditions:
    - {metric: "memcpy_percent", op: ">", threshold: 20}
    - {metric: "overhead_percent", op: ">", threshold: 10}
  failure_modes:
    - "Async copies without pinned host memory can add overhead instead of hiding it"
    - "Missing stream synchronization can create correctness bugs that the baseline trace did not have"
    - "Tiny transfers may not amortize stream-management cost"
  fixture_pair:
    baseline_db:    "tests/fixtures/proven_optimizations/hip_stream_overlap_async_h2d.baseline.db"
    optimized_db:   "tests/fixtures/proven_optimizations/hip_stream_overlap_async_h2d.optimized.db"
    description_md: "tests/fixtures/proven_optimizations/hip_stream_overlap_async_h2d.md"
```

This example is intentionally small but realistic: it shows the level of
specificity reviewers expect for technique naming, conservative speedup ranges,
and failure modes.

## Fixture construction

Use `tests/fixtures/proven_optimizations/_build_fixtures.py` as the
reference implementation. Synthetic SQLite fixtures must:

- Be < 1 MB each
- Follow rocpd UUID-based schema (see CLAUDE.md §SQLite + rocpd)
- Produce metrics that satisfy your declared preconditions on baseline
- Show the claimed speedup on optimized
- Be regeneratable from the script (commit the generator alongside the .db)

## Schema constraints (CI-enforced)

- Schema validation: `tests/test_knowledge/test_proven_optimizations.py`
- All 5 gates: `tests/test_integration/test_proven_optimizations.py`
- False-positive rate: regression gate ≤ 5% across the corpus

## Tests you must add

The parametrized tests already iterate every case — just add the YAML entry
+ the fixture pair and the corpus tests will exercise your case.

## Review requirements

- 1 core reviewer
- CI green (schema + integration + gate cascade)
- Measured speedup citation is verifiable (paper link, in-house experiment ID, or AMD doc ref)

## Common pitfalls

- Fixture counter names must match `knowledge/counter_catalog.yaml`
- Failure_modes are load-bearing — a case with no failure_modes is incomplete
- `measured_speedup_range` is inclusive; keep it conservative — outliers are not the median
- Prefer smaller synthetic fixtures to real traces unless you're comfortable shipping real kernel names

## Related docs

- Spec §6.5 (proven optimization case studies)
- `tests/fixtures/proven_optimizations/_build_fixtures.py` (reference generator)
