# Walkthrough: adding a new bottleneck class (`io_bound`)

## Overview

This guide walks through adding `io_bound` as a new bottleneck type in perfxpert.
The change touches 7 files and requires no architectural review (RFC) — pure
knowledge + tool + agent addition under the narrow-scope model.

## Prerequisites

- Your development environment set up (see [CONTRIBUTING.md](../../CONTRIBUTING.md) TL;DR)
- Familiarity with the three agents (Analysis, Recommendation, and Correctness)

## Step 1: Extend bottleneck types

Edit `perfxpert/knowledge/bottleneck_types.yaml` and add a new entry:

```yaml
- id: io_bound
  name: "I/O-bound (memory-controller or PCIe throttle)"
  description: |
    Kernel performance is constrained by system I/O (pinned memory, 
    peer-to-peer transfers, or host-device copies). Indicators: low 
    compute utilization, high memory-controller busy, PCIe saturation.
  classification_priority: 4
  typical_preconditions:
    - metric: "memory_controller_percent_busy"
      op: ">"
      threshold: 70
  specialist_agent: "io_specialist"
```

**Tests:** The schema validator will pick this up automatically. No separate test needed for YAML entry.

## Step 2: Extend the bottleneck classifier tool

Edit `perfxpert/tools/bottleneck.py` and extend the `classify_from_metrics()` function:

```python
# SKIP-SAMPLE — walkthrough snippet (assumes an existing file context)
def classify_from_metrics(metrics: Dict[str, float]) -> str:
    """Classify bottleneck from raw hardware metrics.
    
    Returns one of: 'compute_bound', 'memory_transfer', 'latency', 'io_bound', 'mixed'.
    """
    # ... existing logic ...
    
    # NEW: I/O bound classifier
    if metrics.get("memory_controller_percent_busy", 0) > 70:
        if metrics.get("kernel_occupancy", 1.0) < 0.5:
            return "io_bound"
    
    if metrics.get("pcie_utilization_percent", 0) > 60:
        if metrics.get("gpu_to_gpu_bandwidth_gbs", 0) > 100:
            return "io_bound"
    
    # ... rest of logic ...
```

**Tests:** Add tests to `tests/test_tools/test_bottleneck.py`:

```python
def test_classify_io_bound_high_memctrl():
    metrics = {
        "memory_controller_percent_busy": 85,
        "kernel_occupancy": 0.3,
        "compute_utilization": 0.2,
    }
    assert classify_from_metrics(metrics) == "io_bound"


def test_classify_io_bound_pcie_saturation():
    metrics = {
        "pcie_utilization_percent": 75,
        "gpu_to_gpu_bandwidth_gbs": 120,
        "kernel_occupancy": 0.4,
    }
    assert classify_from_metrics(metrics) == "io_bound"
```

## Step 3: Create knowledge entry for I/O techniques

Create `perfxpert/knowledge/io_techniques.yaml` with a list of optimization
techniques specific to I/O-bound workloads:

```yaml
version: "1.0"
metadata:
  description: "Optimization techniques for I/O-bound kernels"
  maintained_by: aelwazir
  last_updated: "2026-04-18"

techniques:
  - id: overlap_compute_and_transfer
    name: "Overlap compute and data transfer"
    description: |
      Use multiple streams or async transfers to hide PCIe/memory-controller 
      latency. While one stream transfers data to GPU, another computes.
    applicable_when:
      - io_bound
    speedup_range: [1.1, 1.4]
    
  - id: pinned_memory_pool
    name: "Allocate pinned memory pool"
    description: |
      Pre-allocate host memory as pinned (page-locked) to improve 
      host-to-device transfer bandwidth by 2-3×.
    applicable_when:
      - io_bound
    speedup_range: [1.2, 1.5]

  - id: reduce_transfer_size
    name: "Reduce data transfer size via compression or filtering"
    description: |
      Use run-length encoding, delta compression, or on-device filtering 
      to reduce PCIe bandwidth pressure.
    applicable_when:
      - io_bound
    speedup_range: [1.15, 1.6]
```

Create schema `perfxpert/knowledge/_schemas/io_techniques.schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "version": {"type": "string"},
    "metadata": {
      "type": "object",
      "properties": {
        "description": {"type": "string"},
        "maintained_by": {"type": "string"},
        "last_updated": {"type": "string"}
      },
      "required": ["description"]
    },
    "techniques": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "description": {"type": "string"},
          "applicable_when": {"type": "array", "items": {"type": "string"}},
          "speedup_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
        },
        "required": ["id", "name"]
      }
    }
  },
  "required": ["techniques"]
}
```

**Tests:** Add to `tests/test_knowledge/test_io_techniques.py`:

```python
import yaml
import json
import jsonschema

def test_io_techniques_loads():
    with open("perfxpert/knowledge/io_techniques.yaml") as f:
        data = yaml.safe_load(f)
    assert "techniques" in data

def test_io_techniques_validates_against_schema():
    with open("perfxpert/knowledge/io_techniques.yaml") as f:
        data = yaml.safe_load(f)
    with open("perfxpert/knowledge/_schemas/io_techniques.schema.json") as f:
        schema = json.load(f)
    jsonschema.validate(data, schema)

def test_io_techniques_ids_unique():
    with open("perfxpert/knowledge/io_techniques.yaml") as f:
        data = yaml.safe_load(f)
    ids = [t["id"] for t in data["techniques"]]
    assert len(ids) == len(set(ids))
```

## Step 4: Create I/O specialist agent

Create `perfxpert/agents/io_specialist.py`:

```python
# SKIP-SAMPLE — walkthrough snippet (references modules not yet created)
"""IO Specialist — optimizes I/O-bound kernels."""

from typing import Any, Dict

from perfxpert.agents.framework import Agent, AgentSpec
from perfxpert.agents.schemas import HandoffSchema
from perfxpert.tools import bottleneck, query_knowledge


class IOSpecialistHandoff(HandoffSchema):
    """Handoff from IO Specialist to Correctness."""
    technique_id: str
    narrative: str
    estimated_speedup: float


AGENT_SPEC = AgentSpec(
    name="io_specialist",
    fence_path="perfxpert/fence/slices/io_specialist.md",
    tools=[bottleneck, query_knowledge],
    handoff_schema=IOSpecialistHandoff,
    description="Expert on reducing I/O-bound bottlenecks via memory management and overlapping"
)


async def run(context: Dict[str, Any]) -> IOSpecialistHandoff:
    """Main entry point for I/O specialist."""
    # Call tools to select a technique from io_techniques.yaml
    # Return handoff
    return IOSpecialistHandoff(
        technique_id="overlap_compute_and_transfer",
        narrative="Use async streams to hide PCIe latency.",
        estimated_speedup=1.25
    )
```

Create `perfxpert/fence/slices/io_specialist.md` (≤ 400 lines):

```markdown
# I/O Specialist Agent

## Purpose
You are the I/O specialist. When the Analysis agent identifies an 
I/O-bound bottleneck (memory-controller or PCIe saturation), you 
recommend techniques to reduce the I/O stall time.

## Tools
1. `bottleneck()` — confirms the I/O-bound classification
2. `query_knowledge()` — fetches `io_techniques.yaml` entries

## Instructions

Analyze the input metrics:
- If `memory_controller_percent_busy > 70` and `kernel_occupancy < 0.5`:
  Focus on techniques that reduce bandwidth pressure (overlap or compression).
- If `pcie_utilization_percent > 60`:
  Recommend PCIe optimization (async transfers, pinned memory).

Choose one technique and explain the specific code change needed.
Estimate speedup conservatively (1.1–1.5× range).

Output your recommendation as a handoff specifying:
- `technique_id` (from io_techniques.yaml)
- `narrative` (1–2 sentences on how to apply it)
- `estimated_speedup` (float in [1.1, 1.5])
```

**Tests:** Add to `tests/test_agents/test_io_specialist.py`:

```python
# SKIP-SAMPLE — walkthrough: perfxpert.agents.io_specialist does not yet exist
import pytest
from perfxpert.agents.io_specialist import run, AGENT_SPEC

async def test_io_specialist_spec_valid():
    assert AGENT_SPEC.name == "io_specialist"
    assert len(AGENT_SPEC.tools) <= 5

async def test_io_specialist_runs_and_returns_handoff():
    context = {
        "metrics": {"memory_controller_percent_busy": 85, "kernel_occupancy": 0.3},
        "kernel_name": "my_kernel",
    }
    result = await run(context)
    assert result.technique_id in ["overlap_compute_and_transfer", "pinned_memory_pool"]
    assert result.estimated_speedup > 1.0

def test_io_specialist_fence_under_400_lines():
    with open("perfxpert/agents/fence/io_specialist.md") as f:
        lines = f.readlines()
    assert len(lines) <= 400
```

## Step 5: Register I/O specialist in Recommendation agent

Edit `perfxpert/agents/recommendation.py` and add `io_specialist` to the
handoff allowlist. (This is already done if you edited `bottleneck_types.yaml`
with `specialist_agent: "io_specialist"` — the recommendation router will
auto-register it.)

If manual, add:

```python
# SKIP-SAMPLE — walkthrough snippet (context has no AGENT_SPEC defined)
from perfxpert.agents import io_specialist

AGENT_SPEC.tools.append(io_specialist)
```

**Tests:** Existing integration tests in `tests/test_integration/` will exercise
this automatically. No new test needed.

## Step 6: Add a proven-optimization case (optional but recommended)

Append to `perfxpert/knowledge/proven_optimizations.yaml`:

```yaml
- id: io_bound_async_streams
  bottleneck_type: io_bound
  technique: "Async stream overlap (PCIe + compute)"
  description: >
    Using HIP streams to overlap GPU compute with host-to-device transfers.
    Baseline: sequential transfers followed by kernel. Optimized: transfer 
    while prior kernel runs.
  measured_speedup_range: [1.15, 1.30]
  source_citation: "internal-experiment-2026-Q2-MI300X-003"
  preconditions:
    - {metric: "pcie_utilization_percent", op: ">", threshold: 50}
  failure_modes:
    - "Synchronization overhead dominates if kernel < 10ms"
    - "Host CPU cannot feed transfers fast enough"
  fixture_pair:
    baseline_db: "tests/fixtures/proven_optimizations/io_bound_async_streams.baseline.db"
    optimized_db: "tests/fixtures/proven_optimizations/io_bound_async_streams.optimized.db"
    description_md: "tests/fixtures/proven_optimizations/io_bound_async_streams.md"
```

(Fixture creation is a separate task; see `docs/contributing/proven_optimizations.md`.)

## Step 7: Add stub tests

Run the test suite to verify your changes:

```bash
# SKIP-SAMPLE — walkthrough: tests don't exist yet in a fresh checkout
cd experimental/python/perfxpert

# Test the new bottleneck type
pytest tests/test_tools/test_bottleneck.py::test_classify_io_bound_high_memctrl -v

# Test the new knowledge entry
pytest tests/test_knowledge/test_io_techniques.py -v

# Test the agent spec
pytest tests/test_agents/test_io_specialist.py -v

# Integration: full pipeline includes io_specialist
pytest tests/test_integration/ -v -k io_bound
```

All tests must PASS.

## Verification checklist

Before committing:

- [ ] `perfxpert/knowledge/bottleneck_types.yaml` has `io_bound` entry
- [ ] `perfxpert/knowledge/io_techniques.yaml` and schema created
- [ ] `perfxpert/tools/bottleneck.py` updated with `io_bound` classifier
- [ ] `perfxpert/agents/io_specialist.py` and fence created
- [ ] Fence file ≤ 400 lines
- [ ] Tests added and passing (7 files total touched)
- [ ] `perfxpert/agents/recommendation.py` registered (auto or manual)
- [ ] `pytest` green on levels 0-4

## Files touched

1. `perfxpert/knowledge/bottleneck_types.yaml` — 1 line added
2. `perfxpert/knowledge/io_techniques.yaml` — new file
3. `perfxpert/knowledge/_schemas/io_techniques.schema.json` — new file
4. `perfxpert/tools/bottleneck.py` — ~10 lines added
5. `perfxpert/agents/io_specialist.py` — new file
6. `perfxpert/fence/slices/io_specialist.md` — new file
7. Tests: 3 new test files (bottleneck, knowledge, agent)

**Total:** 7 core files, 3 test files. No architectural review needed.

## Commit message

```
feat(perfxpert): add io_bound bottleneck class + IO specialist agent

- New bottleneck type: io_bound (memory-controller or PCIe saturation)
- IO specialist agent with async-transfer and pinned-memory techniques
- New knowledge: io_techniques.yaml + schema
- Classifier extension: detect io_bound from memory_controller_percent_busy + pcie_utilization
- Tests: 3 files (bottleneck, knowledge, agent)
- Fence: io_specialist.md ≤ 400 lines

Touches 7 files. No RFC required (narrow-scope addition under spec §2).
```

---

## Troubleshooting

### Test import errors

If `test_io_specialist.py` can't import `io_specialist`:
- Verify `perfxpert/agents/__init__.py` exports it
- Run `pip install -e .` to refresh the package

### YAML schema validation fails

- Ensure `.schema.json` is valid JSON Schema draft-07
- Run `jsonschema.validate()` offline to debug

### Classifier returns wrong bottleneck type

- Add debug print statements to `classify_from_metrics()`
- Check threshold values against real fixture metrics
- Use `pytest -vv` to see the full assertion

---

## Next steps

After this walkthrough:
1. **Commit** the 7 core files + tests
2. **Open a PR** with this commit
3. **Request 1 core reviewer** (not architectural, so no 3-person gate)
4. Once approved, **merge** into the main branch

The `io_bound` bottleneck is now part of the system and will automatically
be exercised by nightly benchmarking and regression tests.
