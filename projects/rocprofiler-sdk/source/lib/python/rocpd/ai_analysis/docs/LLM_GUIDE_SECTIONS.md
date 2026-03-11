# LLM Reference Guide — Section Tagging System

This document explains how `llm-reference-guide.md` is split into context-aware
sections to reduce per-call LLM token cost by 18–51%.

---

## Why Section Filtering Exists

The full reference guide is ~72 KB / ~18,000 tokens. Sending it with every LLM call
is wasteful: a Tier 1 trace-only analysis does not need the Hardware Counter Reference
(7,979 chars) or the Compiler Optimization section (10,873 chars).

Context-aware filtering selects only the sections relevant to the current analysis,
saving 18–51% of token cost depending on the scenario:

| Scenario | Approx. token saving |
|----------|---------------------|
| Tier 1 trace-only | ~47% |
| Tier 0 source-only | ~51% |
| Tier 1 + compiler trigger | ~32% |
| Tier 2 full analysis (no compiler) | ~18% |

---

## Tag Vocabulary

Each `## Section` in `llm-reference-guide.md` carries a tag comment on the line
immediately after the heading:

```text
## Hardware Counter Reference
<!-- rocpd-context: tier2 -->
```

| Tag | Meaning | Sections |
|-----|---------|----------|
| `always` | Included in every LLM call | Critical rules, role, output format, what not to do, summary |
| `tier1` | Trace data available (Tier 1+) | Profiling workflow, tool reference, common bottleneck types |
| `tier2` | PMC counter data available | Hardware counters, memory hierarchy, perf models, GPU specs, AMD optimizations |
| `compiler` | Compiler optimization is relevant | Compiler Optimization Flags and Options |
| `source` | Reserved for future Tier 0 guidance | *(empty — no sections use this tag yet)* |

**Fallback rule:** A section with **no tag comment** is always included. This
ensures user-added sections are never silently dropped.

---

## `AnalysisContext` Fields

`AnalysisContext` (importable from `rocpd.ai_analysis`) tells the system which tags
to activate:

| Field | Type | Controls |
|-------|------|---------|
| `tier` | `int` | `0` → source + compiler tags; `1` → tier1; `≥2` → tier1 + tier2 |
| `has_counters` | `bool` | `True` adds `tier2` even when `tier == 1` |
| `bottleneck_type` | `str \| None` | `"compute"` or `"memory"` adds `compiler` tag |
| `gpu_arch` | `str \| None` | Reserved for future per-GPU section filtering |
| `custom_prompt` | `str \| None` | Adds `compiler` tag when it contains compiler/flag/build/compile |

---

## How to Add a New Section

1. Add the section to `llm-reference-guide.md` with a `## ` heading.
2. On the **very next line** (line 1 of the section body), add:
   ```
   <!-- rocpd-context: TAG -->
   ```
   where TAG is one of the known vocabulary values above.
3. If unsure which tag to use, use `always` — the section will always be included.
4. Run the integrity tests to confirm no typos:
   ```bash
   PYTHONPATH=/opt/rocm-7.2.0/lib/python3.12/site-packages \
   pytest --noconftest tests/rocprofv3/rocpd/test_guide_filter_standalone.py \
     -v -k "TestGuideIntegrity"
   ```

---

## How to Add a New Tag

1. Add the new tag to `_select_tags()` in `source/lib/python/rocpd/ai_analysis/llm_analyzer.py`.
2. Add the tag to `TestGuideIntegrity.KNOWN_TAGS` in `tests/rocprofv3/rocpd/test_guide_filter_standalone.py`.
3. Add a row to the tag vocabulary table above.
4. Update the `AnalysisContext` docstring if the new tag is driven by a new field.

---

## Debugging: Verbose Mode

Pass `verbose=True` to `LLMAnalyzer` to see which sections were loaded:

```python
analyzer = LLMAnalyzer(provider="anthropic", api_key="...", verbose=True)
analyzer.analyze_with_llm(data, context=ctx)
# → [LLM] Guide filtered: 34800 / 72513 chars (48% of full guide)
```

---

## Tag Selection Logic (for reference)

```
tier == 0                          → always + source + compiler
tier >= 1                          → always + tier1
has_counters == True OR tier >= 2  → also adds tier2
bottleneck_type in compute/memory  → also adds compiler
custom_prompt contains
  compiler/flag/build/compile      → also adds compiler
```
