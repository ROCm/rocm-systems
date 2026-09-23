# Roofline: "Limiting" option for the AI axis dropdown

## Problem Statement

In the interactive roofline HTML report, each kernel can appear as multiple dots on the chart — one per memory (cache) level it has traffic for (e.g. L0, L1, L2, HBM, LDS), each plotted at that level's own arithmetic intensity (AI) with the kernel's performance. The "AI axis" dropdown lets a user pick one specific level to display, or "All peaks" to show every level's dot for every kernel at once.

There is no option that shows, for every kernel simultaneously, only the one dot that actually reflects its bottleneck. A user who wants to scan the whole chart for "what's actually limiting each kernel" today has to either pick a single memory level (which may not be the level that kernel is actually bound by) or view "All peaks" and mentally filter the clutter of every kernel's non-limiting dots. This mirrors the trade-off between Intel Advisor's single-dot CARM view and its multi-dot Integrated/Memory-Level roofline view, discussed as prior art earlier in this workstream.

## Solution

Add a new `"Limiting"` entry to the existing "AI axis" dropdown. When selected, every kernel is drawn with exactly one dot: the point corresponding to whichever memory level (or, for compute-bound/unknown kernels, a well-defined fallback point) the backend has already identified as that kernel's limiter. This reuses the limiter/limiter_category detection already computed for the hover tooltip ("Limited by ...") — it is not new bottleneck-detection logic, only a new way to visualize the existing determination.

## User Stories

1. As a performance engineer scanning a roofline chart with many kernels, I want an option to show only each kernel's limiting dot, so that I can see at a glance which kernels are bandwidth-bound vs. compute-bound without visual clutter from non-limiting cache levels.
2. As a performance engineer, I want the "Limiting" option to sit in the same "AI axis" dropdown I already use to switch between cache levels, so that I don't have to learn a new control.
3. As a performance engineer, I want the existing per-level options and "All peaks" to keep working exactly as before, so that adding this option doesn't change behavior I already rely on.
4. As a performance engineer, I want the default dropdown selection to remain unchanged (still the current default level, e.g. HBM if present), so that opening a report I've already generated doesn't silently change its appearance.
5. As a performance engineer, I want a memory-bound kernel's lone dot in "Limiting" mode to sit exactly at that cache level's own AI/performance point, so that its horizontal position still means what it means everywhere else on the chart.
6. As a performance engineer, I want a compute-bound kernel (no single cache level is its bottleneck) to still show exactly one dot rather than vanishing from the chart, so that I don't lose visibility into compute-bound kernels while in this mode.
7. As a performance engineer, I want that fallback dot (for compute-bound or "Unknown"-limiter kernels) to be the leftmost (lowest-AI) of the kernel's available points, so that the fallback is deterministic and documented rather than arbitrary.
8. As a performance engineer, I want the dot's color to still reflect its cache level (the same color coding used everywhere else, including in single-kernel isolation view), so that color continues to mean the same thing across every view of the chart.
9. As a performance engineer, I want the hover tooltip on a "Limiting"-mode dot to be unchanged (still "Limited by {category}: {name}", Performance, AI, Details/Bandwidth/Dispatch/Duration), so that the information I already rely on in hover text isn't disrupted by this new mode.
10. As a performance engineer, I want to click/isolate a single kernel while in "Limiting" mode and see all of that kernel's cache-level dots (not just the limiting one), so that isolating a kernel still gives me the full multi-level picture, consistent with how isolation already overrides the other specific-level dropdown options.
11. As a performance engineer, I want the "Limiting" mode to leave the bandwidth-roof lines and compute-peak (flat roof) lines completely untouched, so that the roofline geometry I compare dots against never changes based on the AI-axis selection.
12. As a performance engineer, I want "Limiting" mode to apply consistently across the whole report regardless of which datatypes are currently toggled visible via the separate "Precision" control, so that the two controls compose predictably instead of interacting in surprising ways.
13. As a developer maintaining this codebase, I want the "which single point is limiting" decision made once, server-side, in Python, so that the client-side JavaScript only needs a trivial lookup rather than re-implementing bottleneck-detection logic.
14. As a developer maintaining this codebase, I want the new field driving this decision to be unit-testable via the existing kernel-trace-building tests, so that regressions in limiter-to-dot mapping are caught the same way regressions in the existing limiter/hover logic already are.

## Implementation Decisions

- **New dropdown option**: add `"Limiting"` as an entry in the "AI axis" `<select>` (`roofline-peak-select`), positioned first in the option list (before the per-level entries, before "All peaks"). It is additive: existing per-level options, the `"All peaks"` option, and the current default selection logic are unchanged.
- **New special peak value**: introduce a new sentinel value (analogous to the existing `ALL_PEAKS_VALUE = "all"` sentinel) to represent this mode, e.g. a `LIMITING_PEAK_VALUE` constant, threaded through the same places `ALL_PEAKS_VALUE` currently is (model construction in Python, `buildPeakOptions()`/`effectivePeak()`/`pointsForCurrentPeak()` in JS).
- **Server-side resolution of the limiting point** (the core of this feature): in `_build_kernel_traces`, at the point where `_determine_kernel_limiter` is already called once per kernel (using `level_ai` and `compute_peaks`), resolve its result to a concrete point selector and add it to the kernel's model as a new plain field (e.g. `limitingPeak`), holding a value that matches one of that kernel's own `point["peak"]` values:
  - If `limiter_category == "Memory"`: `limitingPeak` = the limiter's cache-level name (already directly equal to one of the kernel's `point["peak"]` values, by construction of `_determine_kernel_limiter`).
  - If `limiter_category` is `"Compute"` or `"Unknown"`: `limitingPeak` = the `peak` of the kernel's leftmost point (the point with the minimum `ai` value among its own `points` list).
- **No change to limiter detection itself**: `_determine_kernel_limiter`'s algorithm (eligible candidates, minimum-roof selection, `("Unknown", "Unknown", None)` fallback) is unchanged. This feature only adds a downstream mapping from its existing result to a displayable point.
- **Client-side filtering**: `pointsForCurrentPeak(kernel)` (or equivalent) gains one more branch: when the effective peak is the new sentinel, return only the point whose `peak === kernel.limitingPeak`, instead of all points or a single specific-level match. No bottleneck logic is duplicated client-side — it's a direct equality filter against the pre-resolved field.
- **Kernel isolation unaffected**: `effectivePeak()`'s existing override (forcing `"all"` whenever exactly one kernel is isolated, regardless of the dropdown's value) is not special-cased for the new sentinel — it already applies uniformly to every non-`"all"` dropdown value, including this new one.
- **Coloring unaffected**: dot coloring continues to key off `point.peak` (`peakColors[point.peak]`) exactly as it does today for every other specific-level selection and for isolation view; no new color/style branch is introduced for the compute/unknown fallback case.
- **No change to roofline line rendering**: bandwidth-roof and compute-peak line traces are entirely independent of the AI-axis selection today and remain so.
- **No change to hover template**: `build_kernel_hover_template` and its inputs (`limiter`, `limiter_category`, per-kernel shared hover cells) are unaffected; the existing "Limited by ..." text already reflects the same underlying determination this feature now also uses to pick a dot.
- **Single combined figure**: confirmed the report renders one combined Plotly figure with one shared "AI axis" dropdown/state (Ops vs. Flops are trace subsets toggled by the separate "Precision" control, not two figures). No per-figure scoping is needed; the new option applies uniformly to whatever kernel traces are currently present.
- **Tooltip copy**: the informational tooltip text next to the "AI axis" label (`PEAK_TITLE`) should be updated to mention the new option, at the implementer's discretion for exact wording — not a decision point requiring further sign-off.

## Testing Decisions

- Tests should assert on externally observable output — the kernel `model` dict returned by the kernel-trace-building code path (`points`, `limitingPeak`, `hoverCells`) and/or the rendered `hovertemplate` string — not on internal call sequencing.
- **Primary seam**: the existing `kernel_traces()` test helper in `tests/unit/roofline/test_roofline_main.py`, which already wraps `_build_kernel_traces` and is used by the existing limiter tests (`test_kernel_traces_score_against_the_roof_that_binds_not_the_tallest_drawn`, `test_kernel_traces_name_the_specific_peak_not_the_tallest_stacked_one`, `test_kernel_traces_ignore_a_memory_roof_the_kernel_already_exceeds`, `test_kernel_traces_name_the_roof_that_binds`, `test_kernel_traces_share_one_performance_value_across_a_kernels_points`). New tests should follow the same pattern and assert the new `limitingPeak` field, covering:
  - A memory-bound kernel: `limitingPeak` equals the cache level `_determine_kernel_limiter` names.
  - A compute-bound kernel: `limitingPeak` equals the `peak` of the point with the lowest `ai` value.
  - An "Unknown"-limiter kernel (no ceiling/compute data, mirroring the existing "Unknown"/"Unknown" test case): `limitingPeak` falls back the same way as the compute-bound case.
  - A kernel with a single available point: `limitingPeak` trivially equals that point's `peak` regardless of category.
- **No JS test infrastructure exists in this repo** (no jest/vitest/node test runner; `roofline_plot.js` is currently only indirectly covered by Python tests that regex-match against the JS source text, e.g. `test_the_controller_looks_up_controls_the_page_renders`). This spec deliberately keeps the decision logic server-side so the client-side change (`point.peak === kernel.limitingPeak`) is trivial enough not to require introducing new JS test infrastructure. If a maintainer judges dedicated JS coverage is warranted anyway, that's a follow-up decision outside this spec, not a blocker for it.
- Existing tests asserting the current dropdown options, default peak selection, and "All peaks" behavior must continue to pass unmodified — this feature must not alter any of that existing behavior.

## Out of Scope

- Any change to how `_determine_kernel_limiter` decides what counts as a candidate roof/peak, or how it picks among candidates.
- Any change to bandwidth-roof or compute-peak line rendering.
- Any change to the "Precision" dropdown or its interaction model with dataset filtering.
- Any change to the hover tooltip content or ordering (already settled in prior work: Limited by → Performance → AI → Details → Bandwidth/Dispatch Count/Duration).
- A legend or on-chart visual distinction between "this dot is the true limiter" vs. "this dot is a leftmost fallback" for compute-bound/unknown kernels — the fallback is silent (no special styling), per the confirmed design.
- Introducing JS test infrastructure (jest/vitest/node) to this repo.
- Any change to per-kernel isolation's existing behavior of always showing all cache-level dots.
- Publishing this spec to an external issue tracker — per explicit request, this spec is written to a local markdown file instead.

## Further Notes

- This feature is intentionally a pure "visualization of an existing determination," not new bottleneck-detection logic — keeping the surface area small was an explicit design goal reached during the interview that produced this spec.
- The design was validated against Intel Advisor's CARM (single aggregate dot) vs. Integrated/Memory-Level Roofline (one dot per cache level) distinction as prior art; "Limiting" mode is closer in spirit to Advisor's Integrated model filtered down to just the binding level, rather than to CARM's L1-anchored cumulative-traffic AI (this codebase's per-level AI values are not being replaced or recomputed — only filtered down to one per kernel).
- The exact constant name (`LIMITING_PEAK_VALUE` or similar) and the new model field name (`limitingPeak` or similar) are naming suggestions, not binding decisions — an implementer should follow this codebase's existing naming conventions when landing them.
