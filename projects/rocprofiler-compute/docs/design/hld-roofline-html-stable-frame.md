# Stable Axes for Interactive Roofline HTML

## System context

`rocprof-compute analyze` produces a self-contained empirical roofline HTML
artifact. Python computes kernel arithmetic-intensity points, bandwidth roofs,
and compute ceilings as Plotly traces. The generated page embeds Plotly.js and
a custom controller that provides kernel filtering, memory-level selection,
compute-precision selection, zoom controls, theme selection, and PNG export.

The post-[#10723](https://github.com/ROCm/rocm-systems/pull/10723) design
consolidates floating-point and integer roofs into one HTML file and provides a
multi-select **Precision** control. That control chooses which compute ceilings
are visible. It does not choose which precision is used to calculate a kernel's
percent-of-roofline or limiter. The proposed follow-up control for kernel
comparison precision is intentionally outside this design.

Before this change, Python and JavaScript independently fitted axes to the
current run's kernel points. JavaScript repeated that fit on page load, reset,
double-click, and resize.

## Problem statement

Data-fitted axes prevent visual comparison:

- Two workloads from the same GPU open with different bounds when their kernel
  points differ.
- A faster kernel can appear in nearly the same screen position because the
  axes move with it.
- The same HTML can open with different bounds at different window sizes.
- Python and JavaScript contain separate framing recipes that can diverge.

The design explored in
[#11103](https://github.com/ROCm/rocm-systems/pull/11103) sought to make the
frame independent of kernel data. That direction is correct, but the branch did
not land cleanly on current develop. This design and
[#11418](https://github.com/ROCm/rocm-systems/pull/11418) port the same goal
onto post-[#10723](https://github.com/ROCm/rocm-systems/pull/10723) standalone
HTML and close the blocking defects found in review.

## Why PR 11103 was not sufficient

PR 11103 identified the right invariant — axes must not depend on kernel
points — but several implementation and integration defects prevented merge.
The table below contrasts the earlier branch with this design.

| Area | PR 11103 | PR 11418 (this design) |
| --- | --- | --- |
| Integration base | Built through older develop merges while [#10723](https://github.com/ROCm/rocm-systems/pull/10723) precision work was still landing | Rebased on current `rocprofiler-compute-develop` with the one-file Precision selector already present |
| Standalone HTML loads | **Blocking:** `buildOffPlotBadge()` in `roofline_plot.js` is missing a closing `}`; `node --check` fails and the controller never runs | Controller passes syntax checks and is exercised by a Node harness with Plotly stubs |
| Precision selector | Dead first `buildPrecisionOptions()` helper; lowering the selected compute cap could leave bandwidth-roof samples above the new knee | Immutable per-roof source coordinates; every precision change rebuilds roofs without exceeding the selected cap |
| `roofline.csv` device rows | `machine_ceilings()` and `calc_ceilings()` could resolve different rows when device IDs are sparse or reordered | Shared `RooflineCsvData` parsing; one semantic device ID → row index used everywhere |
| CSV validation | Device ID treated as a row index in roof construction | Semantic device-ID match; reject malformed row widths, duplicate headers, and duplicate device IDs |
| Invalid ceiling cells | Sanitized when collecting ceilings, but `calc_ceilings()` still called `float()` on `N/A`, `nan`, and `inf` | `sanitize_ai_value()` applied consistently in ceiling collection and roof construction |
| Stacked datatype titles | `_extend_stacked_title()` defined but never called; stacked figures kept only the first datatype in the title | One combined FLOP+OP HTML document (post-10723); one canonical frame and subtitle for the whole page |
| Dash / WebUI | Regressed `roofline_data_type` wiring; WebUI compute peaks could be empty and mis-label limiters | Dash explicitly out of scope; standalone HTML is the supported path |
| Fallback subtitle | Still said “Axes fixed to this GPU” when `DEFAULT_AXIS_BOUNDS` was used | Tracks whether the frame came from machine ceilings and uses neutral fallback copy |
| Off-plot badge zoom | Zoomed only the current memory-level kernel points | Zooms all valid kernel points across memory levels |
| Zoom aspect | Fits each axis to the target points independently, so the angle where a bandwidth roof meets a compute ceiling changes with the zoom target | Zooms keep the canonical frame's decade ratio and only widen, so that angle holds |
| Kernel panel controls | Off-plot badge nested inside a row `role="button"` | Separate primary row action and off-plot badge controls for keyboard and screen readers |
| Async frame races | No guard against stale `Plotly.relayout` callbacks clearing a newer frame apply | Operation counter invalidates superseded relayout work |
| Browser behavior tests | No executable JS tests for reset, resize, Fit to data, or precision changes | `tests/unit/roofline/test_roofline_plot.py` runs the real controller in Node |
| Scope | Precision-selector documentation mixed into an axis-scaling change | Design doc, scope exclusions, and deferred kernel comparison-precision control documented explicitly |

Shared limitations that remain open in both efforts:

- Decade snapping can still move a bound by one decade when benchmark inputs sit
  near a power-of-ten cliff.
- One combined frame still pools FLOP and OP peaks, which can leave unused
  vertical space for lower-precision views.
- View-model trace indices remain coupled to Plotly trace insertion order.
- Terminal `plotext` output still data-fits axes; only standalone HTML is
  stabilized here.

## Requirements

### Functional

- Derive the opening frame only from bandwidth and compute ceilings in
  `roofline.csv`.
- Snap all four bounds to complete base-10 decades.
- Use the same canonical frame for every kernel set and every visible precision
  on one GPU.
- Ship the frame in both the Plotly layout and the embedded page model.
- Make load, reset, and double-click return to the canonical frame.
- Permit resize to widen an axis for readability, but never crop the canonical
  frame.
- Keep off-frame kernel points at their true coordinates.
- Warn about off-frame kernels and add a kernel-panel badge that zooms to one.
- Add a one-shot **Fit to data** action. Reset must undo that zoom.
- Keep the canonical frame's decade ratio in every zoom, so the angle between a
  bandwidth roof and a compute ceiling does not change with the zoom target.
- Include the canonical bounds in the plot subtitle so exported images identify
  the comparison frame.
- Preserve the post-[#10723](https://github.com/ROCm/rocm-systems/pull/10723)
  compute-roof Precision selector without adding a kernel
  comparison-precision selector.

### Non-functional

- Keep the generated HTML self-contained and usable without a server.
- Keep deterministic frame calculation separate from CSV file I/O.
- Do not require Dash/WebUI compatibility work; that frontend is outside scope.
- Do not clamp, synthesize, or otherwise misrepresent kernel coordinates.
- Preserve Python 3.8-compatible syntax and existing Ruff rules.

## Design

### Canonical frame

The frame is a pure function of all positive finite base bandwidth and compute
ceiling values for the selected device:

```text
x_low  = 1e-2
x_high = 10 ** ceil(log10(max_peak / min_bandwidth))
y_low  = 10 ** floor(log10(x_low * min_bandwidth))
y_high = 10 ** ceil(log10(max_peak))
```

Degenerate spans are widened by one decade. If usable bandwidths or peaks are
missing, the caller uses the existing default frame, labels the subtitle
`Default axes - benchmark ceilings unavailable`, and emits a CSV-reading
warning.

`machine_ceilings` is the thin I/O wrapper. It reads `roofline.csv`, selects the
row whose device ID equals the requested device, and returns usable bandwidths
and peaks. Parsing rejects malformed row widths, duplicate headers or device
IDs, and non-integral device IDs rather than silently assigning ceilings to the
wrong device. Ordinary roof construction uses the same whole-file validation.
Failing closed prevents a malformed row or ambiguous device mapping from
silently combining ceilings from different devices. `canonical_frame` performs
no I/O.

### Python ownership

Python calculates one canonical frame before constructing the Plotly figures.
The same tuple is:

1. converted to log10 ranges in the Plotly layout;
2. represented in data coordinates as `{"x": [low, high], "y": [low, high]}`
   in `RooflineViewModel`; and
3. rendered as a title subtitle.

Kernel data is not an input to this calculation. Changing the visible compute
precisions therefore cannot move the canonical frame.

Python checks every kernel point against the canonical bounds. Each point stays
unchanged. A warning identifies kernels with points outside the frame and
describes the signed overflow in decades.

### Browser ownership

The browser converts `model.frame` to Plotly log-axis coordinates. Load, reset,
and double-click use that range. `shapeToPlotArea` may widen one axis to keep
diagonal roofs readable in the current viewport, but cannot return a range
narrower than the canonical frame.

Two explicit one-shot zoom operations are separate from canonical framing:

- **Fit to data** frames all currently drawn kernel points.
- An off-plot badge is visible when a currently drawn point for its kernel is
  outside the canonical frame. Clicking it frames all valid points for that
  kernel across memory levels.

These operations mark the view as manually framed. A resize preserves them.
Reset or double-click restores the canonical frame.

Both zoom operations keep the canonical frame's ratio of x decades to y
decades, widening whichever axis is too narrow. A roof is drawn as a slope, so
that ratio decides the angle at which a bandwidth roof meets a compute ceiling.
Fitting each axis to the points independently would redraw that knee whenever a
kernel moved further in performance than in intensity. Because the adjustment
only widens, the zoomed points stay in view.

Off-plot status is measured against the canonical frame, not viewport padding,
so badges and counts do not change with window size.

### Precision selector

The post-[#10723](https://github.com/ROCm/rocm-systems/pull/10723) Precision
selector remains a multi-select visibility control for compute-ceiling traces.
It may extend visible bandwidth roof segments to the highest selected ceiling,
but it does not:

- change the canonical axis frame;
- choose the compute precision used for kernel comparison; or
- change kernel coordinates.

The opening selection belongs to the server. `_combined_html_figure` records it
in the model as `defaultPrecisions` and paints the figure to match: ceilings for
unselected precisions ship hidden, and bandwidth roofs ship clipped to the
selected cap. Without that, the document would paint every ceiling at the
highest cap on load and then drop to one precision once the controller's first
restyle landed. Each roof also carries its full `sampleAi` grid, so selecting a
taller precision re-extends the diagonal at the original sample density rather
than from the clipped trace. The Dash figures are left untouched, because the
WebUI has no Precision control to restore a hidden ceiling.

The separate kernel comparison-precision dropdown discussed alongside
[#10723](https://github.com/ROCm/rocm-systems/pull/10723) is deferred to a
future design.

### Scope exclusions

- Dash/WebUI-specific changes.
- Persisted frame files or a `--roofline-frame` argument.
- Fixed CSS aspect ratio.
- Cross-machine frame selection.
- Trace-index identity refactoring.

## Implementation phases

1. Add pure frame calculation, ceiling loading, and unit tests.
2. Apply the frame to Plotly and the page model; test layout/model agreement.
3. Replace browser data fitting with canonical reset and one-shot zoom actions.
4. Add off-plot warnings, badges, subtitle, and focused tests.
5. Generate a standalone demo HTML from test workload data.

## Validation and debuggability

- Unit-test decade snapping, invalid ceilings, degenerate spans, and signed
  off-frame overflow.
- Build two figures with different kernel sets and assert identical opening
  ranges and embedded frames.
- Assert off-frame points retain their original coordinates.
- Parse the embedded page model and compare it with Plotly layout ranges.
- Run `node --check` on the controller.
- Run focused roofline unit and integration tests, Ruff, and formatting checks.
- Generate an HTML artifact and verify that its model contains the canonical
  frame and precision list, and that the controller contains the new controls.
- The local demo was generated through production
  `Roofline.construct_plotly_figures`, `_combined_html_figure`, and
  `build_interactive_document`, using the committed MI200 `roofline.csv` plus
  representative inline kernel data. There is no checked-in generator script.

## Open questions and deferred work

- Empirical measurements near a decade boundary can still select adjacent
  frames across runs. Persisting or explicitly supplying a frame is the robust
  future solution.
- Combining floating-point and integer peaks gives one stable frame for the
  single HTML document, but may leave unused space for lower-precision views.
- View-model trace indices remain coupled to Plotly trace insertion order.
- The subtitle reports canonical bounds. A viewport may display wider padded
  bounds, by design.
