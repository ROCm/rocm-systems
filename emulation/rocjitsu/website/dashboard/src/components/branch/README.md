# Published Branch Runs v11

`views/BranchRunsView.jsx` receives `{ data, state, onOpenComparison }`. It consumes validated schema-2 runs from `data.allRuns` and the three selectors in `src/data/branchSelectors.js`; it has no preview-data dependency. The shell owns fixture disclosure, resource retry/error UI, and page navigation.

## State boundary

`state.branchSelection` is the atomic `{ branch, candidateId, referenceId, manual, target, mode, detail }` snapshot. The view derives missing initial defaults without calling the setter. Every user selection passes a complete snapshot to `state.setBranchSelection`. Existing missing IDs remain visible and unavailable rather than changing the pair.

Branch changes select that branch's latest published execution and automatic reference. Candidate changes recompute only an automatic reference; manual overrides persist. Reference choices are explicitly manual, with a restore action. Full comparison receives exact `{ candidateId, baselineId, target, mode, suites }`, including an intentional empty suite selection. The integrator owns URL push/pop history.

## Composition

- `BranchList` / `BranchPicker`: activity-filtered published entries, selectable rows, scroll events, count-free fixed heading/activity note, internal scrolling. At 1000px available container width the 304px picker collapses leftward to a 44px rail; the same named, focusable control reopens it. Collapse only hides mounted picker content and never writes the pair/route or clears searches. Tablet uses a bounded list above detail; phone list/detail behavior remains separate.
- `BranchDetail` / `BranchRunSelector`: exact SHA and attempt identities, branch-scoped candidate search, all-published reference search, unavailable identity disclosure, local suite/search/sort state.
- `ConfigurationMatrix`: gfx1250/gfx950 columns, ST/MT rows using shared categorical shapes; unavailable cells remain inspectable. Its selected cell and difference rows use one pair/suite/search scope.
- `BenchmarkDifferences`: shared matched sums, separate signed Abs change (candidate minus baseline seconds) and Pct change columns with MUI sort headers, phone stacked entries and equivalent sort controls, exclusions, and full-comparison callback. Default sorting is signed seconds descending; each column toggles descending/ascending. Missing values stay last in either direction and test IDs break ties. Contiguous real-suite sections may recur to preserve global sorted order. Zero reference has no percentage. Empty coverage never renders success or synthesized totals.
- `BranchEnvironment`: key-union generic facts, literal scalar values, source clocks/attempts/configuration, safe supplied repository/PR links.
- `branchSelection.js` and `branchPresentation.js`: pure state/presentation helpers, covered separately from JSX rendering.

Phones use list/detail CSS states without unmounting the list or local filter state. The view saves/restores list scroll position and focus; its only effect updates UI focus/scroll, never route state.

## Verification boundary

Focused tests: `tests/unit/branch-v10-{selection,presentation,view,actions}.test.js`. RED→GREEN commands and outputs are recorded by the parent artifact recorder. Server rendering and direct callbacks verify semantics, not browser geometry. The integrator must run both-theme responsive/keyboard/browser-history checks under its exclusive build/browser lease, including actual desktop panel bottom alignment and phone Back scroll restoration. No production schema-2 publication currently exists; fixtures stay in test-only inputs and the shell's conspicuously marked fixture mode.
