## ADDED Requirements

### Requirement: gfx950 enhanced memory view uses Pattern D and compressed Pattern A

For gfx950, the CLI or TUI memory presentation SHALL place **BW-oriented** indicators on **key memory path edges** (Pattern D) and **latency**, **hit/util**, and **stall** information as **compressed** per-node slots on diagram blocks where terminal width allows (Pattern A). If layout constraints prevent inline slots, the implementation MUST fall back to node labels plus the post-chart section without dropping data.

#### Scenario: Narrow terminal fallback

- **WHEN** the effective terminal width is below a documented minimum for inline four-slot rows
- **THEN** the UI MUST still render the diagram and MUST emit the post-chart table with the same structured fields

### Requirement: Post-chart section carries full numbers and hints

Immediately after the memory chart, the presentation MUST include a **post-chart** section listing, per relevant block or edge: native metrics for BW / latency / hit-util / stall (as available), **only** `display_selected` stall or bottleneck reasons per `display_family_id`, and optional `hint_text` when provided by the backend.

#### Scenario: Non-selected family members are hidden from hints

- **WHEN** a family has multiple true raw reasons but only one `display_selected` winner
- **THEN** the post-chart section MUST NOT list non-selected siblings as separate user-facing bottlenecks

### Requirement: Primary vs possible contributors

The post-chart section MUST include a short ranked summary with at most one **primary** item (highest confidence-adjusted native impact among winners, per product rules) and up to four **possible contributors**, each showing `uncertainty_label` and native `impact_value` + `unit`.

#### Scenario: User receives prioritized pointers

- **WHEN** at least one candidate has `uncertainty_label` other than `insufficient_data`
- **THEN** the summary MUST list a primary and/or contributors in ranked order without exceeding the documented limits

### Requirement: gfx115x-style presentation patterns may be reused in TUI

The TUI implementation SHOULD reuse or adapt **gfx115x** memory chart presentation patterns (e.g. Rich layout, grouping, scroll regions) where they improve readability, without requiring the gfx9 ASCII renderer to match gfx115x pixel-for-pixel.

#### Scenario: Arch routing unchanged for non-gfx950

- **WHEN** `gpu_arch` is not gfx950
- **THEN** the enhanced presentation behavior MUST NOT alter existing memory chart behavior unless separately specified
