## ADDED Requirements

### Requirement: Bottleneck candidates expose category-native impact

The memory insight backend SHALL emit zero or more bottleneck **candidates** per kernel (after user-selected aggregation over dispatches). Each candidate MUST include: stable `reason_id`, `impact_kind` ∈ {`bw`,`latency`,`hit_util`,`stall`}, `impact_value` (float), and `unit` string describing the physical meaning of `impact_value`.

#### Scenario: Stall candidate carries stall-native severity

- **WHEN** a stall-related reason is evaluated for a kernel
- **THEN** the candidate MUST use `impact_kind` = `stall` and `impact_value` MUST represent stall-native severity (e.g. stall rate or cycle fraction) in the declared `unit`, not a BW-only scalar

### Requirement: Display families collapse redundancy with YAML-defined trees

Each candidate MUST declare `display_family_id`. For each family and aggregation scope, the backend SHALL apply a **fixed ordered rule set** (from gfx950 perf analysis YAML or arch tuning tables) so that **at most one** candidate per `display_family_id` is marked `display_selected=true` for UI/ranking purposes.

#### Scenario: TCP stall subtree picks a single winner

- **WHEN** multiple TCP-related reasons in the same `display_family_id` satisfy their predicates
- **THEN** exactly one candidate MUST have `display_selected=true` according to the ordered tree (more specific branch wins over generic parent)

### Requirement: Confidence and uncertainty states are explicit

Each candidate MUST include `confidence` in the closed interval [0, 1] (or documented discrete mapping) and `uncertainty_label` ∈ {`high`,`medium`,`low`,`insufficient_data`} derived from documented rules (e.g. margin beyond threshold, cross-metric agreement, missing inputs).

#### Scenario: Missing required inputs

- **WHEN** any required input metric for a candidate is absent
- **THEN** the candidate MUST be marked `uncertainty_label` = `insufficient_data` and MUST NOT be promoted to `primary` in exported summaries

### Requirement: Optional derived rank score is secondary to native impact

The backend MAY compute optional `rank_score` for merged ordering. If present, consumers MUST still be able to reproduce ordering rationale from `impact_value`, `impact_kind`, `confidence`, and family-winner flags without using `rank_score` alone.

#### Scenario: Optiq consumes native fields

- **WHEN** a consumer reads the export payload
- **THEN** `impact_value` and `unit` for the winning candidate MUST be present even if `rank_score` is omitted

### Requirement: Threshold highlights are distinct from raw candidacy

Predicate thresholds in YAML MAY set `threshold_highlight=true` on a candidate. Highlight MUST NOT be the only signal: continuous `impact_value` remains authoritative for severity ranking within a category.

#### Scenario: Below-threshold but non-zero stall

- **WHEN** stall severity is below the “elevated” threshold
- **THEN** the candidate MAY still exist with `threshold_highlight=false` and non-zero `impact_value` if the pipeline emits continuous candidates

### Requirement: Engineering-owned hint strings

Optional field `hint_text` MAY be attached per `reason_id` sourced from YAML. If absent, consumers SHALL omit hints rather than synthesize product copy in the backend core.

#### Scenario: Hint present only for selected reasons

- **WHEN** `display_selected=true` and YAML defines `hint_text` for that `reason_id`
- **THEN** the export MUST include `hint_text` exactly as authored
