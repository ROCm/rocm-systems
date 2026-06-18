## 1. Joint schema and topology

- [ ] 1.1 Align with Optiq on draft JSON: `schema_version`, `topology_version`, `gpu_arch`, `node_id`, `edge_id`, and `bindings.metric_slot` shape
- [ ] 1.2 Publish gfx950 reference topology document (nodes, edges, GL1/GL2/EA mapping to `node_id`) in repo or shared registry path agreed with Optiq
- [ ] 1.3 Document changelog policy for incompatible schema bumps

## 2. Backend: candidates, impact, families, confidence

- [ ] 2.1 Define export schema (JSON or equivalent) for `reason_id`, `impact_kind`, `impact_value`, `unit`, `display_family_id`, `display_selected`, `confidence`, `uncertainty_label`, optional `hint_text`, optional `rank_score`
- [ ] 2.2 Implement gfx950 candidate generation from mem-bw analysis outputs with **category-native** impact fields
- [ ] 2.3 Encode **fixed exclusion trees** in `gfx950` YAML (or tuning tables): ordered winner rules per `display_family_id`
- [ ] 2.4 Implement family winner selection and merge policy for cross-category Top-N (primary + contributors) per design
- [ ] 2.5 Wire export into analysis run / DB or sidecar as agreed (minimize schema churn; prefer metrics + JSON companion if possible)

## 3. Reference validation

- [ ] 3.1 Author **reference workloads** and expected invariants (family winner, acceptable top sets, uncertainty states)—not omniscient labels
- [ ] 3.2 Automate checks where feasible (unit tests or golden JSON comparison on synthetic inputs)

## 4. CLI / TUI presentation

- [ ] 4.1 Implement Pattern D (BW on key edges) + compressed Pattern A on nodes for gfx950, with narrow-terminal fallback
- [ ] 4.2 Add post-chart section: full numbers, `display_selected` reasons only, optional hints, primary + contributors summary
- [ ] 4.3 Reuse or adapt **gfx115x** memory chart presentation patterns in TUI where stack allows; preserve behavior for non-gfx950 arches

## 5. User study readiness

- [ ] 5.1 Define user-study protocol (tasks, success metrics, build pin) referencing this change
- [ ] 5.2 Run pilot sessions and capture findings for threshold/tree tuning backlog