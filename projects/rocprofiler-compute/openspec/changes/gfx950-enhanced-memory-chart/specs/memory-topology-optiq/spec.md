## ADDED Requirements

### Requirement: Topology document is versioned and arch-scoped

The topology export MUST include `schema_version`, `topology_version`, and `gpu_arch` (e.g. `gfx950`). Consumers MUST reject or warn on unknown incompatible `schema_version` per agreed Optiq rules.

#### Scenario: gfx950 topology load

- **WHEN** Optiq loads a topology JSON with `gpu_arch` = `gfx950`
- **THEN** the document MUST parse `schema_version` and MUST use gfx950 node/edge sets as defined in the joint schema

### Requirement: Stable identifiers for nodes and edges

Every drawable block MUST have a stable `node_id`. Every connective path shown as an edge MUST have a stable `edge_id`. Display labels MAY differ from `node_id` but MUST NOT be used as the primary key in metric bindings.

#### Scenario: Metric binding references node

- **WHEN** a metric is bound to the GL1-equivalent block
- **THEN** the binding MUST reference `node_id` agreed with Optiq, not only a human label such as “Vector L1 Cache”

### Requirement: Bindings declare metric slots per node and edge

The topology MUST include a `bindings` object mapping `node_id` and optionally `edge_id` to ordered lists of `metric_slot` entries. Each slot MUST declare: `slot_kind` ∈ {`bw`,`latency`,`hit_util`,`stall`}, `metric_id` or `reason_id`, and optional `role` (e.g. `primary` | `diagnostic`).

#### Scenario: BW slot on an edge

- **WHEN** a BW metric applies to the link between L1 and L2
- **THEN** bindings MUST attach that metric to the corresponding `edge_id` with `slot_kind` = `bw`

### Requirement: Joint schema ownership with Optiq

Breaking changes to `schema_version` or required fields MUST be coordinated with Optiq. rocprofiler-compute SHALL document a changelog entry for each incompatible bump.

#### Scenario: Field deprecation

- **WHEN** a required JSON field is deprecated
- **THEN** the change MUST specify a **Migration** note in the changelog and maintain at least one transitional release if Optiq requires it

### Requirement: Separation of topology from per-run values

Topology JSON MUST NOT embed per-kernel metric values. Per-run data MUST flow through the analysis metric stream or companion payload referencing the same `metric_id` / `reason_id` keys as bindings.

#### Scenario: Same topology for multiple runs

- **WHEN** two workloads share gfx950
- **THEN** an identical topology document MAY be reused with different per-kernel metric value exports
