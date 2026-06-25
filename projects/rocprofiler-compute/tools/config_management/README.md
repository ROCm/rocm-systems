# ROCProfiler-Compute Configuration Management

This directory contains the authoritative configuration-management system for ROCProfiler-Compute analysis configurations.

It is designed to guarantee:

- **Structural correctness** across GPU architectures
- **Byte-level immutability** of panel YAMLs enforced via hashes
- **CI enforcement** of all invariants

All workflows are orchestrated by a single sequential driver script:

```bash
tools/config_management/master_config_workflow_script.py
```

## Repository Layout

```bash
rocprofiler-compute/
├── src/rocprof_compute_soc/
│   └── analysis_configs/
│       ├── gfx908/
│       │   └── 0000_top_stats.yaml
│       ├── gfx90a/
│       ├── gfx940/
│       ├── gfx950/
│       ├── gfx115x/                     # RDNA 3.5 shared analysis configs
│       ├── gfx9_config_template.yaml    # CDNA (gfx9) panel contract
│       └── gfx11_config_template.yaml   # RDNA 3.5 (gfx115*) panel contract
│
├── src/utils/
│   ├── hash_checker.py
│   ├── .config_hashes.json
│
└── tools/config_management/
    ├── master_config_workflow_script.py
    ├── parse_config_template.py
    ├── verify_against_config_template.py
    ├── hash_manager.py
    └── README.md
```

## Core Concepts

### Panel YAMLs

- Live under:
```bash
analysis_configs/<arch>/*.yaml
```
- Must conform strictly to the template schema
- Are edited in-place using ruamel.yaml round-trip mode

### Hash Database

- Stored at:
```bash
src/utils/.config_hashes.json
```
- Records md5 hashes of panel YAMLs per arch
- Machine-generated only
- Enforced in CI and pytest

## Architecture Diagram (End-to-End Flow)
```text
                   ┌──────────────────────────┐
                   │  analysis_configs/       │
                   │  gfx9_config_template    │
                   └───────────┬──────────────┘
                               │
                               ▼
                 ┌───────────────────────────────┐
                 │ verify_against_config_template│
                 │ (structural validation)       │
                 └───────────┬───────────────────┘
                             │
                             ▼
                 ┌───────────────────────────────┐
                 │ hash_manager.py --compute-all │
                 │ (refresh .config_hashes.json) │
                 └───────────┬───────────────────┘
                             │
                             ▼
                 ┌───────────────────────────────┐
                 │ hash_checker.py               │
                 │ (panel-file hash consistency) │
                 └───────────────────────────────┘
```

## Contributor Quick Start

> [!NOTE]
> **Required Python Dependency**
> This configuration management system requires the `ruamel.yaml` Python package.
> It is used to safely modify YAML files while preserving comments, ordering,
> and formatting. The workflow scripts will not function correctly without it.
>
> See [Development dependencies](../../README.md#development-dependencies) for install instructions.

### 1. Validate the current state

Before making **any** config changes:
```bash
python tools/config_management/master_config_workflow_script.py --validate-only
```

This must pass.

### 2. Editing an existing architecture

Edit panel YAMLs **directly** under:
```bash
src/rocprof_compute_soc/analysis_configs/<arch>/
```

Rules:

- Preserve structure
- Preserve ordering
- Use multiline `>-` formatting for metric descriptions
- Do **not** regenerate entire files

After editing:
```bash
python tools/config_management/master_config_workflow_script.py --validate-only
python tools/config_management/hash_manager.py --compute-all src/rocprof_compute_soc/analysis_configs
```

The first command verifies structural conformance. The second refreshes the hash DB so `hash_checker.py` agrees.

### 3. Hash checks (fast local / CI)
```bash
python tools/config_management/master_config_workflow_script.py --hash-only
```

or:
```bash
python tools/config_management/master_config_workflow_script.py --ci
```

This runs panel-file hash validation only.

## Automated Testing

### Pytest Hash Integrity Test

Located at:
```bash
tests/test_autogen_config.py
```

This test:

- parses `.config_hashes.json`
- verifies **byte-for-byte** integrity of panel YAMLs
- fails on:
  - missing files
  - changed content
  - stale hash DB

## Contributor Rules (Strict)

- Do **not** edit `.config_hashes.json` manually
- Do **not** regenerate full YAMLs unnecessarily
- Use in-place edits (ruamel round-trip)
- Expect CI to reject inconsistent states

## Metric Descriptions

Metric descriptions flow unidirectionally from panel YAMLs to documentation:

```
Panel YAMLs (src/) → Per-Arch YAMLs (tools/) → Docs YAMLs (docs/) → Sphinx HTML
```

**Files:**
```bash
tools/per_arch_metric_definitions/
  ├── gfx{908,90a,942,950,1151}_metrics_description.yaml   # plain + rst + unit

docs/data/metrics/
  └── <arch>_metrics.yaml                                  # per-arch rst + unit (generated)
```

**After editing panel `metrics_description` sections:**
```bash
python tools/config_management/metric_description_manager.py --sync-all src/rocprof_compute_soc/analysis_configs
python tools/config_management/metric_description_manager.py --generate-docs
```

To regenerate the docs YAML for a single architecture without rewriting the others:

```bash
python tools/config_management/metric_description_manager.py --generate-docs --docs-arch <arch>
```

**Manual RST edits:** Edit per-arch YAMLs directly. The framework preserves an edit when its `rst` differs from `plain`.
