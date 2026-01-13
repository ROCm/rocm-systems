# ROCProfiler-Compute Configuration Management

This directory contains the authoritative configuration-management system for ROCProfiler-Compute analysis configurations.

It is designed to guarantee:

- **Structural correctness** across GPU architectures
- **Deterministic deltas** relative to a single latest architecture
- **Byte-level immutability** enforced via hashes
- **Safe promotion** of a new latest architecture with rollback
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
│       │   ├── 0000_top_stats.yaml
│       │   └── config_delta/
│       │       └── <latest_arch>_diff.yaml
│       ├── gfx90a/
│       ├── gfx940/
│       ├── gfx950/                      # latest_arch
│       └── gfx9_config_template.yaml    # single source of truth
│
├── src/util/
│   ├── hash_checker.py
│   ├── .config_hashes.json
│
└── tools/config_management/
    ├── master_config_workflow_script.py
    ├── parse_config_template.py
    ├── verify_against_config_template.py
    ├── generate_config_deltas.py
    ├── apply_config_deltas.py
    ├── hash_manager.py
    ├── TESTING.md
    └── README.md
```

## Core Concepts
### Latest Architecture

- Exactly one architecture is considered *latest*
- Defined in:
```bash
src/rocprof_compute_soc/analysis_configs/gfx9_config_template.yaml
```

### Panel YAMLs

- Live under:
```bash
analysis_configs/<arch>/*.yaml
```
- Must conform strictly to the template schema
- Are edited in-place using ruamel.yaml round-trip mode

### Delta YAMLs

- Represent differences from latest → older architecture
- Live under:
```bash
analysis_configs/<older_arch>/config_delta/
```
- Exactly one delta file per arch
- Always named:
```bash
<latest_arch>_diff.yaml
```

### Hash Database

- Stored at:
```bash
src/utils/.config_hashes.json
```
- Records:
  - md5 hashes of panel YAMLs per arch
  - md5 hash of the delta YAML (or null for latest)
- Machine-generated only
- Enforced in CI and pytest

## Architecture Diagram (End-to-End Flow)
```pqsql
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
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌────────────────────┐               ┌──────────────────────┐
│ edit-existing mode │               │ promotion mode       │
│ (local dev only)   │               │ (authoritative path) │
└─────────┬──────────┘               └──────────┬───────────┘
          │                                     │
          ▼                                     ▼
┌────────────────────┐               ┌─────────────────────────────┐
│ generate / apply   │               │ parse_config_template.py    │
│ deltas manually    │               │ (update latest_arch)        │
└────────────────────┘               └──────────┬──────────────────┘
                                                 │
                                                 ▼
                               ┌──────────────────────────────────┐
                               │ generate_config_deltas.py        │
                               │ latest → all older arches        │
                               │ (<latest>_diff.yaml only)        │
                               └──────────┬───────────────────────┘
                                          │
                                          ▼
                               ┌──────────────────────────────────┐
                               │ verify_against_config_template   │
                               │ (post-promotion validation)      │
                               └──────────┬───────────────────────┘
                                          │
                                          ▼
                               ┌──────────────────────────────────┐
                               │ hash_manager.py --compute-all    │
                               │ (new steady state)               │
                               └──────────┬───────────────────────┘
                                          │
                                          ▼
                               ┌──────────────────────────────────┐
                               │ hash_checker.py                  │
                               │ (semantic consistency)           │
                               └──────────────────────────────────┘
```

## Contributor Quick Start

> [!NOTE]
> **Required Python Dependency**
> This configuration management system requires the `ruamel.yaml` Python package.
> It is used to safely modify YAML files while preserving comments, ordering,
> and formatting. The workflow scripts will not function correctly without it.
>
> Install it via:
> ```bash
> pip install ruamel.yaml
> ```

### 1. Validate the current state

Before making **any** config changes:
```bash
python tools/config_management/master_config_workflow_script.py --validate-only
```

This must pass.

### 2. Editing an existing architecture (most common)

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
```

### 3. Generating or applying deltas (advanced / optional)

For local experimentation only:
```bash
python tools/config_management/master_config_workflow_script.py --edit-existing
```

This mode:

- never updates the template
- never updates hashes
- always re-validates after application

### 4. Promoting a new latest architecture (rare, gated)

Promotion changes **global invariants** and must use the master script:
```bash
python tools/config_management/master_config_workflow_script.py --promote <latest_arch>
```

The script will:

1. Update `latest_arch` in the template
2. Regenerate deltas for all older arches
3. Remove stale delta files
4. Re-validate everything
5. Rebuild the hash database
6. Verify semantic consistency

If anything fails:

- all changes are rolled back
- no partial state remains

### 5. Hash checks (fast local / CI)
```bash
python tools/config_management/master_config_workflow_script.py --hash-only
```

or:
```bash
python tools/config_management/master_config_workflow_script.py --ci
```

This runs semantic hash validation only.

## Automated Testing
### Pytest Hash Integrity Test

Located at:
```bash
tests/test_autogen_config.py
```

This test:

- parses `.config_hashes.json`
- verifies **byte-for-byte** integrity of:
  - panel YAMLs
  - delta YAMLs
- fails on:
  - missing files
  - changed content
  - stale hash DB
Semantic correctness is enforced separately by `hash_checker.py`.

## Contributor Rules (Strict)

- Do **not** edit `.config_hashes.json` manually
- Do **not** create multiple delta files per arch
- Do **not** rename delta files arbitrarily
- Do **not** regenerate full YAMLs unnecessarily
- Use in-place edits (ruamel round-trip)
- Use the master script for promotions
- Expect CI to reject inconsistent states

## Summary

This system guarantees:

- A **single source of truth** for latest architecture
- Deterministic, reviewable deltas
- Stable diffs for Git review
- Hash-backed immutability
- Safe, transactional promotions
- CI-enforced correctness

All correctness flows through:
```bash
master_config_workflow_script.py
```
