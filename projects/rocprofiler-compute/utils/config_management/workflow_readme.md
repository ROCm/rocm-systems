# Architecture Configuration Workflow

This document explains the master workflow system for managing architecture-specific metric configurations.

## Overview

The workflow system manages changes to architecture configurations located in `src/rocprof_compute_soc/analysis_configs/gfx<arch>/`. It handles:

- **Metric changes** (additions, deletions, modifications)
- **Metric description changes** (plain text + RST documentation)
- **New architecture additions**
- **Template updates**
- **Config delta generation** for version control

## Files Overview

### Core Scripts

1. **`master_config_workflow_script.py`** - Main orchestrator script
2. **`hash_manager.py`** - Tracks file changes via MD5 hashes
3. **`metric_description_manager.py`** - Syncs metric descriptions across files
4. **`config_workflow.yaml`** - Configuration file
5. **`parse_config_template.py`** - Parses base config template from latest arch
6. **`generate_config_deltas.py`** - Generates config deltas between two archs
7. **`apply_config_deltas.py`** - Applies config deltas to genearte new arch configs
8. **`verify_against_config_template.py`** - Validates configs against template

## Quick Start

### Initial Setup

1. Create the hash database:
```bash
python hash_manager.py --compute-all src/rocprof_compute_soc/analysis_configs
```

2. Ensure `analysis_config_template.yaml` has metadata:
```yaml
latest_arch: gfx950
panels:
  - file: top_stats.yaml
    panel_id: 0
    ...
```

### Making Changes

Simply run the master workflow after making any changes:

```bash
python master_config_workflow_script.py
```

The script will:
- Detect what changed
- Prompt you for confirmation
- Apply changes
- Validate results
- Update all necessary files

### Dry Run Mode

To see what would happen without making changes:

```bash
python master_config_workflow_script.py --dry-run
```

## Usage Scenarios

### Scenario A: Add Metrics to Latest Arch (gfx950)

**Method 1: Direct Edit**

1. Edit `src/rocprof_compute_soc/analysis_configs/gfx950/0700_wavefront.yaml`
2. Add your metric to the appropriate table
3. Add description to `metrics_description` section
4. Run: `python master_config_workflow_script.py`
5. Answer prompts

**Method 2: Using Delta**

1. Create `src/rocprof_compute_soc/analysis_configs/gfx950/config_delta/gfx950_diff.yaml`:
```yaml
Addition:
  - Panel Config:
      id: 700
      title: Wavefront
    metric_tables:
      - metric_table:
          id: 701
          title: Wavefront Launch Stats
          metrics:
            - New Metric:
                avg: AVG(something)
                unit: Units
    metric_descriptions:
      New Metric:
        plain: "Description text"
        rst: "Description with :ref:`RST markup <link>`"  # Optional

Deletion:
  []

Modification:
  []
```

2. Run: `python master_config_workflow_script.py`

**What Happens:**
- Changes applied to gfx950
- Template updated
- Deltas regenerated for all previous archs (gfx940, gfx941, etc.)
- Metric descriptions synced to:
  - `utils/per_arch_metric_definitions/gfx950_metrics_description.yaml`
  - `docs/data/metrics_description.yaml`
- All archs validated
- Hashes updated

### Scenario B: Modify Metrics in Older Arch (gfx940)

**Method 1: Direct Edit**

1. Edit `src/rocprof_compute_soc/analysis_configs/gfx940/0700_wavefront.yaml`
2. Make your changes
3. Run: `python master_config_workflow_script.py`

**Method 2: Using Delta**

1. Create `src/rocprof_compute_soc/analysis_configs/gfx940/config_delta/gfx950_diff.yaml`
2. Run: `python master_config_workflow_script.py`

**What Happens:**
- Changes applied to gfx940 only
- Validated against template (must still match structure)
- Metric descriptions synced to `utils/per_arch_metric_definitions/gfx940_metrics_description.yaml`
- Hashes updated for gfx940 only

### Scenario C: Add New Architecture (gfx951)

**Method 1: Create Directory with YAMLs**

1. Create `src/rocprof_compute_soc/analysis_configs/gfx951/`
2. Copy/create YAML files
3. Run: `python master_config_workflow_script.py`
4. Confirm this is the new latest arch

**Method 2: Using Delta from Latest**

1. Create delta showing differences from gfx950
2. Place in `src/rocprof_compute_soc/analysis_configs/gfx951/config_delta/gfx951_diff.yaml`
3. Run: `python master_config_workflow_script.py`
4. Confirm this is the new latest arch

**What Happens:**
- gfx951 becomes new latest arch
- Template updated with gfx951 as source
- Deltas generated: gfx951 → gfx950, gfx951 → gfx940, etc.
- All archs validated
- Metric descriptions synced
- Hashes updated

### Scenario D: Update Metric Descriptions Only

1. Edit description in config YAML:
```yaml
metrics_description:
  Grid Size: "Updated description text"
```

2. Run: `python master_config_workflow_script.py`

**What Happens:**
- Same workflow as metric changes
- Plain text stored in config YAMLs
- RST version generated and stored in docs/utils files

## Delta YAML Structure

### Complete Example

```yaml
Addition:
  - Panel Config:
      id: 1100
      title: Compute Units - Compute Pipeline
    metric_tables:
      - metric_table:
          id: 1103
          title: Arithmetic Operations
          metrics:
            - F8 OPs:
                avg: AVG(((512 * SQ_INSTS_VALU_MFMA_MOPS_F8) / $denom))
                min: MIN(((512 * SQ_INSTS_VALU_MFMA_MOPS_F8) / $denom))
                max: MAX(((512 * SQ_INSTS_VALU_MFMA_MOPS_F8) / $denom))
                unit: (OPs + $normUnit)
    metric_descriptions:
      F8 OPs:
        plain: "Number of 8-bit floating point operations"
        rst: "Number of 8-bit floating point operations per :ref:`normalization unit <normalization-units>`"

Deletion:
  - Panel Config:
      id: 1100
      title: Compute Units - Compute Pipeline
    metric_tables:
      - metric_table:
          id: 1103
          title: Arithmetic Operations
          metrics:
            - Old Metric:
                avg: AVG(something)
    metric_descriptions:
      Old Metric:
        plain: "Old description"

Modification:
  - Panel Config:
      id: 1100
      title: Compute Units - Compute Pipeline
    metric_tables:
      - metric_table:
          id: 1103
          title: Arithmetic Operations
          metrics:
            - Existing Metric:
                avg: AVG(new_formula)  # Changed field only
    metric_descriptions:
      Existing Metric:
        plain: "Updated description"
        rst: "Updated description with **RST**"
```

### Rules for Deltas

1. **Must have all three sections**: Addition, Deletion, Modification (can be empty lists)
2. **Metric descriptions**:
   - `plain` field is required
   - `rst` field is optional (defaults to copy of plain)
3. **Delta filename**: Must be `<target_arch>_diff.yaml`
4. **Location**: `src/rocprof_compute_soc/analysis_configs/gfx<arch>/config_delta/`

## Standalone Tool Usage

### Hash Manager

```bash
# Compute hashes for all archs
python hash_manager.py --compute-all src/rocprof_compute_soc/analysis_configs

# Detect changes
python hash_manager.py --detect-changes src/rocprof_compute_soc/analysis_configs

# Update hashes for specific arch
python hash_manager.py --update gfx950 src/rocprof_compute_soc/analysis_configs
```

### Metric Description Manager

```bash
# Sync descriptions for specific arch
python metric_description_manager.py --sync-arch gfx950 src/rocprof_compute_soc/analysis_configs --latest-arch gfx950

# Sync all archs
python metric_description_manager.py --sync-all src/rocprof_compute_soc/analysis_configs --latest-arch gfx950

# Validate descriptions
python metric_description_manager.py --validate gfx950 src/rocprof_compute_soc/analysis_configs
```

### Parse Config Template

```bash
# Generate template with metadata
python parse_config_template.py src/rocprof_compute_soc/analysis_configs/gfx950 \
    utils/config_management/analysis_config_template.yaml \
    --latest-arch gfx950
```

### Generate Delta

```bash
# Generate delta from current arch to previous arch
python generate_config_deltas.py \
    src/rocprof_compute_soc/analysis_configs/gfx950 \
    src/rocprof_compute_soc/analysis_configs/gfx940
```

### Apply Delta

```bash
# Apply delta to base arch
python apply_config_deltas.py \
    src/rocprof_compute_soc/analysis_configs/gfx940 \
    src/rocprof_compute_soc/analysis_configs/gfx940/config_delta/gfx950_diff.yaml \
    output_dir
```

### Verify Against Template

```bash
# Validate all archs
python verify_against_config_template.py \
    src/rocprof_compute_soc/analysis_configs \
    utils/config_management/analysis_config_template.yaml
```

## File Structure

```
.
├── src/rocprof_compute_soc/analysis_configs/
│   ├── gfx940/
│   │   ├── 0700_wavefront.yaml           # Config with plain descriptions
│   │   └── config_delta/
│   │       └── gfx950_diff.yaml          # Delta to apply changes
│   ├── gfx941/
│   └── gfx950/                           # Latest arch
│       ├── 0700_wavefront.yaml
│       └── config_delta/
│           └── gfx950_diff.yaml          # Optional delta for modifications
│
├── utils/
│   ├── config_management/
│   │   ├── .config_hashes.json           # Hash database (auto-generated)
│   │   ├── analysis_config_template.yaml # Template with metadata
│   │   ├── hash_manager.py
│   │   ├── metric_description_manager.py
│   │   ├── parse_config_template.py
│   │   ├── generate_config_deltas.py
│   │   ├── apply_config_deltas.py
│   │   ├── verify_against_config_template.py
│   │   ├── master_config_workflow_script.py
│   │   └── config_workflow.yaml
│   │
│   └── per_arch_metric_definitions/
│       ├── gfx940_metrics_description.yaml  # RST only
│       ├── gfx941_metrics_description.yaml
│       └── gfx950_metrics_description.yaml
│
├── docs/data/
│   └── metrics_description.yaml          # RST only, latest arch only
│
└── .backups/                             # Auto-generated backups
    └── 20250115_143022/                  # Timestamped backup
```

## Configuration

Edit `config_workflow.yaml` to customize paths and behavior:

```yaml
paths:
  template: utils/config_management/analysis_config_template.yaml
  configs_root: src/rocprof_compute_soc/analysis_configs
  backups: .backups
  hashes: utils/config_management/.config_hashes.json
  per_arch_metrics: utils/per_arch_metric_definitions
  docs_metrics: docs/data/metrics_description.yaml

validation:
  strict_mode: true              # Fail on warnings
  verify_after_changes: true     # Validate after operations

behavior:
  require_confirmation: true     # Prompt before changes
```

## Error Handling

### Validation Failures

If validation fails:
1. All changes are automatically reverted
2. Backup is restored
3. Detailed error report is printed
4. Fix the issue and run again

### Hash Mismatches

If hashes are out of sync:
```bash
# Recompute all hashes
python hash_manager.py --compute-all src/rocprof_compute_soc/analysis_configs
```

### Description Validation Errors

Common issues:
- **Missing descriptions**: Warning only (won't fail)
- **Invalid RST syntax**: Error (will fail and revert)
- **Missing plain text**: Error (plain is required)

## Best Practices

1. **Always use master_config_workflow_script.py** - Don't run individual scripts manually unless debugging
2. **Test with --dry-run first** - See what will happen before committing
3. **Use deltas for complex changes** - Easier to review and version control
4. **Keep descriptions updated** - Plain text in configs, RST in docs
5. **One change at a time** - If multiple archs need updates, do them sequentially
6. **Check validation output** - Review warnings even if they don't fail

## Troubleshooting

### "No changes detected"

- Check that files were actually modified
- Ensure you're in the correct directory
- Verify hash database exists: `utils/config_management/.config_hashes.json`

### "Validation failed"

- Review the error output carefully
- Check that new metrics match template structure
- Ensure panel IDs are correct
- Verify data source ordering

### "Failed to sync metric descriptions"

- Check RST syntax in descriptions
- Ensure all metrics have descriptions
- Verify section_panel_map includes your table ID

### Changes not detected after manual edit

```bash
# Force recompute hashes
python hash_manager.py --compute-all src/rocprof_compute_soc/analysis_configs

# Then run workflow
python master_config_workflow_script.py
```

## Development Notes

### Adding New Architecture Support

When adding a completely new architecture line:

1. Ensure table IDs are in `metric_description_manager.py`'s `SECTION_PANEL_MAP`
2. Follow existing naming conventions (gfxXXX)
3. Create complete YAML set (don't start with partial configs)

### Modifying the Workflow

If you need to modify the workflow behavior:

1. Edit `config_workflow.yaml` for path/behavior changes
2. Edit `master_config_workflow_script.py` for workflow logic changes
3. Test with `--dry-run` extensively
4. Update this README

## Summary

The master workflow provides:

✅ Automatic change detection
✅ Sequential prompts for multiple changes
✅ Automatic backup/restore on failure
✅ Complete validation
✅ Metric description sync
✅ Delta generation
✅ Template updates
✅ Hash tracking

Simply make your changes and run `python master_config_workflow_script.py` - the system handles the rest!