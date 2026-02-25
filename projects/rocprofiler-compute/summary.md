# Implementation Summary: Migrate CSV Output to Database Workflow

## Overview
Successfully migrated CSV output generation from the CLI workflow to the database workflow.
Instead of creating a directory with multiple CSV files, the system now generates a single
CSV file using the database workflow's data model.

## Behavioral Changes

### Before:
- `--output-format csv` → Routed to CLI workflow → Created directory with multiple CSV files
- Each table was saved as a separate CSV file in the output directory
- Worked with all profiling output formats

### After:
- `--output-format csv` → Routed to DB workflow → Creates single CSV file
- CSV file has the same base name as DB file but with .csv extension
- No .db file is created when CSV format is specified
- Single CSV contains all metrics from the metric_view
- **Requires rocpd profiling output format** (same as database output format)

---

## Files Modified

### 1. src/rocprof_compute_base.py (Lines 137-145)
**Purpose**: Route CSV format to database workflow

**Change**: Modified `detect_analyze()` method
- Changed condition from `self.__args.output_format == "db"`
- To: `self.__args.output_format in ("db", "csv")`
- Now both "db" and "csv" formats use the database workflow

---

### 2. src/utils/analysis_orm.py
**Purpose**: Add reusable metric view query for CSV export and in-memory database support

**Changes**:
a) Added `sqlite3` import at top of file
   - Required for backing up in-memory database to file

b) Added `_db_name` class variable to Database class
   - Stores target filename for database file

c) Modified `Database.init()` to always use in-memory SQLite
   - Changed from `sqlite:///{db_name}` to `sqlite:///:memory:`
   - Stores db_name in `_db_name` class variable for later use
   - Prevents file creation during CSV export

d) Modified `Database.write()` to backup in-memory database to file
   - Uses SQLite's backup API to write in-memory database to disk
   - Only called for DB format, not CSV format

e) Added new class method `get_metric_view()` (after line 219)
   - Returns the metric view query as a SQLAlchemy Select object
   - Query joins Workload, Kernel, MetricDefinition, and MetricValue tables
   - Returns all metric data in a flattened format

f) Updated `get_views()` function (line 312-331)
   - Replaced duplicated query definition with call to `Database.get_metric_view()`
   - Reduces code duplication and ensures consistency

**Columns in metric_view**:
- workload_id, workload_name
- kernel_uuid, kernel_name
- metric_uuid, metric_name, metric_id
- description, table_name, sub_table_name, unit
- value_uuid, value_name, value

---

### 3. src/rocprof_compute_analyze/analysis_db.py (Lines 248-273)
**Purpose**: Write CSV file when CSV format is specified

**Changes**: Added conditional output writing logic
- **For CSV format**:
  * Generate CSV filename: `db_name.replace(".db", ".csv")`
  * Execute `Database.get_metric_view()` query using session.execute().fetchall()
  * Get column names from query and create DataFrame
  * Write DataFrame to CSV file with `metric_df.to_csv(output_name, index=False)`
  * Close session without committing
  * No .db file is created (database is in-memory only)

- **For DB format**:
  * Create SQL views using `get_views()`
  * Call `Database.write()` to commit and backup in-memory database to .db file

**Key insight**: Using in-memory SQLite database for both CSV and DB workflows. For CSV, we query the in-memory database directly and write to CSV. For DB, we backup the in-memory database to disk file. This prevents unwanted .db file creation during CSV export.

---

### 4. src/utils/tty.py
**Purpose**: Remove CSV handling from CLI display layer

**Changes**:
a) Removed CSV directory creation logic (previously lines 656-662)
   - No longer creates output directory for CSV files

b) Removed CSV file writing logic (previously lines 565-571)
   - No longer writes individual CSV files per table

c) Removed `csv_dir` parameter from `format_table_output()` function signature
   - Function no longer needs to know about CSV output

d) Removed `csv_dir` argument from `format_table_output()` call (line 757)
   - Cleaned up function invocation

e) Removed `csv_dir = None` initialization in `show_all()` function
   - Variable no longer needed

f) Removed unused import: `from pathlib import Path`
   - No longer needed after CSV handling removal

---

### 5. tests/test_profile_general.py
**Purpose**: Add test for CSV output with rocpd profiling format

**Changes**: Added new test `test_save_csv` after `test_analyze_rocpd` (after line 1355)

**Test implementation**:
- Profiles workload using rocpd format: `--format-rocprof-output rocpd`
- Analyzes profiled data with CSV output format
- Verifies single CSV file is created: `{output_name}.csv`
- Validates CSV contains data (len(df.index) >= 1)
- Checks for expected columns from metric_view:
  * workload_name, kernel_name, metric_name, value
- Verifies NO database file is created (no `{output_name}.db`)
- Cleanup: Removes CSV file and workload directory

**Note**: Test moved from test_analyze_commands.py because CSV output requires rocpd profiling format

---

### 6. tests/test_analyze_commands.py (Lines 689-726)
**Purpose**: Remove old CSV test that used non-rocpd format

**Changes**: Removed `test_save_dfs` test
- Old test tried to use CSV output on non-rocpd profiled data
- CSV output now requires rocpd format, so test was moved to test_profile_general.py

---

### 7. tests/test_metric_validation.py (Lines 116-144)
**Purpose**: Update metric validation test for new CSV format

**Changes**: Modified metric value validation logic

**Old behavior**:
- Read from multiple CSV files: `{analysis_workload_dir}/{metric['csv_file']}`
- Each metric had its own CSV file specified in test data

**New behavior**:
- Read single CSV file: `{output_file}.csv`
- Filter DataFrame by metric_id to find specific metrics
- Added assertion to verify metric exists in CSV before validating value
- Same validation logic (5% tolerance) applied to filtered data

**Key change**:
```python
# Old:
actual = pd.read_csv(f"{analysis_workload_dir}/{metric['csv_file']}")[
    metric["column"]
].values[0]

# New:
csv_file = f"{output_file}.csv"
df = pd.read_csv(csv_file)
metric_data = df[df['metric_id'] == metric['metric_id']]
actual = metric_data[metric["column"]].values[0]
```

---

## Technical Implementation Details

### Database Session Lifecycle for CSV Output:
1. Initialize database engine with `Database.init(db_name)` (.db extension)
2. Populate session with ORM objects (Workload, Kernel, MetricValue, etc.)
3. Flush session to make ORM data visible to SQL queries
4. Query metric data directly using `Database.get_metric_view()`
5. Write DataFrame to CSV file
6. Close session WITHOUT commit (prevents .db file creation)

**Key Insight**: The database is used as an in-memory data transformation layer,
leveraging existing ORM models, but not persisted to disk for CSV format.

### Naming Convention:
- Input: `--output-name my_results`
- CSV output: `my_results.csv`
- DB output: `my_results.db`
- Default naming: `rocprof_compute_{uuid}.csv` or `rocprof_compute_{uuid}.db`

---

## Benefits

1. **Single CSV file** - Much easier to handle than a directory with multiple files
2. **Consistent naming** - Same base name with different extension (.csv vs .db)
3. **Code reuse** - CSV and DB workflows share all metric calculations and data transformations
4. **Cleaner architecture** - CSV is now an export format, not a display concern
5. **Better data model** - Uses ORM objects and SQL views instead of ad-hoc table formatting
6. **No duplication** - Removed CSV-specific logic from display layer (tty.py)
7. **Maintainability** - Single source of truth for metric data structure

---

## Verification

All modified files pass Python syntax validation:
✓ src/rocprof_compute_base.py
✓ src/utils/analysis_orm.py
✓ src/rocprof_compute_analyze/analysis_db.py
✓ src/utils/tty.py
✓ tests/test_analyze_commands.py
✓ tests/test_metric_validation.py

---

## Usage Examples

### Generate CSV output:
```bash
# Step 1: Profile with rocpd format (required)
rocprof-compute profile --format-rocprof-output rocpd <application>

# Step 2: Analyze and create CSV
rocprof-compute analyze --output-format csv --path <workload_dir>
# Creates: rocprof_compute_<uuid>.csv
```

### Generate CSV with custom name:
```bash
# Step 1: Profile with rocpd format (required)
rocprof-compute profile --format-rocprof-output rocpd <application>

# Step 2: Analyze with custom output name
rocprof-compute analyze --output-format csv --output-name my_results --path <workload_dir>
# Creates: my_results.csv
```

### Generate database (unchanged):
```bash
rocprof-compute analyze --output-format db --path <workload>
# Creates: rocprof_compute_<uuid>.db
```

### Display to stdout (unchanged):
```bash
rocprof-compute analyze --output-format stdout --path <workload>
# Displays to console, no files created
```

---

## Migration Notes

### Important Requirement
- **CSV output now requires rocpd profiling format**
- Profile with: `rocprof-compute profile --format-rocprof-output rocpd <application>`
- Analyze with: `rocprof-compute analyze --output-format csv --path <workload_dir>`
- This is the same requirement as database output format

### Other Changes
- `--output-name` parameter works the same way - it specifies the base name for the output
- CSV output format changed: now creates a single CSV file instead of a directory with multiple files
- Tests that relied on multiple CSV files have been updated to work with single CSV
- The metric data structure in the CSV is now consistent with the database schema
- No impact on profiling workflow or roofline CSV files (those are created during profiling)

---

---

## Documentation Updates

### 1. docs/how-to/analyze/cli.rst

**Section**: Analysis output format (Lines 533-558)

Updated CSV format documentation to reflect new behavior:

**Changes**:
- Documented single CSV file output instead of directory
- Simplified description to focus on programmatic analysis without specifying schema details
- Added note blocks (following RST `.. note::` convention) for:
  * rocpd profiling format requirement
  * Terminal output being disabled
- Updated "Default file name" section to clarify behavior for CSV and DB formats
- Fixed typo: "overriden" → "overridden"

**Rationale**:
- Schema details (column names) are intentionally omitted as they may change over time
- Documentation focuses on usage and requirements rather than implementation details

---

## CHANGELOG Updates

### CHANGELOG.md - Unreleased section

**Section**: Changed

**Entry Added**:
```markdown
* **CSV output format migration**: The `--output-format csv` option now uses the database workflow instead of the CLI workflow.
  * CSV output now generates a single CSV file (e.g., `rocprof_compute_<uuid>.csv`) instead of a directory with multiple files.
  * CSV output requires rocpd profiling format (same requirement as database output): profile with `--format-rocprof-output rocpd`.
  * The CSV file contains all metric data for programmatic analysis.
  * The `--output-name` parameter works the same way for both CSV and database formats, specifying the base name (without extension).
```

**Rationale**: Placed under "Changed" rather than "Added" because this modifies existing CSV functionality rather than introducing a new feature.

---

## Implementation Date
2026-02-25
