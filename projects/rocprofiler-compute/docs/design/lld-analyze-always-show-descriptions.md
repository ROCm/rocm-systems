# LLD: Always show metric descriptions in analyze output

## Motivation

Analyze terminal output hides the metric `Description` column. Users have to
pass `--include-cols Description` to see what a metric means. With the TUI and
GUI removed, terminal users have no other way to read a metric's description
during analysis, so the column should always be shown.

The two column options are not worth keeping:

- `--include-cols` can only un-hide `Description`, the one entry in
  `HIDDEN_COLUMNS_CLI`. Any other name is silently ignored.
- `--cols` picks columns by position, and the same index points at different
  columns in different tables. `--output-format csv` and `db` already give
  full control over columns.

## CLI options

Delete `--cols` and `--include-cols` from the analyze options in
`src/argparser.py`. Both are removed outright, with no deprecation period.

## Hidden columns

Delete `HIDDEN_COLUMNS_CLI` from `src/config.py` and `resolve_hidden_columns`
from `src/utils/tty.py`. Remove the `hidden_cols` parameter from
`is_roofline_shown` and `process_table_data`, and the `args.cols` filter in
`process_table_data`.

`--output-format txt` writes the same tables as the terminal, so it shows
descriptions too. `csv` and `db` output already store descriptions in the
`metric_definition` table and do not change.

## Blank Description columns

If a panel YAML has `metrics_description`, the parser adds a `Description`
column to every metric table in that panel, even tables whose metrics have no
description, such as the per-channel L2 tables in block 18.
`format_table_output` hides any table that has an all-blank column, so showing
descriptions would make those tables disappear.

Add a helper in `src/utils/tty.py` that drops the `Description` column when
every cell is blank. Call it:

- in `format_table_output`, before the empty-column check, which stays as is;
- in `is_roofline_shown`, where the hidden-column loop used to drop columns.

Tables where only some metrics have a description keep the column, with blank
cells for the rest. Doing this at display time also covers metric filters,
since `-b` can leave only undescribed rows in a table that has descriptions.

## Docs and skills

- `docs/how-to/analyze/cli.rst`: delete the "Show Description column" example.
- `skills/rocprof-compute-speed-of-light/SKILL.md`: add `Description` to the
  System Speed-of-Light column list.

## CHANGELOG

- Changed: analyze terminal output always shows the metric `Description`
  column.
- Removed: the `--include-cols` and `--cols` analyze options.

## Tests

Unit tests in `tests/unit/utils/test_tty.py`:

- Delete the `resolve_hidden_columns` tests and `include_cols` from the args
  helpers.
- A table whose `Description` column is all blank is shown without that column.
- A table with some descriptions keeps the column.
- A table with an all-blank value column is still hidden.

Integration tests in `tests/integration/test_analyze_commands.py`: delete
`test_col_1` to `test_col_3` and the `col` marker in `pyproject.toml`.

## Out of scope

Showing descriptions in `--list-metrics` is covered by
<https://github.com/ROCm/rocm-systems/pull/12566>.
