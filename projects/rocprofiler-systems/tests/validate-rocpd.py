#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import json
import os
import re
import sys
import sqlite3
from pathlib import Path
from typing import Any, Optional

# Gfx names that have gfx-profiles/<gfx>.json shipped in this repo. Keep in sync with that directory.
# Profile JSON is loaded only when detect_gpu() reports an architecture in this set.
ROCPD_SPECIAL_VALIDATION_GFX: frozenset[str] = frozenset(("gfx1151",))


class validation_rule:
    """Class to represent a validation rule as defined in JSON file"""

    def __init__(
        self,
        description,
        query,
        expected_result,
        comparison,
        error_message,
        requires=None,
    ):
        self.description = description
        self.query = query
        self.expected_result = expected_result
        self.comparison = comparison
        self.error_message = error_message
        self.requires = requires

    def __repr__(self):
        return f"validation_rule(description={self.description}, query={self.query})"

    def validate_query(self, result):
        """
        Validate the actual result against expected using the specified comparison
        defined in validation_queries in rules definition.
        NOTE: see default_rules.json
        """
        if self.comparison == "equals":
            return result == self.expected_result
        elif self.comparison == "greater_than":
            return result > self.expected_result
        elif self.comparison == "less_than":
            return result < self.expected_result
        elif self.comparison == "greater_than_or_equal":
            return result >= self.expected_result
        elif self.comparison == "less_than_or_equal":
            return result <= self.expected_result
        elif self.comparison == "not_equals":
            return result != self.expected_result
        else:
            raise ValueError(f"Unknown comparison operator: {self.comparison}")


class required_table:
    """Class to represent a required table as defined in JSON rules file"""

    def __init__(
        self, name, name_prefix, required_columns, min_rows=1, validation_queries=None
    ):
        if name is None and name_prefix is None:
            raise ValueError("Either 'name' or 'name_prefix' must be specified")
        if name is not None and name_prefix is not None:
            raise ValueError("Cannot specify both 'name' and 'name_prefix'")

        self.name = name
        self.name_prefix = name_prefix
        self.required_columns = required_columns
        self.min_rows = min_rows
        self.validation_queries = validation_queries or []

    def __repr__(self):
        identifier = (
            f"name={self.name}" if self.name else f"name_prefix={self.name_prefix}"
        )
        return f"required_table({identifier}, required_columns={self.required_columns})"

    def get_table_identifier(self):
        """Returns the table identifier (name or prefix) for display purposes"""
        return self.name if self.name else f"{self.name_prefix}*"


def _normalize_gfx_token(gfx: str) -> str:
    """Normalize a user/chip gfx string for profile file lookup (e.g. gfx1151, GFX11.5.1)."""
    t = gfx.strip().lower()
    t = re.sub(r"[^0-9a-z.]", "", t)
    t = t.replace(".", "")
    if t and not t.startswith("gfx") and t[0].isdigit():
        t = "gfx" + t
    return t


def _default_gfx_profiles_dir() -> Path:
    return (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "rocpd-validation-rules"
        / "gfx-profiles"
    )


def load_gfx_profile_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"GFX profile must be a JSON object: {path}")
    return data


def load_gfx_profile(gfx_token: Optional[str]) -> Optional[dict[str, Any]]:
    if not gfx_token:
        return None

    token = _normalize_gfx_token(gfx_token)
    if not token:
        print(f"Warning: empty GFX token after normalize from {gfx_token!r}; ignoring")
        return None

    candidate = _default_gfx_profiles_dir() / f"{token}.json"
    if not candidate.exists():
        print(
            f"Warning: No bundled GFX profile for '{gfx_token}' "
            f"(looked for {candidate}). Continuing without gfx overrides."
        )
        return None

    profile = load_gfx_profile_file(candidate)
    print(f"Loaded GFX profile for '{gfx_token}' from {candidate}")
    return profile


def build_gfx_config(profile: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Turn a loaded profile dict into the structure used during validation."""
    if not profile:
        return None

    skip = profile.get("skip_requires") or []
    if isinstance(skip, str):
        skip = [skip]
    skip_set = set(skip)

    min_ov = profile.get("min_rows_overrides") or {}
    min_rules = profile.get("min_rows_rules") or []
    exp_ov = profile.get("expected_result_overrides") or {}

    return {
        "skip_requires": skip_set,
        "min_rows_overrides": min_ov,
        "min_rows_rules": min_rules,
        "expected_result_overrides": exp_ov,
    }


def _effective_min_rows(
    rule: required_table, gfx_config: Optional[dict[str, Any]]
) -> int:
    if not gfx_config:
        return rule.min_rows

    for entry in gfx_config.get("min_rows_rules") or []:
        pref = entry.get("name_prefix")
        nm = entry.get("name")
        if pref and rule.name_prefix != pref:
            continue
        if nm and rule.name != nm:
            continue
        if not pref and not nm:
            continue
        if "when_min_rows" in entry and entry["when_min_rows"] != rule.min_rows:
            continue
        return int(entry["min_rows"])

    ov = gfx_config.get("min_rows_overrides") or {}
    if rule.name_prefix and rule.name_prefix in ov:
        return int(ov[rule.name_prefix])
    if rule.name and rule.name in ov:
        return int(ov[rule.name])
    return rule.min_rows


def _effective_expected_result(
    validation_query: validation_rule, gfx_config: Optional[dict[str, Any]]
):
    if not gfx_config or not validation_query.requires:
        return validation_query.expected_result
    ov = gfx_config.get("expected_result_overrides") or {}
    if validation_query.requires in ov:
        return ov[validation_query.requires]
    return validation_query.expected_result


def _should_skip_validation_query(
    validation_query: validation_rule,
    available_metrics: Optional[set],
    gfx_config: Optional[dict[str, Any]],
) -> tuple[bool, Optional[str]]:
    """Returns (skip, reason)."""
    if (
        validation_query.requires
        and gfx_config
        and validation_query.requires in gfx_config.get("skip_requires", set())
    ):
        return True, "gfx profile skip_requires"

    if (
        validation_query.requires
        and available_metrics is not None
        and validation_query.requires not in available_metrics
    ):
        return True, "metric not available on platform"

    return False, None


def _detect_gpu_for_validation():
    """Return rocprofsys GPUInfo from rocminfo, or None if import/runtime fails."""
    try:
        _pytest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytest")
        if _pytest_dir not in sys.path:
            sys.path.insert(0, _pytest_dir)
        from rocprofsys.gpu import detect_gpu

        return detect_gpu()
    except Exception as e:
        print(f"Warning: detect_gpu unavailable ({e})")
        return None


def print_help():
    """Print out the help message"""
    print(f"""
    ROCPD Database Validation Tool

    DESCRIPTION:
        This tool validates ROCm Profiler Database (ROCPD) files against a set of predefined rules.
        It checks for required tables, columns, minimum row counts, and executes custom validation queries.

    USAGE:
        {os.path.basename(__file__)} --database <path_to_database> [OPTIONS]

    REQUIRED ARGUMENTS:
        -db, --database PATH        Path to the ROCPD database file (.db) to validate

    OPTIONAL ARGUMENTS:
        -r, --validation_rules PATH [PATH ...]  One or more JSON rules files (default: default_rules.json)
        -h, --help                  Show this help message and exit

    Gfx-specific rules load when detect_gpu() matches ROCPD_SPECIAL_VALIDATION_GFX (see script).

    EXAMPLES:
        # Validate database with default rules
        {os.path.basename(__file__)} --database my_profile.db

        # Validate database with custom rules file
        {os.path.basename(__file__)} --database my_profile.db -r custom_rules.json

        # Validate database with multiple rules files
        {os.path.basename(__file__)} --database my_profile.db -r validation_rules.json amd_smi_rules.json

    VALIDATION FEATURES:
        - Checks for presence of required tables
        - Verifies required columns exist in each table
        - Ensures minimum row count requirements are met
        - Executes custom SQL validation queries
        - Supports various comparison operators (equals, greater_than, less_than, etc.)

    EXIT CODES:
        0  - All validations passed successfully
        64 - Invalid command line arguments (EX_USAGE)
        65 - Validation failures detected (EX_DATAERR)
        1  - General error (database connection, file not found, etc.)
    """)


def validate_table(cursor, rule, tables, available_metrics=None, gfx_config=None) -> bool:
    """
    Validates a database table against a set of rules.
    This function checks if a table specified by `rule` exists in the provided `tables` list,
    verifies that all required columns are present, ensures the table meets a minimum row count,
    and executes custom validation queries defined in the rule.

    Args:
        cursor: Database cursor used to execute SQL queries.
        rule: An object containing validation rules for the table.
        bool: True if the table passes all validation checks, False otherwise.

    Returns:
        bool: True if table is found in the database and if all validation queries pass,
              False if any validation fails or matching table not found in database.
    """

    matching_tables = []

    if rule.name:
        for table in tables:
            if table["name"] == rule.name:
                matching_tables.append(table)
                break
    elif rule.name_prefix:
        for table in tables:
            if table["name"].startswith(rule.name_prefix):
                matching_tables.append(table)

    if not matching_tables:
        if rule.name:
            print(f"❌ ERROR: Required table '{rule.name}' not found in database")
        elif rule.name_prefix:
            print(
                f"❌ ERROR: No tables found with prefix '{rule.name_prefix}' in database"
            )
        return False

    all_tables_passed = True

    for matching_table in matching_tables:
        table_name = matching_table["name"]

        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col["name"] for col in columns]

            missing_columns = [
                col for col in rule.required_columns if col not in column_names
            ]
            if missing_columns:
                print(
                    f"❌ ERROR: Table '{table_name}' missing required columns: {missing_columns}"
                )
                all_tables_passed = False
                continue
            else:
                print(
                    f"✅ All required columns present in '{table_name}': {rule.required_columns}"
                )

            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            row_count = cursor.fetchone()["count"]

            min_rows = _effective_min_rows(rule, gfx_config)
            if row_count < min_rows:
                print(
                    f"❌ ERROR: Table '{table_name}' has {row_count} rows, minimum required: {min_rows}"
                )
                all_tables_passed = False
                continue
            else:
                print(
                    f"✅ Row count check passed for '{table_name}': {row_count} rows (minimum: {min_rows})"
                )

            all_queries_passed = True
            for validation_query in rule.validation_queries:
                skip, skip_reason = _should_skip_validation_query(
                    validation_query, available_metrics, gfx_config
                )
                if skip:
                    req = validation_query.requires or ""
                    print(
                        f"⏭️  Skipping '{validation_query.description}' on '{table_name}' "
                        f"({skip_reason}; requires '{req}')"
                    )
                    continue

                try:
                    query = validation_query.query.replace("{table_name}", table_name)
                    cursor.execute(query)
                    result = cursor.fetchone()

                    if result and "count" in result.keys():
                        actual_result = result["count"]
                    else:
                        actual_result = result[0] if result else None

                    expected_val = _effective_expected_result(
                        validation_query, gfx_config
                    )
                    prev_expected = validation_query.expected_result
                    validation_query.expected_result = expected_val
                    try:
                        query_ok = validation_query.validate_query(actual_result)
                    finally:
                        validation_query.expected_result = prev_expected

                    if not query_ok:
                        print(
                            f"❌ ERROR: {validation_query.error_message} (Table: '{table_name}')"
                        )
                        print(
                            f"   Expected: {validation_query.comparison} {expected_val}, Got: {actual_result}"
                        )
                        all_queries_passed = False
                    else:
                        print(
                            f"✅ Validation query passed for '{table_name}': {validation_query.description}"
                        )

                except sqlite3.Error as e:
                    print(
                        f"❌ ERROR: Failed to execute validation query on '{table_name}': {e}"
                    )
                    print(f"Query: {validation_query.query}")
                    all_queries_passed = False

            if not all_queries_passed:
                all_tables_passed = False

        except sqlite3.Error as e:
            print(f"❌ ERROR: Failed to validate table '{table_name}': {e}")
            all_tables_passed = False

    return all_tables_passed


def validate_rocpd(
    cursor, rules, tables, available_metrics=None, gfx_config=None
) -> bool:
    """
    Validation of a ROCPD database by applying a set of validation rules to specified tables.
    It iterates through each rule, validates the corresponding table, and provides feedback on the validation status.

    Args:
        cursor: Database cursor object for executing SQL queries
        rules: List of validation rule objects containing validation criteria for a specific table
        tables: Collection of table definitions or table objects to validate against

    Returns:
        bool: True if all validation checks pass for all tables,
              False if any validation fails.
    """

    print("Starting ROCPD database validation...")
    db_valid = True

    for rule in rules:
        print(f"\nValidating table: {rule.get_table_identifier()}")
        table_valid = validate_table(cursor, rule, tables, available_metrics, gfx_config)
        db_valid = db_valid and table_valid

    if db_valid:
        print("\n✅ All validation checks passed!")
    else:
        print("\n❌ Some validation checks failed!")

    return db_valid


def load_validation_rules(validation_rules) -> list:
    """
    Load validation rules from a JSON file and convert them to validation objects.

    Args:
        rules_file: Path to the JSON rules file containing validation configuration.

    Returns:
        list: A list of required_table objects.
              Returns empty list if any file doesn't exist or on error.
    """
    all_rules = []

    for rules_file in validation_rules:
        try:
            rules_path = Path(rules_file)
            if not rules_path.exists():
                print(
                    f"Warning: Rules file '{rules_file}' not found, using default rules"
                )
                return []

            with open(rules_path, "r") as f:
                rules_data = json.load(f)
                rules = []

                for table_data in rules_data["required_tables"]:
                    validation_queries = []
                    for vq in table_data.get("validation_queries", []):
                        validation_query_obj = validation_rule(
                            description=vq["description"],
                            query=vq["query"],
                            expected_result=vq["expected_result"],
                            comparison=vq.get("comparison", "equals"),
                            error_message=vq["error_message"],
                            requires=vq.get("requires", None),
                        )
                        validation_queries.append(validation_query_obj)

                    required_table_obj = required_table(
                        name=table_data.get("name", None),
                        name_prefix=table_data.get("name_prefix", None),
                        required_columns=table_data["required_columns"],
                        min_rows=table_data.get("min_rows", 1),
                        validation_queries=validation_queries,
                    )
                    rules.append(required_table_obj)
                    print(f"Loaded required table rule: {required_table_obj}")

                all_rules.extend(rules)

        except Exception as e:
            print(f"Error loading rules file: {e}")
            return []

    if not all_rules:
        print("Warning: No validation rules loaded from any file")
    else:
        print(f"Total rules loaded: {len(all_rules)}")

    return all_rules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "-db", "--database", type=Path, help="Database file to validate", default=None
    )

    parser.add_argument(
        "-r",
        "--validation-rules",
        type=Path,
        nargs="+",
        help="Rules against which to validate database",
        default=[
            Path(
                f"{os.path.dirname(os.path.abspath(__file__))}/rocpd-validation-rules/default-rules.json"
            )
        ],
    )

    parser.add_argument(
        "-h", "--help", action="store_true", help="Prints out the help message"
    )

    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(os.EX_OK)

    if not args.database:
        print("Database file not provided!")
        print_help()

        sys.exit(os.EX_USAGE)

    # Auto-detect available GPU metrics via amd-smi
    available_metrics = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from check_amd_smi_metrics import get_available_metrics

        gpus = get_available_metrics()
        available_metrics = set()
        from check_amd_smi_metrics import collect_metric_names

        print("\n--- Platform GPU Metric Availability ---")
        for gpu in gpus:
            gpu_metrics = collect_metric_names(gpu)
            available_metrics |= gpu_metrics
            print(f"GPU {gpu.gpu_id}:")
            print(
                f"  Activity:    gfx={gpu.gfx_activity}  umc={gpu.umc_activity}  mm={gpu.mm_activity}"
            )
            print(
                f"  Temperature: hotspot={gpu.hotspot_temperature}  edge={gpu.edge_temperature}"
            )
            print(f"  Power:       socket={gpu.current_socket_power}")
            print(
                f"  VCN/JPEG:    vcn_activity={gpu.vcn_activity}  vcn_busy={gpu.vcn_busy}  jpeg_activity={gpu.jpeg_activity}  jpeg_busy={gpu.jpeg_busy}"
            )
            print(
                f"  Other:       mem_usage={gpu.mem_usage}  xgmi={gpu.xgmi}  pcie={gpu.pcie}"
            )
        print(
            f"Detected available metrics (union): {', '.join(sorted(available_metrics))}"
        )
        print("---\n")
    except Exception as e:
        print(f"Warning: Could not detect GPU metrics ({e}), running all queries")

    detected_gpu = _detect_gpu_for_validation()
    profile_token = None
    if detected_gpu is not None and detected_gpu.available:
        matched = ROCPD_SPECIAL_VALIDATION_GFX.intersection(detected_gpu.architectures)
        if matched:
            profile_token = sorted(matched)[0]
    profile_data = load_gfx_profile(profile_token)
    gfx_config = build_gfx_config(profile_data)
    if gfx_config:
        print(f"GFX profile rules active (gfx: {profile_token}).")

    print(f"Validating ROCPD. Database file: {args.database}")

    db_path = args.database
    validation_rules_files = args.validation_rules
    rules = load_validation_rules(validation_rules_files)

    if not rules:
        print("❌ No validation rules loaded. Exiting.")
        sys.exit(1)

    try:
        if not Path(db_path).exists():
            print(f"❌ Error: Database file '{db_path}' not found")
            sys.exit(1)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        print(f"✅ Successfully connected to database: {db_path}")

        cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');")
        tables = cursor.fetchall()

        validation_result = validate_rocpd(
            cursor, rules, tables, available_metrics, gfx_config
        )

        conn.close()

        if validation_result:
            print(f"✅ {db_path} validated")
        else:
            print(f"❌ Failure validating {db_path}")

        sys.exit(os.EX_OK if validation_result else os.EX_DATAERR)

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
