#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
import sys
import sqlite3
from pathlib import Path


class validation_rule:
    """Class to represent a validation rule as defined in JSON file"""

    def __init__(self, description, query, expected_result, comparison, error_message):
        self.description = description
        self.query = query
        self.expected_result = expected_result
        self.comparison = comparison
        self.error_message = error_message

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


def print_help():
    """Print out the help message"""
    print(
        f"""
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
    """
    )


def validate_table(cursor, rule, tables) -> bool:
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

            if row_count < rule.min_rows:
                print(
                    f"❌ ERROR: Table '{table_name}' has {row_count} rows, minimum required: {rule.min_rows}"
                )
                all_tables_passed = False
                continue
            else:
                print(
                    f"✅ Row count check passed for '{table_name}': {row_count} rows (minimum: {rule.min_rows})"
                )

            all_queries_passed = True
            for validation_query in rule.validation_queries:
                try:
                    query = validation_query.query.replace("{table_name}", table_name)
                    cursor.execute(query)
                    result = cursor.fetchone()

                    if result and "count" in result.keys():
                        actual_result = result["count"]
                    else:
                        actual_result = result[0] if result else None

                    if not validation_query.validate_query(actual_result):
                        print(
                            f"❌ ERROR: {validation_query.error_message} (Table: '{table_name}')"
                        )
                        print(
                            f"   Expected: {validation_query.comparison} {validation_query.expected_result}, Got: {actual_result}"
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


def validate_rocpd(cursor, rules, tables) -> bool:
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
        table_valid = validate_table(cursor, rule, tables)
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
    import json

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

        validation_result = validate_rocpd(cursor, rules, tables)

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
