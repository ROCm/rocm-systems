# Python Coding Style Guidelines

This document outlines coding conventions and best practices for Python development in the ROCprofiler-compute project.

## Table of Contents

- [Function Length](#function-length)
- [Naming Conventions](#naming-conventions)
- [When to Use Helper Functions](#when-to-use-helper-functions)
- [When NOT to Extract Helper Functions](#when-not-to-extract-helper-functions)
- [Single Responsibility Principle](#single-responsibility-principle)
- [Levels of Abstraction](#levels-of-abstraction)
- [Avoiding Deep Nesting](#avoiding-deep-nesting)
- [Code Organization](#code-organization)
- [Key Principles Summary](#key-principles-summary)

## Function Length

Focus on what the function does, not arbitrary line counts. The right measure is whether the function has a single, clear purpose — not whether it exceeds a particular number of lines.

Break code into smaller functions to improve readability and reusability.

**Good:** Each function has a focused responsibility

```python
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw data from a CSV file."""
    return pd.read_csv(filepath)

def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that fail validation checks."""
    return df[df["status"] == "valid"]
```

**Bad:** One function doing too much

```python
def load_and_process_data(filepath: str, output_path: str) -> None:
    """Load data AND validate it AND transform it AND save it."""
    df = pd.read_csv(filepath)
    # 15 lines of validation
    # 20 lines of transformation
    # 10 lines of output formatting
    df.to_csv(output_path)
    # Result: 50+ line function that is hard to test or reuse
```

## Naming Conventions

Prefer descriptive names over brevity. A few extra characters in a name can save significant time when reading and understanding code.

**Rule of thumb:**

- The smaller the scope of a function, the longer and more specific its name should be — a private helper called only once can afford to be very explicit about what it does.
- The larger the scope of a variable, the longer and more specific its name should be — a variable visible across many lines or modules should leave no ambiguity about its purpose.

### Example 1: Concise Helper Functions

**Good:** Functions are focused and concise

```python
def load_sys_info(filepath: str) -> pd.DataFrame:
    """Load system running info from a CSV file."""
    return pd.read_csv(filepath)
```

**Bad:** Combining multiple responsibilities in one function

```python
def load_and_process_sys_info(filepath: str, filter_cols: list[str]) -> pd.DataFrame:
    """Load, validate, filter, and transform sys info."""
    df = pd.read_csv(filepath)
    df = df[filter_cols]
    # 10 lines of validation
    # 20 lines of transformation
    # Result: 35+ line function doing too much
    return df
```

### Example 2: Breaking Down Complex Operations

**Good:** Extract calculation into separate function

```python
def version_to_numeric(version_parts: list[int], max_len: int) -> int:
    """Convert version tuple to numeric value using base-1000 positional system."""
    version_numeric = 0
    for i, part in enumerate(version_parts):
        version_numeric += part * (1000 ** (max_len - i - 1))
    return version_numeric
```

**Bad:** Inline complex logic without extraction

```python
def resolve_library_path(library_path: Optional[str]) -> Optional[str]:
    if not library_path:
        return library_path

    # ... 50+ lines of logic including version parsing, comparison,
    # and selection all in one function.
    # This makes the function hard to test and understand.
```

## When to Use Helper Functions

The main reasons to break code into smaller functions are:

- **Make code more readable** by defining function signatures with clear inputs and outputs, making it easier to understand high-level logic without reading each function's implementation.
- **Simplify code reuse (open-closed principle)** — new logic can be written by reusing existing functions in different combinations with minimal or no changes to existing code. Fewer changes to existing functions means a lower probability of regressions.

Always consider breaking code into smaller functions even if the code is used only once.

Extract helper functions when you encounter:

### 1. Repeated Code Patterns

**Good:** Extract repeated logic into a reusable function

```python
def parse_formula_counters(formula: str) -> tuple[bool, list[str]]:
    """Extract counter names from a formula string."""
    if not isinstance(formula, str):
        return False, []
    visited = False
    counters = []
    # ... parsing logic
    return visited, counters

# Reused in multiple places
for formula in formulas:
    visited, counters = parse_formula_counters(formula)
```

**Bad:** Copy-pasting the same logic

```python
# In function A:
if not isinstance(formula, str):
    return False, []
tree = ast.parse(formula)
# ... 20 lines of parsing

# In function B:
if not isinstance(formula, str):  # Duplicated!
    return False, []
tree = ast.parse(formula)  # Duplicated!
# ... same 20 lines of parsing
```

### 2. Complex Expressions That Need Clarification

**Good:** Name the operation

```python
def to_max(*args: Any) -> Union[float, np.ndarray]:
    """Return the maximum value across scalar or Series arguments."""
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].max()
    elif len(args) == 2 and (
        isinstance(args[0], pd.Series) or isinstance(args[1], pd.Series)
    ):
        return np.maximum(args[0], args[1])
    elif max(args) is None:
        return np.nan
    else:
        return max(args)
```

**Bad:** Inline complex conditional with unclear purpose

```python
# What does this do? Hard to understand at a glance
result = (args[0].max() if len(args) == 1 and isinstance(args[0], pd.Series)
          else np.maximum(args[0], args[1]) if len(args) == 2
          and (isinstance(args[0], pd.Series) or isinstance(args[1], pd.Series))
          else np.nan if max(args) is None else max(args))
```

### 3. Improving Testability

**Good:** Separate calculation from I/O

```python
def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

# Pure function — easily tested with any file path
```

**Bad:** Mixing I/O and computation

```python
def process_and_hash_file(filepath: Path) -> tuple:
    """Does too much — hard to unit test."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        data = f.read()
        md5.update(data)

    # Also validates
    is_valid = validate_data(data)

    # Also transforms
    transformed = transform_data(data)

    return md5.hexdigest(), is_valid, transformed
```

## When NOT to Extract Helper Functions

Don't extract if:

- The function name doesn't add value or clarify intent
- The operation is trivially simple and the intent is already obvious from context

```python
# Bad: Unnecessary helper — the name adds no clarity
def is_empty(lst: list) -> bool:
    return len(lst) == 0

# Good: Direct usage is equally readable
if len(my_list) == 0:
    pass
```

```python
# Bad: Helper that just wraps an already-clear operation
def get_column_names(df: pd.DataFrame) -> list[str]:
    return list(df.columns)

# Good: Direct usage is just as clear
column_names = list(df.columns)
```

## Single Responsibility Principle

Each function should do **ONE** thing well. If you use "and" to describe what it does, it likely needs splitting.

### Example 1: Clear Single Purpose

**Good:** Does one thing — creates a dataframe

```python
def create_df_pmc(
    raw_data_root_dir: str,
    nodes: Optional[list[str]],
    spatial_multiplexing: bool,
    kernel_verbose: int,
    verbose: int,
    config_dict: dict[str, Any],
) -> pd.DataFrame:
    """Load all raw pmc counters and join into one dataframe."""
    # Single responsibility: create and return a DataFrame.
    # Delegates the details to a focused helper.
    return _create_single_df_pmc(...)
```

**Bad:** Multiple responsibilities

```python
def load_validate_transform_and_save_data(
    raw_data_dir: str,
    output_path: str,
    validation_rules: dict,
    transform_config: dict,
) -> bool:
    """Load data AND validate it AND transform it AND save it."""
    df = pd.read_csv(raw_data_dir)

    if not validate(df, validation_rules):
        return False

    df = transform(df, transform_config)
    df.to_csv(output_path)
    return True
```

### Example 2: Focused Data Processing

**Good:** Single purpose per function

```python
def pre_processing(self) -> None:
    """Perform pre-processing steps prior to analysis."""
    super().pre_processing()
    args = self.get_args()

    for path_info in args.path:
        workload = self._runs[path_info[0]]

        # Each operation delegated to a focused helper
        workload.raw_pmc = file_io.create_df_pmc(...)

        if args.spatial_multiplexing:
            workload.raw_pmc = self.spatial_multiplex_merge_counters(
                workload.raw_pmc
            )

        file_io.create_df_kernel_top_stats(...)
        kernel_name_shortener(workload.raw_pmc, args.kernel_verbose)
        parser.load_table_data(...)
```

**Bad:** Doing everything inline

```python
def pre_processing(self) -> None:
    """Do everything in one place."""
    args = self.get_args()

    for path_info in args.path:
        workload = self._runs[path_info[0]]

        # 50 lines of inline DataFrame creation logic
        dfs = []
        for csv_file in Path(path_info[0]).rglob("*.csv"):
            # ... lots of processing

        # 30 lines of inline merge logic
        if args.spatial_multiplexing:
            # ... complex merging

        # 40 lines of inline stats creation
        # ... more processing

        # Result: 150+ line function that is hard to test or understand
```

## Levels of Abstraction

An abstraction level refers to how much detail a piece of code exposes. High-level code describes *what* to do (e.g., `setup_configuration()`, `run_analysis()`), while low-level code describes *how* to do it (e.g., opening files, manipulating strings, iterating bytes). Mixing these within a single function makes the overall flow harder to follow.

Keep functions at one abstraction level. Don't mix high-level operations with low-level details.

### Example 1: Consistent Abstraction Level

**Good:** High-level coordination

```python
def run_profiler(
    config_files: Union[list[str], str],
    profiler_options: dict,
    workload_dir: str,
    # ...
) -> None:
    """High-level profiling orchestration."""
    # All operations at the same abstraction level
    setup_counter_definitions(config_files, workload_dir)

    if is_live_attach_mode:
        perform_attach_detach(env, options)
    else:
        success, output = capture_subprocess_output(app_cmd, env)

    process_profiling_results(workload_dir)
```

**Bad:** Mixing abstraction levels

```python
def run_profiler(config_files, profiler_options, workload_dir) -> None:
    """Mixes high and low level operations."""
    # High-level: setup counter definitions
    setup_counter_definitions(config_files, workload_dir)

    # Suddenly drops to low-level details inline
    if options.get("attach_pid") is not None:
        with open("/tmp/config.yaml", "w") as f:
            yaml.dump(counter_defs, f)
        env["METRICS_PATH"] = str(
            Path(tempfile.mkdtemp(prefix="profiler_", dir="/tmp"))
        )
```

### Example 2: Delegation to Appropriate Level

**Good:** Delegate low-level details to libraries

```python
def convert_counter_format(
    counter_file: str,
    agent_info_filepath: str,
    output_file: str,
) -> None:
    """Convert counter file from v3 to v2 format."""
    counters = pd.read_csv(counter_file)
    agent_info = pd.read_csv(agent_info_filepath)

    # pandas handles the low-level details
    result = counters.pivot_table(...)
    result = result.merge(agent_info[...], ...)
    result.rename(columns=name_mapping, inplace=True)
    result.to_csv(output_file, index=False)
```

**Bad:** Implementing low-level logic when libraries exist

```python
def convert_counter_format(counter_file: str, output_file: str) -> None:
    """Don't reinvent the wheel."""
    rows = []
    with open(counter_file) as f:
        for line in f:
            fields = []
            current = ""
            in_quotes = False
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == "," and not in_quotes:
                    fields.append(current)
                    current = ""
                # ... 50 more lines of manual CSV parsing
```

## Avoiding Deep Nesting

- Limit nesting to 2-3 levels, with 2 levels being preferable
- Use early returns (guard clauses) — check for invalid/edge-case inputs at the top of a function and return immediately, keeping the main logic at a shallow level
- Invert conditions when possible — prefer `if not condition: return` over `if condition: { entire body }`
- Extract nested blocks into functions when nesting would otherwise exceed the limit

### Example 1: Early Returns

**Good:** Guard clauses reduce nesting

```python
def resolve_library_path(library_path: Optional[str]) -> Optional[str]:
    """Use early returns to avoid deep nesting."""
    if not library_path:
        return library_path

    path = Path(library_path)

    if path.exists():
        return str(path)

    # Continue with main logic at a low nesting level
    matches = glob.glob(f"{glob.escape(library_path)}.*")
    version_tuples: list[tuple[list[int], str]] = []

    for candidate in matches:
        if not candidate.startswith(library_path):
            continue  # Early continue
        # Process at shallow nesting
```

**Bad:** Deep nesting

```python
def resolve_library_path(library_path: Optional[str]) -> Optional[str]:
    """Avoid this deep nesting."""
    if library_path:  # Level 1
        path = Path(library_path)
        if path.exists():  # Level 2
            return str(path)
        else:  # Level 2
            matches = glob.glob(f"{glob.escape(library_path)}.*")
            if matches:  # Level 3 - already at limit
                for candidate in matches:  # Level 4 - exceeds recommendation
                    if candidate.startswith(library_path):  # Level 5
                        suffix = candidate[len(library_path):]
                        if suffix.startswith("."):  # Level 6
                            # ...
```

### Example 2: Extract Nested Logic

**Good:** Extract complex nested logic into helpers

```python
def create_filtered_stats(
    df_in: dict[str, pd.DataFrame],
    filter_nodes: Optional[list[str]],
    filter_gpu_ids: Optional[list[str]],
    filter_dispatch_ids: Optional[list[str]],
) -> None:
    """Main function delegates complex nested operations."""
    df = df_in["raw"].copy()

    if filter_nodes:
        df = df.loc[df["Node"].astype(str).isin(filter_nodes)]

    if filter_gpu_ids:
        df = df.loc[df["GPU_ID"].astype(str).isin(filter_gpu_ids)]

    if filter_dispatch_ids:
        df = apply_dispatch_filter(df, filter_dispatch_ids)
```

**Bad:** Everything nested inline (exceeds 2-3 level recommendation)

```python
def create_filtered_stats(df_in, filter_nodes, ...) -> None:
    """Deep nesting makes this hard to follow."""
    df = df_in["raw"].copy()

    if filter_nodes:  # Level 1
        if isinstance(filter_nodes, list):  # Level 2
            for node in filter_nodes:  # Level 3 - at limit
                if node in df["Node"].values:  # Level 4 - exceeds recommendation
                    # ... filter logic
                    if validate_node(node):  # Level 5
                        # Too deep!
                        pass
```

## Code Organization

### None Comparisons

Always use `is None` and `is not None` — never `== None` or `!= None`. `None` is a singleton and identity comparison is both correct and more explicit.

```python
# Good
if value is None:
    return default

# Bad
if value == None:  # Caught by Ruff (E711)
    return default
```

### File Structure

Organize Python modules with this structure. Prefer public functions first, then private/helper functions — private/helper functions are implementation details, and if their names are self-explanatory, readers often don't need to examine them.

```python
# 1. Docstring
"""Module description."""

# 2. Imports (grouped and sorted)
import os
import sys

import pandas as pd
import yaml

from mypackage.logger import log
from mypackage.specs import MachineSpecs

# 3. Constants
MAX_SIZE = 100
DEFAULT_TIMEOUT = 30

# 4. Public functions
def public_api_function() -> None:
    """Public API exposed to other modules."""
    pass

# 5. Helper functions (private, prefixed with _)
def _internal_helper() -> str:
    """Private helper function."""
    pass

# 6. Classes
class DataProcessor:
    """Class for processing data."""
    pass
```

### Example

**Good:** Well-organized module structure

```python
#!/usr/bin/env python3
# License header...

"""
Hash manager for tracking configuration file changes.
Can be used standalone or imported by the master workflow.

Usage:
    python hash_manager.py --compute-all <configs_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

# Constants
DEFAULT_HASH_DB = "config/.hashes.json"

# Public functions in logical order
def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    # ...

def compute_arch_hashes(arch_dir: Path) -> dict:
    """Compute hashes for all YAML files in an arch directory."""
    # ...

def load_hash_db(hash_file: Path) -> dict:
    """Load hash database from file."""
    # ...
```

**Bad:** Disorganized structure

```python
# No docstring
import sys
from mypackage.logger import log  # Mixed import groups
DEFAULT_HASH_DB = "config/.hashes.json"  # Constant placed before imports
import hashlib  # Import after constant
import argparse

# Function used before it's defined
def main():
    result = compute_hash()  # Defined later

# Private function mixed with public
def _helper():
    pass

def compute_hash():
    return _other_helper()  # Defined even later

def _other_helper():
    pass
```

## Key Principles Summary

| Principle | Guideline |
|-----------|-----------|
| **Readability first** | Code is read far more than written — optimize for clarity and maintainability over brevity |
| **Single responsibility** | Each function should do exactly one thing well |
| **Consistent abstraction** | Keep operations at the same conceptual level within a function |
| **Shallow nesting** | Use guard clauses and early returns to keep nesting to 2-3 levels |
| **Meaningful extraction** | Extract helpers when they improve clarity, reusability, or testability |
| **Appropriate naming** | Prefer descriptive names; readability matters more than brevity |
