# Adding Experimental Features
New features that aren't yet stable can be introduced behind the `--experimental` flag. This lets users opt in to preview functionality while keeping the default experience stable.

## How It Works
The `--experimental` flag acts as a master toggle:

- Experimental options are **hidden** from help output unless `--experimental` is passed.
- Attempting to use an experimental flag without `--experimental` raises a clear error.
- A warning is displayed when an experimental feature is active.

To see available experimental features:

```bash
rocprof-compute profile --experimental --help
```

## Adding a New Experimental Feature
Follow these three steps to add a new experimental flag.

**Step 1 — Register it in the `--experimental` help text**
In `src/argparser.py`, update the `add_general_group()` function:

```python
general_group.add_argument(
    "--experimental",
    action="store_true",
    default=False,
    help=(
        "Enable experimental feature(s):\n"
        "   Your feature name (--your-flag)\n"  # Add this line
    ),
)
```

**Step 2 — Add the argument using `ExperimentalAction`**
Add your flag to the relevant parser group (profile, analyze, etc.):

```python
# For a flag that accepts a value
profile_group.add_argument(
    "--your-flag",
    dest="your_flag",
    required=False,
    default=None,
    action=ExperimentalAction,
    experimental_enabled=experimental_enabled,
    feature_label="Your feature name",
    base_action="store",  # Required — see supported actions below
    type=str,
    nargs="*",
    metavar="",
    help="\t\t\tDescription of your feature",
)

# For a boolean toggle flag
analyze_group.add_argument(
    "--your-flag",
    dest="your_flag",
    required=False,
    default=False,
    action=ExperimentalAction,
    experimental_enabled=experimental_enabled,
    feature_label="Your feature description",
    base_action="store_const",  # Required — see supported actions below
    nargs=0,
    const=True,
    help="\t\tDescription of your feature",
)
```

The `base_action` parameter is required and must be one of:

| Value | Behavior |
|---|---|
| `store` | Store a value (standard argparse default) |
| `store_const` | Store a fixed constant; consumes no arguments |
| `store_true` | Store `True` when the flag is present |
| `store_false` | Store `False` when the flag is present |
| `append` | Append each value to a list |
| `append_const` | Append a constant to a list |
| `count` | Count occurrences (e.g. `-vvv`) |
| `extend` | Extend a list with multiple values |

**Step 3 — Verify behavior**
Confirm the flag is hidden without `--experimental` and visible with it:

```bash
# Should not appear
rocprof-compute profile --help

# Should appear with EXPERIMENTAL: prefix
rocprof-compute profile --experimental --help
```

### Promoting a Feature to Stable
When a feature is ready for general availability:

1. Remove it from the `--experimental` help text in `src/argparser.py`.
2. Replace `action=ExperimentalAction` with a standard argparse action (e.g. `action="store"`).
3. Remove the `experimental_enabled`, `feature_label`, and `base_action` parameters.
4. Update documentation and tests accordingly.
