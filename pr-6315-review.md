# PR #6315 Review — Add test filter standardization to hip-tests

**PR:** https://github.com/ROCm/rocm-systems/pull/6315
**Author:** dileepr1
**Branch:** `users/dravindr/tf_hiptests` → `develop`
**Files changed:** 3 (+386 / -5)

## Overview

Adds a post-processing step to the hip-tests build chain. After `parse_config.py` generates `hip_test_config.hh`, a new `inject_category_tags.py` rewrites the header in-place to append tier tags (`[quick]`, `[standard]`, `[comprehensive]`, `[full]`) and GPU/OS-exclusion labels driven by a new `test_categories.yaml`. Because Catch2 uses `ADD_TAGS_AS_LABELS` + `PRE_TEST` discovery, these compiled-in tags become CTest labels at runtime, enabling `ctest -L quick`.

**Approach trade-off:** The PR deliberately keeps `parse_config.py` untouched and bolts on a second pass. This preserves the existing tested codepath, but rewriting generated headers in-place via regex is fragile and a non-obvious convention to maintain. A single-pass solution (moving the YAML read into `parse_config.py` or `common.py`) would be simpler and removes a class of bugs. Worth flagging to the author.

## Issues

### Bugs / Correctness

- **`inject_category_tags.py:191-200` — misleading indentation.** The `# GPU exclusion labels` comment sits inside the `for tier` loop body but the `for ex_tag` loop is dedented to the outer scope. The code runs correctly, but reads as if the second loop is nested. Fix indentation of the comment so it visually belongs to its loop.
- **`inject_category_tags.py:121` — silent fallback to "all tiers".** `resolve_tier_tags` returns `list(TIER_ORDER)` if `tier_patterns` is empty. A missing or empty `test_categories.yaml` would silently tag every test with every tier, defeating the filter. Prefer `sys.exit(1)` with an error, or at minimum a warning.
- **`inject_category_tags.py:179` — silent skip of unrecognized `#define` lines.** `_DEFINE_RE` makes strong assumptions about the format emitted by `parse_config.py`. If `parse_config.py` ever changes (e.g., adds a field), every line would silently skip and the script would report `+0 tier tag(s)` without failing the build. Suggest asserting at the end that `added_tiers > 0` (or that *some* lines matched) so a format drift fails the build.
- **In-place write without atomic rename.** `patch_header` opens the same file for read then for write. If the script crashes mid-write, the header is corrupt but the file mtime is still newer than the dependencies — incremental builds will compile garbage. Write to `header_path + ".tmp"` and `os.replace` it.
- **Inconsistent matching semantics.** Tier `test_patterns` are treated as regexes (anchored with `^...$`); `exclude_arch` `test_patterns` are matched by exact dict lookup (`gpu_exclusion_tags.get(test_name, ())`). The YAML doesn't make this distinction obvious — e.g., a future contributor adding `"Unit_hipMemset.*"` under `exclude_arch` would silently match nothing. Either document it inline in the YAML or use regex matching consistently.

### Configuration / YAML

- **`test_categories.yaml` — orphan `exclude:` keys.** Every tier has an empty `exclude:` line followed by a sibling `labels:`. The `exclude:` key has no value and isn't read by the script. If it's intended placeholder, add a TODO comment; otherwise delete.
- **Naming mismatch.** The PR description says "rename `exclude_gpu` → `exclude_arch`," but only the parent section is renamed; the individual keys still use `exclude_gpu_*` (matched by `_EXCLUDE_GPU_KEY_RE`). Either rename the keys to `exclude_arch_*` for consistency or note the inconsistency in a comment.
- **Quick-tier patterns are aspirational.** `Unit_[^_]+_Positive_Basic.*` requires the component name to be a single underscore-free token. `Unit_hip_async_Positive_Basic` (if it existed) wouldn't match. The TODO is acknowledged in the file, but worth confirming the placeholder is intentional for landing rather than something to refine before merge.

### Style / Minor

- **`inject_category_tags.py:81` / `:101`** — `tier_patterns: dict = {}` bare-`dict` annotation. Use `dict[str, list[re.Pattern]]` or omit.
- **`inject_category_tags.py:108`** — `is_linux` is computed but only used as a positive check; could simplify to `is_windows` alone since the keys only support those two suffixes.
- **No unit tests** for `_DEFINE_RE`, `parse_exclude_gpu_key`, or `resolve_tier_tags`. These are pure functions with clear inputs/outputs and would be cheap to test; given the fragility of regex-against-generated-code, tests would be high-value.

## Test Coverage

The PR links a CI run but adds no automated tests for the new script. Given the project already has `check_config.py` for YAML validation, consider adding a similar check that:

1. Validates `test_categories.yaml` schema (required keys, regex compilability).
2. Asserts every tier pattern matches at least one test (catches stale patterns).

## Security

No concerns — script reads local YAML, writes a local header. No shell-out, no network.

## Recommendation

Request changes for the silent-fallback and silent-skip bugs (these will mask real problems in CI), the atomic-rename, and the indentation/orphan-config cleanup. Architecture concern (post-processing vs single-pass) is worth raising as a discussion but not blocking.
