---
description: Run clang-tidy on changes using the repository .clang-tidy config
argument-hint: [preset=debug] [base=develop]
---

Preset: `${1:-debug}`. Base ref: `${2:-develop}`.

1. Confirm cwd is `projects/rocprofiler-systems/`.
2. Confirm `build/${1:-debug}/compile_commands.json` exists. Configure and
   build that preset first if it does not.
3. Resolve the base commit. Local `develop` can be stale, so use the fork point:
   `base=$(git merge-base --fork-point ${2:-develop} || git merge-base ${2:-develop} HEAD)`
4. Run clang-tidy on changed lines. Do not pass `--checks`; omitting it makes
   `scripts/clang-tidy-check.py` use `.clang-tidy`:
   `./scripts/clang-tidy-check.py --build-path build/${1:-debug} --base "$base" -j $(nproc)`
5. Report the files and line ranges checked, each diagnostic (check name,
   location, message), and the exit status.
6. Fix diagnostics that violate `.clang-tidy`. Do not silence them with a
   `--checks` override or a new NOLINT suppression.

See `cpp-conventions` for the C++ lint rules.
