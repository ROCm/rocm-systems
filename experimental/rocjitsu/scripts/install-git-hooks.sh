#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install rocjitsu git hooks into the enclosing git repository.
#
# Symlinks scripts/git-hooks/pre-commit into the repo's hooks directory
# so git invokes it on every commit. Refuses to overwrite an existing
# hook unless --force is passed.
#
# Run this once per clone. The symlink (rather than a copy) means future
# edits to the tracked hook script take effect with no re-install.
#
# Usage:
#   ./scripts/install-git-hooks.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK_SRC="$SCRIPT_DIR/git-hooks/pre-commit"

if [ ! -f "$HOOK_SRC" ]; then
  echo "error: hook script not found at $HOOK_SRC" >&2
  exit 1
fi

force=0
for arg in "$@"; do
  case "$arg" in
    --force|-f) force=1 ;;
    -h|--help)
      sed -n '5,15p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "error: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# `git rev-parse --git-path hooks` returns the hooks dir for the
# enclosing repo (handles non-standard layouts: submodules, worktrees,
# GIT_DIR overrides). The path it returns is relative to the cwd, so we
# anchor cwd to SCRIPT_DIR before resolving and converting to absolute.
HOOKS_DIR_REL="$(git -C "$SCRIPT_DIR" rev-parse --git-path hooks)"
HOOKS_DIR="$(cd "$SCRIPT_DIR" && cd "$HOOKS_DIR_REL" && pwd)"
HOOK_DST="$HOOKS_DIR/pre-commit"

# Refuse to clobber whatever's already there — could be another
# subproject's hook in this super-repo. -L catches dangling symlinks
# that -e misses.
if [ -e "$HOOK_DST" ] || [ -L "$HOOK_DST" ]; then
  if [ "$force" -ne 1 ]; then
    echo "error: $HOOK_DST already exists" >&2
    echo "       re-run with --force to overwrite" >&2
    exit 1
  fi
  rm -f "$HOOK_DST"
fi

ln -s "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_SRC"
echo "installed: $HOOK_DST -> $HOOK_SRC"
