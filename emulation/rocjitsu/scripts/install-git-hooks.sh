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
    -h|--help) echo "Usage: $0 [--force]"; exit 0 ;;
    *) echo "error: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Resolve the hooks directory git will actually invoke. `core.hooksPath`,
# if set, overrides the default `<gitdir>/hooks/` (used by tools like
# husky and corporate dotfiles). `--git-path hooks` does NOT honor that
# config, so we have to check it ourselves — otherwise the install would
# silently no-op.
HOOKS_DIR="$(git -C "$SCRIPT_DIR" config --get core.hooksPath || true)"
if [ -z "$HOOKS_DIR" ]; then
  HOOKS_DIR_REL="$(git -C "$SCRIPT_DIR" rev-parse --git-path hooks)"
  # That path is relative to cwd; anchor it to SCRIPT_DIR and absolutize.
  HOOKS_DIR="$(cd "$SCRIPT_DIR" && cd "$HOOKS_DIR_REL" && pwd)"
else
  # core.hooksPath may itself be relative (to the worktree root). Make it
  # absolute *before* mkdir, since the directory may not yet exist.
  case "$HOOKS_DIR" in
    /*) ;; # already absolute, use as-is
    *)  HOOKS_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)/$HOOKS_DIR" ;;
  esac
fi
mkdir -p "$HOOKS_DIR"
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
echo "installed: $HOOK_DST -> $HOOK_SRC"
