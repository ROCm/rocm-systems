#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Install or remove the rocprof-compute Agent Skills in an agent's
# personal skills folder.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") --agent <agent> [--uninstall]

Copies the rocprof-compute-* skills next to this script into the personal
skills folder of <agent>, replacing any rocprof-compute-* skills already there.

  --agent <agent>   Agent to install for.
                    Values: claude (~/.claude/skills), codex (~/.agents/skills),
                    cursor (~/.cursor/skills)
  --uninstall       Remove the rocprof-compute-* skills instead.
  -h, --help        Show this help.
EOF
}

agent=""
uninstall=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            agent="${2:-}"
            shift 2 || { usage >&2; exit 1; }
            ;;
        --uninstall)
            uninstall=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
done

case "$agent" in
    claude) target="$HOME/.claude/skills" ;;
    codex) target="$HOME/.agents/skills" ;;
    cursor) target="$HOME/.cursor/skills" ;;
    *)
        usage >&2
        exit 1
        ;;
esac

rm -rf "$target"/rocprof-compute-*
if [[ "$uninstall" == true ]]; then
    echo "Removed rocprof-compute skills from $target"
    exit 0
fi

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docs_dir="$(dirname "$source_dir")/docs"
# Escape the characters sed treats specially in the replacement.
docs_dir_sed="$(printf '%s' "$docs_dir" | sed 's/[\\&|]/\\&/g')"
mkdir -p "$target"
for skill in "$source_dir"/rocprof-compute-*/; do
    name="$(basename "$skill")"
    cp -r "$skill" "$target/$name"
    rm -rf "${target:?}/$name/evals"
    # Point the relative docs links at the docs shipped next to the skills.
    sed -i "s|](\.\./\.\./docs/|](${docs_dir_sed}/|g" "$target/$name"/*.md
    echo "Installed $name into $target"
done
