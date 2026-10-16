#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install or remove the rocprofiler-systems Agent Skills in an agent's
# personal skills folder.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") --agent <agent> [--uninstall]

Copies the skills next to this script (every folder that contains a SKILL.md)
into the personal skills folder of <agent>, replacing any skill of the same
name already there. Other skills in that folder are left alone.

  --agent <agent>   Agent to install for.
                    Values: claude (~/.claude/skills), codex (~/.agents/skills),
                    cursor (~/.cursor/skills)
  --uninstall       Remove these skills instead.
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

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skills=()
for skill in "$source_dir"/*/; do
    if [[ -f "${skill}SKILL.md" ]]; then
        skills+=("$(basename "$skill")")
    fi
done
if [[ ${#skills[@]} -eq 0 ]]; then
    echo "No skills found next to $0" >&2
    exit 1
fi

for name in "${skills[@]}"; do
    rm -rf "${target:?}/$name"
done
if [[ "$uninstall" == true ]]; then
    echo "Removed rocprofiler-systems skills from $target"
    exit 0
fi

mkdir -p "$target"
for name in "${skills[@]}"; do
    cp -r "$source_dir/$name" "$target/$name"
    rm -rf "${target:?}/$name/evals"
    echo "Installed $name into $target"
done
