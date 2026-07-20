#!/usr/bin/env bash
# rocm_lttng_enable.sh -- enable curated ROCm LTTng tracepoints by
# category and/or explicit API name (schema v1).
#
# Each curated API is a single combined LTTng event (`<api>`, or
# `<api>`, `<api>_2`, `<api>_3`, ... for APIs split into chunks by the
# codegen's field-budget limit). Enabling an event name turns on BOTH the
# ENTER and EXIT records for that API (a `phase` field discriminates), so
# fully tracing one API means enabling ALL of its chunk event names --
# this script takes care of that expansion for you.
#
# Categories are pattern-based groupings of curated APIs, defined in
# shared/lttng/scripts/lttng_categories.yaml and resolved (including chunk
# expansion) into a per-provider JSON manifest by
# lttng_gen_category_manifest.py:
#   projects/clr/hipamd/scripts/lttng_categories_manifest.json           (hip)
#   projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_categories_manifest.json (hsa)
#
# Usage:
#   rocm_lttng_enable.sh --session <name> --provider hip|hsa \
#       [--channel <name>] [--category NAME]... [--api NAME]... \
#       [--manifest <path>] [--dry-run]
#   rocm_lttng_enable.sh --provider hip|hsa --list [--manifest <path>]
#
# Options:
#   --session NAME    LTTng tracing session name. The session (and its
#                      channel) must already exist -- this script does not
#                      create sessions/channels, only enables events on one.
#                      Required unless --list is given.
#   --provider NAME    "hip" or "hsa". Required.
#   --channel NAME     LTTng channel name to enable events on. Optional;
#                      when omitted, `-c` is NOT passed and LTTng uses its
#                      own default userspace channel (normally "channel0").
#                      Pass --channel NAME to force a specific channel.
#   --category NAME    Enable every event for every API in category NAME
#                      (repeatable).
#   --api NAME         Enable every chunk event for the single curated API
#                      NAME (repeatable). NAME is the base API name (e.g.
#                      "hipMalloc"), not a chunk-suffixed event name.
#   --manifest PATH    Override the default per-provider manifest path. By
#                      default the manifest is discovered from the installed
#                      share path (<prefix>/share/rocm-lttng/<provider>/)
#                      first, then the in-source path.
#   --list             Print the available categories for --provider (with
#                      per-category API counts) and exit. Ignores
#                      --session/--channel/--category/--api.
#   --dry-run          Print the `lttng enable-event` command(s) that would
#                      be run, without running them.
#
# At least one --category or --api is required unless --list is given.
#
# Examples:
#   rocm_lttng_enable.sh --provider hip --list
#   rocm_lttng_enable.sh --session mysess --provider hip \
#       --category hip_memory --category hip_streams
#   rocm_lttng_enable.sh --session mysess --provider hip \
#       --api hipMalloc --api hipMemcpy --channel mychan
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Manifest discovery order (first existing wins), unless --manifest overrides:
#   1. Installed share path, relative to this script's install location. When
#      installed the script lives in <prefix>/bin and manifests in
#      <prefix>/share/rocm-lttng/<provider>/lttng_categories_manifest.json.
#   2. Source-tree path (running the script straight out of the repo).
INSTALL_PREFIX="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALLED_MANIFEST_HIP="$INSTALL_PREFIX/share/rocm-lttng/hip/lttng_categories_manifest.json"
INSTALLED_MANIFEST_HSA="$INSTALL_PREFIX/share/rocm-lttng/hsa/lttng_categories_manifest.json"

# Source-tree layout: this script is at shared/lttng/scripts/, three levels
# below the repo root.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SOURCE_MANIFEST_HIP="$REPO_ROOT/projects/clr/hipamd/scripts/lttng_categories_manifest.json"
SOURCE_MANIFEST_HSA="$REPO_ROOT/projects/rocr-runtime/runtime/hsa-runtime/scripts/lttng_categories_manifest.json"

# Return the first existing manifest for a provider: installed share path
# first, then the in-source path. Prints nothing if neither exists.
resolve_default_manifest() {
    local provider="$1"
    local installed source
    if [ "$provider" = "hip" ]; then
        installed="$INSTALLED_MANIFEST_HIP"; source="$SOURCE_MANIFEST_HIP"
    else
        installed="$INSTALLED_MANIFEST_HSA"; source="$SOURCE_MANIFEST_HSA"
    fi
    if [ -f "$installed" ]; then
        printf '%s' "$installed"
    elif [ -f "$source" ]; then
        printf '%s' "$source"
    fi
}

# Max event names batched into a single `lttng enable-event` invocation, to
# keep command lines from growing unreasonably long for big categories.
BATCH_SIZE=50

usage() {
    # Print the leading comment help block (from line 2 up to the line before
    # `set -euo pipefail`), stripping the leading `# `. Computed dynamically so
    # edits to the help block don't silently truncate the output.
    local end
    end="$(grep -n '^set -euo pipefail' "${BASH_SOURCE[0]}" | head -1 | cut -d: -f1)"
    sed -n "2,$((end - 1))p" "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

SESSION=""
PROVIDER=""
# Channel is UNSET by default: we omit `-c` so LTTng picks its own default
# userspace channel (normally "channel0"), rather than forcing a wrong name.
CHANNEL=""
MANIFEST=""
LIST=0
DRY_RUN=0
CATEGORIES=()
APIS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --session)
            SESSION="$2"; shift 2 ;;
        --provider)
            PROVIDER="$2"; shift 2 ;;
        --channel)
            CHANNEL="$2"; shift 2 ;;
        --category)
            CATEGORIES+=("$2"); shift 2 ;;
        --api)
            APIS+=("$2"); shift 2 ;;
        --manifest)
            MANIFEST="$2"; shift 2 ;;
        --list)
            LIST=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            usage 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage 2 ;;
    esac
done

if [ -z "$PROVIDER" ]; then
    echo "ERROR: --provider is required" >&2
    usage 2
fi
if [ "$PROVIDER" != "hip" ] && [ "$PROVIDER" != "hsa" ]; then
    echo "ERROR: --provider must be 'hip' or 'hsa', got: $PROVIDER" >&2
    exit 2
fi

if [ -z "$MANIFEST" ]; then
    MANIFEST="$(resolve_default_manifest "$PROVIDER")"
fi
if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found for provider '$PROVIDER'." >&2
    echo "       Looked in the installed share path and the source tree." >&2
    echo "       Pass --manifest PATH, or regenerate with" >&2
    echo "       lttng_gen_category_manifest.py." >&2
    exit 2
fi

# --------------------------------------------------------------------------
# --list mode
# --------------------------------------------------------------------------
if [ "$LIST" -eq 1 ]; then
    python3 - "$MANIFEST" <<'PYEOF'
import json
import sys

manifest_path = sys.argv[1]
with open(manifest_path) as f:
    manifest = json.load(f)

print(f"Provider: {manifest['provider']}")
print(f"Manifest: {manifest_path}")
print()
print(f"{'CATEGORY':<24} {'APIS':>6} {'EVENTS':>7}")
for cat in sorted(manifest['categories']):
    events = manifest['categories'][cat]
    bases = set()
    for ev in events:
        parts = ev.rsplit('_', 1)
        base = parts[0] if len(parts) == 2 and parts[1].isdigit() else ev
        bases.add(base)
    print(f"{cat:<24} {len(bases):>6} {len(events):>7}")

uncategorized = manifest.get('uncategorized_apis', [])
if uncategorized:
    print()
    print(f"Uncategorized APIs ({len(uncategorized)}):")
    for api in uncategorized:
        print(f"  - {api}")
PYEOF
    exit 0
fi

# --------------------------------------------------------------------------
# enable mode
# --------------------------------------------------------------------------
if [ -z "$SESSION" ]; then
    echo "ERROR: --session is required (unless --list is given)" >&2
    usage 2
fi
if [ "${#CATEGORIES[@]}" -eq 0 ] && [ "${#APIS[@]}" -eq 0 ]; then
    echo "ERROR: at least one --category or --api is required" >&2
    usage 2
fi

# Resolve the requested categories/APIs into the deduplicated, sorted list
# of concrete event names via the manifest. Errors on any unknown
# category/API name (listing what IS available) and exits nonzero.
EVENTS_FILE="$(mktemp)"
trap 'rm -f "$EVENTS_FILE"' EXIT

python3 - "$MANIFEST" "$EVENTS_FILE" "${#CATEGORIES[@]}" "${CATEGORIES[@]}" "${#APIS[@]}" "${APIS[@]}" <<'PYEOF'
import json
import sys

manifest_path = sys.argv[1]
out_path = sys.argv[2]
n_cats = int(sys.argv[3])
rest = sys.argv[4:]
categories = rest[:n_cats]
n_apis = int(rest[n_cats])
apis = rest[n_cats + 1:n_cats + 1 + n_apis]

with open(manifest_path) as f:
    manifest = json.load(f)

known_categories = manifest['categories']

# Group all known event names by base API (strip trailing chunk suffix
# `_<digits>`), same convention as lttng_gen_category_manifest.py.
by_base = {}
for ev in manifest['all_apis']:
    parts = ev.rsplit('_', 1)
    base = parts[0] if len(parts) == 2 and parts[1].isdigit() else ev
    by_base.setdefault(base, []).append(ev)

errors = []
selected = set()

for cat in categories:
    if cat not in known_categories:
        errors.append(
            f"unknown category '{cat}'. Available categories: "
            + ', '.join(sorted(known_categories)))
        continue
    selected.update(known_categories[cat])

for api in apis:
    if api not in by_base:
        errors.append(
            f"unknown API '{api}' for provider {manifest['provider']} "
            "(expected a base curated API name, e.g. 'hipMalloc', not an "
            "LTTng event/chunk name)")
        continue
    selected.update(by_base[api])

if errors:
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

with open(out_path, 'w') as f:
    for ev in sorted(selected):
        f.write(ev + '\n')
PYEOF

mapfile -t EVENTS < "$EVENTS_FILE"

if [ "${#EVENTS[@]}" -eq 0 ]; then
    echo "ERROR: no events resolved from the given --category/--api selection" >&2
    exit 1
fi

PROV_TAG="rocm_$PROVIDER"

echo "Resolved ${#EVENTS[@]} event(s) for provider $PROV_TAG:" >&2
for ev in "${EVENTS[@]}"; do
    echo "  $ev" >&2
done

# Batch into BATCH_SIZE-sized chunks so the enable-event command line stays
# a reasonable length for very large category selections.
total="${#EVENTS[@]}"
for ((start = 0; start < total; start += BATCH_SIZE)); do
    batch=("${EVENTS[@]:start:BATCH_SIZE}")
    joined=""
    for ev in "${batch[@]}"; do
        if [ -z "$joined" ]; then
            joined="${PROV_TAG}:${ev}"
        else
            joined="${joined},${PROV_TAG}:${ev}"
        fi
    done

    # Only pass `-c <channel>` when the user explicitly requested one;
    # otherwise let LTTng use its own default userspace channel.
    channel_args=()
    if [ -n "$CHANNEL" ]; then
        channel_args=(-c "$CHANNEL")
    fi
    cmd=(lttng enable-event --userspace -s "$SESSION" "${channel_args[@]}" "$joined")

    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s' "lttng enable-event --userspace -s $SESSION"
        if [ -n "$CHANNEL" ]; then
            printf '%s' " -c $CHANNEL"
        fi
        printf " '%s'\n" "$joined"
    else
        echo "+ ${cmd[*]}" >&2
        "${cmd[@]}"
    fi
done

if [ "$DRY_RUN" -eq 0 ]; then
    if [ -n "$CHANNEL" ]; then
        echo "Enabled ${#EVENTS[@]} event(s) on session '$SESSION' channel '$CHANNEL'." >&2
    else
        echo "Enabled ${#EVENTS[@]} event(s) on session '$SESSION' (default channel)." >&2
    fi
fi
