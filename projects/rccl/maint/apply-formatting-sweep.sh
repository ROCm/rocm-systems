#!/usr/bin/env bash
#
# apply-formatting-sweep.sh — Reproduce the tree-wide formatting sweep.
#
# Runs clang-format and the pre-commit fixer hooks over projects/rccl, proves
# that what they changed really is only formatting, and lands the result as two
# commits: the sweep itself, and an entry for it in .git-blame-ignore-revs.
#
# Nothing here is routine. A sweep is for the one-off occasion when a formatting
# rule changes, or a hook is adopted and the backlog has to be cleared in a
# single reviewable commit instead of dribbling into unrelated pull requests.
# Day to day, the hooks fix the files you touch and that is enough.
#
#   projects/rccl/maint/apply-formatting-sweep.sh
#   projects/rccl/maint/apply-formatting-sweep.sh --no-commit   # leave it unstaged
#   projects/rccl/maint/apply-formatting-sweep.sh --check       # verify only
#
# The working tree must be clean: the checks work by comparing every file
# against its committed version, so uncommitted edits would be indistinguishable
# from the formatter's own work and would be swept into the commit.
#
# What gets formatted is not restated here. The clang-format hook's `files:`
# pattern in .pre-commit-config.yaml is the single owner of that, and this
# script reads it, along with the clang-format version pinned by the hook's
# `rev:`. See maint/README-rccl.md.

set -euo pipefail

RCCL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${RCCL_ROOT}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="projects/rccl/.pre-commit-config.yaml"
BLAME_IGNORE="projects/rccl/.git-blame-ignore-revs"
VENV_DIR="${TMPDIR:-/tmp}/rccl-formatting-sweep-venv"

COMMIT=1
CHECK_ONLY=0

usage() {
  echo "Usage: $0 [--no-commit | --check]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-commit) COMMIT=0 ;;
    --check) CHECK_ONLY=1; COMMIT=0 ;;
    -h | --help) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree is not clean." >&2
  echo "       Every check below compares a file against its committed version," >&2
  echo "       so an uncommitted edit cannot be told apart from the formatter's" >&2
  echo "       own work, and would be swept into the formatting commit." >&2
  exit 2
fi

if ! command -v pre-commit &>/dev/null; then
  echo "ERROR: pre-commit not found. Install it with 'pip install pre-commit'" >&2
  echo "       (or 'uv tool install pre-commit')." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Tooling, pinned to what the hooks themselves use
# ---------------------------------------------------------------------------
# A sweep has to produce exactly what the clang-format hook would accept, and
# different clang-format releases disagree about the same file. Taking whatever
# is on PATH would make the result depend on the machine it ran on, so the
# version comes from the hook's own `rev:` and is installed to match. Set
# CLANG_FORMAT to override, accepting that you then own the difference.
CLANG_FORMAT_VERSION="$(
  python3 - "$CONFIG" <<'PY'
import re, sys
# Deliberately not a YAML parse: this runs before the venv that would provide
# PyYAML exists. The rev always directly follows the mirror's repo line.
text = open(sys.argv[1]).read()
m = re.search(r'repo:\s*\S*mirrors-clang-format\s*\n\s*rev:\s*v?(\S+)', text)
if not m:
    sys.exit("cannot find the clang-format hook's rev in " + sys.argv[1])
print(m.group(1))
PY
)"

if [[ -n "${CLANG_FORMAT:-}" ]]; then
  echo "Using CLANG_FORMAT=${CLANG_FORMAT} (overriding the pinned ${CLANG_FORMAT_VERSION})"
else
  echo "Installing clang-format==${CLANG_FORMAT_VERSION} (pinned by ${CONFIG})..."
  python3 -m venv "$VENV_DIR"
  "${VENV_DIR}/bin/pip" install --quiet \
    "clang-format==${CLANG_FORMAT_VERSION}" pyyaml
  CLANG_FORMAT="${VENV_DIR}/bin/clang-format"
fi

PYTHON="${VENV_DIR}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3

echo "clang-format: $("$CLANG_FORMAT" --version)"

# ---------------------------------------------------------------------------
# Work out which files the clang-format hook claims
# ---------------------------------------------------------------------------
# Derived from the hook's own patterns rather than restated, so this cannot
# drift from what CI checks. Honours the config's top-level files/exclude too,
# since pre-commit applies those to every hook.
echo "Resolving the clang-format file set from ${CONFIG}..."
IN_SCOPE="$(mktemp)"
trap 'rm -f "$IN_SCOPE"' EXIT

"$PYTHON" - "$CONFIG" "$IN_SCOPE" <<'PY'
import re, subprocess, sys, yaml

config_path, out_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(config_path))

hook = next(
    h for r in cfg["repos"] for h in r.get("hooks", [])
    if h.get("id") == "clang-format"
)

def compiled(pattern):
    # pre-commit compiles these with re.VERBOSE, which is what lets the shared
    # configs write a commented (?x) block.
    return re.compile(pattern, re.VERBOSE) if pattern else None

includes = [compiled(p) for p in (cfg.get("files"), hook.get("files")) if p]
excludes = [compiled(p) for p in (cfg.get("exclude"), hook.get("exclude")) if p]

listing = subprocess.run(
    ["git", "ls-files", "-z"], check=True, capture_output=True,
).stdout.decode()
names = [n for n in listing.split("\0") if n]
keep = [
    n for n in names
    if all(p.search(n) for p in includes) and not any(p.search(n) for p in excludes)
]
open(out_path, "w").write("".join(n + "\n" for n in sorted(keep)))
print(f"  {len(keep)} files in the clang-format hook's scope")
PY

# ---------------------------------------------------------------------------
# Record what clang-format would produce, before anything is touched
# ---------------------------------------------------------------------------
# Captured now, from the pristine tree, so the verification afterwards compares
# the result against an independent expectation rather than against a rerun of
# the same command on the file it just wrote. Files matched by
# .clang-format-ignore come back empty: clang-format prints nothing for them,
# which is how it reports a skip, so they are held aside and checked as
# whitespace-only along with everything else.
EXPECTED_DIR="$(mktemp -d)"
IGNORED_LIST="$(mktemp)"
trap 'rm -f "$IN_SCOPE" "$IGNORED_LIST"; rm -rf "$EXPECTED_DIR"' EXIT

echo "Capturing expected clang-format output..."
while IFS= read -r f; do
  rel="${f#projects/rccl/}"
  dest="${EXPECTED_DIR}/${f}"
  mkdir -p "$(dirname "$dest")"
  (cd "$RCCL_ROOT" && "$CLANG_FORMAT" --style=file "$rel") > "$dest"
  if [[ ! -s "$dest" ]]; then
    echo "$f" >> "$IGNORED_LIST"
    rm -f "$dest"
  fi
done < "$IN_SCOPE"
echo "  $(grep -c . "$IGNORED_LIST" 2>/dev/null || echo 0) skipped via .clang-format-ignore"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo
  echo "--check: tree is clean and the expectation was computed; nothing applied."
  exit 0
fi

# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
# clang-format first, then the whitespace hooks. The order is not arbitrary:
# end-of-file-fixer has to run last, because clang-format leaves a file without
# a trailing newline exactly as it found it, and the sweep would otherwise land
# in a state the hooks still reject.
echo
echo "Applying clang-format..."
(cd "$RCCL_ROOT" && tr '\n' '\0' < "$IN_SCOPE" \
  | sed -z 's|^projects/rccl/||' \
  | xargs -0 --no-run-if-empty "$CLANG_FORMAT" -i --style=file)

echo "Applying the pre-commit fixer hooks..."
# These exit non-zero precisely when they change something, which here is the
# point rather than a failure.
pre-commit run --config "$CONFIG" --all-files || true

if git diff --quiet; then
  echo
  echo "Nothing to do: the tree is already formatted."
  exit 0
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
# "I ran the formatter" and "this diff contains only what the formatter does"
# are different claims, and a sweep is reviewed by trusting the second one. So
# every changed file is held to one of two standards, and anything meeting
# neither stops the script before a commit exists.
echo
echo "Verifying..."
CHANGED="$(mktemp)"
VERIFIED="$(mktemp)"
trap 'rm -f "$IN_SCOPE" "$IGNORED_LIST" "$CHANGED" "$VERIFIED"; rm -rf "$EXPECTED_DIR"' EXIT
git diff --name-only > "$CHANGED"

fail=0

# Standard one: a file in clang-format's scope must be byte-identical to the
# output captured from its committed version, give or take the trailing newline
# that end-of-file-fixer adds afterwards.
exact=0
while IFS= read -r f; do
  expected="${EXPECTED_DIR}/${f}"
  [[ -f "$expected" ]] || continue
  grep -qxF "$f" "$CHANGED" || continue
  if cmp -s "$expected" "$f" \
    || { printf '%s\n' "$(cat "$expected")" | cmp -s - "$f"; }; then
    echo "$f" >> "$VERIFIED"
    exact=$((exact + 1))
  else
    echo "  NOT CLANG-FORMAT OUTPUT: $f" >&2
    fail=1
  fi
done < "$IN_SCOPE"
echo "  ${exact} file(s) are exactly clang-format's output"

# Standard two: everything else may differ only in whitespace. --ignore-all-space
# alone would not do, since a hunk that only adds or removes a blank line still
# shows up; --ignore-blank-lines covers that, and what remains is real content.
ws=0
while IFS= read -r f; do
  grep -qxF "$f" "$VERIFIED" 2>/dev/null && continue
  if [[ -z "$(git diff --ignore-all-space --ignore-blank-lines -- "$f" \
              | grep '^[+-][^+-]' || true)" ]]; then
    ws=$((ws + 1))
  else
    echo "  NOT WHITESPACE-ONLY: $f" >&2
    fail=1
  fi
done < "$CHANGED"
echo "  ${ws} file(s) changed in whitespace only"

if [[ "$fail" -ne 0 ]]; then
  echo >&2
  echo "ERROR: the diff contains changes neither tool accounts for, so it is not" >&2
  echo "       a formatting sweep. Nothing has been committed; inspect with" >&2
  echo "       'git diff' and reset with 'git checkout -- .'." >&2
  exit 1
fi

# The hooks must now be satisfied by their own output. If this fails the sweep
# is not a fixed point, and committing it would leave the next pull request to
# trip over the remainder.
echo "  re-running the hooks against the result..."
if ! pre-commit run --config "$CONFIG" --all-files > /dev/null 2>&1; then
  echo "ERROR: the hooks still report changes after the sweep. Nothing has been" >&2
  echo "       committed. Re-run them to see what is left:" >&2
  echo "         pre-commit run --config ${CONFIG} --all-files" >&2
  exit 1
fi
echo "  hooks pass against the swept tree"

echo
echo "Swept $(wc -l < "$CHANGED") file(s)."

if [[ "$COMMIT" -eq 0 ]]; then
  echo "--no-commit: the changes are in the working tree, uncommitted."
  exit 0
fi

# ---------------------------------------------------------------------------
# Commit, then ignore that commit in blame
# ---------------------------------------------------------------------------
# Two commits, because a commit cannot contain its own hash. Keeping the sweep
# free of the blame-ignore edit also keeps it mechanically checkable: the whole
# of it satisfies the standards above, with nothing hand-written mixed in.
echo
echo "Committing..."
git add --update
git commit --quiet --file - <<EOF
[AICOMRCCL-3539] [rccl] Apply clang-format and the whitespace hooks

Formatting only, produced by maint/apply-formatting-sweep.sh, and
recorded in .git-blame-ignore-revs by the commit that follows.

Every changed file was checked to be either byte-identical to
clang-format's output for its previous content, or different in
whitespace alone. The hooks pass against the result.

Co-authored-by: Pi Agent
EOF

SWEEP_SHA="$(git rev-parse HEAD)"

printf '\n# [Auto] Apply clang-format and the whitespace hooks\n%s\n' \
  "$SWEEP_SHA" >> "$BLAME_IGNORE"

git add "$BLAME_IGNORE"
git commit --quiet --file - <<EOF
[AICOMRCCL-3539] [rccl] Ignore the formatting sweep in git blame

Records $(git rev-parse --short "$SWEEP_SHA") so that \`git blame\` attributes the
lines it touched to whoever last changed them for a reason, rather
than to the formatter.

Co-authored-by: Pi Agent
EOF

echo
git --no-pager log --oneline -2
