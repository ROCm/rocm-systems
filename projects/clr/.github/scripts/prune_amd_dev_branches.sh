#!/usr/bin/env bash
#
# Prune remote/local branches whose names start with a given prefix (default: amd/dev)
# ONLY if:
#   1. They have NO open pull requests (checked via GitHub CLI `gh`).
#   2. Their last commit is older than N days (default: 10).
#
# Additional Safety:
#   - Never deletes the remote's default branch.
#   - Never deletes branches whose names start with any protected prefix:
#         amd-
#         release/rocm-rel
#
# Remote URL Parsing:
#   Supports SSH and HTTPS forms, including enterprise hosts, e.g.:
#     git@github.com:owner/repo.git
#     git@github.mycompany.net:group/repo.git
#     ssh://git@github.mycompany.net/owner/repo.git
#     https://github.com/owner/repo.git
#     https://github.mycompany.net/owner/repo
#
# Options:
#   --remote <name>       Remote name (default: origin)
#   --prefix <prefix>     Candidate branch prefix to prune (default: amd/dev)
#   --dry-run             Show actions; perform no deletions
#   --force               Force delete local branches even if unmerged (-D)
#   --quiet               Suppress ALL stdout (errors still on stderr)
#   --age-days <n>        Minimum age in days since last commit (default: 10)
#   -h | --help           Show this help text
#
# Exit code:
#   0 on success, non-zero on errors
#
set -euo pipefail

remote="origin"
prefix="amd/dev"
dry_run=0
force_delete=0
quiet=0
min_age_days=10
FORCE_LOCAL_FLAG="-d"

# Protected prefixes (hard-coded per request)
protected_prefixes=(
  "amd-"
  "release/rocm-rel"
)

log() { [[ $quiet -eq 0 ]] && echo "$@"; }

show_help() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) remote="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --force) force_delete=1; FORCE_LOCAL_FLAG="-D"; shift ;;
    --quiet) quiet=1; shift ;;
    --age-days) min_age_days="$2"; shift 2 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: Not inside a git repository." >&2
  exit 1
fi

remote_url="$(git remote get-url "$remote" 2>/dev/null || true)"
if [[ -z "$remote_url" ]]; then
  echo "Error: Cannot get URL for remote '$remote'." >&2
  exit 1
fi

# Robust owner/repo parser.
# Strategy:
#   1. Strip protocol (ssh://, git://, https://, http://).
#   2. Strip leading 'git@'.
#   3. Convert the first ':' after host (scp-like syntax) into '/'.
#   4. Remove any port specification (host:port) in ssh:// style (already handled by step 1 normalization).
#   5. After host/, take first two path components as owner + repo.
#   6. Strip trailing .git from repo.
parse_owner_repo() {
  local url="$1"
  local work owner repo path_part

  work="$url"

  # Remove protocol schemes
  work="${work#ssh://}"
  work="${work#git://}"
  work="${work#https://}"
  work="${work#http://}"

  # Remove optional user@
  work="${work#git@}"

  # If it is scp-like host:owner/repo, replace first ':' with '/'
  # but be careful not to replace a colon that might be part of ipv6 (unlikely here)
  if [[ "$work" == *:*/* ]]; then
    # host:owner/repo form
    work="${work/:/\/}"
  fi

  # Now work should look like: host/owner/repo(.git)
  # Strip host
  if [[ "$work" == */* ]]; then
    path_part="${work#*/}"
  else
    return 1
  fi

  # Extract owner and repo (only first two path components)
  IFS='/' read -r owner repo _rest <<< "$path_part"
  if [[ -z "${owner:-}" || -z "${repo:-}" ]]; then
    return 1
  fi

  # Trim possible .git
  repo="${repo%.git}"

  # Basic validation: disallow spaces and slashes
  if [[ "$owner" =~ [[:space:]] || "$repo" =~ [[:space:]] || -z "$repo" ]]; then
    return 1
  fi

  printf '%s/%s' "$owner" "$repo"
}


if ! repo="$(parse_owner_repo "$remote_url")"; then
  echo "Error: Unable to parse owner/repo from remote URL: $remote_url" >&2
  exit 1
fi

log "==> Using repository: $repo"
log "==> Fetching latest refs from '$remote'..."
git fetch --prune "$remote" >/dev/null 2>&1

log "==> Enumerating remote branches with prefix '$prefix'..."
mapfile -t remote_branches < <(git for-each-ref --format='%(refname:short)' "refs/remotes/$remote/$prefix" | sed "s|^$remote/||")

if [[ ${#remote_branches[@]} -eq 0 ]]; then
  log "No matching remote branches found for prefix '$prefix'."
  exit 0
fi

log "Found ${#remote_branches[@]} candidate branch(es). Minimum age: ${min_age_days}d."

default_branch="$(git symbolic-ref refs/remotes/$remote/HEAD 2>/dev/null | sed "s|^refs/remotes/$remote/||" || true)"

now_epoch=$(date +%s)
min_age_seconds=$(( min_age_days * 24 * 60 * 60 ))

deleted_count=0
skipped_pr=0
skipped_age=0
skipped_error=0
skipped_default=0
skipped_protected=0

is_protected_branch() {
  local b="$1"
  for p in "${protected_prefixes[@]}"; do
    if [[ "$b" == "$p"* ]]; then
      return 0
    fi
  done
  return 1
}

for branch in "${remote_branches[@]}"; do
  log ""
  log "--> Evaluating: $branch"

  if [[ "$branch" == "$default_branch" ]]; then
    log "    Skipping default branch."
    ((skipped_default++))
    continue
  fi

  if is_protected_branch "$branch"; then
    log "    Skipping protected branch (matches protected prefix)."
    ((skipped_protected++))
    continue
  fi

  if ! git show-ref --verify --quiet "refs/remotes/$remote/$branch"; then
    log "    Warning: Remote ref vanished; skipping."
    ((skipped_error++))
    continue
  fi

  last_commit_sha="$(git rev-parse "$remote/$branch")"
  if ! last_commit_epoch="$(git show -s --format=%ct "$remote/$branch" 2>/dev/null)"; then
    log "    Warning: Unable to read commit timestamp; skipping."
    ((skipped_error++))
    continue
  fi

  age_seconds=$(( now_epoch - last_commit_epoch ))
  if (( age_seconds < min_age_seconds )); then
    days_old=$(printf "%.1f" "$(echo "$age_seconds / 86400" | bc -l 2>/dev/null || echo 0)")
    log "    Recent activity: last commit $(date -d @"$last_commit_epoch" +'%Y-%m-%d %H:%M:%S') (~${days_old}d) < ${min_age_days}d -> skip."
    ((skipped_age++))
    continue
  fi

  if gh pr list --state open --head "$branch" --limit 1 --json number --repo "$repo" 2>/dev/null | grep -q '"number"'; then
    log "    Open PR exists -> skip."
    ((skipped_pr++))
    continue
  fi

  log "    Eligible (no open PR, age >= ${min_age_days}d, not protected)."
  log "    Last commit: $last_commit_sha $(date -d @"$last_commit_epoch" +'%Y-%m-%d %H:%M:%S')"

  if [[ $dry_run -eq 1 ]]; then
    log "    DRY RUN: Would delete remote/$branch and local (if exists)."
    continue
  fi

  if git push "$remote" --delete "$branch" >/dev/null 2>&1; then
    log "    Remote deleted."
  else
    echo "Warning: Failed to delete remote branch $branch (possibly protected or already removed)." >&2
    ((skipped_error++))
    continue
  fi

  if git show-ref --verify --quiet "refs/heads/$branch"; then
    if git branch "$FORCE_LOCAL_FLAG" "$branch" >/dev/null 2>&1; then
      log "    Local deleted."
    else
      echo "Warning: Failed to delete local branch $branch." >&2
      ((skipped_error++))
    fi
  else
    log "    No local branch present."
  fi

  ((deleted_count++))
done

if [[ $quiet -eq 0 ]]; then
  log ""
  log "Summary:"
  log "  Deleted:                 $deleted_count"
  log "  Skipped (open PR):       $skipped_pr"
  log "  Skipped (too young):     $skipped_age"
  log "  Skipped (default):       $skipped_default"
  log "  Skipped (protected):     $skipped_protected"
  log "  Skipped (errors):        $skipped_error"
  [[ $dry_run -eq 1 ]] && log "  Mode: DRY RUN"
  [[ $force_delete -eq 1 ]] && log "  Local deletes forced (-D)"
  log "Done."
fi