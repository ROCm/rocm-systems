#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Check the include-what-you-use (IWYU) runs that iwyu-wrapper.sh recorded
#
# Usage: check-iwyu.sh <log dir> [<summary file>]
#
# Fails if IWYU didn't run, or if any run failed or reported an error
# (including a mapping file it couldn't open), since either means that
# building with HIPFILE_USE_IWYU no longer works. IWYU's suggestions are
# advisory and don't fail the check; it lists the files IWYU suggests
# changing.
#
# If a summary file (such as $GITHUB_STEP_SUMMARY) is given, a Markdown
# summary is appended to it. In GitHub Actions, the result is also
# reported with a workflow annotation.

set -eu

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <log dir> [<summary file>]" >&2
  exit 2
fi
log_dir=$1
summary=${2:-/dev/null}

# Paths are shown relative to the hipFile source directory
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Print a GitHub Actions workflow command (an annotation)
annotate() {
  if [[ ${GITHUB_ACTIONS:-} == true ]]; then
    echo "::$1 title=include-what-you-use::$2"
  fi
}

# Each run has a file holding its exit status, named run.XXXXXX (see
# iwyu-wrapper.sh), and its other files are named after it
shopt -s nullglob
runs=("${log_dir}"/run.??????)
shopt -u nullglob

if [[ ${#runs[@]} -eq 0 ]]; then
  msg="IWYU didn't run. Check that HIPFILE_USE_IWYU is on and that CMake runs iwyu-wrapper.sh."
  echo "${msg}"
  annotate error "${msg}"
  {
    echo "## include-what-you-use"
    echo
    echo "**${msg}**"
  } >> "${summary}"
  exit 1
fi

# A run failed if IWYU exited with a nonzero status or reported an error.
# An empty status file means the wrapper didn't finish.
failed=()
for run in "${runs[@]}"; do
  status=$(< "${run}")
  if [[ ${status} != 0 ]] \
    || grep -qE '(^|: )(fatal )?error: |^Cannot open mapping file' \
      "${run}.stdout" "${run}.stderr"; then
    failed+=("${run}")
  fi
done

# IWYU prints a list of lines to add and a list to remove for each file
# it checks, even when a list is empty. A file needs changing if either
# list has an entry before the blank line that ends it.
mapfile -t suggested < <(
  awk '
    FNR == 1 { listing = 0 }
    / should (add|remove) these lines:$/ {
      file = $0
      sub(/ should (add|remove) these lines:$/, "", file)
      listing = 1
      next
    }
    /^$/ { listing = 0; next }
    listing { print file }
  ' "${runs[@]/%/.stdout}" "${runs[@]/%/.stderr}" \
    | sort -u
)
for i in "${!suggested[@]}"; do
  suggested[i]=${suggested[i]#"${root}/"}
done

echo "IWYU ran on ${#runs[@]} source files."

if [[ ${#failed[@]} -gt 0 ]]; then
  echo
  for run in "${failed[@]}"; do
    echo "IWYU failed (exit status $(< "${run}")):"
    cat "${run}.cmd"
    cat "${run}.stdout" "${run}.stderr"
    echo
  done
  annotate error "IWYU failed on ${#failed[@]} of ${#runs[@]} source files. The Check IWYU step's log has the details."
fi

echo
echo "IWYU suggests changes to ${#suggested[@]} files."
if [[ ${#suggested[@]} -gt 0 ]]; then
  printf '  %s\n' "${suggested[@]}"
  annotate warning "IWYU suggests changes to ${#suggested[@]} files. The suggestions are advisory; the job summary lists the files, and the Build step's log has the suggestions."
fi

{
  echo "## include-what-you-use"
  echo
  echo "IWYU ran on ${#runs[@]} source files."
  if [[ ${#failed[@]} -gt 0 ]]; then
    echo
    echo "**IWYU failed on ${#failed[@]} of them, so building with" \
      "\`HIPFILE_USE_IWYU\` is broken.** The Check IWYU step's log has the" \
      "details."
  fi
  echo
  echo "IWYU suggests changes to ${#suggested[@]} files. The suggestions" \
    "are advisory and don't fail this job. The Build step's log has them."
  if [[ ${#suggested[@]} -gt 0 ]]; then
    echo
    echo "<details><summary>Files IWYU suggests changing</summary>"
    echo
    printf -- "- \`%s\`\n" "${suggested[@]}"
    echo
    echo "</details>"
  fi
} >> "${summary}"

if [[ ${#failed[@]} -gt 0 ]]; then
  exit 1
fi
