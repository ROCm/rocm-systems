#!/usr/bin/env bash

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Start of configuration
preamble="Copyright +(\([cC]\) +)?"
postamble=",? +Advanced +Micro +Devices, +Inc\."
find_pattern="$preamble([0-9]{4}-)?[0-9]{4}$postamble"
# printf format string, receives the current year as a parameter
uptodate_pattern="$preamble([0-9]{4}-)?%d$postamble"
# <pattern>/<replacement> interpreted with sed syntax, also passed to printf
# printf interprets '\' escape sequences so they must be escaped
# The capture groups are as follows:
#   - \1 is the whole preamble text
#   - \3 is the start year, \2 is skipped because it is used for an optional part of the preamble
#   - \5 is the end of the copyright statement after the end year, \4 would be the original end year
#     as written in the file, it is replaced by the current year instead.
replace_pattern="($preamble)([0-9]{4})(-[0-9]{4})?($postamble)/\\\1\\\3-%d\\\5"
# End of configuration

print_help() { printf -- \
"\033[36musuage\033[0m: \033[33mcheck_year.sh [-h] [-u] [-a] [-d <SHA>] [-k] [-v]\033[0m
\033[36mdescription\033[0m: Checks for if the copyright year in the staged files is up to date and displays the files with out-of-date copyright statements. Exits with '0' if successful and with '1' if something is out of date.
\033[36moptions\033[0m:
  \033[34m-h\033[0m       Displays this message.
  \033[34m-u\033[0m       Automatically updates the copyright year
  \033[34m-a\033[0m       Automatically applies applies the changes to current staging environment. Implies '-u' and '-c'.
  \033[34m-c\033[0m       Compare files to the index instead of the working tree.
  \033[34m-d <SHA>\033[0m Compare using the diff of a hash.
  \033[34m-k\033[0m       Compare using the fork point: where this branch and 'remotes/origin/HEAD' diverge.
  \033[34m-q\033[0m       Suppress updates about progress.
  \033[34m-v\033[0m       Verbose output.
Use '\033[33mgit config --local hooks.updateCopyright <true|false>\033[0m' to automatically apply copyright changes on commit.
"
}

# argument parsing
apply=false
update=false
verbose=false
forkdiff=false
quiet=false
cached=false

while getopts "auhvkqcd:" arg; do
    case $arg in
        a) update=true;apply=true;cached=true;;
        u) update=true;;
        v) verbose=true;;
        k) forkdiff=true;;
        q) quiet=true;;
        c) cached=true;;
        d) diff_hash=${OPTARG};;
        h) print_help; exit;;
        *) print help; exit 1;;
    esac
done

# If set, check all files changed since the fork point
if $forkdiff; then
    branch="$(git rev-parse --abbrev-ref HEAD)"
    remote="$(git config --local --get "branch.$branch.remote" || echo 'origin')"
    source_commit="remotes/$remote/HEAD"

    # don't use fork-point for finding fork point (lol)
    # see: https://stackoverflow.com/a/53981615
    diff_hash="$(git merge-base "$source_commit" "$branch")"
fi

if [ -n "${diff_hash}" ]; then
    $verbose && printf -- "Using base commit: %s\n" "${diff_hash}"
else
    diff_hash="HEAD"
fi

# Current year
year="$(date +%Y)"

# Enable rename detection with full matches only, this skips copyright checks for file name only
# changes.
diff_opts=(-z --name-only '--diff-filter=MA' '--find-renames=100%')
git_grep_opts=(-z --extended-regexp --ignore-case --no-recursive -I)
if $cached; then
    diff_opts+=(--cached)
    git_grep_opts+=(--cached)
fi

! $quiet && printf -- "Checking if copyright statements are up-to-date... "
mapfile -d $'\0' changed_files < <(git diff-index "${diff_opts[@]}" "$diff_hash" | LANG=C.UTF-8 sort -z)

if ! (( ${#changed_files[@]} )); then
    ! $quiet && printf -- "\033[32mDone!\033[0m\n"
    $verbose && printf -- "\033[36mNo changed files found.\033[0m\n"
    exit 0
fi;

mapfile -d $'\0' found_copyright < <(                                      \
    git grep "${git_grep_opts[@]}" --files-with-matches -e "$find_pattern" \
        -- "${changed_files[@]}" |                                         \
    LANG=C.UTF-8 sort -z)

outdated_copyright=()
if (( ${#found_copyright[@]} )); then
    # uptodate_pattern variable holds the format string using it as such is intentional
    # shellcheck disable=SC2059
    printf -v uptodate_pattern -- "$uptodate_pattern" "$year"
    mapfile -d $'\0' outdated_copyright < <(                                        \
        git grep "${git_grep_opts[@]}" --files-without-match -e "$uptodate_pattern" \
            -- "${found_copyright[@]}" |                                            \
        LANG=C.UTF-8 sort -z)
fi

! $quiet && printf -- "\033[32mDone!\033[0m\n"
if $verbose; then
    # Compute the files that don't have a copyright as the set difference of
    # `changed_files and `found_copyright`
    mapfile -d $'\0' notfound_copyright < <(                                   \
        printf -- '%s\0' "${changed_files[@]}" |                               \
        LANG=C.UTF-8 comm -z -23 - <(printf -- '%s\0' "${found_copyright[@]}"))

    if (( ${#notfound_copyright[@]} )); then
        printf -- "\033[36mCouldn't find a copyright statement in %d file(s):\033[0m\n" \
            "${#notfound_copyright[@]}"
        printf -- '  - %q\n' "${notfound_copyright[@]}"
    fi

    # Similarly the up-to-date files are the difference of `found_copyright` and `outdated_copyright`
    mapfile -d $'\0' uptodate_copyright < <(                                       \
        printf -- '%s\0' "${found_copyright[@]}" |                                 \
        LANG=C.UTF-8 comm -z -23 - <(printf -- '%s\0' "${outdated_copyright[@]}"))

    if (( ${#uptodate_copyright[@]} )); then
        printf -- "\033[36mThe copyright statement was already up to date in %d file(s):\033[0m\n" \
            "${#uptodate_copyright[@]}"
        printf -- '  - %q\n' "${uptodate_copyright[@]}"
    fi
fi

if ! (( ${#outdated_copyright[@]} )); then
    exit 0
fi

printf -- \
"\033[31m==== COPYRIGHT OUT OF DATE ====\033[0m
\033[36m%d file(s) need(s) to be updated:\033[0m\n" "${#outdated_copyright[@]}"
printf -- '  - %q\n' "${outdated_copyright[@]}"

# If we don't need to update, we early exit.
if ! $update; then
    printf -- \
"\nRun '\033[33m%s -u\033[0m' to update the copyright statement(s). See '-h' for more info,
or set '\033[33mgit config --local hooks.updateCopyright true\033[0m' to automatically update copyrights when committing.\n" \
"${BASH_SOURCE[0]}"
    exit 1
fi

if $apply; then
    ! $quiet && printf -- "Updating copyrights and staging changes... "
else
    ! $quiet && printf -- "Updating copyrights... "
fi

# replace_pattern variable holds a format string, using it as such is intentional
# shellcheck disable=SC2059
printf -v replace_pattern -- "$replace_pattern" "$year"
# Just update the files in place if only touching the working-tree
if ! $apply; then
    sed --regexp-extended --separate "s/$replace_pattern/g" -i "${outdated_copyright[@]}"
    printf -- "\033[32mDone!\033[0m\n"
    exit 0
fi

generate_patch() {
    # Sed command to create a hunk for a copyright statement fix
    # expects input to be line number then copyright statement on the next line
    to_hunk_cmd="{# Print hunk header, move to the next line
                  s/.+/@@ -&,1 +&,1 @@/;n
                  # Print removed line by prepending '-' to it
                  ;s/^/-/;p
                  # Print added line, replace the '-' with '+' and replace the copyright statement
                  s/^-/+/;s/$replace_pattern/g}"

    # Run file-names through git ls-files, just to get a (possibly) quoted name for each
    mapfile -t -d $'\n' quoted_files < <(git ls-files --cached -- "${outdated_copyright[@]}")
    for ((i = 0;i < ${#outdated_copyright[@]}; i++)); do
        file="${outdated_copyright["$i"]}"
        quoted="${quoted_files["$i"]}"
        # Drop the quote from the start and end (to avoid quoting twice)
        escaped="${quoted#\"}"; escaped="${escaped%\"}"
        a="\"a/$escaped\""
        b="\"b/$escaped\""

        printf -- "diff --git %s %s\n--- %s\n+++ %s\n" "$a" "$b" "$a" "$b"

        # Print line number and line for each line with a copyright statement
        git cat-file blob ":$file" |                               \
            sed --quiet --regexp-extended "/$find_pattern/{=;p}" | \
            sed --regexp-extended "$to_hunk_cmd"
    done
}

patch_file="$(git rev-parse --git-dir)/copyright-fix.patch"
generate_patch > "$patch_file"

# Cleanup patch file when the script exits
finish () {
    rm -f "$patch_file"
}
# The trap will be invoked whenever the script exits, even due to a signal, this is a bash only
# feature
trap finish EXIT

if ! git apply --unidiff-zero < "$patch_file"; then
    printf -- "\033[31mFailed to apply changes to working tree.
Perhaps the fix is already applied, but not yet staged?\n\033[0m"
    exit 1
fi

if ! git apply --cached --unidiff-zero < "$patch_file"; then
    printf -- "\033[31mFailed to apply change to the index.\n\033[0m"
    exit 1
fi

! $quiet && printf -- "\033[32mDone!\033[0m\n"
exit 0
