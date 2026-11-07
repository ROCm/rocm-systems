# Sourced by lstt-hw-health.yml.

LSTT_PUBLIC_RELEASES="${BASH_SOURCE[0]%/*}/public-releases.txt"

# show <component> <version>: print the version if "<component>: <version>" is
# a line of public-releases.txt, otherwise a placeholder, so an unlisted
# version never reaches a log. Whole-line, fixed-string match: versions contain
# spaces and dots.
show() {
  if grep -qxF -- "$1: $2" "$LSTT_PUBLIC_RELEASES"; then
    printf '%s' "$2"
  else
    printf 'withheld (not a listed public release)'
  fi
}
