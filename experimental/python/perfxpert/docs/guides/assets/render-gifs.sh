#!/usr/bin/env bash
set -euo pipefail

ASSETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAPES_DIR="$ASSETS_DIR/tapes"
SCRIPTS_DIR="$ASSETS_DIR/scripts"
REPO_ROOT="$(git -C "$ASSETS_DIR" rev-parse --show-toplevel)"
CONTAINER_NAME="${PERFXPERT_GUIDE_VHS_CONTAINER:-perfxpert-guide-vhs}"
ROCM_IMAGE="${PERFXPERT_GUIDE_VHS_IMAGE:-rocm/dev-ubuntu-22.04:7.2.2}"
declare -a tmp_tapes=()

cleanup() {
  rm -f "${tmp_tapes[@]:-}"
  "$SCRIPTS_DIR/stop-demo-container.sh" || true
}

make_temp_tape() {
  local tape="$1"
  local marker=""
  local prelude=""
  local tmp_tape

  if grep -Fq 'Set Shell "./vhs-demo-shell.sh"' "$tape"; then
    marker='Set Shell "./vhs-demo-shell.sh"'
    prelude=$(
      cat <<EOF
Type "$SCRIPTS_DIR/start-demo-container.sh"
Enter
Sleep 200ms
Type "docker exec -it $CONTAINER_NAME bash --noprofile --norc"
Enter
Sleep 500ms
EOF
    )
  elif grep -Fq 'Set Shell "./vhs-install-shell.sh"' "$tape"; then
    marker='Set Shell "./vhs-install-shell.sh"'
    prelude=$(
      cat <<EOF
Type "docker run --rm -it -v $REPO_ROOT:/src:ro $ROCM_IMAGE bash -lc 'set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; apt-get update >/dev/null; apt-get install -y --no-install-recommends git jq less ca-certificates >/dev/null; cp -a /src /tmp/src; exec bash --noprofile --norc'"
Enter
Sleep 25s
EOF
    )
  else
    return 1
  fi

  tmp_tape="$(mktemp "$TAPES_DIR/.render-XXXXXX-$(basename "$tape")")"
  tmp_tapes+=("$tmp_tape")

  awk -v marker="$marker" -v prelude="$prelude" '
    $0 == marker {
      print "Set Shell \"bash\""
      next
    }
    !inserted && $0 == "Hide" {
      print
      if (prelude != "") print prelude
      inserted = 1
      next
    }
    { print }
  ' "$tape" > "$tmp_tape"

  printf '%s\n' "$tmp_tape"
}

declare -a tapes=()
if [ "$#" -eq 0 ]; then
  tapes=("$TAPES_DIR"/*.tape)
else
  for arg in "$@"; do
    if [ -f "$arg" ]; then
      tapes+=("$arg")
    else
      tapes+=("$TAPES_DIR/$arg")
    fi
  done
fi

need_demo_container=0
for tape in "${tapes[@]}"; do
  if [ "$(basename "$tape")" != "01-install.tape" ]; then
    need_demo_container=1
    break
  fi
done

if [ "$need_demo_container" -eq 1 ]; then
  trap cleanup EXIT
  "$SCRIPTS_DIR/start-demo-container.sh"
else
  trap cleanup EXIT
fi

cd "$TAPES_DIR"
for tape in "${tapes[@]}"; do
  if temp_tape="$(make_temp_tape "$tape")"; then
    vhs "$(basename "$temp_tape")"
  else
    vhs "$(basename "$tape")"
  fi
done
