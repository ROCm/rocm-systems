#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build the amdsmi PyPI wheel reproducibly from a rocm-systems checkout.
#
# Usage: tools/build_pypi_release.sh <rocm-systems checkout> <output dir>
#
# Pins the build image, the build tools (pypi_release_constraints.txt), the
# netlink packages that auditwheel copies into the wheel, the archive
# timestamps (commit time), the length of the embedded commit id and the
# source path (the library embeds source file names), so two builds of the
# same commit produce the same file. Fails if the third-party notice does not
# list every library copied into amdsmi.libs/.
set -euo pipefail

REPO=$(cd "$1" && pwd -P)
mkdir -p "$2"
OUT=$(cd "$2" && pwd -P)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
IMAGE=quay.io/pypa/manylinux_2_28_x86_64@sha256:1e102f2e9ed9b7fd8d64b29f798c76078c3fa8f14d34914d7a4045e39419d8e0
RPMS="libnl3-3.7.0-1.el8 libnl3-devel-3.7.0-1.el8 libmnl-1.0.4-6.el8 libmnl-devel-1.0.4-6.el8"
GIT_DIR=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)
SOURCE_DATE_EPOCH=$(git -C "$REPO" log -1 --format=%ct)

# A checkout made under umask 002 leaves group-writable files in the wheel.
chmod -R go-w "$REPO/projects/amdsmi"

docker run --rm \
    -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" -e REPO=/src/rocm-systems -e IMAGE="$IMAGE" \
    -e RPMS="$RPMS" -e OWNER="$(id -u):$(id -g)" \
    -v "$REPO:/src/rocm-systems" -v "$GIT_DIR:$GIT_DIR:ro" -v "$OUT:/out" \
    -v "$HERE/pypi_release_constraints.txt:/constraints.txt:ro" \
    "$IMAGE" bash -c '
set -euo pipefail
trap "chown -R $OWNER $REPO/projects/amdsmi /out" EXIT
dnf install -y -q $RPMS >/dev/null
rpm -q $RPMS >/dev/null
git config --global --add safe.directory "*"
git config --global core.abbrev 11
PY=/opt/python/cp310-cp310/bin/python3
printf "[global]\nconstraint = /constraints.txt\n" >"$($PY -c "import sys; print(sys.prefix)")/pip.conf"
$PY -m pip install -q pip setuptools wheel auditwheel
echo "SOURCE $(git -C "$REPO" rev-parse HEAD) SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH"
echo "IMAGE $IMAGE"
$PY -m pip freeze --all | grep -i -E "^(pip|setuptools|wheel|auditwheel|pyelftools|packaging)==" | sed "s/^/TOOL /"
rpm -q $RPMS | sed "s/^/RPM /"
$PY "$REPO/projects/amdsmi/tools/build_wheel.py" --project-dir "$REPO/projects/amdsmi" \
    --build-dir /tmp/amdsmi-build --output-dir /out/dist --os-variant AlmaLinux8 --repair --release
mkdir -p /out/raw
cp /tmp/raw-wheels/*.whl /out/raw/
cd /out/dist
sha256sum ./*.whl >SHA256SUMS
$PY - ./*.whl <<"PY"
import sys, zipfile
wheel = zipfile.ZipFile(sys.argv[1])
notice = wheel.read("amdsmi/THIRD_PARTY_LICENSES/NOTICE.md").decode()
missing = [n for n in wheel.namelist() if n.startswith("amdsmi.libs/") and n.endswith(tuple("0123456789")) and n not in notice]
if missing:
    sys.exit("NOTICE.md does not list: " + ", ".join(missing))
print("NOTICE lists every vendored library")
PY
' >"$OUT/build.log" 2>&1 || { tail -20 "$OUT/build.log"; exit 1; }

grep -E "^(SOURCE|IMAGE|TOOL|RPM) |^NOTICE lists" "$OUT/build.log"
cat "$OUT/dist/SHA256SUMS"
