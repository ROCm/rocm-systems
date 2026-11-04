#!/bin/bash
# Build the nccl4py (RCCL port, $SRC/projects/rccl/bindings/nccl4py) and nccl-extensions (nccl.m2n) wheels
# into $WORK/whl. Run as yourself inside the container after the patch is applied to $NCCLX (section 13);
# then install both as root: pip install --no-deps --force-reinstall $WORK/whl/*.whl
set -eo pipefail
D=$(dirname "$(readlink -f "$0")")
. $D/py_env.sh
W=$WORK/whl; B=/tmp/m2n-py-build-$(id -u)
rm -rf $B $W && mkdir -p $B $W
cp -r --no-preserve=all $SRC/projects/rccl/bindings/nccl4py $NCCLX/python $B/
(cd $B/nccl4py && SETUPTOOLS_SCM_PRETEND_VERSION=${NCCL4PY_VERSION:-0.2.0} \
  python -m pip wheel -q --no-build-isolation --no-deps -w $W .)
# nccl-extensions' setup.py selects the ROCm build when nccl._hip_compat (the port) is importable.
python -m pip install -q --no-deps --target $B/site $W/nccl4py-*.whl
(cd $B/python && PYTHONPATH=$B/site python -m pip wheel -q --no-build-isolation --no-deps -w $W .)
ls $W
