# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

HIP_FILE=$1

if [[ "$HIP_FILE" =~ .*/src/device/.*\.h ]]; then
  perl -pi -e 's/(template\s*<typename T, typename RedOp(?:, (?:typename|int) Proto)?)(, bool isNetOffload.*?)?>/\1, int USE_ACC, int COLL_UNROLL, int Pipeline\2>/g' "$HIP_FILE"
  perl -pi -e 's/(template\s*<typename T, typename RedOp(?:, (?:typename|int) Proto)?(?:, int RCCLMetadata)?)(, bool isNetOffload.*?)?>/\1, int USE_ACC, int COLL_UNROLL, int Pipeline\2>/g' "$HIP_FILE"
  perl -0777 -pi -e 's/(ProtoSimple<[^,]*?,[^,]+?)>/\1, USE_ACC, COLL_UNROLL>/g' "$HIP_FILE"
  perl -0777 -pi -e 's/(runRing<T.*?)((,[\s\\]*(true|false))?>\()/\1, USE_ACC, COLL_UNROLL\2/sg' "$HIP_FILE"
  perl -pi -e 's/(runTreeUpDown<T.*?)>\(/\1, USE_ACC, COLL_UNROLL>(/' "$HIP_FILE"
  perl -pi -e 's/(runTreeSplit<T.*?)>\(/\1, USE_ACC, COLL_UNROLL>(/' "$HIP_FILE"

  perl -pi -e 's/(runTreeSplit<T, RedOp, (ProtoLL|ProtoLL128), USE_ACC, COLL_UNROLL.*?)>/\1, 0>/' "$HIP_FILE"
  perl -pi -e 's/(runTreeUpDown<T, RedOp, (ProtoLL|ProtoLL128), USE_ACC, COLL_UNROLL.*?)>/\1, 0>/' "$HIP_FILE"
  perl -pi -e 's/(runRing<T, RedOp, (ProtoLL|ProtoLL128), USE_ACC, COLL_UNROLL.*?)>/\1, 0>/' "$HIP_FILE"
  perl -pi -e 's/(runRing<T, RedOp, (ProtoLL|ProtoLL128), (RCCL_ONE_NODE_RING_SIMPLE|RCCL_METADATA_EMPTY), USE_ACC, COLL_UNROLL.*?)>/\1, 0>/' "$HIP_FILE"

  perl -pi -e 's/(runRing<T, RedOp, Proto, (RCCL_ONE_NODE_RING_SIMPLE|RCCL_METADATA_EMPTY), USE_ACC, COLL_UNROLL.*?)>/\1, Pipeline>/' "$HIP_FILE"
  perl -pi -e 's/(runRing<T, RedOp, Proto, USE_ACC, COLL_UNROLL.*?)>/\1, Pipeline>/' "$HIP_FILE"
  perl -pi -e 's/(runTreeSplit<T, RedOp, Proto, USE_ACC, COLL_UNROLL.*?)>/\1, Pipeline>/' "$HIP_FILE"
  perl -pi -e 's/(runTreeUpDown<T, RedOp, Proto, USE_ACC, COLL_UNROLL.*?)>/\1, Pipeline>/' "$HIP_FILE"

  # AllGatherV: inject unroll/pipeline params, strip the ones setDataPtrsHelper can't deduce, add prims acc arg.
  perl -pi -e 's/(runAllGatherV<T.*?)(>\()/\1, USE_ACC, COLL_UNROLL\2/g' "$HIP_FILE"
  perl -pi -e 's/(runAllGatherV<T, RedOp, (ProtoLL|ProtoLL128), USE_ACC, COLL_UNROLL.*?)>/\1, 0>/' "$HIP_FILE"
  perl -pi -e 's/(runAllGatherV<T, RedOp, Proto, USE_ACC, COLL_UNROLL.*?)>/\1, Pipeline>/' "$HIP_FILE"
  perl -0777 -pi -e 's/(typename Proto), int USE_ACC, int COLL_UNROLL, int Pipeline(>[\s\w]*?setDataPtrsHelper)/$1$2/' "$HIP_FILE"
  perl -0777 -pi -e 's/(int COLL_UNROLL), int Pipeline(>[\s\w]*?setDataPtrsHelper)/$1$2/' "$HIP_FILE"
  perl -pi -e 's/(prims\.setDataPtrs\(srcBuf, dstBuf, redOpArg, nullptr, 0, 0)\)/\1, nullptr)/g' "$HIP_FILE"

  sed -i "s/\\(struct RunWorkBatch<ncclFunc[^>]*\\)>*/\\1, USE_ACC, COLL_UNROLL, Pipeline, UserRegMode>/" "$HIP_FILE"
  sed -i "s/\\(RunWorkColl<[^,]*,[^,]*,[^,]*,[^,]*,[^>]*\\)>/\\1, USE_ACC, COLL_UNROLL, Pipeline, UserRegMode>/" "$HIP_FILE"

  # Declare the UserRegMode template parameter on RunWorkColl / RunWorkBatch
  # partial specialization headers (those immediately followed by a
  # `struct RunWork...`). Partial specializations may not carry default template
  # arguments, so this is added without a default (the primary templates in
  # common.h provide the default). Must run after the USE_ACC/COLL_UNROLL/Pipeline
  # header expansion above. The optional params between RedOp and USE_ACC cover
  # specializations that carry an extra `int Proto` (e.g. AllGatherV).
  perl -0777 -pi -e 's/(template\s*<typename T, typename RedOp[^>]*?, int USE_ACC, int COLL_UNROLL, int Pipeline)>(\s*\nstruct RunWork)/\1, int UserRegMode>\2/g' "$HIP_FILE"

  # Thread a compile-time UserRegMode parameter onto runRing so LL/LL128 can select
  # their user-buffer access path (0=runtime, 1=registered, 2=non-registered)
  # without a runtime dual code path. Append it (defaulted) to every (already
  # expanded) runRing/runTree header, right after the Pipeline / isNetOffload tail.
  perl -pi -e 's/(template\s*<typename T, typename RedOp, typename Proto[^>]*int Pipeline[^>]*)>/\1, int UserRegMode = 0>/g' "$HIP_FILE"

  # LL128 reg/noreg split: LL (ProtoLL) keeps the baseline single runtime path and
  # LL128 is generated as two separate kernels (see generate.py). The RunWorkColl
  # specialization is instantiated per UserRegMode, so these dispatch helpers just
  # forward the compile-time UserRegMode template parameter into runRing/runTreeSplit.
  #   all_reduce ring : <T, RedOp, Proto, RCCLMetadata, USE_ACC, COLL_UNROLL, Pipeline, UserRegMode>
  #   all_reduce tree : <T, RedOp, Proto, USE_ACC, COLL_UNROLL, Pipeline, UserRegMode>
  #   all_gather ring : <T, RedOp, Proto, USE_ACC, COLL_UNROLL, Pipeline, isNetOffload, UserRegMode>
  #   broadcast  ring : <T, RedOp, Proto, USE_ACC, COLL_UNROLL, Pipeline, UserRegMode>
  perl -pi -e 's/runARRingLL128<T, RedOp>\(/runRing<T, RedOp, ProtoLL128, RCCL_METADATA_EMPTY, USE_ACC, COLL_UNROLL, 0, UserRegMode>(/g' "$HIP_FILE"
  perl -pi -e 's/runARTreeLL128<T, RedOp>\(/runTreeSplit<T, RedOp, ProtoLL128, USE_ACC, COLL_UNROLL, 0, UserRegMode>(/g' "$HIP_FILE"
  perl -pi -e 's/runAGRingLL128<T, RedOp>\(/runRing<T, RedOp, ProtoLL128, USE_ACC, COLL_UNROLL, 0, false, UserRegMode>(/g' "$HIP_FILE"
  perl -pi -e 's/runBcastRingLL128<T, RedOp>\(/runRing<T, RedOp, ProtoLL128, USE_ACC, COLL_UNROLL, 0, UserRegMode>(/g' "$HIP_FILE"
fi