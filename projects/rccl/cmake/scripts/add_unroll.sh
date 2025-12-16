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
  perl -pi -e 's/(template<typename T, typename RedOp(?:, typename Proto)?)(, bool isNetOffload.*?)?>/\1, int USE_ACC, int COLL_UNROLL, int Pipeline\2>/g' "$HIP_FILE"
  perl -pi -e 's/(template<typename T, typename RedOp(?:, typename Proto)?(?:, int RCCLMetadata)?)(, bool isNetOffload.*?)?>/\1, int USE_ACC, int COLL_UNROLL, int Pipeline\2>/g' "$HIP_FILE"
  perl -pi -e 's/(ProtoSimple<[^,]*?,[^,]+?)>/\1, USE_ACC, COLL_UNROLL>/g' "$HIP_FILE"
  perl -pi -e 's/(runRing<T.*?)((, (true|false))?>\()/\1, USE_ACC, COLL_UNROLL\2/g' "$HIP_FILE"
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
  sed -i "s/\\(struct RunWorkBatch<ncclFunc[^>]*\\)>*/\\1, USE_ACC, COLL_UNROLL, Pipeline>/" "$HIP_FILE"
  sed -i "s/\\(RunWorkColl<[^,]*,[^,]*,[^,]*,[^,]*,[^>]*\\)>/\\1, USE_ACC, COLL_UNROLL, Pipeline>/" "$HIP_FILE"
fi