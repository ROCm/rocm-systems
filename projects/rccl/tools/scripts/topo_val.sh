#!/bin/bash
# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for i in {0..88}
do
	if [[ $i -eq 50 ]] || [[ $i -eq 51 ]]
	then
		NCCL_COLLNET_ENABLE=1 $DIR/../topo_expl/topo_expl -m $i > "topo_m$i.log"
	elif [[ $i -eq 54 ]]
	then
		RCCL_ENABLE_MULTIPLE_SAT=1 NCCL_COLLNET_ENABLE=1 $DIR/../topo_expl/topo_expl -m $i > "topo_m$i.log"
	else
		$DIR/../topo_expl/topo_expl -m $i > "topo_m$i.log"
	fi
	$DIR/../TopoVisual/topo_visual.sh -i "topo_m$i.log"
done
