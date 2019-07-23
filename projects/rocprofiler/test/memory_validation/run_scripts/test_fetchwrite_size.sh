#!/bin/bash

###############################################################################
# Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $BASE_DIR/global.cfg
initialize

Ns_M="8192 131072|512"

headers="Mem32Bwrites FetchSize WriteSize"

# set up outputs
OUT_DIR="outs"; [ -d $OUT_DIR ] && rm -rf $OUT_DIR; mkdir $OUT_DIR
log_file=$OUT_DIR/"prof.log"; >$log_file

kerns="cache_test_RO cache_test_WO"
# append '/' if a non-default/empty path is specified
if [[ ! -z $ROCP_PATH ]]; then ROCP_PATH=$ROCP_PATH"/"; fi

function one_run
{
    local N=$1
    local s=$2
    local M=$3
    local level=$4
    rst_sym="N${N}_s${s}_M${M}k"
    rst_file=$OUT_DIR/"${rst_sym}.csv"
    #echo "$N: $s -- $rst_file"
    printf "\n Traverse %5s-int array in %5s-int stride with %4s accesses" \
            $N $s "${M}K"
    ${ROCP_PATH}rocprof -i ${BASE_DIR}/../pmc_config_files/cache_pmc.txt -o \
      ${BASE_DIR}/../$rst_file $PATH_CACHE_BENCH/cache $s $N $M \
      >> $log_file

    # check the profiling result
    checkProfRun $rst_file $log_file

    colIds=$(getColIds $rst_file)

    sed -i 's/(.*)/(args)/g' $rst_file

    for kern in $kerns
    do
        mc32wrs=0; fetchsize=0; writesize=0
        values=`grep $kern $rst_file | sed 's/,/ /g'`
        for colIdStr in $colIds
        do
          colId=`echo $colIdStr | cut -f1 -d'|'`
          colStr=`echo $colIdStr | cut -f2 -d'|'`
          colVal=`echo $values | cut -f$colId -d' '`

          if [[ $kern == cache_test_RO || $kern == cache_test_WO ]]; then
            if [[ $colStr == Mem32Bwrites ]]; then mc32wrs=$colVal
            elif [[ $colStr == FetchSize ]]; then fetchsize=$colVal
            elif [[ $colStr == WriteSize ]]; then writesize=$colVal; fi
          fi
        done

        rstdiff=1        # check result (0: pass/no-difference, 1: fail)

        line=$b_tcp; if (( $s > $b_tcp)); then line=$s; fi
        coldmisses=`echo "scale=0; $N/$line" | bc`
        #-- use kernel 'cache_test_RO' to validate fetch size
        if [[ $kern == cache_test_RO ]]; then
          # program-level expectation: coldmisses*cacheline_size
          expect_fetchKB=$(awk -v n=$coldmisses \
                          'BEGIN{printf("%.0f", 64*n/1024)}')
          # profiled value
          profile_fetchKB=$fetchsize

          printf "\n\tFetch-Size: expected=%4s KB, profiled=%4s KB, " \
                    $expect_fetchKB $profile_fetchKB
          if (( $profile_fetchKB == $expect_fetchKB )); then
            printf "test [${GREEN}PASS${NC}]"
          else printf "test [${RED}FAIL${NC}]"; fi
        #-- use kernel 'cache_test_WO' to validate write size
        elif [[ $kern == cache_test_WO ]]; then
          # program-level expectation: coldmisses*req_size
          expect0Max_writeKB=$(awk -v n=$coldmisses \
                          'BEGIN{printf("%.0f", 64*n/1024)}')
          expect0Min_writeKB=$(awk -v n=$coldmisses \
                          'BEGIN{printf("%.0f", 32*n/1024)}')
          expect1_writeKB=$(awk -v wr32B=$mc32wrs \
                          'BEGIN{printf("%.0f", (32*wr32B)/1024)}')
          profile_writeKB=$writesize

          # stride is less then a line, always write 64B
          if (( $s < $b_tcp )); then expect1_writeKB=$expect0Max_writeKB; fi

          rstdiff=1        #different by default
          expect_writeKB=$expect1_writeKB  #expected size
          if (( $profile_writeKB >= $expect0Min_writeKB )) \
                  && (( $profile_writeKB <= $expect0Max_writeKB )) \
                  && (( $profile_writeKB == $expect1_writeKB )); then
            rstdiff=0
          # not fall in expected range (min, max)
          elif (( $profile_writeKB == $expect1_writeKB )); then
            rstdiff=1; expect_writeKB=-1
          # in the range, but not as desired
          else rstdiff=1; fi

          if (( $expect_writeKB == -1 )); then
            printf "\n\tWrite-Size: expected>=%3s KB, profiled=%4s KB, " \
                    $expect0Min_writeKB $profile_writeKB
            printf "test [${RED}FAIL${NC}]"
          else
            printf "\n\tWrite-Size: expected=%4s KB, profiled=%4s KB, " \
                    $expect_writeKB $profile_writeKB
            if [[ $rstdiff == 0 ]]; then printf "test [${GREEN}PASS${NC}]"
            else printf "test [${RED}FAIL${NC}]"; fi
          fi
        fi

    done
}

Ns=`echo $Ns_M | cut -f1 -d'|'`                             #-- array sizes
M=`echo $Ns_M | cut -f2 -d'|' | cut -f1 -d'@'`              #-- array accesses
S=""
if [[ $Ns_M == *@* ]]; then S=`echo $Ns_M | cut -f2 -d'@'`; fi  #-- stride
#echo $S

printf "\n\t=========================================================\n"
printf "\t================ Test [fetch/write size] ================\n"
printf "\t========================================================="

for N in $Ns
do
    if [[ x$S == x ]]; then
      m_stride=$N
      for (( s=1; s<=$m_stride/32; s*=2 ))
      do
        one_run $N $s $M $cache
      done
    else
      one_run $N $S $M $cache
    fi
done
printf "\n"
