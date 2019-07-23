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
REQ_DIFF=40     # at most 30 TCC reads are not from TCP

#-- test kernel:
#--     single thread, pointer chase
#-- test settings (format: Ns|M):
#--     issue M=512k accesses access the array with N elements using different
#--      strides (s=1/2/4/...)
#-- miss rate patterns:
#--     1) if N <= C: r = (N/b)/M
#--         array fits into the cache, causing no replacement, and thus misses
#--          only happens when the line is being loaded,
#--         i.e., cold misses
#--     2) if N > C:
#--         a). r = s/b,  s in [1, b)
#--         b). r = 100%, s in [b, N/a)
#--         c). r = 0%,   s in [N/a, ]
CACHES="TCP TCC"        # which caches to test
input_args="$1 $2"
caches="${input_args// }"
if [[ ! -z $caches ]]; then
    if [[ $caches == TCP ]] || [[ $caches == TCC ]]; then CACHES=$caches
    elif [[ $caches != TCPTCC ]]; then
        printf "${RED}Supported caches are TCP and TCC ...${NC}\n"; exit
    fi
fi

#-- TCP
TCP_Ns_M="64 16384|512"
#-- TCC
TCC_Ns_M="4096 8192 16384 32768 65536 131072 2097152 4194304|512@16"


headers="TCC_HIT_sum TCC_MISS_sum"

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
    printf "\n Traverse %7s-int array in %5s-int stride with %4s accesses" \
                $N $s "${M}K"
    ${ROCP_PATH}rocprof -i ${BASE_DIR}/../pmc_config_files/cache_pmc.txt -o \
      ${BASE_DIR}/../$rst_file $PATH_CACHE_BENCH/cache $s $N $M \
      >> $log_file

    # check the profiling result
    checkProfRun $rst_file $log_file

    colIds=$(getColIds $rst_file)

    sed -i 's/(.*)/(args)/g' $rst_file

    totTcpRds=0; totTcpWrs=0                    # tcp rds/wrs
    missTcpRds=0; missTcpWrs=0                  # tcp rd/wr misses
    hitTccReqs=0; missTccReqs=0; totTccReqs=0   # tcc hits/misses/reqs
    totTccRds=0; totTccWrs=0                    # tcc rds/wrs
    missTccRds=0; missTccWrs=0                  # tcc rd/wr misses

    for kern in $kerns
    do
        values=`grep $kern $rst_file | sed 's/,/ /g'`
        for colIdStr in $colIds
        do
          colId=`echo $colIdStr | cut -f1 -d'|'`
          colStr=`echo $colIdStr | cut -f2 -d'|'`
          colVal=`echo $values | cut -f$colId -d' '`

          if [[ $kern == cache_test_RO || $kern == cache_test_WO ]]; then
            if [[ $colStr == TCC_HIT_sum ]]; then hitTccReqs=$colVal
            elif [[ $colStr == TCC_MISS_sum ]]; then missTccReqs=$colVal; fi
          fi
        done

        rstdiff=1        # check result (0: pass/no-difference, 1: fail)
        totTccReqs=$((hitTccReqs + missTccReqs))

        #-- use kernel 'cache_test_RO' to validate read miss rate
        if [[ $kern == cache_test_RO ]]; then
          totTcpRds=$(( $M*1024 )); totTcpWrs=1     # tcp rds/wrs
          totTccWrs=1; missTccWrs=1                 # one write, and miss
          missTccRds=$(($missTccReqs-$missTccWrs))  # remaining are read misses
          totTccRds=$(($totTccReqs - $totTccWrs))   # remaining are reads
          missTcpRds=$totTccRds                     # tcp rd misses + other

          mn=0; md=0        # miss rate denoted using numerator and denomiator
          line=$b_tcp; if (( $s > $b_tcp)); then line=$s; fi
          expectedMissTcpRds=0
          if (( $N > $C_tcp )) && (( $s >= $b_tcp )); then
            # array size is larger than cache capacity, and stride is larger
            #  than a cacheline size (N>C && s>=b)
            #  100% miss if s in [b, N/a), only code misses if s is [N/a,]
            if (($missTcpRds - $totTcpRds < $REQ_DIFF)) \
                    && (($totTcpRds - $missTcpRds < $REQ_DIFF)); then
             rstdiff=0; expectedMissTcpRds=$totTcpRds
            elif (($missTcpRds - $N/$s < $REQ_DIFF)) \
                    && (($N/$s - $missTcpRds < $REQ_DIFF)); then
             rstdiff=0; expectedMissTcpRds=$(($N/$s)); fi
          else
            # array size is no larger than cache size (N<=C): only cold misses
            # array size is larger than cache capacity, and stride is less than
            # a cacheline size (N>C && s<b)
            #   always miss when a line is loaded into cache
            if (( $N <= $C_tcp )); then
              mn=$(( $N/$line )); md=$totTcpRds; expectedMissTcpRds=$mn
            else
              mn=$s; md=$b_tcp
              expectedMissTcpRds=$(awk -v s=$s -v b=$b_tcp -v rd=$totTcpRds \
                              'BEGIN{printf("%.0f", s*rd/b)}')
            fi
            if (($missTcpRds - $expectedMissTcpRds < $REQ_DIFF)) \
                    && (($expectedMissTcpRds - $missTcpRds < $REQ_DIFF)); then
                rstdiff=0; fi
          fi

          # tcp validation
          if [[ $level == TCP ]]; then
            printf "\n\tTCP-READ  : expected=%6s±%s, profiled=%6s, " \
                    $expectedMissTcpRds $REQ_DIFF $missTcpRds
            if (( $rstdiff == 0 )); then printf "test [${GREEN}PASS${NC}]"
            else printf "test [${RED}FAIL${NC}]"; fi
          # tcc validation
          elif [[ $level == TCC ]]; then
            if (( $rstdiff != 0 )); then
              printf "\n\tTCP-READ  : test [${RED}FAIL${NC}]"; fi
            # tcp miss rate
            tcprdmissrate=$(awk -v mr=$missTcpRds -v rd=$totTcpRds \
                            'BEGIN{printf("%f", mr/rd)}')
            # tcc miss rate
            tccrdmissrate=$(awk -v mr=$missTccRds -v rd=$totTccRds \
                            'BEGIN{printf("%f", mr/rd)}')

            coldmisses=`echo "scale=0; $N/$line" | bc`

            expectedMissTccRds=$coldmisses
            if (( $(echo "$tccrdmissrate > 0.98" | bc -l) )) \
                    && (( $(echo "$tcprdmissrate > 0.98" | bc -l) )); then
                expectedMissTccRds=$totTcpRds
                printf "\n\tTCC-READ  : expected=%6s±%s, profiled=%6s, " \
                    $expectedMissTccRds ".5%" $missTccRds
            else
                printf "\n\tTCC-READ  : expected=%6s±%s, profiled=%6s, " \
                    $expectedMissTccRds $REQ_DIFF $missTccRds
            fi

            # absolute difference between profiled and expected
            diff=$(( $missTccRds - $coldmisses ))
            if (( $(echo "$diff < 0" | bc -l) )); then
                diff=`echo "$diff*-1" | bc -l`; fi
            if (( $(echo "$tccrdmissrate > 0.98" | bc -l) )); then
              printf "test [${GREEN}PASS${NC}]"
            elif (( $diff < $REQ_DIFF )); then
              printf "test [${GREEN}PASS${NC}]"
            else printf "test [${RED}FAIL${NC}]"; fi
          fi
        #-- use kernel 'cache_test_WO' to validate TCP write miss rate
        elif [[ $kern == cache_test_WO ]]; then
          totTcpRds=0; totTcpWrs=$(( $M*1024 )); # tcp rds/wrs
          totTccRds=0; missTccRds=$totTccRds         # no reads from tcp
          totTccWrs=$(($totTccReqs - $totTccRds))    # remaining are writes
          missTccWrs=$(($missTccReqs - $missTccRds)) # remaining are write mis
          missTcpWrs=$totTccWrs                      # all tcc wrs are from tcp

          if (($missTcpWrs - $totTcpWrs < $REQ_DIFF)) \
                  && (($totTcpWrs - $missTcpWrs < $REQ_DIFF)); then
            rstdiff=0; fi
          # tcp is write through
          expectedMissTcpWrs=$totTcpWrs

          if [[ $level == TCP ]]; then
            printf "\n\tTCP-WRITE : expected=%6s±%s, profiled=%6s, " \
                    $expectedMissTcpWrs $REQ_DIFF $missTcpWrs
            if (( $rstdiff == 0 )); then printf "test [${GREEN}PASS${NC}]"
            else printf "test [${RED}FAIL${NC}]"; fi
          # tcc validation
          elif [[ $level == TCC ]]; then
            if (( $rstdiff != 0 )); then
                    printf "\n\tTCP-WRITE  : test [${RED}FAIL${NC}]"; fi
            tccwrmissrate=$(awk -v mw=$missTccWrs -v wr=$totTccWrs \
                            'BEGIN{printf("%f", mw/wr)}')

            coldmisses=`echo "scale=0; $N/$line" | bc`

            expectedMissTccWrs=$coldmisses
            if (( $(echo "$tccwrmissrate > 0.98" | bc -l) )); then
                expectedMissTccWrs=$totTcpWrs;
                printf "\n\tTCC-WRITE : expected=%6s±%s, profiled=%6s, " \
                        $expectedMissTccWrs ".5%" $missTccWrs
            else
                printf "\n\tTCC-WRITE : expected=%6s±%s, profiled=%6s, " \
                        $expectedMissTccWrs $REQ_DIFF $missTccWrs
            fi
            if (($missTccWrs - $coldmisses < $REQ_DIFF)) \
                    && (($missTccWrs - $coldmisses < $REQ_DIFF)); then
              printf "test [${GREEN}PASS${NC}]"
            elif (( $(echo "$tccwrmissrate > 0.98" | bc -l) )); then
              printf "test [${GREEN}PASS${NC}]"
            else printf "test [${RED}FAIL${NC}]"; fi
          fi
        fi

    done
}

for cache in $CACHES
do
{
    cfgname="${cache}_Ns_M"
    Ns_M=${!cfgname}

    Ns=`echo $Ns_M | cut -f1 -d'|'`                         #-- array sizes
    M=`echo $Ns_M | cut -f2 -d'|' | cut -f1 -d'@'`          #-- array accesses
    S=""
    if [[ $Ns_M == *@* ]]; then S=`echo $Ns_M | cut -f2 -d'@'`; fi  #-- stride
    #echo $S

    printf "\n\t=========================================================\n"
    printf "\t==================== Test [$cache miss] ====================\n"
    printf "\t========================================================="

    for N in $Ns
    do
      if [[ x$S == x ]]; then
        m_stride=$N
        for (( s=1; s<=$m_stride; s*=2 ))
        do
            one_run $N $s $M $cache
        done
      else
        one_run $N $S $M $cache
      fi
    done
    printf "\n"
}
done
