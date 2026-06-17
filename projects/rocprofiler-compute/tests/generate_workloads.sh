#!/bin/bash
declare -A commands=(
    [path]=' '
    [no_roof]='--no-roof'
    [kernel_names]='--roof-only'
    [device_filter]='--device 0'
    [kernel]='--kernel vecCopy'
    [ipblocks_SQ]='-b cu_ins cu_pipe'
    [ipblocks_SQC]='-b ins_cache sl1d'
    [ipblocks_TA]='-b tatd'
    [ipblocks_TD]='-b tatd'
    [ipblocks_TCP]='-b vl1d'
    [ipblocks_TCC]='-b l2 l2_per_channel'
    [ipblocks_SPI]='-b spi'
    [ipblocks_CPC]='-b cpc'
    [ipblocks_CPF]='-b cpc'
    [ipblocks_SQ_CPC]='-b cu_ins cu_pipe cpc'
    [ipblocks_SQ_TA]='-b cu_ins cu_pipe tatd'
    [ipblocks_SQ_SPI]='-b cu_ins cu_pipe spi'
    [ipblocks_SQ_SQC_TCP_CPC]='-b cu_ins cu_pipe ins_cache sl1d vl1d cpc'
    [ipblocks_SQ_SPI_TA_TCC_CPF]='-b cu_ins cu_pipe spi tatd l2 l2_per_channel cpc'
    [dispatch_0]='--dispatch 1'
    [dispatch_0_1]='--dispatch 1:3'
    [dispatch_2]='--dispatch 2'
    [kernel_verbose_0]='--kernel-verbose 0'
    [kernel_verbose_1]='--kernel-verbose 1'
    [kernel_verbose_2]='--kernel-verbose 2'
    [kernel_verbose_3]='--kernel-verbose 3'
    [kernel_verbose_4]='--kernel-verbose 4'
    [kernel_verbose_5]='--kernel-verbose 5'
    [join_type_grid]='--join-type grid'
    [join_type_kernel]='--join-type kernel'
    [sort_dispatches]='--roof-only --sort dispatches'
    [sort_kernels]='--roof-only --sort kernels'
    [mem_levels_HBM]='--roof-only --mem-level HBM'
    [mem_levels_L2]='--roof-only --mem-level L2'
    [mem_levels_vL1D]='--roof-only --mem-level vL1D'
    [mem_levels_LDS]='--roof-only --mem-level LDS'
    [mem_levels_HBM_LDS]='--roof-only --mem-level HBM LDS'
    [mem_levels_vL1d_LDS]='--roof-only --mem-level vL1D LDS'
    [mem_levels_L2_vL1d_LDS]='--roof-only --mem-level L2 vL1D LDS'
    #########################################################
    #         Attempt to break rocprofiler-compute          #
    #########################################################
    [dispatch_7]='--dispatch 7'
    [dispatch_6_8]='--dispatch 6:8'
    [dispatch_inv]='--dispatch invalid'
    [kernel_substr]='--kernel vecCopy'
    [kernel_inv_str]='--kernel vecPaste'
    [kernel_inv_int]='--kernel 42'
    [device_inv]='--device invalid' # does not generate a workload
    [device_inv_int]='--device -1'
    )
soc=MI300X_A1
echo "starting"
for key in "${!commands[@]}"; do
    echo profiling $key;
    command="${commands[$key]}"
    echo "$key = ./src/rocprof-compute profile -n $key ${dirs[@]}"
    ./src/rocprof-compute profile -n $key $command --output-directory tests/workloads/$key/$soc  -- ./tests/vcopy -n 1048576 -b 256 -i 3 ;
echo "done" ; done
