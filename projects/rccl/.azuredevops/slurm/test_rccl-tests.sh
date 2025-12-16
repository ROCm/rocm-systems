#!/bin/bash
#SBATCH --job-name=rccl-tests
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gt

short_id=$(hostname | cut -d'.' -f1 | cut -d'-' -f3-)
echo "Node identifier: $short_id"

source /etc/profile.d/lmod.sh
module load rocm/6.4.1

cd ${PIPELINE_WORKSPACE}/TestResults
mkdir -p ${PIPELINE_WORKSPACE}/TestResults/rccl-tests_logs
export WORKDIR=${PIPELINE_WORKSPACE}/TestResults/rccl-tests_logs

export PATH="$BINARIES_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$BINARIES_DIR/lib:$LD_LIBRARY_PATH"

### create hostlist
#nodelist=($(scontrol show hostnames))
#echo "SLURM nodes:"
#echo ${nodelist[@]}
#echo ""
#
#hosts_8ppn=()
#for node in "${nodelist[@]}"
#do
#    hosts_8ppn+=("${node}:8")
#done
#echo ${hosts_8ppn[@]}

### Run multi- and single-node RCCL-Tests
## Run single-node RCCL-Tests
for n in 1
do
    total=$((n*8))
    #h_8ppn=`echo ${hosts_8ppn[@]:0:${n}} | tr ' ' ','`

    for coll in all_reduce all_gather reduce_scatter alltoall alltoallv broadcast gather reduce scatter sendrecv
    do
        for dtype in float bfloat16 half fp8_e5m2
        do
            out_filename="${WORKDIR}/rccl-tests_${coll}_1KB-16GB_nodes${n}_gpus${total}_${dtype}.log"
            #cmd="${MPI_HOME}/bin/mpirun -np ${total} --host ${h_8ppn} -mca pml ob1 -mca btl ^openib -mca oob_tcp_if_exclude docker,lo -mca btl_tcp_if_exclude docker,lo -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=VERSION -x NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7 -x NCCL_IGNORE_CPU_AFFINITY=1 -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_IB_GID_INDEX=3 ${BINARIES_DIR}/bin/${coll}_perf -b 1K -e 16G -f 2 -g 1 -n 100 -w 50 -d ${dtype} -Z json -x ${WORKDIR}/rccl-tests_${coll}_nodes${n}_gpus${total}_${dtype}.json"
            cmd="${MPI_HOME}/bin/mpirun -np ${total} -mca pml ^ucx -mca osc ^ucx -mca btl ^openib -mca oob_tcp_if_exclude docker,lo -mca btl_tcp_if_exclude docker,lo -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=VERSION -x NCCL_IGNORE_CPU_AFFINITY=1 -x HSA_NO_SCRATCH_RECLAIM=1 ${BINARIES_DIR}/bin/${coll}_perf -b 1K -e 16G -f 2 -g 1 -n 100 -w 50 -d ${dtype} -Z json -x ${WORKDIR}/rccl-tests_${coll}_nodes${n}_gpus${total}_${dtype}.json"

            echo "Running ${coll}" 2>&1 | tee ${out_filename}
            echo "Run cmd: ${cmd}" 2>&1 | tee -a ${out_filename}
            eval ${cmd} 2>&1 | tee -a ${out_filename}

            sleep 2
        done
    done
done

## To add
### Summarize results
### Convert to junit
