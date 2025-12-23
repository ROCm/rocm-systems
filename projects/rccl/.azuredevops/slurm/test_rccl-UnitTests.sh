#!/bin/bash
#SBATCH --job-name=rccl-UnitTests
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gt

short_id=$(hostname | cut -d'.' -f1 | cut -d'-' -f3-)
echo "Node identifier: $short_id"

source /etc/profile.d/lmod.sh
module load rocm/6.4.1
cd "$BINARIES_DIR/bin"
LD_LIBRARY_PATH="$BINARIES_DIR/lib:$LD_LIBRARY_PATH" NCCL_DEBUG=INFO RCCL_ENABLE_SIGNALHANDLER=1 HSA_NO_SCRATCH_RECLAIM=1 ./rccl-UnitTests --gtest_output=xml:$PIPELINE_WORKSPACE/rccl-UnitTests_output.xml --gtest_color=yes
